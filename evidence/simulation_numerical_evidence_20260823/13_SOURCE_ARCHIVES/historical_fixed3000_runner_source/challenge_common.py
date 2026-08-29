from __future__ import annotations

import importlib.util
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from benchmark_core import (
    MAX_STEPS,
    MAX_VOLUME_ML,
    MIN_VOLUME_ML,
    NeuralVolumePolicy,
    PolicyEnvironment,
    StressScenario,
    Task,
    _unwrap_state_dict,
    generate_tasks,
    seed_everything,
)


MULTIPLIERS = np.asarray([0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00], dtype=np.float32)


SCENARIOS: dict[str, StressScenario] = {
    "nominal": StressScenario("nominal"),
    "analyte_low": StressScenario("analyte_low", analyte_conc_scale=0.30),
    "analyte_high": StressScenario("analyte_high", analyte_conc_scale=3.00),
    "volume_small": StressScenario("volume_small", initial_volume_scale=0.50),
    "volume_large": StressScenario("volume_large", initial_volume_scale=2.00),
    "titrant_low": StressScenario("titrant_low", titrant_conc_scale=0.50),
    "actuator_under": StressScenario("actuator_under", actuator_scale=0.75),
    "actuator_over": StressScenario("actuator_over", actuator_scale=1.25),
    "actuator_random": StressScenario("actuator_random", actuator_log_sd=0.15),
    "noise_003": StressScenario("noise_003", measurement_noise_sd=0.03),
    "noise_005": StressScenario("noise_005", measurement_noise_sd=0.05),
    "noise_010": StressScenario("noise_010", measurement_noise_sd=0.10),
    "bias_010": StressScenario("bias_010", sensor_bias_sd=0.10),
    "drift_001": StressScenario("drift_001", drift_sd_per_step=0.01),
    "partial_response": StressScenario("partial_response", response_fraction=0.60),
    "tetraprotic": StressScenario("tetraprotic", acid_family="tetraprotic"),
    "out_of_range": StressScenario("out_of_range", acid_family="outside_pka_range"),
    "close_pka": StressScenario("close_pka", acid_family="close_pka"),
    # These composite conditions are held out from training and are not selected post hoc.
    "low_conc_noise": StressScenario("low_conc_noise", analyte_conc_scale=0.30, measurement_noise_sd=0.05),
    "high_conc_under": StressScenario("high_conc_under", analyte_conc_scale=3.00, actuator_scale=0.75),
    "large_volume_drift": StressScenario("large_volume_drift", initial_volume_scale=2.00, drift_sd_per_step=0.01),
    "partial_bias": StressScenario("partial_bias", response_fraction=0.60, sensor_bias_sd=0.10),
    "tetra_noise": StressScenario("tetra_noise", acid_family="tetraprotic", measurement_noise_sd=0.03),
    "close_random_actuator": StressScenario("close_random_actuator", acid_family="close_pka", actuator_log_sd=0.15),
}

TRAIN_SCENARIOS = [
    "nominal", "analyte_low", "analyte_high", "volume_small", "volume_large", "titrant_low",
    "actuator_under", "actuator_over", "actuator_random", "noise_003", "noise_005", "bias_010",
    "drift_001", "partial_response", "tetraprotic", "out_of_range", "close_pka",
]
CONFIRM_STRESS_SCENARIOS = [
    "nominal", "analyte_low", "analyte_high", "volume_small", "volume_large", "titrant_low",
    "actuator_under", "actuator_over", "actuator_random", "noise_005", "noise_010", "bias_010",
    "drift_001", "partial_response", "tetraprotic", "out_of_range", "close_pka",
    "low_conc_noise", "high_conc_under", "large_volume_drift", "partial_bias", "tetra_noise", "close_random_actuator",
]


def load_bayesian_module(path: Path):
    spec = importlib.util.spec_from_file_location("challenge_bayesian_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Bayesian controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_task_pool(seed: int, total_tasks: int, robust: bool) -> list[tuple[Task, StressScenario]]:
    names = TRAIN_SCENARIOS if robust else ["nominal"]
    base_count, remainder = divmod(total_tasks, len(names))
    pool: list[tuple[Task, StressScenario]] = []
    for index, name in enumerate(names):
        scenario = SCENARIOS[name]
        count = base_count + int(index < remainder)
        if count == 0:
            continue
        tasks = generate_tasks(seed * 1009 + index * 7919, count, scenario)
        pool.extend((task, scenario) for task in tasks)
    rng = np.random.default_rng(seed + 937)
    rng.shuffle(pool)
    return pool


def make_features(env: Any, mode: str, base_volume: float | None = None, history: deque | None = None) -> np.ndarray:
    raw = np.asarray(env.state(), dtype=np.float32)
    if mode == "basic":
        features = raw
    else:
        total_volume = float(getattr(env, "acid_added_ml", 0.0) + getattr(env, "base_added_ml", 0.0))
        overshoots = float(getattr(env, "overshoots", 0.0))
        maximum = float(env.maximum_requested_volume())
        secondary = float(getattr(env, "use_secondary", False))
        extra = np.asarray([
            float(getattr(env, "steps", 0.0)) / MAX_STEPS,
            overshoots / 10.0,
            total_volume / 50.0,
            maximum / MAX_VOLUME_ML,
            secondary,
        ], dtype=np.float32)
        features = np.concatenate([raw, extra])
        if mode == "history":
            if history is None:
                history = deque(maxlen=3)
            prior = list(history)
            while len(prior) < 3:
                prior.insert(0, np.zeros(5, dtype=np.float32))
            features = np.concatenate([features] + prior[-3:])
        elif mode == "filtered":
            prior = list(history or [])[-3:]
            window = prior + [raw]
            while len(window) < 4:
                window.insert(0, raw)
            values = np.asarray(window, dtype=np.float32)
            ph = values[:, 0]
            error = values[:, 3]
            delta = values[:, 2]
            volume = values[:, 4]
            slope = float((-1.5 * error[0] - 0.5 * error[1] + 0.5 * error[2] + 1.5 * error[3]) / 5.0)
            filtered = np.asarray([
                np.median(ph), np.mean(ph), np.std(ph),
                np.median(error), np.mean(error), np.std(error),
                np.mean(delta), np.std(delta), delta[-1],
                np.mean(volume), np.std(volume), volume[-1],
                slope, np.ptp(ph), float(np.sign(error[-1]) == np.sign(np.median(error))),
            ], dtype=np.float32)
            features = np.concatenate([features, filtered])
    if base_volume is not None:
        maximum = max(MIN_VOLUME_ML, float(env.maximum_requested_volume()))
        features = np.concatenate([features, np.asarray([base_volume / MAX_VOLUME_ML, base_volume / maximum], dtype=np.float32)])
    return features.astype(np.float32, copy=False)


def feature_dim(mode: str, residual: bool = False) -> int:
    base = {"basic": 5, "augmented": 10, "history": 25, "filtered": 25}[mode]
    return base + (2 if residual else 0)


class CategoricalActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )


class GaussianActor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.7], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        mean = self.mean(hidden)
        return mean, self.log_std.expand_as(mean).clamp(-4.0, 1.0)


class DeterministicActor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return (self.net(state) + 1.0) / 2.0


class QNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action_fraction: torch.Tensor) -> torch.Tensor:
        if action_fraction.ndim == 1:
            action_fraction = action_fraction.unsqueeze(-1)
        return self.net(torch.cat([state, action_fraction], dim=-1))


def initialize_from_imitation(model: CategoricalActorCritic, weights: Path, device: torch.device) -> None:
    payload = torch.load(weights, map_location=device)
    state = _unwrap_state_dict(payload)
    with torch.no_grad():
        source_w = state["net.0.weight"]
        source_b = state["net.0.bias"]
        source_w2 = state["net.2.weight"]
        source_b2 = state["net.2.bias"]
        source_w3 = state["net.4.weight"]
        source_b3 = state["net.4.bias"]
        actor_layers = [model.actor[0], model.actor[2], model.actor[4]]
        actor_layers[0].weight.zero_()
        actor_layers[0].weight[:, : min(source_w.shape[1], actor_layers[0].weight.shape[1])].copy_(source_w[:, : actor_layers[0].weight.shape[1]])
        actor_layers[0].bias.copy_(source_b)
        if actor_layers[1].weight.shape == source_w2.shape:
            actor_layers[1].weight.copy_(source_w2)
            actor_layers[1].bias.copy_(source_b2)
        if actor_layers[2].weight.shape == source_w3.shape:
            actor_layers[2].weight.copy_(source_w3)
            actor_layers[2].bias.copy_(source_b3)


def categorical_action(model: CategoricalActorCritic, features: np.ndarray, max_volume: float, device: torch.device, stochastic: bool) -> tuple[int, float, float, float]:
    limit = max(1, min(1000, int(math.floor(max_volume / 0.01 + 1e-9))))
    state = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model.actor(state)[:, :limit]
    distribution = torch.distributions.Categorical(logits=logits)
    action_tensor = distribution.sample() if stochastic else torch.argmax(logits, dim=1)
    action = int(action_tensor.item())
    return action, (action + 1) * 0.01, float(distribution.log_prob(action_tensor).item()), float(model.critic(state).item())


def residual_action(model: CategoricalActorCritic, features: np.ndarray, base_volume: float, max_volume: float, device: torch.device, stochastic: bool) -> tuple[int, float, float, float]:
    state = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model.actor(state)
    distribution = torch.distributions.Categorical(logits=logits)
    action_tensor = distribution.sample() if stochastic else torch.argmax(logits, dim=1)
    action = int(action_tensor.item())
    volume = float(np.clip(base_volume * MULTIPLIERS[action], MIN_VOLUME_ML, max_volume))
    return action, volume, float(distribution.log_prob(action_tensor).item()), float(model.critic(state).item())


def continuous_action(actor: GaussianActor, features: np.ndarray, max_volume: float, device: torch.device, stochastic: bool) -> tuple[float, float]:
    state = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    mean, log_std = actor(state)
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = normal.sample() if stochastic else mean
    squashed = torch.tanh(raw)
    fraction = float(((squashed + 1.0) / 2.0).clamp(0.0, 1.0).item())
    log_prob = float((normal.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + 1e-6)).sum().item())
    volume = MIN_VOLUME_ML + (max_volume - MIN_VOLUME_ML) * fraction
    return volume, log_prob


def deterministic_action(actor: DeterministicActor, features: np.ndarray, max_volume: float, device: torch.device, noise_sd: float = 0.0) -> tuple[float, float]:
    state = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    fraction = float(actor(state).clamp(0.0, 1.0).item())
    if noise_sd > 0.0:
        fraction = float(np.clip(fraction + np.random.normal(0.0, noise_sd), 0.0, 1.0))
    volume = MIN_VOLUME_ML + (max_volume - MIN_VOLUME_ML) * fraction
    return volume, fraction


def reward_value(previous_error: float, current_error: float, volume: float, overshoot: bool, done: bool, true_success: bool, measured_success: bool, steps: int, risk_aware: bool, profile: str = "standard") -> float:
    progress = float(np.clip(previous_error - current_error, -3.0, 3.0))
    conservative = profile == "conservative"
    reward = progress - (0.008 if conservative else 0.005)
    reward -= (0.006 if conservative else (0.004 if risk_aware else 0.001)) * volume
    reward -= (0.45 if conservative else (0.25 if risk_aware else 0.08)) * float(overshoot)
    if done:
        if true_success:
            reward += (11.0 if conservative else 8.0) + max(0.0, (MAX_STEPS - steps) / MAX_STEPS)
        else:
            reward -= 8.0 if conservative else 5.0
        if measured_success and not true_success:
            reward -= 10.0 if conservative else 4.0
        if conservative:
            reward -= 2.0 * min(current_error, 2.0)
    return float(np.clip(reward, -10.0, 10.0))


@dataclass
class BayesianStep:
    volume: float
    reagent: str


class BayesianAdapter:
    """Runs the submitted Bayesian controller against the common stress environment."""

    def __init__(self, module_or_path: Any, particles: int, seed: int) -> None:
        self.module = load_bayesian_module(Path(module_or_path)) if isinstance(module_or_path, (str, Path)) else module_or_path
        self.particles = particles
        self.seed = seed
        self.controller = None

    def reset(self, task: Task) -> None:
        np.random.seed(int(self.seed) % (2**32 - 1))
        self.controller = self.module.PHAdjustmentEnv(num_particles=self.particles)
        self.controller.initialize(task.acid_type, list(task.pka_values), task.initial_ph, task.target_ph, self.module.MAX_STEPS)

    def select(self, env: PolicyEnvironment) -> BayesianStep:
        assert self.controller is not None
        measured = float(env.measured_ph)
        last_measured = getattr(self.controller, "last_measured_ph", None)
        if last_measured is None or not math.isclose(float(last_measured), measured, rel_tol=0.0, abs_tol=1e-12):
            self.controller.update_exp_ph(measured)
        base_reagent, base_volume = self.controller.select_best_action()[0]
        base_conc = float(self.controller.reagents[base_reagent])
        external_conc = float(env.secondary_titrant_m if env.use_secondary else env.primary_titrant_m)
        volume = base_volume * base_conc / max(external_conc, 1e-12)
        volume = float(np.clip(volume, MIN_VOLUME_ML, env.maximum_requested_volume()))
        return BayesianStep(volume=volume, reagent=base_reagent)

    def observe(self, env: PolicyEnvironment, requested_volume: float, reagent: str) -> None:
        assert self.controller is not None
        self.controller.done = False
        try:
            self.controller.step((reagent, float(requested_volume)), mode="Simulate")
        except Exception:
            pass
        self.controller.current_ph = float(env.measured_ph)
        self.controller.last_measured_ph = float(env.measured_ph)
        self.controller.prev_measured_ph = float(env.previous_measured_ph)
        self.controller.done = False
        self.controller.update_posteriors((reagent, float(requested_volume)), float(env.measured_ph))
        self.controller.done = bool(env.done)


def bayesian_feature_vector(controller: Any, base_volume: float, max_volume: float, mode: str) -> np.ndarray:
    raw = np.asarray(controller.get_state(), dtype=np.float32)
    if mode == "basic":
        features = raw
    else:
        total = float(controller.acid_volume + controller.base_volume)
        extra = np.asarray([
            float(controller.steps_taken) / MAX_STEPS,
            float(controller.oscillation_count) / 10.0,
            total / 50.0,
            max_volume / MAX_VOLUME_ML,
            float(controller.use_secondary_reagents),
        ], dtype=np.float32)
        features = np.concatenate([raw, extra])
        if mode == "history":
            # The residual controller is trained with the same fixed-length format;
            # zeros are used for the unavailable pre-history in the external teacher.
            features = np.concatenate([features, np.zeros(15, dtype=np.float32)])
    maximum = max(MIN_VOLUME_ML, max_volume)
    return np.concatenate([features, np.asarray([base_volume / MAX_VOLUME_ML, base_volume / maximum], dtype=np.float32)]).astype(np.float32)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
