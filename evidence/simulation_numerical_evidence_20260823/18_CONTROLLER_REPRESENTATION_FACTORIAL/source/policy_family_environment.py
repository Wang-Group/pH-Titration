from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SUCCESS_TOLERANCE = 0.10
STRICT_TOLERANCE = 0.05
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0
SENSOR_RESOLUTION_PH = 0.01


def task_row(task) -> dict:
    if isinstance(task, dict):
        return dict(task)
    if is_dataclass(task):
        return asdict(task)
    return dict(vars(task))


def task_object(row: dict):
    return SimpleNamespace(**row)


def load_tasks(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [task_object(json.loads(line)) for line in handle if line.strip()]


def save_tasks(path: Path, tasks) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task_row(task), separators=(",", ":")) + "\n")


def load_independent_generator(path: Path):
    spec = importlib.util.spec_from_file_location("independent_j123_policy_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenericControlEnvironment:
    def __init__(self, task, rng: np.random.Generator, domain=None):
        self.task = task
        self.row = task_row(task)
        self.rng = rng
        self.domain = domain
        self.observation_noise_sd = 0.0 if domain is None else float(domain.observation_noise_sd)
        self.actuator_log_sd = 0.0 if domain is None else float(domain.actuator_log_sd)
        self.titrant_scale = 1.0 if domain is None else float(domain.titrant_scale)
        self.response_fraction = 1.0 if domain is None else float(domain.response_fraction)
        self.target_ph = float(self.row["target_ph"])
        self.base_moles = float(self.row["initial_base_moles"])
        self.acid_moles = 0.0
        self.base_added_ml = 0.0
        self.acid_added_ml = 0.0
        self.steps = 0
        self.overshoots = 0
        self.last_requested_volume_ml = 0.0
        self.true_ph = float(self.row["initial_ph"])
        self.previous_true_ph = self.true_ph
        self.measured_ph = self._measure(self.true_ph, self.true_ph)
        self.previous_measured_ph = self.measured_ph
        self.done = abs(self.measured_ph - self.target_ph) <= SUCCESS_TOLERANCE
        self.stop_reason = "initial_success" if self.done else "running"

    def _measure(self, true_ph: float, previous_measured: float) -> float:
        responded = previous_measured + self.response_fraction * (true_ph - previous_measured)
        noisy = responded + float(self.rng.normal(0.0, self.observation_noise_sd))
        clipped = float(np.clip(noisy, 0.0, 14.0))
        return float(np.round(clipped / SENSOR_RESOLUTION_PH) * SENSOR_RESOLUTION_PH)

    def _truth_ph(self) -> float:
        from chemistry_model import SolutionState, solve_ph_scalar

        state = SolutionState(
            float(self.row["initial_volume_ml"]) + self.base_added_ml + self.acid_added_ml,
            self.base_moles,
            self.acid_moles,
        )
        if "component_concentrations_m" not in self.row:
            return float(
                solve_ph_scalar(
                    float(self.row["analyte_conc_m"]),
                    self.row["pka_values"],
                    float(self.row["initial_volume_ml"]),
                    state,
                )
            )
        from independent_mixture_pf import solve_independent_ph_scalar

        return float(
            solve_independent_ph_scalar(
                np.asarray(self.row["component_concentrations_m"], dtype=float),
                np.asarray(self.row["pka_values"], dtype=float),
                float(self.row["initial_volume_ml"]),
                state,
            )
        )

    def state(self) -> np.ndarray:
        return np.asarray(
            [
                self.measured_ph,
                self.target_ph,
                self.measured_ph - self.previous_measured_ph,
                self.measured_ph - self.target_ph,
                self.last_requested_volume_ml,
            ],
            dtype=np.float32,
        )

    def step(self, requested_volume_ml: float) -> dict:
        if self.done:
            return {"crossed_target": False, "actual_volume_ml": 0.0}
        requested = float(np.clip(requested_volume_ml, 0.01, 10.0))
        actual = requested * float(self.rng.lognormal(0.0, self.actuator_log_sd))
        remaining = MAX_TOTAL_DOSE_ML - self.base_added_ml - self.acid_added_ml
        if remaining <= 1e-12:
            self.done = True
            self.stop_reason = "dose_limit"
            return {"crossed_target": False, "actual_volume_ml": 0.0}
        actual = float(np.clip(actual, 0.0, remaining))
        self.previous_true_ph = self.true_ph
        self.previous_measured_ph = self.measured_ph
        if self.measured_ph < self.target_ph:
            self.base_moles += 0.1 * self.titrant_scale * actual / 1000.0
            self.base_added_ml += actual
        else:
            self.acid_moles += 0.1 * self.titrant_scale * actual / 1000.0
            self.acid_added_ml += actual
        self.true_ph = self._truth_ph()
        self.measured_ph = self._measure(self.true_ph, self.previous_measured_ph)
        self.steps += 1
        self.last_requested_volume_ml = requested
        crossed = (self.previous_true_ph - self.target_ph) * (self.true_ph - self.target_ph) < 0.0
        self.overshoots += int(crossed)
        total_added = self.base_added_ml + self.acid_added_ml
        if abs(self.measured_ph - self.target_ph) <= SUCCESS_TOLERANCE:
            self.done = True
            self.stop_reason = "measured_success"
        elif self.steps >= MAX_STEPS:
            self.done = True
            self.stop_reason = "max_steps"
        elif total_added >= MAX_TOTAL_DOSE_ML - 1e-9:
            self.done = True
            self.stop_reason = "dose_limit"
        return {"crossed_target": crossed, "actual_volume_ml": actual}

    def metrics(self) -> dict:
        true_error = abs(self.true_ph - self.target_ph)
        measured_error = abs(self.measured_ph - self.target_ph)
        return {
            "true_success": int(true_error <= SUCCESS_TOLERANCE),
            "strict_success": int(true_error <= STRICT_TOLERANCE),
            "severe_failure": int(true_error > 0.50),
            "measured_success": int(measured_error <= SUCCESS_TOLERANCE),
            "false_stop": int(measured_error <= SUCCESS_TOLERANCE and true_error > SUCCESS_TOLERANCE),
            "steps": self.steps,
            "overshoots": self.overshoots,
            "final_abs_error": true_error,
            "total_volume_ml": self.base_added_ml + self.acid_added_ml,
            "acid_added_ml": self.acid_added_ml,
            "base_added_ml": self.base_added_ml,
            "final_true_ph": self.true_ph,
            "final_measured_ph": self.measured_ph,
            "stop_reason": self.stop_reason,
        }


def actor_volume(actor, normalizer, state, device, stochastic=False) -> float:
    import torch

    tensor = torch.as_tensor(
        normalizer.transform_numpy(state), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        logits = actor(tensor)
        if stochastic:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            action = int(torch.argmax(logits, dim=1).item())
    return (action + 1) * 0.01


def evaluate_actor(actor, normalizer, tasks, device, seed_offset=0, domain=None):
    rows = []
    actor.eval()
    for task in tasks:
        row = task_row(task)
        seed = seed_offset + int(row["seed"]) * 1_000_003 + int(row["task_id"])
        env = GenericControlEnvironment(task, np.random.default_rng(seed), domain)
        while not env.done:
            env.step(actor_volume(actor, normalizer, env.state(), device))
        rows.append(
            {
                "task_seed": int(row["seed"]),
                "task_id": int(row["task_id"]),
                "acid_type": row["acid_type"],
                "difficulty": row["difficulty"],
                "direction": row["direction"],
                "pka_family": row["pka_family"],
                "true_pair_count": int(row.get("component_count", len(row["pka_values"]))),
                "true_concentration_m": float(row["analyte_conc_m"]),
                "initial_volume_ml": float(row["initial_volume_ml"]),
                "initial_ph": float(row["initial_ph"]),
                "target_ph": float(row["target_ph"]),
                **env.metrics(),
            }
        )
    return rows
