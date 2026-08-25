from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - neural analyses require PyTorch.
    torch = None
    nn = None


SUCCESS_THRESHOLD = 0.1
MAX_STEPS = 50
PRIMARY_TITRANT_M = 0.1
SECONDARY_TITRANT_M = 0.01
MIN_VOLUME_ML = 0.01
MAX_VOLUME_ML = 10.0


@dataclass(frozen=True)
class Task:
    seed: int
    task_id: int
    acid_type: str
    pka_values: tuple[float, ...]
    initial_ph: float
    target_ph: float
    initial_volume_ml: float = 11.0
    analyte_conc_m: float = 0.1


@dataclass(frozen=True)
class StressScenario:
    name: str
    analyte_conc_scale: float = 1.0
    initial_volume_scale: float = 1.0
    titrant_conc_scale: float = 1.0
    actuator_scale: float = 1.0
    actuator_log_sd: float = 0.0
    measurement_noise_sd: float = 0.0
    sensor_bias_sd: float = 0.0
    drift_sd_per_step: float = 0.0
    response_fraction: float = 1.0
    acid_family: str = "nominal"


@dataclass
class EpisodeResult:
    seed: int
    task_id: int
    scenario: str
    method: str
    acid_type: str
    pka_values: str
    initial_ph: float
    target_ph: float
    final_true_ph: float
    final_measured_ph: float
    steps: int
    true_success: bool
    measured_success: bool
    overshoots: int
    acid_added_ml: float
    base_added_ml: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def acid_charge_factor(ph: float, pka_values: Sequence[float]) -> float:
    hydrogen = 10.0 ** (-ph)
    kas = [10.0 ** (-float(value)) for value in sorted(pka_values)]
    n = len(kas)
    coefficients = [1.0]
    running = 1.0
    for ka in kas:
        running *= ka
        coefficients.append(running)
    terms = [coefficients[index] * hydrogen ** (n - index) for index in range(n + 1)]
    denominator = sum(terms)
    if denominator == 0:
        return 0.0
    return sum(index * terms[index] for index in range(n + 1)) / denominator


def solve_ph(
    pka_values: Sequence[float],
    initial_volume_ml: float,
    analyte_conc_m: float,
    base_moles: float = 0.0,
    acid_moles: float = 0.0,
    added_volume_ml: float = 0.0,
) -> float:
    total_volume_l = (initial_volume_ml + added_volume_ml) / 1000.0
    analyte_moles = initial_volume_ml / 1000.0 * analyte_conc_m
    c_analyte = analyte_moles / total_volume_l
    c_na = base_moles / total_volume_l
    c_hcl = acid_moles / total_volume_l

    def balance(ph: float) -> float:
        hydrogen = 10.0 ** (-ph)
        hydroxide = 1e-14 / hydrogen
        negative_charge = c_analyte * acid_charge_factor(ph, pka_values)
        return hydrogen + c_na - hydroxide - c_hcl - negative_charge

    lo, hi = 0.0, 14.0
    f_lo = balance(lo)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = balance(mid)
        if abs(f_mid) < 1e-12:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def _sample_pkas(rng: np.random.Generator, family: str) -> tuple[str, tuple[float, ...]]:
    if family == "tetraprotic":
        return "tetraprotic", (
            float(rng.uniform(1.5, 3.0)),
            float(rng.uniform(3.0, 5.0)),
            float(rng.uniform(5.0, 7.0)),
            float(rng.uniform(7.0, 9.0)),
        )
    if family == "outside_pka_range":
        acid_type = str(rng.choice(["monoprotic", "diprotic", "triprotic"]))
        if acid_type == "monoprotic":
            return acid_type, (float(rng.choice([rng.uniform(0.5, 1.8), rng.uniform(6.2, 9.0)])),)
        if acid_type == "diprotic":
            return acid_type, (float(rng.uniform(0.5, 2.0)), float(rng.uniform(7.0, 10.0)))
        return acid_type, (
            float(rng.uniform(0.5, 2.0)),
            float(rng.uniform(3.0, 6.0)),
            float(rng.uniform(8.0, 11.0)),
        )
    if family == "close_pka":
        center = float(rng.uniform(3.0, 7.0))
        count = int(rng.choice([2, 3]))
        offsets = np.sort(rng.uniform(-0.35, 0.35, size=count))
        label = "diprotic" if count == 2 else "triprotic"
        return label, tuple(float(center + offset) for offset in offsets)

    acid_type = str(rng.choice(["monoprotic", "diprotic", "triprotic"]))
    if acid_type == "monoprotic":
        return acid_type, (float(rng.uniform(2.0, 6.0)),)
    if acid_type == "diprotic":
        return acid_type, (float(rng.uniform(2.0, 4.0)), float(rng.uniform(4.0, 7.0)))
    return acid_type, (
        float(rng.uniform(2.0, 4.0)),
        float(rng.uniform(4.0, 6.0)),
        float(rng.uniform(6.0, 8.0)),
    )


def generate_tasks(seed: int, count: int, scenario: StressScenario | None = None) -> list[Task]:
    scenario = scenario or StressScenario("nominal")
    rng = np.random.default_rng(seed)
    tasks: list[Task] = []
    for task_id in range(1, count + 1):
        acid_type, pka_values = _sample_pkas(rng, scenario.acid_family)
        initial_volume_ml = 11.0 * scenario.initial_volume_scale
        analyte_conc_m = 0.1 * scenario.analyte_conc_scale
        initial_ph = solve_ph(pka_values, initial_volume_ml, analyte_conc_m)
        target_ph = float(np.round(rng.uniform(2.0, 11.0), 2))
        tasks.append(
            Task(
                seed=seed,
                task_id=task_id,
                acid_type=acid_type,
                pka_values=tuple(float(value) for value in pka_values),
                initial_ph=float(np.round(initial_ph, 2)),
                target_ph=target_ph,
                initial_volume_ml=initial_volume_ml,
                analyte_conc_m=analyte_conc_m,
            )
        )
    return tasks


def load_tasks_csv(path: Path, seed: int) -> list[Task]:
    import ast
    import csv

    tasks: list[Task] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            raw = row.get("Acid_Params", row.get("Acid params", ""))
            parsed = ast.literal_eval(raw)
            values = parsed if isinstance(parsed, list) else [parsed]
            tasks.append(
                Task(
                    seed=seed,
                    task_id=int(row.get("Experiment", index)),
                    acid_type=row.get("Acid_Type", row.get("Acid type", "unknown")).lower(),
                    pka_values=tuple(float(value) for value in values),
                    initial_ph=float(row.get("Initial_pH", row.get("Initial p h", "nan"))),
                    target_ph=float(row.get("Target_pH", row.get("Target p h", "nan"))),
                )
            )
    return tasks


if nn is not None:
    class DiscreteVolumeRegressor(nn.Module):
        def __init__(self, input_dim: int = 5) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1000),
            )

        def forward(self, inputs):
            return self.net(inputs)
else:  # pragma: no cover
    class DiscreteVolumeRegressor:  # type: ignore[no-redef]
        pass


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("The weight file does not contain a state dictionary.")
    for key in ("state_dict", "model_state_dict", "policy_state_dict", "actor_state_dict"):
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        new_key = str(key)
        for prefix in ("module.", "policy_model.", "model.", "actor."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


class NeuralVolumePolicy:
    def __init__(self, weights: Path, device: str = "cpu") -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for neural-policy evaluation.")
        self.device = torch.device(device)
        self.model = DiscreteVolumeRegressor().to(self.device)
        payload = torch.load(weights, map_location=self.device)
        state_dict = _unwrap_state_dict(payload)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.volumes = np.round(np.arange(1, 1001) * 0.01, 2)

    def select_volume(self, state: np.ndarray, maximum_ml: float = MAX_VOLUME_ML) -> float:
        upper_index = max(1, min(1000, int(math.floor(maximum_ml / 0.01 + 1e-9))))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(tensor)[0, :upper_index]
            index = int(torch.argmax(logits).item())
        return float(self.volumes[index])


class PolicyEnvironment:
    def __init__(self, task: Task, scenario: StressScenario, rng: np.random.Generator) -> None:
        self.task = task
        self.scenario = scenario
        self.rng = rng
        self.target_ph = task.target_ph
        self.primary_titrant_m = PRIMARY_TITRANT_M * scenario.titrant_conc_scale
        self.secondary_titrant_m = SECONDARY_TITRANT_M * scenario.titrant_conc_scale
        self.base_moles = 0.0
        self.acid_moles = 0.0
        self.base_added_ml = 0.0
        self.acid_added_ml = 0.0
        self.steps = 0
        self.overshoots = 0
        self.last_action_volume = 0.0
        self.overshoot_threshold_ml: float | None = None
        self.oscillation_count = 0
        self.use_secondary = False
        self.sensor_bias = float(rng.normal(0.0, scenario.sensor_bias_sd))
        self.sensor_drift = 0.0
        self.true_ph = solve_ph(task.pka_values, task.initial_volume_ml, task.analyte_conc_m)
        self.previous_true_ph = self.true_ph
        self.measured_ph = self._measure(self.true_ph, previous=None)
        self.previous_measured_ph = self.measured_ph
        self.done = abs(self.measured_ph - self.target_ph) <= SUCCESS_THRESHOLD

    def _measure(self, true_ph: float, previous: float | None) -> float:
        self.sensor_drift += float(self.rng.normal(0.0, self.scenario.drift_sd_per_step))
        reference = true_ph if previous is None else previous
        response = reference + self.scenario.response_fraction * (true_ph - reference)
        measured = (
            response
            + self.sensor_bias
            + self.sensor_drift
            + float(self.rng.normal(0.0, self.scenario.measurement_noise_sd))
        )
        return float(np.clip(measured, 0.0, 14.0))

    def state(self) -> np.ndarray:
        delta = self.measured_ph - self.previous_measured_ph
        error = self.measured_ph - self.target_ph
        return np.array(
            [self.measured_ph, self.target_ph, delta, error, self.last_action_volume],
            dtype=np.float32,
        )

    def maximum_requested_volume(self) -> float:
        if self.overshoot_threshold_ml is None:
            return MAX_VOLUME_ML
        return max(MIN_VOLUME_ML, min(MAX_VOLUME_ML, self.overshoot_threshold_ml))

    def step(self, requested_volume_ml: float) -> None:
        if self.done:
            return
        requested = float(np.clip(requested_volume_ml, MIN_VOLUME_ML, self.maximum_requested_volume()))
        actual = requested * self.scenario.actuator_scale
        if self.scenario.actuator_log_sd > 0:
            actual *= float(self.rng.lognormal(mean=0.0, sigma=self.scenario.actuator_log_sd))
        actual = max(MIN_VOLUME_ML, actual)
        concentration = self.secondary_titrant_m if self.use_secondary else self.primary_titrant_m
        add_base = self.measured_ph < self.target_ph

        self.previous_true_ph = self.true_ph
        self.previous_measured_ph = self.measured_ph
        if add_base:
            self.base_moles += concentration * actual / 1000.0
            self.base_added_ml += actual
        else:
            self.acid_moles += concentration * actual / 1000.0
            self.acid_added_ml += actual
        total_added_ml = self.base_added_ml + self.acid_added_ml
        self.true_ph = solve_ph(
            self.task.pka_values,
            self.task.initial_volume_ml,
            self.task.analyte_conc_m,
            base_moles=self.base_moles,
            acid_moles=self.acid_moles,
            added_volume_ml=total_added_ml,
        )
        self.measured_ph = self._measure(self.true_ph, previous=self.previous_measured_ph)
        self.steps += 1
        self.last_action_volume = requested

        sign_change = (self.previous_true_ph - self.target_ph) * (self.true_ph - self.target_ph) < 0
        error_increased = abs(self.true_ph - self.target_ph) > abs(self.previous_true_ph - self.target_ph)
        if sign_change:
            self.overshoots += 1
        if sign_change or error_increased:
            candidate = max(actual / 2.0, MIN_VOLUME_ML)
            if self.overshoot_threshold_ml is None:
                self.overshoot_threshold_ml = candidate
            else:
                self.overshoot_threshold_ml = min(self.overshoot_threshold_ml, candidate)

        if requested <= MIN_VOLUME_ML + 1e-9 and sign_change and abs(self.true_ph - self.previous_true_ph) > 0.1:
            self.oscillation_count += 1
            if self.oscillation_count >= 3:
                self.use_secondary = True

        if abs(self.measured_ph - self.target_ph) <= SUCCESS_THRESHOLD or self.steps >= MAX_STEPS:
            self.done = True


def run_neural_policy(
    policy: NeuralVolumePolicy,
    task: Task,
    scenario: StressScenario,
    method: str,
    rng_seed: int,
) -> EpisodeResult:
    env = PolicyEnvironment(task, scenario, np.random.default_rng(rng_seed))
    while not env.done:
        volume = policy.select_volume(env.state(), env.maximum_requested_volume())
        env.step(volume)
    return EpisodeResult(
        seed=task.seed,
        task_id=task.task_id,
        scenario=scenario.name,
        method=method,
        acid_type=task.acid_type,
        pka_values=json.dumps(task.pka_values),
        initial_ph=task.initial_ph,
        target_ph=task.target_ph,
        final_true_ph=env.true_ph,
        final_measured_ph=env.measured_ph,
        steps=env.steps,
        true_success=abs(env.true_ph - env.target_ph) <= SUCCESS_THRESHOLD,
        measured_success=abs(env.measured_ph - env.target_ph) <= SUCCESS_THRESHOLD,
        overshoots=env.overshoots,
        acid_added_ml=env.acid_added_ml,
        base_added_ml=env.base_added_ml,
    )


def summarize_results(results: Iterable[EpisodeResult]) -> dict[str, float | int]:
    rows = list(results)
    successful = [row for row in rows if row.true_success]
    successful_steps = [row.steps for row in successful]
    total_steps = sum(row.steps for row in rows)
    total_overshoots = sum(row.overshoots for row in rows)
    return {
        "tasks": len(rows),
        "successful_tasks": len(successful),
        "success_rate_percent": 100.0 * len(successful) / len(rows) if rows else math.nan,
        "successful_steps_mean": float(np.mean(successful_steps)) if successful_steps else math.nan,
        "successful_steps_sd": float(np.std(successful_steps, ddof=1)) if len(successful_steps) > 1 else 0.0,
        "overshoot_rate_percent": 100.0 * total_overshoots / total_steps if total_steps else 0.0,
        "final_abs_error_mean": float(np.mean([abs(row.final_true_ph - row.target_ph) for row in rows])) if rows else math.nan,
        "false_stop_rate_percent": 100.0 * sum(row.measured_success and not row.true_success for row in rows) / len(rows) if rows else math.nan,
    }


def exact_mcnemar(success_a: Sequence[bool], success_b: Sequence[bool]) -> dict[str, float | int]:
    from scipy.stats import binomtest

    if len(success_a) != len(success_b):
        raise ValueError("McNemar inputs must contain matched outcomes.")
    a_only = sum(bool(a) and not bool(b) for a, b in zip(success_a, success_b))
    b_only = sum(not bool(a) and bool(b) for a, b in zip(success_a, success_b))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    return {
        "a_success_b_fail": a_only,
        "a_fail_b_success": b_only,
        "discordant": discordant,
        "p_value_exact_two_sided": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted
