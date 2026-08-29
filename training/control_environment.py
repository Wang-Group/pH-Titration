from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .chemistry_model import SolutionState, solve_ph_scalar
    from .task_distribution import ControlTask
except ImportError:  # pragma: no cover - direct script compatibility
    from chemistry_model import SolutionState, solve_ph_scalar
    from task_distribution import ControlTask


SUCCESS_TOLERANCE = 0.10
STRICT_TOLERANCE = 0.05
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0
SENSOR_RESOLUTION_PH = 0.01


@dataclass(frozen=True)
class DomainRandomization:
    observation_noise_sd: float = 0.0
    actuator_log_sd: float = 0.0
    titrant_scale: float = 1.0
    response_fraction: float = 1.0

    def __post_init__(self) -> None:
        values = (
            ("observation_noise_sd", self.observation_noise_sd),
            ("actuator_log_sd", self.actuator_log_sd),
            ("titrant_scale", self.titrant_scale),
            ("response_fraction", self.response_fraction),
        )
        for name, value in values:
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.observation_noise_sd < 0.0 or self.actuator_log_sd < 0.0:
            raise ValueError("noise standard deviations must be nonnegative")
        if self.titrant_scale <= 0.0:
            raise ValueError("titrant_scale must be positive")
        if not 0.0 < self.response_fraction <= 1.0:
            raise ValueError("response_fraction must be in (0, 1]")


def sample_training_domain(rng: np.random.Generator) -> DomainRandomization:
    noise = float(rng.choice([0.0, 0.005, 0.01, 0.02], p=[0.35, 0.25, 0.25, 0.15]))
    return DomainRandomization(
        observation_noise_sd=noise,
        actuator_log_sd=0.02,
        titrant_scale=float(rng.uniform(0.97, 1.03)),
        response_fraction=float(rng.uniform(0.95, 1.0)),
    )


class ControlEnvironment:
    def __init__(
        self,
        task: ControlTask,
        rng: np.random.Generator,
        domain: DomainRandomization | None = None,
    ) -> None:
        self.task = task
        self.rng = rng
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.domain = domain or DomainRandomization()
        self.target_ph = task.target_ph
        self.titrant_m = 0.1 * self.domain.titrant_scale
        self.base_moles = task.initial_base_moles
        self.acid_moles = 0.0
        self.base_added_ml = 0.0
        self.acid_added_ml = 0.0
        self.steps = 0
        self.overshoots = 0
        self.last_requested_volume_ml = 0.0
        self.true_ph = float(task.initial_ph)
        self.previous_true_ph = self.true_ph
        self.measured_ph = self._measure(self.true_ph, self.true_ph)
        self.previous_measured_ph = self.measured_ph
        self.done = abs(self.measured_ph - self.target_ph) < SUCCESS_TOLERANCE
        self.stop_reason = "initial_success" if self.done else "running"

    def _measure(self, true_ph: float, previous_measured: float) -> float:
        responded = previous_measured + self.domain.response_fraction * (true_ph - previous_measured)
        noisy = responded + float(self.rng.normal(0.0, self.domain.observation_noise_sd))
        clipped = float(np.clip(noisy, 0.0, 14.0))
        return float(np.round(clipped / SENSOR_RESOLUTION_PH) * SENSOR_RESOLUTION_PH)

    def state(self) -> np.ndarray:
        delta = self.measured_ph - self.previous_measured_ph
        error = self.measured_ph - self.target_ph
        return np.asarray(
            [self.measured_ph, self.target_ph, delta, error, self.last_requested_volume_ml],
            dtype=np.float32,
        )

    def step(self, requested_volume_ml: float) -> dict:
        if self.done:
            return {"crossed_target": False, "actual_volume_ml": 0.0}
        requested = float(np.clip(requested_volume_ml, 0.01, 10.0))
        actual = requested * float(self.rng.lognormal(0.0, self.domain.actuator_log_sd))
        remaining = MAX_TOTAL_DOSE_ML - self.base_added_ml - self.acid_added_ml
        if remaining <= 1e-12:
            self.done = True
            self.stop_reason = "dose_limit"
            return {"crossed_target": False, "actual_volume_ml": 0.0}
        actual = float(np.clip(actual, 0.0, remaining))
        if actual <= 1e-12:
            self.done = True
            self.stop_reason = "dose_limit"
            return {"crossed_target": False, "actual_volume_ml": 0.0}
        add_base = self.measured_ph < self.target_ph
        self.previous_true_ph = self.true_ph
        self.previous_measured_ph = self.measured_ph
        if add_base:
            self.base_moles += self.titrant_m * actual / 1000.0
            self.base_added_ml += actual
        else:
            self.acid_moles += self.titrant_m * actual / 1000.0
            self.acid_added_ml += actual
        total_added = self.base_added_ml + self.acid_added_ml
        state = SolutionState(
            self.task.initial_volume_ml + total_added,
            self.base_moles,
            self.acid_moles,
        )
        self.true_ph = solve_ph_scalar(
            self.task.analyte_conc_m,
            self.task.pka_values,
            self.task.initial_volume_ml,
            state,
        )
        self.measured_ph = self._measure(self.true_ph, self.previous_measured_ph)
        self.steps += 1
        self.last_requested_volume_ml = requested
        crossed = (self.previous_true_ph - self.target_ph) * (self.true_ph - self.target_ph) < 0.0
        self.overshoots += int(crossed)
        measured_success = abs(self.measured_ph - self.target_ph) < SUCCESS_TOLERANCE
        dose_exhausted = total_added >= MAX_TOTAL_DOSE_ML - 1e-9
        if measured_success:
            self.done = True
            self.stop_reason = "measured_success"
        elif self.steps >= MAX_STEPS:
            self.done = True
            self.stop_reason = "max_steps"
        elif dose_exhausted:
            self.done = True
            self.stop_reason = "dose_limit"
        return {"crossed_target": crossed, "actual_volume_ml": actual}

    def metrics(self) -> dict:
        true_error = abs(self.true_ph - self.target_ph)
        measured_error = abs(self.measured_ph - self.target_ph)
        return {
            "true_success": int(true_error < SUCCESS_TOLERANCE),
            "strict_success": int(true_error < STRICT_TOLERANCE),
            "severe_failure": int(true_error > 0.50),
            "measured_success": int(measured_error < SUCCESS_TOLERANCE),
            "false_stop": int(measured_error < SUCCESS_TOLERANCE and true_error >= SUCCESS_TOLERANCE),
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
