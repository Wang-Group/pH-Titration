from __future__ import annotations

import math
from typing import Any

import numpy as np

from chemistry_model import SolutionState, solve_ph_scalar
from controller_api import (
    MAX_ACTION_VOLUME_ML,
    MIN_ACTION_VOLUME_ML,
    PRIMARY_TITRANT_CONCENTRATION_M,
    ControllerAction,
    normalize_reagent,
    quantize_ph,
    stop_action,
)
from particle_inference import PosteriorEstimate, build_filter


PF_VARIANT = "pf_pka_conc_variable_k"
CONTROL_VOLUME_BISECTION_ITERATIONS = 32


def _solve_volume_root(func, lo: float = 0.0, hi: float = 10.0) -> float:
    f_lo = float(func(lo))
    f_hi = float(func(hi))
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return 0.0
    left, right = float(lo), float(hi)
    left_value = f_lo
    for _ in range(CONTROL_VOLUME_BISECTION_ITERATIONS):
        mid = 0.5 * (left + right)
        mid_value = float(func(mid))
        if abs(mid_value) < 1e-10:
            return mid
        if left_value * mid_value < 0.0:
            right = mid
        else:
            left = mid
            left_value = mid_value
    return 0.5 * (left + right)


class RobustPFController:
    """Deployable form of the selected new-PF plus full dose-shaping controller."""

    def __init__(
        self,
        particles: int = 1000,
        seed: int = 0,
        success_tolerance_ph: float = 0.10,
        max_steps: int = 50,
        max_total_dose_ml: float | None = None,
        titrant_concentration_m: float = PRIMARY_TITRANT_CONCENTRATION_M,
    ) -> None:
        if particles < 60:
            raise ValueError("Variable-K PF requires at least 60 total particles")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if titrant_concentration_m <= 0.0:
            raise ValueError("titrant_concentration_m must be positive")
        self.particles = int(particles)
        self.seed = int(seed) % (2**32 - 1)
        self.success_tolerance_ph = float(success_tolerance_ph)
        self.max_steps = int(max_steps)
        self.max_total_dose_ml = (
            None if max_total_dose_ml is None else float(max_total_dose_ml)
        )
        self.titrant_concentration_m = float(titrant_concentration_m)
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self.inference = None
        self.initial_volume_ml = math.nan
        self.target_ph = math.nan
        self.current_ph = math.nan
        self.previous_ph = math.nan
        self.total_volume_ml = math.nan
        self.base_moles = math.nan
        self.acid_moles = math.nan
        self.base_added_ml = 0.0
        self.acid_added_ml = 0.0
        self.last_action_volume_ml = 0.0
        self.steps = 0
        self.done = False
        self.stop_reason = "not_initialized"
        self.overshoot_threshold_ml: float | None = None
        self.overshoot_occurred = False
        self.overshoot_reagent: str | None = None
        self.pending_action: ControllerAction | None = None
        self.last_update_diagnostics: dict[str, Any] | None = None
        self._legacy_buffer_mean = math.nan

    def reset(
        self,
        initial_measured_ph: float,
        target_ph: float,
        initial_volume_ml: float,
        initial_base_moles: float = 0.0,
        initial_acid_moles: float = 0.0,
    ) -> dict[str, Any]:
        if initial_volume_ml <= 0.0:
            raise ValueError("initial_volume_ml must be positive")
        if initial_base_moles < 0.0 or initial_acid_moles < 0.0:
            raise ValueError("Initial acid/base moles cannot be negative")
        self._reset_runtime()
        self.initial_volume_ml = float(initial_volume_ml)
        self.total_volume_ml = float(initial_volume_ml)
        self.base_moles = float(initial_base_moles)
        self.acid_moles = float(initial_acid_moles)
        self.current_ph = quantize_ph(initial_measured_ph)
        self.previous_ph = self.current_ph
        self.target_ph = float(target_ph)
        if not np.isfinite(self.target_ph):
            raise ValueError("target_ph must be finite")

        # The formal hybrid controller inherited one seeded nuisance factor from
        # the original implementation. Reproduce its draw locally without
        # mutating NumPy's global RNG.
        legacy_rng = np.random.RandomState(self.seed)
        legacy_rng.uniform(2.0, 6.0, size=3)
        legacy_buffer_total_moles = legacy_rng.uniform(1e-6, 0.5, size=3)
        self._legacy_buffer_mean = float(np.mean(legacy_buffer_total_moles))
        filter_seed = (self.seed + 17) % (2**32 - 1)
        self.inference = build_filter(PF_VARIANT, self.particles, filter_seed)
        if abs(self.current_ph - self.target_ph) < self.success_tolerance_ph:
            self.done = True
            self.stop_reason = "initial_success"
        else:
            self.stop_reason = "running"
        return self.status()

    def _require_initialized(self) -> None:
        if self.inference is None:
            raise RuntimeError("Call reset() before requesting an action")

    def posterior_estimate(self) -> PosteriorEstimate:
        self._require_initialized()
        return self.inference.estimate()

    def state_vector(self) -> np.ndarray:
        self._require_initialized()
        return np.asarray(
            [
                self.current_ph,
                self.target_ph,
                self.current_ph - self.previous_ph,
                self.current_ph - self.target_ph,
                self.last_action_volume_ml,
            ],
            dtype=np.float32,
        )

    def _compute_required_volume(self) -> float:
        estimate = self.posterior_estimate()
        direction = "base" if self.current_ph < self.target_ph else "acid"

        def objective(volume_ml: float) -> float:
            if direction == "base":
                state = SolutionState(
                    total_volume_ml=self.total_volume_ml + volume_ml,
                    base_moles=(
                        self.base_moles
                        + self.titrant_concentration_m * volume_ml / 1000.0
                    ),
                    acid_moles=self.acid_moles,
                )
            else:
                state = SolutionState(
                    total_volume_ml=self.total_volume_ml + volume_ml,
                    base_moles=self.base_moles,
                    acid_moles=(
                        self.acid_moles
                        + self.titrant_concentration_m * volume_ml / 1000.0
                    ),
                )
            return (
                solve_ph_scalar(
                    estimate.concentration_m,
                    estimate.pka_values,
                    self.initial_volume_ml,
                    state,
                )
                - self.target_ph
            )

        return _solve_volume_root(objective, 0.0, MAX_ACTION_VOLUME_ML)

    def _stop_if_needed(self) -> None:
        if abs(self.current_ph - self.target_ph) < self.success_tolerance_ph:
            self.done = True
            self.stop_reason = "measured_success"
        elif self.steps >= self.max_steps:
            self.done = True
            self.stop_reason = "max_steps"
        elif (
            self.max_total_dose_ml is not None
            and self.base_added_ml + self.acid_added_ml >= self.max_total_dose_ml - 1e-12
        ):
            self.done = True
            self.stop_reason = "dose_limit"

    def recommend(self) -> ControllerAction:
        self._require_initialized()
        self._stop_if_needed()
        if self.done:
            return stop_action(self.stop_reason, self.status())
        if self.pending_action is not None:
            raise RuntimeError("The previous action has not been acknowledged by observe()")

        if self.overshoot_occurred and self.overshoot_reagent is not None:
            reagent = "acid" if self.overshoot_reagent == "base" else "base"
            self.overshoot_occurred = False
            self.overshoot_reagent = None
        else:
            reagent = "base" if self.current_ph < self.target_ph else "acid"

        candidate_volumes = np.asarray(
            [round(MIN_ACTION_VOLUME_ML * index, 2) for index in range(1, 1001)],
            dtype=float,
        )
        if self.overshoot_threshold_ml is not None:
            filtered = candidate_volumes[
                candidate_volumes <= self.overshoot_threshold_ml
            ]
            if len(filtered):
                candidate_volumes = filtered

        estimate = self.posterior_estimate()
        error = abs(self.current_ph - self.target_ph)
        ph_change = abs(self.current_ph - self.previous_ph)
        bonus_factor = 1.0 + 0.5 * (1.0 - min(ph_change, 1.0))
        average_uncertainty = float(np.mean(estimate.pka_sd))
        uncertainty_factor = 1.0 - 0.1 * min(average_uncertainty, 1.0)
        buffering_factor = float(
            np.clip(1.0 + 0.1 * (self._legacy_buffer_mean - 0.5), 0.95, 1.05)
        )
        alpha = 0.2 * bonus_factor * uncertainty_factor * buffering_factor
        required_volume = float(self._compute_required_volume())
        combined_value = error + 0.1 * required_volume
        ideal_volume = MIN_ACTION_VOLUME_ML + (
            MAX_ACTION_VOLUME_ML - MIN_ACTION_VOLUME_ML
        ) * np.tanh(alpha * combined_value)
        volume = float(candidate_volumes[np.argmin(np.abs(candidate_volumes - ideal_volume))])
        action = ControllerAction(
            stop=False,
            reagent=reagent,
            volume_ml=round(volume, 2),
            titrant_concentration_m=self.titrant_concentration_m,
            diagnostics={
                "state": self.state_vector().tolist(),
                "required_volume_ml": required_volume,
                "ideal_volume_ml": float(ideal_volume),
                "overshoot_threshold_ml": self.overshoot_threshold_ml,
                "estimated_concentration_m": estimate.concentration_m,
                "estimated_pair_count": estimate.pair_count,
                "pair_probabilities": estimate.pair_probabilities.tolist(),
                "estimated_pka": estimate.pka_values.tolist(),
                "estimated_pka_sd": estimate.pka_sd.tolist(),
                "effective_sample_size": estimate.effective_sample_size,
            },
        )
        self.pending_action = action
        return action

    def observe(
        self,
        measured_ph: float,
        actual_volume_ml: float | None = None,
        reagent: str | None = None,
    ) -> dict[str, Any]:
        self._require_initialized()
        if self.pending_action is None:
            raise RuntimeError("Call recommend() before observe()")
        expected_reagent = str(self.pending_action.reagent)
        executed_reagent = (
            expected_reagent if reagent is None else normalize_reagent(reagent)
        )
        if executed_reagent != expected_reagent:
            raise ValueError(
                f"Executed reagent {executed_reagent!r} does not match requested "
                f"reagent {expected_reagent!r}"
            )
        actual = (
            float(self.pending_action.volume_ml)
            if actual_volume_ml is None
            else float(actual_volume_ml)
        )
        if not np.isfinite(actual) or actual <= 0.0:
            raise ValueError("actual_volume_ml must be finite and positive")

        before_state = SolutionState(
            total_volume_ml=self.total_volume_ml,
            base_moles=self.base_moles,
            acid_moles=self.acid_moles,
        )
        added_moles = self.titrant_concentration_m * (actual / 1000.0)
        self.total_volume_ml += actual
        if executed_reagent == "base":
            self.base_moles += added_moles
            self.base_added_ml += actual
        else:
            self.acid_moles += added_moles
            self.acid_added_ml += actual
        after_state = SolutionState(
            total_volume_ml=self.total_volume_ml,
            base_moles=self.base_moles,
            acid_moles=self.acid_moles,
        )

        observed_before = self.current_ph
        observed_after = quantize_ph(measured_ph)
        log_predictive = self.inference.update(
            self.initial_volume_ml,
            before_state,
            after_state,
            observed_before,
            observed_after,
        )
        self.previous_ph = observed_before
        self.current_ph = observed_after
        self.last_action_volume_ml = actual
        self.steps += 1

        sign_change = (
            (observed_before - self.target_ph) * (observed_after - self.target_ph) < 0.0
        )
        error_increased = abs(observed_after - self.target_ph) > abs(
            observed_before - self.target_ph
        )
        if sign_change or error_increased:
            overshoot_volume = (
                added_moles * 1000.0 / self.titrant_concentration_m
            )
            new_threshold = max(
                overshoot_volume / 2.0, MIN_ACTION_VOLUME_ML
            )
            self.overshoot_occurred = True
            self.overshoot_reagent = executed_reagent
            if (
                self.overshoot_threshold_ml is None
                or new_threshold < self.overshoot_threshold_ml
            ):
                self.overshoot_threshold_ml = new_threshold

        estimate = self.posterior_estimate()
        self.last_update_diagnostics = {
            "step": self.steps,
            "observed_before_ph": observed_before,
            "observed_after_ph": observed_after,
            "actual_volume_ml": actual,
            "reagent": executed_reagent,
            "crossed_target": sign_change,
            "error_increased": error_increased,
            "log_predictive": float(log_predictive),
            "estimated_concentration_m": estimate.concentration_m,
            "estimated_concentration_sd_m": estimate.concentration_sd_m,
            "estimated_pair_count": estimate.pair_count,
            "pair_probabilities": estimate.pair_probabilities.tolist(),
            "estimated_pka": estimate.pka_values.tolist(),
            "estimated_pka_sd": estimate.pka_sd.tolist(),
            "effective_sample_size": estimate.effective_sample_size,
            "overshoot_threshold_ml": self.overshoot_threshold_ml,
        }
        self.pending_action = None
        self._stop_if_needed()
        return self.status()

    def status(self) -> dict[str, Any]:
        estimate = None if self.inference is None else self.posterior_estimate()
        return {
            "controller": "new_pf_hybrid_full",
            "initialized": self.inference is not None,
            "done": self.done,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "current_measured_ph": self.current_ph,
            "target_ph": self.target_ph,
            "total_added_volume_ml": self.base_added_ml + self.acid_added_ml,
            "base_added_ml": self.base_added_ml,
            "acid_added_ml": self.acid_added_ml,
            "overshoot_threshold_ml": self.overshoot_threshold_ml,
            "posterior": (
                None
                if estimate is None
                else {
                    "concentration_m": estimate.concentration_m,
                    "concentration_sd_m": estimate.concentration_sd_m,
                    "pair_count": estimate.pair_count,
                    "pair_probabilities": estimate.pair_probabilities.tolist(),
                    "pka_values": estimate.pka_values.tolist(),
                    "pka_sd": estimate.pka_sd.tolist(),
                    "effective_sample_size": estimate.effective_sample_size,
                }
            ),
            "last_update": self.last_update_diagnostics,
        }
