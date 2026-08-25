from __future__ import annotations

from typing import Any

import numpy as np

from chemistry_model import SolutionState, solve_ph_scalar
from particle_inference import PF_VARIANTS, PosteriorEstimate, build_filter
from reference import original_bayesian_controller as original


MAX_STEPS = original.MAX_STEPS
SUCCESS_THRESHOLD = original.SUCCESS_THRESHOLD
ACTION_COUNT = 1000
CONTROL_VOLUME_BISECTION_ITERATIONS = 32


class JointInferenceController(original.PHAdjustmentEnv):
    """Preserved dosing logic with a replaceable robust inference state."""

    def __init__(self, variant: str, num_particles: int = 1000, filter_seed: int = 0):
        if variant not in PF_VARIANTS:
            raise KeyError(f"Unknown controller variant: {variant}")
        super().__init__(num_particles=num_particles)
        self.addition_volumes = [
            round(self.min_addition_volume * index, 2)
            for index in range(1, ACTION_COUNT + 1)
        ]
        self.action_space = [
            (reagent, volume)
            for reagent in self.reagents
            for volume in self.addition_volumes
        ]
        self.variant = variant
        self.filter_seed = int(filter_seed)
        self.initial_volume_ml = float(original.TITRATED_VOLUME)
        self.true_analyte_concentration_m = float(original.ANALYTE_CONC)
        self.inference = None
        self.last_update_diagnostics: dict[str, Any] | None = None

    def initialize_task(self, task, max_steps: int = MAX_STEPS) -> None:
        self.initial_volume_ml = float(task.initial_volume_ml)
        self.true_analyte_concentration_m = float(task.analyte_conc_m)
        observed_initial_ph = float(np.round(task.initial_ph, 2))
        super().initialize(
            task.acid_type,
            list(task.pka_values),
            observed_initial_ph,
            task.target_ph,
            max_steps,
        )
        self.total_volume = self.initial_volume_ml
        self.previous_total_volume = self.initial_volume_ml
        self.inference = build_filter(self.variant, self.num_particles, self.filter_seed)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        estimate = self.posterior_estimate()
        self.pKa_list = estimate.pka_values.copy()
        self.pKa_std = estimate.pka_sd.copy()
        self.ref_pKa = estimate.pka_values.copy()

    def posterior_estimate(self) -> PosteriorEstimate:
        if self.inference is None:
            raise RuntimeError("Controller has not been initialized")
        return self.inference.estimate()

    def get_effective_pka_array(self):
        return self.posterior_estimate().pka_values.copy()

    def simulate_observed_ph(self):
        state = SolutionState(
            total_volume_ml=float(self.total_volume),
            base_moles=float(self.base_added_moles),
            acid_moles=float(self.acid_added_moles),
        )
        value = solve_ph_scalar(
            self.true_analyte_concentration_m,
            self.true_pKas,
            self.initial_volume_ml,
            state,
        )
        return round(value, 2)

    def compute_required_volume(self):
        estimate = self.posterior_estimate()
        estimated_moles = self.initial_volume_ml / 1000.0 * estimate.concentration_m
        effective_pka = estimate.pka_values.tolist()

        if self.current_ph < self.target_ph:
            reagent = "Dilute base 2" if self.use_secondary_reagents else "Dilute base 1"
            concentration = self.reagents[reagent]

            def objective(volume_ml):
                state = SolutionState(
                    total_volume_ml=self.total_volume + volume_ml,
                    base_moles=self.base_added_moles + concentration * volume_ml / 1000.0,
                    acid_moles=self.acid_added_moles,
                )
                model_concentration = estimated_moles / (self.initial_volume_ml / 1000.0)
                return solve_ph_scalar(
                    model_concentration,
                    effective_pka,
                    self.initial_volume_ml,
                    state,
                ) - self.target_ph

            return original.solve_volume_root(
                objective,
                0.0,
                10.0,
                iterations=CONTROL_VOLUME_BISECTION_ITERATIONS,
            )

        reagent = "Dilute acid 2" if self.use_secondary_reagents else "Dilute acid 1"
        concentration = self.reagents[reagent]

        def objective(volume_ml):
            state = SolutionState(
                total_volume_ml=self.total_volume + volume_ml,
                base_moles=self.base_added_moles,
                acid_moles=self.acid_added_moles + concentration * volume_ml / 1000.0,
            )
            model_concentration = estimated_moles / (self.initial_volume_ml / 1000.0)
            return solve_ph_scalar(
                model_concentration,
                effective_pka,
                self.initial_volume_ml,
                state,
            ) - self.target_ph

        return original.solve_volume_root(
            objective,
            0.0,
            10.0,
            iterations=CONTROL_VOLUME_BISECTION_ITERATIONS,
        )

    def update_posteriors(self, action, observed_ph):
        if self.inference is None:
            raise RuntimeError("Controller has not been initialized")
        reagent = str(action[0]).lower()
        before_base_moles = float(self.base_added_moles)
        before_acid_moles = float(self.acid_added_moles)
        if "base" in reagent:
            before_base_moles -= float(self.last_base_added)
        else:
            before_acid_moles -= float(self.last_acid_added)
        before_state = SolutionState(
            total_volume_ml=float(self.previous_total_volume),
            base_moles=before_base_moles,
            acid_moles=before_acid_moles,
        )
        after_state = SolutionState(
            total_volume_ml=float(self.total_volume),
            base_moles=float(self.base_added_moles),
            acid_moles=float(self.acid_added_moles),
        )
        log_predictive = self.inference.update(
            self.initial_volume_ml,
            before_state,
            after_state,
            float(self.previous_ph),
            float(observed_ph),
        )
        self._refresh_summary()
        estimate = self.posterior_estimate()
        self.last_update_diagnostics = {
            "update": int(self.steps_taken),
            "log_predictive": float(log_predictive),
            "concentration_m": estimate.concentration_m,
            "concentration_sd_m": estimate.concentration_sd_m,
            "pair_count": estimate.pair_count,
            "pair_probabilities": estimate.pair_probabilities.copy(),
            "pka_values": estimate.pka_values.copy(),
            "pka_sd": estimate.pka_sd.copy(),
            "effective_sample_size": estimate.effective_sample_size,
        }


def build_controller(variant: str, particles: int, seed: int) -> JointInferenceController:
    return JointInferenceController(variant, num_particles=particles, filter_seed=seed)
