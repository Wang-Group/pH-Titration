from __future__ import annotations

"""Experimental PF representation for mixtures of independent monoprotic acids.

The released controller represents an effective polyprotic analyte using one total
concentration and K successive pKa values.  This module keeps the controller and
likelihood protocol fixed, but replaces that chemical state with J=1, 2, or 3
independent monoprotic components.  Each component has its own concentration and
pKa.  Components are ordered by pKa to remove label switching.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "github_ph_titration"
CONTROLLER_DIR = str(REPO / "controllers")
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from chemistry_model import SolutionState, WATER_KW  # noqa: E402
from controller_api import MAX_ACTION_VOLUME_ML  # noqa: E402
from new_pf_controller import RobustPFController, _solve_volume_root  # noqa: E402
from particle_inference import (  # noqa: E402
    ESS_THRESHOLD_FRACTION,
    LIU_WEST_H,
    PKA_CLIP_HIGH,
    PKA_CLIP_LOW,
    PRIOR_CONC_HIGH_M,
    PRIOR_CONC_LOW_M,
    PRIOR_PKA_HIGH,
    PRIOR_PKA_LOW,
    _log_student_kernel,
    _logsumexp,
    _resample_indices,
    _stable_covariance,
)


FRACTION_LOGIT_CLIP = 8.0
MIN_FRACTION = 1.0e-8


def _fractions_from_alr(alr: np.ndarray, component_count: int) -> np.ndarray:
    """Convert additive-log-ratio coordinates to simplex fractions."""
    count = int(component_count)
    values = np.asarray(alr, dtype=float)
    if count == 1:
        rows = values.shape[0] if values.ndim == 2 else 1
        return np.ones((rows, 1), dtype=float)
    if values.ndim != 2 or values.shape[1] != count - 1:
        raise ValueError("ALR coordinates have incompatible shape")
    logits = np.column_stack([values, np.zeros(values.shape[0], dtype=float)])
    logits -= np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _alr_from_fractions(fractions: np.ndarray) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 2:
        raise ValueError("Fractions must be a matrix")
    if values.shape[1] == 1:
        return np.empty((values.shape[0], 0), dtype=float)
    values = np.maximum(values, MIN_FRACTION)
    values /= np.sum(values, axis=1, keepdims=True)
    return np.log(values[:, :-1]) - np.log(values[:, -1:])


def solve_independent_ph_scalar(
    component_concentrations_m: np.ndarray,
    pka_values: np.ndarray,
    initial_volume_ml: float,
    state: SolutionState,
    iterations: int = 60,
) -> float:
    concentrations = np.asarray(component_concentrations_m, dtype=float)
    pkas = np.asarray(pka_values, dtype=float)
    if concentrations.ndim != 1 or pkas.shape != concentrations.shape:
        raise ValueError("Component concentrations and pKa values must be matching vectors")
    total_volume_l = float(state.total_volume_ml) / 1000.0
    dilution = float(initial_volume_ml) / float(state.total_volume_ml)
    current_concentrations = concentrations * dilution
    kas = np.power(10.0, -pkas)
    c_na = float(state.base_moles) / total_volume_l
    c_hcl = float(state.acid_moles) / total_volume_l

    def balance(ph: float) -> float:
        hydrogen = 10.0 ** (-ph)
        hydroxide = WATER_KW / hydrogen
        charge = float(np.sum(current_concentrations * kas / (kas + hydrogen)))
        return hydrogen + c_na - hydroxide - c_hcl - charge

    lo, hi = 0.0, 14.0
    f_lo = balance(lo)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def solve_independent_ph_particles(
    component_concentrations_m: np.ndarray,
    pka_matrix: np.ndarray,
    initial_volume_ml: float,
    state: SolutionState,
    iterations: int = 60,
) -> np.ndarray:
    concentrations = np.asarray(component_concentrations_m, dtype=float)
    pkas = np.asarray(pka_matrix, dtype=float)
    if concentrations.ndim != 2 or pkas.shape != concentrations.shape:
        raise ValueError("Particle component concentrations and pKa matrix must match")
    total_volume_l = float(state.total_volume_ml) / 1000.0
    dilution = float(initial_volume_ml) / float(state.total_volume_ml)
    current_concentrations = concentrations * dilution
    kas = np.power(10.0, -pkas)
    c_na = float(state.base_moles) / total_volume_l
    c_hcl = float(state.acid_moles) / total_volume_l

    lo = np.zeros(pkas.shape[0], dtype=float)
    hi = np.full(pkas.shape[0], 14.0, dtype=float)

    def balance(ph: np.ndarray) -> np.ndarray:
        hydrogen = np.power(10.0, -ph)
        hydroxide = WATER_KW / hydrogen
        charge = np.sum(
            current_concentrations * kas / (kas + hydrogen[:, None]), axis=1
        )
        return hydrogen + c_na - hydroxide - c_hcl - charge

    f_lo = balance(lo)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        left = f_lo * f_mid < 0.0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi)


def solve_independent_ph_grid(
    component_concentrations_m: np.ndarray,
    pka_values: np.ndarray,
    initial_volume_ml: float,
    total_volume_ml: np.ndarray,
    base_moles: np.ndarray,
    acid_moles: np.ndarray,
    iterations: int = 60,
) -> np.ndarray:
    """Vectorized pH curve for one independent-component parameter vector."""
    concentrations = np.asarray(component_concentrations_m, dtype=float)
    pkas = np.asarray(pka_values, dtype=float)
    if concentrations.ndim != 1 or pkas.shape != concentrations.shape:
        raise ValueError("Component concentrations and pKa values must be matching vectors")
    volumes = np.asarray(total_volume_ml, dtype=float)
    base = np.asarray(base_moles, dtype=float)
    acid = np.asarray(acid_moles, dtype=float)
    volumes, base, acid = np.broadcast_arrays(volumes, base, acid)
    total_volume_l = volumes / 1000.0
    dilution = float(initial_volume_ml) / volumes
    current_concentrations = dilution[..., None] * concentrations
    kas = np.power(10.0, -pkas)
    c_na = base / total_volume_l
    c_hcl = acid / total_volume_l

    lo = np.zeros_like(volumes, dtype=float)
    hi = np.full_like(volumes, 14.0, dtype=float)

    def balance(ph: np.ndarray) -> np.ndarray:
        hydrogen = np.power(10.0, -ph)
        hydroxide = WATER_KW / hydrogen
        charge = np.sum(
            current_concentrations
            * kas
            / (kas + hydrogen[..., None]),
            axis=-1,
        )
        return hydrogen + c_na - hydroxide - c_hcl - charge

    f_lo = balance(lo)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        left = f_lo * f_mid < 0.0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class IndependentMixturePosteriorEstimate:
    concentration_m: float
    concentration_sd_m: float
    component_concentrations_m: np.ndarray
    component_concentration_sd_m: np.ndarray
    component_fractions: np.ndarray
    pka_values: np.ndarray
    pka_sd: np.ndarray
    pair_count: int
    pair_probabilities: np.ndarray
    effective_sample_size: float


class IndependentFixedJParticleFilter:
    def __init__(
        self,
        particle_count: int,
        component_count: int,
        rng: np.random.Generator,
    ) -> None:
        count = int(particle_count)
        if isinstance(particle_count, bool) or float(particle_count) != count or count < 20:
            raise ValueError("At least 20 particles are required")
        if component_count not in (1, 2, 3):
            raise ValueError("component_count must be 1, 2, or 3")
        self.particle_count = count
        self.component_count = int(component_count)
        self.pair_count = self.component_count
        self.rng = rng
        # Keep the J=1 draw order identical to the released fixed-K filter so its
        # chemistry and particle update can be verified task-for-task.
        self.pka_particles = np.sort(
            rng.uniform(
                PRIOR_PKA_LOW,
                PRIOR_PKA_HIGH,
                size=(self.particle_count, self.component_count),
            ),
            axis=1,
        )
        self.log_total_concentration_particles = rng.uniform(
            np.log(PRIOR_CONC_LOW_M),
            np.log(PRIOR_CONC_HIGH_M),
            size=self.particle_count,
        )
        if self.component_count == 1:
            self.fraction_logits_particles = np.empty(
                (self.particle_count, 0), dtype=float
            )
        else:
            fractions = rng.dirichlet(
                np.ones(self.component_count, dtype=float), size=self.particle_count
            )
            self.fraction_logits_particles = _alr_from_fractions(fractions)
        self.weights = np.full(self.particle_count, 1.0 / self.particle_count)
        self.cached_ph: np.ndarray | None = None
        self.update_count = 0
        self.resample_count = 0
        self.last_log_predictive = 0.0

    @property
    def total_concentrations_m(self) -> np.ndarray:
        return np.exp(self.log_total_concentration_particles)

    @property
    def component_fractions(self) -> np.ndarray:
        return _fractions_from_alr(
            self.fraction_logits_particles, self.component_count
        )

    @property
    def component_concentrations_m(self) -> np.ndarray:
        return self.total_concentrations_m[:, None] * self.component_fractions

    def predict(self, initial_volume_ml: float, state: SolutionState) -> np.ndarray:
        return solve_independent_ph_particles(
            self.component_concentrations_m,
            self.pka_particles,
            initial_volume_ml,
            state,
        )

    def update(
        self,
        initial_volume_ml: float,
        before_state: SolutionState,
        after_state: SolutionState,
        observed_before_ph: float,
        observed_after_ph: float,
    ) -> float:
        predicted_before = (
            self.predict(initial_volume_ml, before_state)
            if self.cached_ph is None
            else self.cached_ph
        )
        predicted_after = self.predict(initial_volume_ml, after_state)
        predicted_delta = predicted_after - predicted_before
        observed_delta = float(observed_after_ph) - float(observed_before_ph)
        log_likelihood = _log_student_kernel(observed_delta - predicted_delta)
        log_terms = np.log(self.weights + 1e-300) + log_likelihood
        log_predictive = _logsumexp(log_terms)
        self.weights = np.exp(log_terms - log_predictive)
        self.weights /= np.sum(self.weights)

        ess = float(1.0 / np.sum(self.weights**2))
        if ess < ESS_THRESHOLD_FRACTION * self.particle_count:
            indices = _resample_indices(self.weights, self.rng)
            selected_pka = self.pka_particles[indices]
            pieces = [self.log_total_concentration_particles[indices, None]]
            if self.component_count > 1:
                pieces.append(self.fraction_logits_particles[indices])
            pieces.append(selected_pka)
            transformed = np.column_stack(pieces)
            mean = np.mean(transformed, axis=0)
            covariance = _stable_covariance(transformed)
            a = np.sqrt(1.0 - LIU_WEST_H**2)
            roughening_covariance = LIU_WEST_H**2 * covariance
            transform = np.linalg.cholesky(
                roughening_covariance + np.eye(covariance.shape[0]) * 1e-8
            )
            noise = self.rng.normal(size=transformed.shape) @ transform.T
            regularized = a * transformed + (1.0 - a) * mean + noise

            cursor = 0
            self.log_total_concentration_particles = np.clip(
                regularized[:, cursor],
                np.log(PRIOR_CONC_LOW_M),
                np.log(PRIOR_CONC_HIGH_M),
            )
            cursor += 1
            if self.component_count > 1:
                alr = np.clip(
                    regularized[:, cursor : cursor + self.component_count - 1],
                    -FRACTION_LOGIT_CLIP,
                    FRACTION_LOGIT_CLIP,
                )
                fractions = _fractions_from_alr(alr, self.component_count)
                cursor += self.component_count - 1
            else:
                fractions = np.ones((self.particle_count, 1), dtype=float)
            proposed_pka = np.clip(
                regularized[:, cursor : cursor + self.component_count],
                PKA_CLIP_LOW,
                PKA_CLIP_HIGH,
            )
            order = np.argsort(proposed_pka, axis=1)
            self.pka_particles = np.take_along_axis(proposed_pka, order, axis=1)
            fractions = np.take_along_axis(fractions, order, axis=1)
            self.fraction_logits_particles = _alr_from_fractions(fractions)
            self.weights.fill(1.0 / self.particle_count)
            self.cached_ph = self.predict(initial_volume_ml, after_state)
            self.resample_count += 1
        else:
            self.cached_ph = predicted_after

        self.update_count += 1
        self.last_log_predictive = float(log_predictive)
        return float(log_predictive)

    def estimate(self) -> IndependentMixturePosteriorEstimate:
        total = self.total_concentrations_m
        component = self.component_concentrations_m
        fractions = self.component_fractions
        total_mean = float(np.sum(self.weights * total))
        total_sd = float(
            np.sqrt(np.sum(self.weights * (total - total_mean) ** 2))
        )
        component_mean = np.sum(self.weights[:, None] * component, axis=0)
        component_sd = np.sqrt(
            np.sum(
                self.weights[:, None] * (component - component_mean) ** 2,
                axis=0,
            )
        )
        fraction_mean = np.sum(self.weights[:, None] * fractions, axis=0)
        pka_mean = np.sum(self.weights[:, None] * self.pka_particles, axis=0)
        pka_sd = np.sqrt(
            np.sum(
                self.weights[:, None] * (self.pka_particles - pka_mean) ** 2,
                axis=0,
            )
        )
        return IndependentMixturePosteriorEstimate(
            concentration_m=total_mean,
            concentration_sd_m=total_sd,
            component_concentrations_m=np.asarray(component_mean, dtype=float),
            component_concentration_sd_m=np.asarray(component_sd, dtype=float),
            component_fractions=np.asarray(fraction_mean, dtype=float),
            pka_values=np.asarray(pka_mean, dtype=float),
            pka_sd=np.asarray(pka_sd, dtype=float),
            pair_count=self.component_count,
            pair_probabilities=np.eye(3, dtype=float)[self.component_count - 1],
            effective_sample_size=float(1.0 / np.sum(self.weights**2)),
        )


class IndependentVariableJParticleFilter:
    """Model bank for J=1, 2, and 3 independent monoprotic components."""

    def __init__(self, particle_count: int, rng: np.random.Generator) -> None:
        count = int(particle_count)
        if isinstance(particle_count, bool) or float(particle_count) != count or count < 60:
            raise ValueError("At least 60 total particles are required for variable J")
        allocations = [count // 3] * 3
        for index in range(count - sum(allocations)):
            allocations[index] += 1
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=3, dtype=np.uint32)
        self.banks = {
            component_count: IndependentFixedJParticleFilter(
                allocations[component_count - 1],
                component_count,
                np.random.default_rng(int(seeds[component_count - 1])),
            )
            for component_count in (1, 2, 3)
        }
        self.model_log_probabilities = np.full(3, -np.log(3.0))
        self.update_count = 0

    def update(
        self,
        initial_volume_ml: float,
        before_state: SolutionState,
        after_state: SolutionState,
        observed_before_ph: float,
        observed_after_ph: float,
    ) -> float:
        predictive = np.asarray(
            [
                self.banks[j].update(
                    initial_volume_ml,
                    before_state,
                    after_state,
                    observed_before_ph,
                    observed_after_ph,
                )
                for j in (1, 2, 3)
            ],
            dtype=float,
        )
        updated = self.model_log_probabilities + predictive
        normalization = _logsumexp(updated)
        self.model_log_probabilities = updated - normalization
        self.update_count += 1
        return float(normalization)

    @property
    def model_probabilities(self) -> np.ndarray:
        values = np.exp(self.model_log_probabilities)
        return values / np.sum(values)

    @property
    def resample_count(self) -> int:
        return int(sum(bank.resample_count for bank in self.banks.values()))

    def estimate(self) -> IndependentMixturePosteriorEstimate:
        probabilities = self.model_probabilities
        map_j = int(np.argmax(probabilities) + 1)
        conditional = self.banks[map_j].estimate()
        return IndependentMixturePosteriorEstimate(
            concentration_m=conditional.concentration_m,
            concentration_sd_m=conditional.concentration_sd_m,
            component_concentrations_m=conditional.component_concentrations_m,
            component_concentration_sd_m=conditional.component_concentration_sd_m,
            component_fractions=conditional.component_fractions,
            pka_values=conditional.pka_values,
            pka_sd=conditional.pka_sd,
            pair_count=map_j,
            pair_probabilities=probabilities.copy(),
            effective_sample_size=conditional.effective_sample_size,
        )


class IndependentMixturePFController(RobustPFController):
    """Released dose-shaping controller with an independent-mixture PF state."""

    def reset(
        self,
        initial_measured_ph: float,
        target_ph: float,
        initial_volume_ml: float,
        initial_base_moles: float = 0.0,
        initial_acid_moles: float = 0.0,
    ):
        super().reset(
            initial_measured_ph,
            target_ph,
            initial_volume_ml,
            initial_base_moles,
            initial_acid_moles,
        )
        filter_seed = (self.seed + 17) % (2**32 - 1)
        self.inference = IndependentVariableJParticleFilter(
            self.particles, np.random.default_rng(filter_seed)
        )
        return self.status()

    def _compute_required_volume(self) -> float:
        estimate = self.posterior_estimate()
        direction = "base" if self.current_ph < self.target_ph else "acid"

        def objective(volume_ml: float) -> float:
            if direction == "base":
                state = SolutionState(
                    self.total_volume_ml + volume_ml,
                    self.base_moles
                    + self.titrant_concentration_m * volume_ml / 1000.0,
                    self.acid_moles,
                )
            else:
                state = SolutionState(
                    self.total_volume_ml + volume_ml,
                    self.base_moles,
                    self.acid_moles
                    + self.titrant_concentration_m * volume_ml / 1000.0,
                )
            return (
                solve_independent_ph_scalar(
                    estimate.component_concentrations_m,
                    estimate.pka_values,
                    self.initial_volume_ml,
                    state,
                )
                - self.target_ph
            )

        return float(_solve_volume_root(objective, 0.0, MAX_ACTION_VOLUME_ML))

    def status(self):
        result = super().status()
        result["controller"] = "independent_monoprotic_mixture_pf_hybrid_full"
        result["chemical_representation"] = (
            "J=1,2,3 independent monoprotic components"
        )
        estimate = None if self.inference is None else self.posterior_estimate()
        if (
            estimate is not None
            and result.get("posterior") is not None
            and hasattr(estimate, "component_concentrations_m")
        ):
            result["posterior"]["component_concentrations_m"] = (
                estimate.component_concentrations_m.tolist()
            )
            result["posterior"]["component_concentration_sd_m"] = (
                estimate.component_concentration_sd_m.tolist()
            )
            result["posterior"]["component_fractions"] = (
                estimate.component_fractions.tolist()
            )
        return result


def effective_resampling_diagnostics(controller: IndependentMixturePFController) -> dict:
    inference = controller.inference
    return {
        "total_resamples": int(inference.resample_count),
        "bank_resamples": {
            str(j): int(inference.banks[j].resample_count) for j in (1, 2, 3)
        },
        "bank_updates": {
            str(j): int(inference.banks[j].update_count) for j in (1, 2, 3)
        },
    }
