from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chemistry_model import SolutionState, solve_ph_particles


PRIOR_PKA_LOW = 1.5
PRIOR_PKA_HIGH = 9.0
PKA_CLIP_LOW = PRIOR_PKA_LOW
PKA_CLIP_HIGH = PRIOR_PKA_HIGH
PRIOR_CONC_LOW_M = 0.02
PRIOR_CONC_HIGH_M = 0.30
LIKELIHOOD_SCALE_PH = 0.20
STUDENT_DF = 4.0
ESS_THRESHOLD_FRACTION = 0.50
LIU_WEST_H = 0.15


@dataclass(frozen=True)
class PosteriorEstimate:
    concentration_m: float
    concentration_sd_m: float
    pka_values: np.ndarray
    pka_sd: np.ndarray
    pair_count: int
    pair_probabilities: np.ndarray
    effective_sample_size: float


def _log_student_kernel(residual_ph: np.ndarray) -> np.ndarray:
    standardized = np.asarray(residual_ph, dtype=float) / LIKELIHOOD_SCALE_PH
    return -0.5 * (STUDENT_DF + 1.0) * np.log1p(
        standardized**2 / STUDENT_DF
    )


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    return rng.choice(count, size=count, replace=True, p=weights)


def _stable_covariance(values: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.asarray(np.cov(values, rowvar=False), dtype=float))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


class FixedKParticleFilter:
    def __init__(
        self,
        particle_count: int,
        pair_count: int,
        infer_concentration: bool,
        rng: np.random.Generator,
        fixed_concentration_m: float = 0.1,
    ) -> None:
        if particle_count < 20:
            raise ValueError("At least 20 particles are required")
        if pair_count not in (1, 2, 3):
            raise ValueError("pair_count must be 1, 2, or 3")
        self.particle_count = int(particle_count)
        self.pair_count = int(pair_count)
        self.infer_concentration = bool(infer_concentration)
        self.fixed_concentration_m = float(fixed_concentration_m)
        self.rng = rng
        self.pka_particles = np.sort(
            rng.uniform(
                PRIOR_PKA_LOW,
                PRIOR_PKA_HIGH,
                size=(self.particle_count, self.pair_count),
            ),
            axis=1,
        )
        if self.infer_concentration:
            self.log_concentration_particles = rng.uniform(
                np.log(PRIOR_CONC_LOW_M),
                np.log(PRIOR_CONC_HIGH_M),
                size=self.particle_count,
            )
        else:
            self.log_concentration_particles = np.full(
                self.particle_count,
                np.log(self.fixed_concentration_m),
            )
        self.weights = np.full(self.particle_count, 1.0 / self.particle_count)
        self.cached_ph: np.ndarray | None = None
        self.update_count = 0
        self.last_log_predictive = 0.0

    @property
    def concentrations_m(self) -> np.ndarray:
        return np.exp(self.log_concentration_particles)

    def predict(self, initial_volume_ml: float, state: SolutionState) -> np.ndarray:
        return solve_ph_particles(
            self.concentrations_m,
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
        log_weights = log_terms - log_predictive
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)

        ess = float(1.0 / np.sum(self.weights**2))
        if ess < ESS_THRESHOLD_FRACTION * self.particle_count:
            indices = _resample_indices(self.weights, self.rng)
            selected_pka = self.pka_particles[indices]
            if self.infer_concentration:
                transformed = np.column_stack(
                    [self.log_concentration_particles[indices], selected_pka]
                )
            else:
                transformed = selected_pka
            mean = np.mean(transformed, axis=0)
            covariance = _stable_covariance(transformed)
            a = np.sqrt(1.0 - LIU_WEST_H**2)
            roughening_covariance = LIU_WEST_H**2 * covariance
            transform = np.linalg.cholesky(
                roughening_covariance + np.eye(covariance.shape[0]) * 1e-8
            )
            noise = self.rng.normal(size=transformed.shape) @ transform.T
            regularized = a * transformed + (1.0 - a) * mean + noise
            if self.infer_concentration:
                self.log_concentration_particles = np.clip(
                    regularized[:, 0],
                    np.log(PRIOR_CONC_LOW_M),
                    np.log(PRIOR_CONC_HIGH_M),
                )
                self.pka_particles = regularized[:, 1:]
            else:
                self.pka_particles = regularized
            self.pka_particles = np.clip(
                np.sort(self.pka_particles, axis=1),
                PKA_CLIP_LOW,
                PKA_CLIP_HIGH,
            )
            self.weights.fill(1.0 / self.particle_count)
            self.cached_ph = self.predict(initial_volume_ml, after_state)
        else:
            self.cached_ph = predicted_after

        self.update_count += 1
        self.last_log_predictive = float(log_predictive)
        return float(log_predictive)

    def estimate(self) -> PosteriorEstimate:
        concentration = self.concentrations_m
        concentration_mean = float(np.sum(self.weights * concentration))
        concentration_sd = float(
            np.sqrt(np.sum(self.weights * (concentration - concentration_mean) ** 2))
        )
        pka_mean = np.sum(self.weights[:, None] * self.pka_particles, axis=0)
        pka_sd = np.sqrt(
            np.sum(self.weights[:, None] * (self.pka_particles - pka_mean) ** 2, axis=0)
        )
        return PosteriorEstimate(
            concentration_m=concentration_mean,
            concentration_sd_m=concentration_sd,
            pka_values=np.asarray(pka_mean, dtype=float),
            pka_sd=np.asarray(pka_sd, dtype=float),
            pair_count=self.pair_count,
            pair_probabilities=np.eye(3, dtype=float)[self.pair_count - 1],
            effective_sample_size=float(1.0 / np.sum(self.weights**2)),
        )


class VariableKParticleFilter:
    """Rao-Blackwellized model bank for K=1, 2, and 3 effective pairs."""

    def __init__(self, particle_count: int, rng: np.random.Generator) -> None:
        if particle_count < 60:
            raise ValueError("At least 60 total particles are required for variable K")
        allocations = [particle_count // 3] * 3
        for index in range(particle_count - sum(allocations)):
            allocations[index] += 1
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=3, dtype=np.uint32)
        self.banks = {
            pair_count: FixedKParticleFilter(
                allocations[pair_count - 1],
                pair_count,
                infer_concentration=True,
                rng=np.random.default_rng(int(seeds[pair_count - 1])),
            )
            for pair_count in (1, 2, 3)
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
                self.banks[k].update(
                    initial_volume_ml,
                    before_state,
                    after_state,
                    observed_before_ph,
                    observed_after_ph,
                )
                for k in (1, 2, 3)
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

    def estimate(self) -> PosteriorEstimate:
        probabilities = self.model_probabilities
        map_k = int(np.argmax(probabilities) + 1)
        conditional = self.banks[map_k].estimate()
        return PosteriorEstimate(
            concentration_m=conditional.concentration_m,
            concentration_sd_m=conditional.concentration_sd_m,
            pka_values=conditional.pka_values,
            pka_sd=conditional.pka_sd,
            pair_count=map_k,
            pair_probabilities=probabilities.copy(),
            effective_sample_size=conditional.effective_sample_size,
        )


def build_filter(variant: str, particle_count: int, seed: int):
    rng = np.random.default_rng(int(seed))
    if variant == "pf_pka_only_k3":
        return FixedKParticleFilter(particle_count, 3, False, rng)
    if variant == "pf_pka_conc_k3":
        return FixedKParticleFilter(particle_count, 3, True, rng)
    if variant == "pf_pka_conc_variable_k":
        return VariableKParticleFilter(particle_count, rng)
    raise KeyError(f"Unknown particle-filter variant: {variant}")


PF_VARIANTS = (
    "pf_pka_only_k3",
    "pf_pka_conc_k3",
    "pf_pka_conc_variable_k",
)
