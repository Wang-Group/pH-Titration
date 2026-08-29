from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

from chemistry_model import solve_ph_scalar
from particle_inference import (
    LIKELIHOOD_SCALE_PH,
    PRIOR_CONC_HIGH_M,
    PRIOR_CONC_LOW_M,
    PRIOR_PKA_HIGH,
    PRIOR_PKA_LOW,
    STUDENT_DF,
    PosteriorEstimate,
)


PYMC_VARIANTS = (
    "pymc_pka_only_k3",
    "pymc_pka_conc_k3",
    "pymc_pka_conc_variable_k",
)


@dataclass(frozen=True)
class PyMCFit:
    estimate: PosteriorEstimate
    runtime_seconds: float
    log_evidence_by_k: np.ndarray
    draws: int
    chains: int


def _student_logpdf(residual_ph: float) -> float:
    standardized = residual_ph / LIKELIHOOD_SCALE_PH
    return float(
        gammaln((STUDENT_DF + 1.0) / 2.0)
        - gammaln(STUDENT_DF / 2.0)
        - 0.5 * math.log(STUDENT_DF * math.pi)
        - math.log(LIKELIHOOD_SCALE_PH)
        - 0.5 * (STUDENT_DF + 1.0) * math.log1p(standardized**2 / STUDENT_DF)
    )


def trajectory_log_likelihood(task, transitions, concentration_m: float, pka_values: Sequence[float]) -> float:
    total = 0.0
    pka = np.sort(np.asarray(pka_values, dtype=float))
    for transition in transitions:
        predicted_before = solve_ph_scalar(
            concentration_m,
            pka,
            task.initial_volume_ml,
            transition.before_state,
        )
        predicted_after = solve_ph_scalar(
            concentration_m,
            pka,
            task.initial_volume_ml,
            transition.after_state,
        )
        predicted_delta = predicted_after - predicted_before
        observed_delta = transition.observed_after_ph - transition.observed_before_ph
        total += _student_logpdf(observed_delta - predicted_delta)
    return float(total)


def _extract_log_evidence(idata) -> float:
    if "log_marginal_likelihood" not in idata.sample_stats:
        raise RuntimeError("PyMC SMC result did not contain log_marginal_likelihood")
    values = np.asarray(idata.sample_stats["log_marginal_likelihood"], dtype=float)
    chain_values = []
    for chain in values.reshape(values.shape[0], -1):
        finite = chain[np.isfinite(chain)]
        if len(finite):
            chain_values.append(float(finite[-1]))
    if not chain_values:
        raise RuntimeError("PyMC SMC returned no finite marginal-likelihood estimate")
    return float(logsumexp(chain_values) - math.log(len(chain_values)))


def _fit_fixed_k(task, transitions, pair_count: int, infer_concentration: bool, draws: int, chains: int, seed: int):
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from pytensor.compile.ops import wrap_py
    except ImportError as exc:  # pragma: no cover - exercised on the run device.
        raise RuntimeError(
            "PyMC is not installed. Run INSTALL_ENV.cmd or pip install -r requirements.txt."
        ) from exc

    if infer_concentration:
        @wrap_py(itypes=[pt.dscalar, pt.dvector], otypes=[pt.dscalar])
        def loglike_op(log_concentration, raw_pka):
            value = trajectory_log_likelihood(
                task,
                transitions,
                float(np.exp(log_concentration)),
                np.sort(np.asarray(raw_pka, dtype=float)),
            )
            return np.asarray(value, dtype=np.float64)
    else:
        @wrap_py(itypes=[pt.dvector], otypes=[pt.dscalar])
        def loglike_op(raw_pka):
            value = trajectory_log_likelihood(
                task,
                transitions,
                0.1,
                np.sort(np.asarray(raw_pka, dtype=float)),
            )
            return np.asarray(value, dtype=np.float64)

    with pm.Model() as model:
        pka_raw = pm.Uniform(
            "pka_raw",
            lower=PRIOR_PKA_LOW,
            upper=PRIOR_PKA_HIGH,
            shape=pair_count,
            default_transform=None,
        )
        if infer_concentration:
            log_concentration = pm.Uniform(
                "log_concentration",
                lower=np.log(PRIOR_CONC_LOW_M),
                upper=np.log(PRIOR_CONC_HIGH_M),
                default_transform=None,
            )
            pm.Potential("trajectory_likelihood", loglike_op(log_concentration, pka_raw))
        else:
            pm.Potential("trajectory_likelihood", loglike_op(pka_raw))

        start = time.perf_counter()
        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )
        runtime = time.perf_counter() - start

    pka_draws = np.asarray(idata.posterior["pka_raw"], dtype=float).reshape(-1, pair_count)
    pka_draws = np.sort(pka_draws, axis=1)
    if infer_concentration:
        concentration_draws = np.exp(
            np.asarray(idata.posterior["log_concentration"], dtype=float).reshape(-1)
        )
    else:
        concentration_draws = np.full(len(pka_draws), 0.1)
    estimate = PosteriorEstimate(
        concentration_m=float(np.mean(concentration_draws)),
        concentration_sd_m=float(np.std(concentration_draws, ddof=1)) if len(concentration_draws) > 1 else 0.0,
        pka_values=np.mean(pka_draws, axis=0),
        pka_sd=np.std(pka_draws, axis=0, ddof=1) if len(pka_draws) > 1 else np.zeros(pair_count),
        pair_count=pair_count,
        pair_probabilities=np.eye(3)[pair_count - 1],
        effective_sample_size=float("nan"),
    )
    return estimate, _extract_log_evidence(idata), runtime


def fit_pymc_variant(task, transitions, variant: str, draws: int, chains: int, seed: int) -> PyMCFit:
    if variant == "pymc_pka_only_k3":
        estimate, evidence, runtime = _fit_fixed_k(
            task, transitions, 3, False, draws, chains, seed
        )
        return PyMCFit(estimate, runtime, np.array([np.nan, np.nan, evidence]), draws, chains)
    if variant == "pymc_pka_conc_k3":
        estimate, evidence, runtime = _fit_fixed_k(
            task, transitions, 3, True, draws, chains, seed
        )
        return PyMCFit(estimate, runtime, np.array([np.nan, np.nan, evidence]), draws, chains)
    if variant != "pymc_pka_conc_variable_k":
        raise KeyError(f"Unknown PyMC variant: {variant}")

    estimates = []
    evidences = []
    runtime = 0.0
    for pair_count in (1, 2, 3):
        estimate, evidence, elapsed = _fit_fixed_k(
            task,
            transitions,
            pair_count,
            True,
            draws,
            chains,
            seed + 1009 * pair_count,
        )
        estimates.append(estimate)
        evidences.append(evidence)
        runtime += elapsed
    log_posterior = np.asarray(evidences, dtype=float) - math.log(3.0)
    probabilities = np.exp(log_posterior - logsumexp(log_posterior))
    map_index = int(np.argmax(probabilities))
    selected = estimates[map_index]
    estimate = PosteriorEstimate(
        concentration_m=selected.concentration_m,
        concentration_sd_m=selected.concentration_sd_m,
        pka_values=selected.pka_values,
        pka_sd=selected.pka_sd,
        pair_count=map_index + 1,
        pair_probabilities=probabilities,
        effective_sample_size=float("nan"),
    )
    return PyMCFit(estimate, runtime, np.asarray(evidences), draws, chains)
