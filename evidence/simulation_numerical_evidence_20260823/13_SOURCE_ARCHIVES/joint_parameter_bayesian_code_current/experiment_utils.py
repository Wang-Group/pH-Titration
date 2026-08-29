from __future__ import annotations

import json
import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from chemistry_model import SolutionState, full_base_curve, response_curve
from particle_controllers import SUCCESS_THRESHOLD, build_controller
from particle_inference import PF_VARIANTS, PosteriorEstimate, build_filter


@dataclass(frozen=True)
class Transition:
    step: int
    reagent: str
    requested_volume_ml: float
    before_state: SolutionState
    after_state: SolutionState
    observed_before_ph: float
    observed_after_ph: float


@dataclass
class ControlResult:
    seed: int
    task_id: int
    method: str
    acid_type: str
    true_pkas: str
    true_pair_count: int
    true_concentration_m: float
    initial_ph: float
    target_ph: float
    final_ph: float
    success: bool
    steps: int
    overshoots: int
    acid_added_ml: float
    base_added_ml: float
    decision_time_mean_ms: float
    decision_time_median_ms: float
    update_time_mean_ms: float
    estimated_concentration_m: float
    estimated_pair_count: int
    estimated_pkas: str
    pair_probabilities: str

    def to_dict(self):
        return asdict(self)


def run_control_episode(task, variant: str, particles: int, rng_seed: int, keep_trajectory=False):
    # The task-level seed is deliberately derived from the experiment seed and
    # task id. Normalize it for NumPy's legacy global RNG, which only accepts
    # unsigned 32-bit values; Generator/SeedSequence used elsewhere can retain
    # the full integer seed.
    np.random.seed(int(rng_seed) % (2**32))
    controller = build_controller(variant, particles, rng_seed + 17)
    controller.initialize_task(task)
    if abs(controller.current_ph - controller.target_ph) <= SUCCESS_THRESHOLD:
        controller.done = True
    transitions: list[Transition] = []
    decision_times = []
    update_times = []
    overshoots = 0
    while not controller.done:
        before_state = SolutionState(
            controller.total_volume,
            controller.base_added_moles,
            controller.acid_added_moles,
        )
        observed_before = float(controller.current_ph)
        start = time.perf_counter()
        action, _ = controller.select_best_action()
        decision_times.append((time.perf_counter() - start) * 1000.0)
        current_ph, _, done, info = controller.step(action, mode="Simulate")
        after_state = SolutionState(
            controller.total_volume,
            controller.base_added_moles,
            controller.acid_added_moles,
        )
        start = time.perf_counter()
        controller.update_posteriors(action, current_ph)
        update_times.append((time.perf_counter() - start) * 1000.0)
        overshoots += int(bool(info.get("crossed_target", False)))
        if keep_trajectory:
            transitions.append(
                Transition(
                    step=int(controller.steps_taken),
                    reagent=str(action[0]),
                    requested_volume_ml=float(action[1]),
                    before_state=before_state,
                    after_state=after_state,
                    observed_before_ph=observed_before,
                    observed_after_ph=float(current_ph),
                )
            )
        if done:
            break
    estimate = controller.posterior_estimate()
    result = ControlResult(
        seed=int(task.seed),
        task_id=int(task.task_id),
        method=variant,
        acid_type=str(task.acid_type),
        true_pkas=json.dumps(task.pka_values),
        true_pair_count=len(task.pka_values),
        true_concentration_m=float(task.analyte_conc_m),
        initial_ph=float(task.initial_ph),
        target_ph=float(task.target_ph),
        final_ph=float(controller.current_ph),
        success=abs(controller.current_ph - controller.target_ph) <= SUCCESS_THRESHOLD,
        steps=int(controller.steps_taken),
        overshoots=int(overshoots),
        acid_added_ml=float(controller.acid_volume),
        base_added_ml=float(controller.base_volume),
        decision_time_mean_ms=float(np.mean(decision_times)) if decision_times else 0.0,
        decision_time_median_ms=float(np.median(decision_times)) if decision_times else 0.0,
        update_time_mean_ms=float(np.mean(update_times)) if update_times else 0.0,
        estimated_concentration_m=estimate.concentration_m,
        estimated_pair_count=estimate.pair_count,
        estimated_pkas=json.dumps(estimate.pka_values.tolist()),
        pair_probabilities=json.dumps(estimate.pair_probabilities.tolist()),
    )
    return result, transitions, estimate


def replay_particle_filter(task, transitions: Sequence[Transition], variant: str, particles: int, seed: int):
    inference = build_filter(variant, particles, seed)
    snapshots = {0: inference.estimate()}
    for transition in transitions:
        inference.update(
            task.initial_volume_ml,
            transition.before_state,
            transition.after_state,
            transition.observed_before_ph,
            transition.observed_after_ph,
        )
        snapshots[transition.step] = inference.estimate()
    return inference.estimate(), snapshots


def matched_pka_errors(true_pkas: Sequence[float], estimate: PosteriorEstimate):
    truth = np.asarray(true_pkas, dtype=float)
    predicted = np.asarray(estimate.pka_values, dtype=float)
    cost = np.abs(truth[:, None] - predicted[None, :])
    truth_index, predicted_index = linear_sum_assignment(cost)
    matched = cost[truth_index, predicted_index]
    unmatched = abs(len(truth) - len(predicted))
    penalty = 3.0
    penalized = (float(np.sum(matched)) + penalty * unmatched) / max(len(truth), len(predicted))
    return {
        "pka_matched_mae": float(np.mean(matched)) if len(matched) else math.nan,
        "pka_matched_rmse": float(np.sqrt(np.mean(matched**2))) if len(matched) else math.nan,
        "pka_penalized_mae": float(penalized),
        "matched_pairs": int(len(matched)),
        "unmatched_pair_count": int(unmatched),
    }


def curve_metrics(task, state: SolutionState, estimate: PosteriorEstimate, windows=(0.10, 0.25, 0.50)):
    maximum = max(windows)
    probe = np.arange(-maximum, maximum + 0.005, 0.01)
    true_local = response_curve(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        state,
        probe,
    )
    model_local = response_curve(
        estimate.concentration_m,
        estimate.pka_values,
        task.initial_volume_ml,
        state,
        probe,
    )
    zero = int(np.argmin(np.abs(probe)))
    true_delta = true_local - true_local[zero]
    model_delta = model_local - model_local[zero]
    output = {}
    for window in windows:
        mask = np.abs(probe) <= window + 1e-12
        error = model_delta[mask] - true_delta[mask]
        key = f"{float(window):.2f}".replace(".", "p")
        output[f"local_rmse_{key}ml_ph"] = float(np.sqrt(np.mean(error**2)))
        output[f"local_mae_{key}ml_ph"] = float(np.mean(np.abs(error)))

    grid = np.arange(0.0, 33.0 + 0.1, 0.2)
    true_full = full_base_curve(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        task.initial_ph,
        grid,
    )
    model_full = full_base_curve(
        estimate.concentration_m,
        estimate.pka_values,
        task.initial_volume_ml,
        task.initial_ph,
        grid,
    )
    full_error = model_full - true_full
    output["full_curve_rmse_0_33ml_ph"] = float(np.sqrt(np.mean(full_error**2)))
    output["full_curve_mae_0_33ml_ph"] = float(np.mean(np.abs(full_error)))
    return output


def summarize_control(rows: Iterable[ControlResult]):
    values = list(rows)
    successful = [row for row in values if row.success]
    total_steps = sum(row.steps for row in values)
    return {
        "tasks": len(values),
        "success_percent": 100.0 * np.mean([row.success for row in values]),
        "successful_steps_mean": float(np.mean([row.steps for row in successful])) if successful else math.nan,
        "overshoot_events_per_step_percent": 100.0 * sum(row.overshoots for row in values) / total_steps if total_steps else 0.0,
        "final_abs_error_mean_ph": float(np.mean([abs(row.final_ph - row.target_ph) for row in values])),
        "total_titrant_volume_mean_ml": float(np.mean([row.acid_added_ml + row.base_added_ml for row in values])),
        "decision_time_median_ms": float(np.median([row.decision_time_median_ms for row in values])),
        "update_time_mean_ms": float(np.mean([row.update_time_mean_ms for row in values])),
        "pair_count_accuracy_percent": 100.0 * np.mean([row.true_pair_count == row.estimated_pair_count for row in values]),
        "concentration_relative_error_median_percent": 100.0 * float(np.median([
            abs(row.estimated_concentration_m - row.true_concentration_m) / row.true_concentration_m
            for row in values
        ])),
    }


def exact_sign_flip_p(values: Sequence[float]) -> float:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return math.nan
    observed = abs(float(np.mean(differences)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = differences * np.asarray(signs, dtype=float)
        exceed += abs(float(np.mean(permuted))) >= observed - 1e-12
        total += 1
    return float(exceed / total)


def paired_seed_bootstrap(
    values: Sequence[float],
    repetitions: int = 20_000,
    seed: int = 20260811,
) -> tuple[float, float]:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    means = np.mean(differences[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
