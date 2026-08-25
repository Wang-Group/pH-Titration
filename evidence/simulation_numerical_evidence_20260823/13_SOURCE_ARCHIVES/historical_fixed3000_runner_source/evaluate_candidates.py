from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon

from benchmark_core import (
    MAX_STEPS,
    MIN_VOLUME_ML,
    NeuralVolumePolicy,
    PolicyEnvironment,
    exact_mcnemar,
    generate_tasks,
    holm_adjust,
)
from challenge_common import (
    BayesianAdapter,
    CategoricalActorCritic,
    CONFIRM_STRESS_SCENARIOS,
    DeterministicActor,
    GaussianActor,
    SCENARIOS,
    bayesian_feature_vector,
    categorical_action,
    continuous_action,
    deterministic_action,
    load_bayesian_module,
    make_features,
    residual_action,
)


class LoadedCandidate:
    def __init__(self, path: Path, device: torch.device) -> None:
        payload = torch.load(path, map_location=device)
        self.metadata = dict(payload["metadata"])
        self.name = str(self.metadata["name"])
        self.algorithm = str(self.metadata["algorithm"])
        self.state_mode = str(self.metadata["state_mode"])
        self.residual = bool(self.metadata["residual"])
        self.device = device
        if self.algorithm == "sac":
            self.actor = GaussianActor(int(self.metadata["input_dim"])).to(device)
            self.actor.load_state_dict(payload["actor_state_dict"])
            self.actor.eval()
            self.model = None
        elif self.algorithm == "td3":
            self.actor = DeterministicActor(int(self.metadata["input_dim"])).to(device)
            self.actor.load_state_dict(payload["actor_state_dict"])
            self.actor.eval()
            self.model = None
        else:
            self.model = CategoricalActorCritic(int(self.metadata["input_dim"]), int(self.metadata["action_dim"])).to(device)
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.eval()
            self.actor = None


def result_row(train_seed, eval_seed, scenario, method, task, env, elapsed_ms: float) -> dict[str, Any]:
    final_error = float(env.true_ph - env.target_ph)
    initial_error = float(task.initial_ph - task.target_ph)
    initial_distance = abs(initial_error)
    return {
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "scenario": scenario,
        "method": method,
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "pka_values": json.dumps(task.pka_values),
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "direction": "increase" if initial_error < 0 else "decrease",
        "difficulty_bin": "near" if initial_distance < 2.0 else ("medium" if initial_distance < 5.0 else "far"),
        "initial_abs_error": initial_distance,
        "final_true_ph": float(env.true_ph),
        "final_measured_ph": float(env.measured_ph),
        "true_success": abs(env.true_ph - env.target_ph) <= 0.1,
        "success_005": abs(env.true_ph - env.target_ph) <= 0.05,
        "success_020": abs(env.true_ph - env.target_ph) <= 0.20,
        "severe_failure_050": abs(env.true_ph - env.target_ph) > 0.50,
        "measured_success": abs(env.measured_ph - env.target_ph) <= 0.1,
        "steps": int(env.steps),
        "overshoots": int(env.overshoots),
        "acid_added_ml": float(env.acid_added_ml),
        "base_added_ml": float(env.base_added_ml),
        "total_added_ml": float(env.acid_added_ml + env.base_added_ml),
        "final_abs_error": abs(final_error),
        "final_signed_error": final_error,
        "decision_time_ms": float(elapsed_ms),
    }


def run_candidate(candidate: LoadedCandidate, imitation: NeuralVolumePolicy, task, scenario, method: str, rng_seed: int, train_seed: int, eval_seed: int) -> dict:
    env = PolicyEnvironment(task, scenario, np.random.default_rng(rng_seed))
    history: deque = deque(maxlen=3)
    started = time.perf_counter()
    with torch.no_grad():
        while not env.done:
            raw = env.state().copy()
            maximum = env.maximum_requested_volume()
            if candidate.residual:
                base = imitation.select_volume(raw, maximum)
                features = make_features(env, candidate.state_mode, base_volume=base, history=history)
                _, volume, _, _ = residual_action(candidate.model, features, base, maximum, candidate.device, False)
            elif candidate.algorithm == "sac":
                features = make_features(env, candidate.state_mode, history=history)
                volume, _ = continuous_action(candidate.actor, features, maximum, candidate.device, False)
            elif candidate.algorithm == "td3":
                features = make_features(env, candidate.state_mode, history=history)
                volume, _ = deterministic_action(candidate.actor, features, maximum, candidate.device)
            else:
                features = make_features(env, candidate.state_mode, history=history)
                _, volume, _, _ = categorical_action(candidate.model, features, maximum, candidate.device, False)
            env.step(volume)
            history.append(raw)
    elapsed = (time.perf_counter() - started) * 1000.0
    return result_row(train_seed, eval_seed, scenario.name, method, task, env, elapsed)


def run_reference(policy: NeuralVolumePolicy, task, scenario, method, rng_seed, train_seed, eval_seed) -> dict:
    env = PolicyEnvironment(task, scenario, np.random.default_rng(rng_seed))
    started = time.perf_counter()
    while not env.done:
        env.step(policy.select_volume(env.state(), env.maximum_requested_volume()))
    elapsed = (time.perf_counter() - started) * 1000.0
    return result_row(train_seed, eval_seed, scenario.name, method, task, env, elapsed)


def run_original_bayesian(module, task, particles: int, rng_seed: int, train_seed: int, eval_seed: int, residual: LoadedCandidate | None = None) -> dict:
    np.random.seed(int(rng_seed) % (2**32 - 1))
    env = module.PHAdjustmentEnv(num_particles=particles)
    env.initialize(task.acid_type, list(task.pka_values), task.initial_ph, task.target_ph, module.MAX_STEPS)
    overshoots = 0
    started = time.perf_counter()
    with torch.no_grad():
        while not env.done:
            reagent, base_volume = env.select_best_action()[0]
            volume = float(base_volume)
            if residual is not None:
                maximum = float(env.overshoot_threshold if env.overshoot_threshold is not None else 9.99)
                features = bayesian_feature_vector(env, volume, maximum, residual.state_mode)
                _, volume, _, _ = residual_action(residual.model, features, volume, maximum, residual.device, False)
                volume = round(max(MIN_VOLUME_ML, volume) / 0.01) * 0.01
            current_ph, _, done, info = env.step((reagent, volume), mode="Simulate")
            env.update_posteriors((reagent, volume), current_ph)
            overshoots += int(bool(info.get("crossed_target", False)))
            if done:
                break
    elapsed = (time.perf_counter() - started) * 1000.0
    final_error = float(env.current_ph - env.target_ph)
    initial_error = float(task.initial_ph - task.target_ph)
    initial_distance = abs(initial_error)
    return {
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "scenario": "nominal",
        "method": "bayesian_residual_ppo" if residual is not None else "bayesian_original",
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "pka_values": json.dumps(task.pka_values),
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "direction": "increase" if initial_error < 0 else "decrease",
        "difficulty_bin": "near" if initial_distance < 2.0 else ("medium" if initial_distance < 5.0 else "far"),
        "initial_abs_error": initial_distance,
        "final_true_ph": float(env.current_ph),
        "final_measured_ph": float(env.current_ph),
        "true_success": abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD,
        "success_005": abs(env.current_ph - env.target_ph) <= 0.05,
        "success_020": abs(env.current_ph - env.target_ph) <= 0.20,
        "severe_failure_050": abs(env.current_ph - env.target_ph) > 0.50,
        "measured_success": abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD,
        "steps": int(env.steps_taken),
        "overshoots": overshoots,
        "acid_added_ml": float(env.acid_volume),
        "base_added_ml": float(env.base_volume),
        "total_added_ml": float(env.acid_volume + env.base_volume),
        "final_abs_error": abs(final_error),
        "final_signed_error": final_error,
        "decision_time_ms": float(elapsed),
    }


def run_common_bayesian(module, task, scenario, particles, rng_seed, train_seed, eval_seed, residual: LoadedCandidate | None = None) -> dict:
    env = PolicyEnvironment(task, scenario, np.random.default_rng(rng_seed))
    adapter = BayesianAdapter(module, particles, rng_seed)
    adapter.reset(task)
    history: deque = deque(maxlen=3)
    started = time.perf_counter()
    with torch.no_grad():
        while not env.done:
            raw = env.state().copy()
            base = adapter.select(env)
            volume = base.volume
            if residual is not None:
                features = make_features(env, residual.state_mode, base_volume=volume, history=history)
                _, volume, _, _ = residual_action(residual.model, features, volume, env.maximum_requested_volume(), residual.device, False)
            suffix = "2" if env.use_secondary else "1"
            direction = "base" if env.measured_ph < env.target_ph else "acid"
            reagent = f"Dilute {direction} {suffix}"
            env.step(volume)
            adapter.observe(env, volume, reagent)
            history.append(raw)
    elapsed = (time.perf_counter() - started) * 1000.0
    method = "bayesian_common_residual_ppo" if residual is not None else "bayesian_common"
    return result_row(train_seed, eval_seed, scenario.name, method, task, env, elapsed)


def run_seed_job(job) -> list[dict]:
    (
        train_seed, eval_seed, candidate_dir, imitation_path, submitted_rl_path, ppo_reference_dir,
        bayesian_source, candidate_names, nominal_tasks, stress_tasks, scenario_names,
        nominal_particles, stress_particles, device_name,
    ) = job
    device = torch.device(device_name)
    candidates = {
        name: LoadedCandidate(Path(candidate_dir) / f"{name}_seed{train_seed}.pth", device)
        for name in candidate_names
    }
    residual = candidates.get("ppo_residual_robust")
    imitation = NeuralVolumePolicy(Path(imitation_path), device_name)
    submitted_rl = NeuralVolumePolicy(Path(submitted_rl_path), device_name)
    ppo_reference = NeuralVolumePolicy(Path(ppo_reference_dir) / f"ppo_full_seed{train_seed}.pth", device_name)
    module = load_bayesian_module(Path(bayesian_source))
    rows: list[dict] = []

    nominal = SCENARIOS["nominal"]
    tasks = generate_tasks(eval_seed, nominal_tasks, nominal)
    for index, task in enumerate(tasks, 1):
        rng_seed = eval_seed * 1_000_003 + task.task_id
        rows.append(run_original_bayesian(module, task, nominal_particles, rng_seed, train_seed, eval_seed))
        if residual is not None:
            rows.append(run_original_bayesian(module, task, nominal_particles, rng_seed, train_seed, eval_seed, residual))
        rows.append(run_reference(imitation, task, nominal, "imitation", rng_seed, train_seed, eval_seed))
        rows.append(run_reference(submitted_rl, task, nominal, "submitted_rl", rng_seed, train_seed, eval_seed))
        rows.append(run_reference(ppo_reference, task, nominal, "ppo_reference", rng_seed, train_seed, eval_seed))
        for name, candidate in candidates.items():
            method = "ppo_residual_imitation" if candidate.residual else name
            rows.append(run_candidate(candidate, imitation, task, nominal, method, rng_seed, train_seed, eval_seed))
        if index % 100 == 0:
            print(f"eval seed {eval_seed}, nominal: {index}/{len(tasks)}")

    for scenario_index, scenario_name in enumerate(scenario_names):
        if scenario_name == "nominal":
            continue
        scenario = SCENARIOS[scenario_name]
        tasks = generate_tasks(eval_seed + 10_000 + scenario_index * 101, stress_tasks, scenario)
        for index, task in enumerate(tasks, 1):
            rng_seed = (eval_seed + scenario_index * 31) * 1_000_003 + task.task_id
            rows.append(run_common_bayesian(module, task, scenario, stress_particles, rng_seed, train_seed, eval_seed))
            if residual is not None:
                rows.append(run_common_bayesian(module, task, scenario, stress_particles, rng_seed, train_seed, eval_seed, residual))
            rows.append(run_reference(imitation, task, scenario, "imitation", rng_seed, train_seed, eval_seed))
            rows.append(run_reference(submitted_rl, task, scenario, "submitted_rl", rng_seed, train_seed, eval_seed))
            rows.append(run_reference(ppo_reference, task, scenario, "ppo_reference", rng_seed, train_seed, eval_seed))
            for name, candidate in candidates.items():
                method = "ppo_residual_imitation" if candidate.residual else name
                rows.append(run_candidate(candidate, imitation, task, scenario, method, rng_seed, train_seed, eval_seed))
        print(f"eval seed {eval_seed}, {scenario_name}: {len(tasks)} tasks")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_summary(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(int(row["eval_seed"]), str(row["scenario"]), str(row["method"])) for row in rows})
    for eval_seed, scenario, method in keys:
        subset = [row for row in rows if int(row["eval_seed"]) == eval_seed and row["scenario"] == scenario and row["method"] == method]
        successful = [row for row in subset if bool(row["true_success"])]
        total_steps = sum(int(row["steps"]) for row in subset)
        output.append({
            "eval_seed": eval_seed,
            "scenario": scenario,
            "method": method,
            "tasks": len(subset),
            "success_rate_percent": 100.0 * len(successful) / len(subset),
            "strict_success_005_percent": 100.0 * sum(bool(row["success_005"]) for row in subset) / len(subset),
            "wide_success_020_percent": 100.0 * sum(bool(row["success_020"]) for row in subset) / len(subset),
            "severe_failure_050_percent": 100.0 * sum(bool(row["severe_failure_050"]) for row in subset) / len(subset),
            "successful_steps_mean": float(np.mean([int(row["steps"]) for row in successful])) if successful else math.nan,
            "all_steps_mean": float(np.mean([int(row["steps"]) for row in subset])),
            "steps_p95": float(np.percentile([int(row["steps"]) for row in subset], 95)),
            "overshoot_rate_percent": 100.0 * sum(int(row["overshoots"]) for row in subset) / max(1, total_steps),
            "overshoots_per_task_mean": float(np.mean([int(row["overshoots"]) for row in subset])),
            "total_added_ml_mean": float(np.mean([float(row["total_added_ml"]) for row in subset])),
            "total_added_ml_p95": float(np.percentile([float(row["total_added_ml"]) for row in subset], 95)),
            "final_abs_error_mean": float(np.mean([float(row["final_abs_error"]) for row in subset])),
            "final_abs_error_p95": float(np.percentile([float(row["final_abs_error"]) for row in subset], 95)),
            "final_abs_error_cvar95": upper_cvar([float(row["final_abs_error"]) for row in subset]),
            "total_added_ml_cvar95": upper_cvar([float(row["total_added_ml"]) for row in subset]),
            "steps_cvar95": upper_cvar([float(row["steps"]) for row in subset]),
            "final_signed_error_mean": float(np.mean([float(row["final_signed_error"]) for row in subset])),
            "false_stop_rate_percent": 100.0 * sum(bool(row["measured_success"]) and not bool(row["true_success"]) for row in subset) / len(subset),
            "decision_time_ms_mean": float(np.mean([float(row["decision_time_ms"]) for row in subset])),
            "decision_time_ms_p95": float(np.percentile([float(row["decision_time_ms"]) for row in subset], 95)),
        })
    return output


def aggregate_summary(seed_rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["scenario"], row["method"]) for row in seed_rows})
    metric_names = [
        "success_rate_percent", "strict_success_005_percent", "wide_success_020_percent", "severe_failure_050_percent",
        "successful_steps_mean", "all_steps_mean", "steps_p95", "steps_cvar95", "overshoot_rate_percent",
        "overshoots_per_task_mean", "total_added_ml_mean", "total_added_ml_p95", "final_abs_error_mean",
        "total_added_ml_cvar95", "final_abs_error_p95", "final_abs_error_cvar95", "final_signed_error_mean",
        "false_stop_rate_percent", "decision_time_ms_mean", "decision_time_ms_p95",
    ]
    for scenario, method in keys:
        subset = [row for row in seed_rows if row["scenario"] == scenario and row["method"] == method]
        item: dict[str, Any] = {"scenario": scenario, "method": method, "seeds": len(subset)}
        for metric in metric_names:
            values = [float(row[metric]) for row in subset if not math.isnan(float(row[metric]))]
            item[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            item[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def upper_cvar(values: list[float], alpha: float = 0.95) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(math.ceil((1.0 - alpha) * ordered.size)))
    return float(np.mean(ordered[-count:]))


def stratified_summary(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({
        (str(row["scenario"]), str(row["method"]), str(row["acid_type"]), str(row["direction"]), str(row["difficulty_bin"]))
        for row in rows
    })
    for scenario, method, acid_type, direction, difficulty in keys:
        subset = [
            row for row in rows
            if row["scenario"] == scenario and row["method"] == method and row["acid_type"] == acid_type
            and row["direction"] == direction and row["difficulty_bin"] == difficulty
        ]
        output.append({
            "scenario": scenario,
            "method": method,
            "acid_type": acid_type,
            "direction": direction,
            "difficulty_bin": difficulty,
            "tasks": len(subset),
            "success_rate_percent": 100.0 * sum(bool(row["true_success"]) for row in subset) / len(subset),
            "strict_success_005_percent": 100.0 * sum(bool(row["success_005"]) for row in subset) / len(subset),
            "severe_failure_050_percent": 100.0 * sum(bool(row["severe_failure_050"]) for row in subset) / len(subset),
            "final_abs_error_mean": float(np.mean([float(row["final_abs_error"]) for row in subset])),
            "steps_mean": float(np.mean([float(row["steps"]) for row in subset])),
            "total_added_ml_mean": float(np.mean([float(row["total_added_ml"]) for row in subset])),
        })
    return output


def bootstrap_difference(a: np.ndarray, b: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = a - b
    if differences.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, differences.size, size=(iterations, differences.size))
    samples = differences[indices].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def clustered_bootstrap_difference(differences: np.ndarray, clusters: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    if differences.size == 0 or unique.size == 0:
        return math.nan, math.nan
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected_clusters = rng.choice(unique, size=unique.size, replace=True)
        selected = []
        for cluster in selected_clusters:
            values = differences[clusters == cluster]
            selected.append(rng.choice(values, size=values.size, replace=True))
        samples[index] = float(np.mean(np.concatenate(selected)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    differences = a - b
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue)


def paired_tests(rows: list[dict], bootstrap_iterations: int) -> list[dict]:
    tests: list[dict] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        baseline = "bayesian_original" if scenario == "nominal" else "bayesian_common"
        methods = sorted({str(row["method"]) for row in rows if row["scenario"] == scenario and row["method"] != baseline})
        base_rows = {
            (int(row["eval_seed"]), int(row["task_id"])): row
            for row in rows if row["scenario"] == scenario and row["method"] == baseline
        }
        for method in methods:
            method_rows = {
                (int(row["eval_seed"]), int(row["task_id"])): row
                for row in rows if row["scenario"] == scenario and row["method"] == method
            }
            keys = sorted(set(base_rows) & set(method_rows))
            if not keys:
                continue
            b_success = np.asarray([bool(base_rows[key]["true_success"]) for key in keys], dtype=float)
            m_success = np.asarray([bool(method_rows[key]["true_success"]) for key in keys], dtype=float)
            mcnemar = exact_mcnemar(b_success.astype(bool).tolist(), m_success.astype(bool).tolist())
            success_diff = 100.0 * float((m_success - b_success).mean())
            ci_low, ci_high = bootstrap_difference(100.0 * m_success, 100.0 * b_success, bootstrap_iterations, 991 + len(tests))
            clusters = np.asarray([int(key[0]) for key in keys])
            cluster_low, cluster_high = clustered_bootstrap_difference(100.0 * (m_success - b_success), clusters, bootstrap_iterations, 1991 + len(tests))
            strict = exact_mcnemar(
                [bool(base_rows[key]["success_005"]) for key in keys],
                [bool(method_rows[key]["success_005"]) for key in keys],
            )
            wide = exact_mcnemar(
                [bool(base_rows[key]["success_020"]) for key in keys],
                [bool(method_rows[key]["success_020"]) for key in keys],
            )
            severe = exact_mcnemar(
                [bool(base_rows[key]["severe_failure_050"]) for key in keys],
                [bool(method_rows[key]["severe_failure_050"]) for key in keys],
            )
            metrics = {}
            for name in ("steps", "overshoots", "total_added_ml", "final_abs_error", "decision_time_ms"):
                base_values = np.asarray([float(base_rows[key][name]) for key in keys])
                method_values = np.asarray([float(method_rows[key][name]) for key in keys])
                metrics[f"{name}_mean_difference"] = float(np.mean(method_values - base_values))
                metrics[f"{name}_relative_change_percent"] = 100.0 * float(np.mean(method_values) - np.mean(base_values)) / max(abs(float(np.mean(base_values))), 1e-9)
                metrics[f"{name}_p_value"] = safe_wilcoxon(method_values, base_values)
            tests.append({
                "scenario": scenario,
                "baseline": baseline,
                "method": method,
                "matched_tasks": len(keys),
                "success_difference_pp": success_diff,
                "success_difference_ci95_low": ci_low,
                "success_difference_ci95_high": ci_high,
                "success_difference_cluster_ci95_low": cluster_low,
                "success_difference_cluster_ci95_high": cluster_high,
                "strict_success_005_difference_pp": 100.0 * float(np.mean([bool(method_rows[key]["success_005"]) - bool(base_rows[key]["success_005"]) for key in keys])),
                "strict_success_005_p_value": strict["p_value_exact_two_sided"],
                "wide_success_020_difference_pp": 100.0 * float(np.mean([bool(method_rows[key]["success_020"]) - bool(base_rows[key]["success_020"]) for key in keys])),
                "wide_success_020_p_value": wide["p_value_exact_two_sided"],
                "severe_failure_050_difference_pp": 100.0 * float(np.mean([bool(method_rows[key]["severe_failure_050"]) - bool(base_rows[key]["severe_failure_050"]) for key in keys])),
                "severe_failure_050_p_value": severe["p_value_exact_two_sided"],
                **mcnemar,
                **metrics,
            })
    success_adjusted = holm_adjust([float(row["p_value_exact_two_sided"]) for row in tests])
    strict_adjusted = holm_adjust([float(row["strict_success_005_p_value"]) for row in tests])
    wide_adjusted = holm_adjust([float(row["wide_success_020_p_value"]) for row in tests])
    severe_adjusted = holm_adjust([float(row["severe_failure_050_p_value"]) for row in tests])
    metric_adjusted = {
        name: holm_adjust([float(row[f"{name}_p_value"]) for row in tests])
        for name in ("steps", "overshoots", "total_added_ml", "final_abs_error", "decision_time_ms")
    }
    for index, row in enumerate(tests):
        row["success_p_value_holm"] = success_adjusted[index]
        row["strict_success_005_p_value_holm"] = strict_adjusted[index]
        row["wide_success_020_p_value_holm"] = wide_adjusted[index]
        row["severe_failure_050_p_value_holm"] = severe_adjusted[index]
        for name, values in metric_adjusted.items():
            row[f"{name}_p_value_holm"] = values[index]
        row["clear_success_win"] = row["success_difference_pp"] >= 1.0 and row["success_p_value_holm"] < 0.05
        noninferior = row["success_difference_cluster_ci95_low"] >= -0.5
        efficient = any(
            row[f"{name}_relative_change_percent"] <= -10.0 and row[f"{name}_p_value_holm"] < 0.05
            for name in ("steps", "overshoots", "total_added_ml")
        )
        no_error_harm = row["final_abs_error_mean_difference"] <= 0.01
        row["multiobjective_tradeoff_win"] = bool(noninferior and efficient and no_error_harm)
    return tests


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Confirmatory multi-metric comparison against Bayesian control.")
    parser.add_argument("--candidate-dir", type=Path, default=base / "results_challenge" / "training" / "models")
    parser.add_argument("--candidate-names", nargs="+", default=["ppo_nominal", "ppo_robust", "a2c_robust", "ppo_history_robust", "sac_history_robust", "ppo_residual_robust", "ppo_filtered_robust", "ppo_conservative_robust", "td3_filtered_robust"])
    parser.add_argument("--imitation-weights", type=Path, default=base / "models" / "imitation.pth")
    parser.add_argument("--submitted-rl-weights", type=Path, default=base / "models" / "reinforcement.pth")
    parser.add_argument("--ppo-reference-dir", type=Path, default=base / "models" / "ppo_reference")
    parser.add_argument("--bayesian-source", type=Path, default=base / "inputs" / "bayesian_controller.py")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[7101, 7202, 7303, 7404, 7555])
    parser.add_argument("--nominal-tasks", type=int, default=1000)
    parser.add_argument("--stress-tasks", type=int, default=200)
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=CONFIRM_STRESS_SCENARIOS)
    parser.add_argument("--nominal-particles", type=int, default=500)
    parser.add_argument("--stress-particles", type=int, default=200)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=base / "results_challenge" / "evaluation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.train_seeds) != len(args.eval_seeds):
        raise SystemExit("--train-seeds and --eval-seeds must have the same length.")
    if args.workers > 1 and args.device != "cpu":
        raise SystemExit("Multiple evaluation workers require --device cpu.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            train_seed, eval_seed, str(args.candidate_dir.resolve()), str(args.imitation_weights.resolve()),
            str(args.submitted_rl_weights.resolve()), str(args.ppo_reference_dir.resolve()), str(args.bayesian_source.resolve()),
            tuple(args.candidate_names), args.nominal_tasks, args.stress_tasks, tuple(args.scenarios),
            args.nominal_particles, args.stress_particles, args.device,
        )
        for train_seed, eval_seed in zip(args.train_seeds, args.eval_seeds)
    ]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            for part in executor.map(run_seed_job, jobs):
                rows.extend(part)
    else:
        for job in jobs:
            rows.extend(run_seed_job(job))
    write_csv(args.output_dir / "per_task_results.csv", rows)
    seed_rows = seed_summary(rows)
    write_csv(args.output_dir / "per_seed_summary.csv", seed_rows)
    aggregate = aggregate_summary(seed_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    write_csv(args.output_dir / "stratified_summary.csv", stratified_summary(rows))
    tests = paired_tests(rows, args.bootstrap_iterations)
    write_csv(args.output_dir / "paired_tests.csv", tests)
    settings = vars(args).copy()
    for key in ("candidate_dir", "imitation_weights", "submitted_rl_weights", "ppo_reference_dir", "bayesian_source", "output_dir"):
        settings[key] = str(settings[key])
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
