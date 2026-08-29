from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from benchmark_core import exact_mcnemar, holm_adjust
from chemistry_model import SolutionState
from experiment_utils import (
    curve_metrics,
    exact_sign_flip_p,
    matched_pka_errors,
    paired_seed_bootstrap,
    replay_particle_filter,
    run_control_episode,
)
from io_utils import read_json, write_csv, write_json
from particle_inference import PF_VARIANTS
from task_generation import DEFAULT_SEEDS, generate_comparison_tasks


CHECKPOINTS = ("after_step4", "after_step8", "last_decision", "final")


def checkpoint_definitions(transitions, snapshots, initial_volume_ml):
    count = len(transitions)
    if count == 0:
        initial_state = SolutionState(float(initial_volume_ml), 0.0, 0.0)
        return {name: (snapshots[0], initial_state, 0) for name in CHECKPOINTS}

    def after_step(wanted):
        actual = min(wanted, count)
        state = transitions[actual - 1].after_state if actual else transitions[0].before_state
        return snapshots[actual], state, actual

    last_prefix = max(0, count - 1)
    last_state = transitions[-1].before_state
    final_state = transitions[-1].after_state
    return {
        "after_step4": after_step(4),
        "after_step8": after_step(8),
        "last_decision": (snapshots[last_prefix], last_state, last_prefix),
        "final": (snapshots[count], final_state, count),
    }


def run_seed_job(job):
    seed, count, particles, minimum_error, shard_directory, resume = job
    config = {
        "seed": seed,
        "tasks_per_seed": count,
        "particles": particles,
        "minimum_initial_error_ph": minimum_error,
        "distribution": "variable_concentration",
        "pka_prior_support": [1.5, 9.0],
    }
    shard_path = Path(shard_directory) / f"curve_seed{seed}.json"
    expected_rows = count * len(PF_VARIANTS) * len(CHECKPOINTS)
    if resume and shard_path.exists():
        try:
            payload = read_json(shard_path)
            stored_rows = payload.get("rows", [])
            if payload.get("config") == config and len(stored_rows) == expected_rows:
                return stored_rows
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    tasks = generate_comparison_tasks(
        seed,
        count,
        "variable_concentration",
        minimum_initial_error_ph=minimum_error,
    )
    rows = []
    for index, task in enumerate(tasks, 1):
        trajectory_seed = seed * 1_000_003 + task.task_id
        baseline_result, transitions, _ = run_control_episode(
            task,
            "pf_pka_only_k3",
            particles,
            trajectory_seed,
            keep_trajectory=True,
        )
        for variant_index, variant in enumerate(PF_VARIANTS):
            estimate, snapshots = replay_particle_filter(
                task,
                transitions,
                variant,
                particles,
                trajectory_seed + 10_000 * (variant_index + 1),
            )
            definitions = checkpoint_definitions(transitions, snapshots, task.initial_volume_ml)
            for checkpoint, (checkpoint_estimate, state, observed_steps) in definitions.items():
                metrics = curve_metrics(task, state, checkpoint_estimate)
                row = {
                    "seed": seed,
                    "task_id": task.task_id,
                    "method": variant,
                    "acid_type": task.acid_type,
                    "true_pair_count": len(task.pka_values),
                    "true_pkas": json.dumps(task.pka_values),
                    "true_concentration_m": task.analyte_conc_m,
                    "trajectory_success": baseline_result.success,
                    "trajectory_steps": len(transitions),
                    "checkpoint": checkpoint,
                    "observed_updates": observed_steps,
                    "estimated_pair_count": checkpoint_estimate.pair_count,
                    "pair_probability_k1": checkpoint_estimate.pair_probabilities[0],
                    "pair_probability_k2": checkpoint_estimate.pair_probabilities[1],
                    "pair_probability_k3": checkpoint_estimate.pair_probabilities[2],
                    "estimated_concentration_m": checkpoint_estimate.concentration_m,
                    "concentration_abs_error_m": abs(checkpoint_estimate.concentration_m - task.analyte_conc_m),
                    "concentration_relative_error_percent": 100.0 * abs(checkpoint_estimate.concentration_m - task.analyte_conc_m) / task.analyte_conc_m,
                    "estimated_pkas": json.dumps(checkpoint_estimate.pka_values.tolist()),
                    **matched_pka_errors(task.pka_values, checkpoint_estimate),
                    **metrics,
                }
                rows.append(row)
        if index % 25 == 0:
            print(f"curve seed {seed}: {index}/{count}", flush=True)
    write_json(shard_path, {"config": config, "rows": rows, "status": "COMPLETE"})
    return rows


def summarize(rows):
    metrics = [
        "local_rmse_0p10ml_ph",
        "local_rmse_0p25ml_ph",
        "local_rmse_0p50ml_ph",
        "full_curve_rmse_0_33ml_ph",
        "concentration_relative_error_percent",
        "pka_matched_mae",
        "pka_penalized_mae",
    ]
    output = []
    for variant in PF_VARIANTS:
        for checkpoint in CHECKPOINTS:
            subset = [row for row in rows if row["method"] == variant and row["checkpoint"] == checkpoint]
            summary = {
                "method": variant,
                "checkpoint": checkpoint,
                "tasks": len(subset),
                "pair_count_accuracy_percent": 100.0 * float(np.mean([
                    int(row["true_pair_count"]) == int(row["estimated_pair_count"])
                    for row in subset
                ])),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                summary[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            output.append(summary)
    return output


def final_by_acid_type(rows):
    selected = [row for row in rows if row["checkpoint"] == "final"]
    output = []
    for variant in PF_VARIANTS:
        for acid_type in ("monoprotic", "diprotic", "triprotic"):
            subset = [row for row in selected if row["method"] == variant and row["acid_type"] == acid_type]
            if not subset:
                continue
            output.append(
                {
                    "method": variant,
                    "acid_type": acid_type,
                    "tasks": len(subset),
                    "full_curve_rmse_median_ph": float(np.median([row["full_curve_rmse_0_33ml_ph"] for row in subset])),
                    "local_rmse_0p10ml_median_ph": float(np.median([row["local_rmse_0p10ml_ph"] for row in subset])),
                    "concentration_relative_error_median_percent": float(np.median([row["concentration_relative_error_percent"] for row in subset])),
                    "pka_matched_mae_median": float(np.nanmedian([row["pka_matched_mae"] for row in subset])),
                    "pair_count_accuracy_percent": 100.0 * float(np.mean([
                        int(row["true_pair_count"]) == int(row["estimated_pair_count"])
                        for row in subset
                    ])),
                }
            )
    return output


def confusion_matrix(rows):
    selected = [row for row in rows if row["checkpoint"] == "final" and row["method"] == "pf_pka_conc_variable_k"]
    output = []
    for true_k in (1, 2, 3):
        for estimated_k in (1, 2, 3):
            count = sum(
                int(row["true_pair_count"]) == true_k and int(row["estimated_pair_count"]) == estimated_k
                for row in selected
            )
            output.append({"true_k": true_k, "estimated_k": estimated_k, "tasks": count})
    return output


def paired_tests(rows):
    selected = [row for row in rows if row["checkpoint"] == "final"]
    lookup = {
        (row["method"], int(row["seed"]), int(row["task_id"])): row
        for row in selected
    }
    output = []
    metrics = (
        "local_rmse_0p10ml_ph",
        "full_curve_rmse_0_33ml_ph",
        "concentration_relative_error_percent",
        "pka_penalized_mae",
    )
    for first_index, first in enumerate(PF_VARIANTS):
        for second in PF_VARIANTS[first_index + 1:]:
            keys = sorted(
                set((int(row["seed"]), int(row["task_id"])) for row in selected if row["method"] == first)
                & set((int(row["seed"]), int(row["task_id"])) for row in selected if row["method"] == second)
            )
            first_correct = [
                int(lookup[(first, *key)]["estimated_pair_count"]) == int(lookup[(first, *key)]["true_pair_count"])
                for key in keys
            ]
            second_correct = [
                int(lookup[(second, *key)]["estimated_pair_count"]) == int(lookup[(second, *key)]["true_pair_count"])
                for key in keys
            ]
            count_test = exact_mcnemar(first_correct, second_correct)
            output.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "metric": "pair_count_correct",
                    "matched_tasks": len(keys),
                    **count_test,
                }
            )
            for metric in metrics:
                first_values = np.asarray([float(lookup[(first, *key)][metric]) for key in keys])
                second_values = np.asarray([float(lookup[(second, *key)][metric]) for key in keys])
                difference = second_values - first_values
                try:
                    p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
                except ValueError:
                    p_value = np.nan
                output.append(
                    {
                        "method_a": first,
                        "method_b": second,
                        "metric": metric,
                        "matched_tasks": len(keys),
                        "median_a": float(np.median(first_values)),
                        "median_b": float(np.median(second_values)),
                        "paired_difference_b_minus_a_median": float(np.median(difference)),
                        "fraction_b_lower_percent": 100.0 * float(np.mean(second_values < first_values)),
                        "wilcoxon_p_two_sided": p_value,
                    }
                )
    for metric in sorted({str(row["metric"]) for row in output}):
        indices = [index for index, row in enumerate(output) if row["metric"] == metric]
        raw = [
            float(output[index].get("p_value_exact_two_sided", output[index].get("wilcoxon_p_two_sided", np.nan)))
            for index in indices
        ]
        finite = [(index, value) for index, value in zip(indices, raw) if np.isfinite(value)]
        adjusted = holm_adjust([value for _, value in finite])
        for (index, _), value in zip(finite, adjusted):
            output[index]["holm_p_within_metric"] = value
    return output


def per_seed_final_summary(rows):
    selected = [row for row in rows if row["checkpoint"] == "final"]
    metrics = (
        "local_rmse_0p10ml_ph",
        "local_rmse_0p25ml_ph",
        "local_rmse_0p50ml_ph",
        "full_curve_rmse_0_33ml_ph",
        "concentration_relative_error_percent",
        "pka_matched_mae",
        "pka_penalized_mae",
    )
    output = []
    for variant in PF_VARIANTS:
        for seed in sorted({int(row["seed"]) for row in selected}):
            subset = [row for row in selected if row["method"] == variant and int(row["seed"]) == seed]
            summary = {
                "method": variant,
                "seed": seed,
                "tasks": len(subset),
                "pair_count_accuracy_percent": 100.0 * float(np.mean([
                    int(row["true_pair_count"]) == int(row["estimated_pair_count"]) for row in subset
                ])),
            }
            for metric in metrics:
                summary[f"{metric}_median"] = float(np.nanmedian([float(row[metric]) for row in subset]))
            output.append(summary)
    return output


def seed_level_paired_tests(seed_rows):
    lookup = {(row["method"], int(row["seed"])): row for row in seed_rows}
    metrics = ["pair_count_accuracy_percent"] + [
        key for key in seed_rows[0] if key.endswith("_median")
    ]
    output = []
    for first_index, first in enumerate(PF_VARIANTS):
        for second in PF_VARIANTS[first_index + 1:]:
            seeds = sorted(
                set(seed for method, seed in lookup if method == first)
                & set(seed for method, seed in lookup if method == second)
            )
            for metric_index, metric in enumerate(metrics):
                differences = np.asarray([
                    float(lookup[(second, seed)][metric]) - float(lookup[(first, seed)][metric])
                    for seed in seeds
                ])
                finite = differences[np.isfinite(differences)]
                ci_low, ci_high = paired_seed_bootstrap(finite, seed=20261811 + metric_index)
                output.append(
                    {
                        "method_a": first,
                        "method_b": second,
                        "metric": metric,
                        "independent_seeds": len(finite),
                        "difference_b_minus_a_mean": float(np.mean(finite)) if len(finite) else np.nan,
                        "difference_b_minus_a_seed_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                        "bootstrap_95ci_low": ci_low,
                        "bootstrap_95ci_high": ci_high,
                        "exact_sign_flip_p_two_sided": exact_sign_flip_p(finite),
                    }
                )
    return output


def plot_results(rows, output_dir):
    selected = [row for row in rows if row["checkpoint"] == "final"]
    labels = ["pKa only, K=3", "concentration + pKa, K=3", "concentration + pKa + K"]
    colors = ["#527A9A", "#D18C3C", "#3C8B67"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    for ax, metric, title in (
        (axes[0, 0], "local_rmse_0p10ml_ph", "Local response RMSE (+/-0.10 mL)"),
        (axes[0, 1], "full_curve_rmse_0_33ml_ph", "Full 0-33 mL curve RMSE"),
        (axes[1, 0], "concentration_relative_error_percent", "Concentration relative error"),
    ):
        groups = [
            [float(row[metric]) for row in selected if row["method"] == variant]
            for variant in PF_VARIANTS
        ]
        boxes = ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        if "rmse" in metric:
            ax.set_yscale("log")

    confusion = np.zeros((3, 3), dtype=int)
    variable = [row for row in selected if row["method"] == "pf_pka_conc_variable_k"]
    for row in variable:
        confusion[int(row["true_pair_count"]) - 1, int(row["estimated_pair_count"]) - 1] += 1
    image = axes[1, 1].imshow(confusion, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, str(confusion[i, j]), ha="center", va="center")
    axes[1, 1].set_xticks(range(3), [1, 2, 3])
    axes[1, 1].set_yticks(range(3), [1, 2, 3])
    axes[1, 1].set_xlabel("Estimated K")
    axes[1, 1].set_ylabel("True K")
    axes[1, 1].set_title("Variable-K confusion matrix")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)
    fig.suptitle("Fixed-trajectory parameter and response recovery", weight="bold")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"pf_curve_recovery_summary.{extension}", dpi=260 if extension == "png" else None)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fixed-trajectory PF parameter and curve recovery audit")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks-per-seed", type=int, default=300)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--minimum-initial-error-ph", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or base / "results" / f"pf_curve_recovery_{args.tasks_per_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_directory = output_dir / "_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    jobs = [
        (seed, args.tasks_per_seed, args.particles, args.minimum_initial_error_ph, str(shard_directory), resume)
        for seed in args.seeds
    ]
    start = time.perf_counter()
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            for job_rows in executor.map(run_seed_job, jobs):
                rows.extend(job_rows)
    else:
        for job in jobs:
            rows.extend(run_seed_job(job))
    elapsed = time.perf_counter() - start

    expected_rows = len(args.seeds) * args.tasks_per_seed * len(PF_VARIANTS) * len(CHECKPOINTS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} checkpoint rows, found {len(rows)}")

    summary = summarize(rows)
    by_acid = final_by_acid_type(rows)
    confusion = confusion_matrix(rows)
    paired = paired_tests(rows)
    seed_rows = per_seed_final_summary(rows)
    seed_paired = seed_level_paired_tests(seed_rows)
    write_csv(output_dir / "pf_curve_recovery_per_task_checkpoint.csv", rows)
    write_csv(output_dir / "pf_curve_recovery_summary.csv", summary)
    write_csv(output_dir / "pf_curve_recovery_final_by_acid_type.csv", by_acid)
    write_csv(output_dir / "pf_variable_k_confusion_matrix.csv", confusion)
    write_csv(output_dir / "pf_curve_recovery_paired_tests.csv", paired)
    write_csv(output_dir / "pf_curve_recovery_per_seed.csv", seed_rows)
    write_csv(output_dir / "pf_curve_recovery_seed_paired_tests.csv", seed_paired)
    plot_results(rows, output_dir)
    payload = {
        "settings": vars(args) | {
            "distribution": "variable_concentration (log-uniform 0.03-0.25 M)",
            "trajectory": "same pKa-only K=3 controller trajectory replayed by every inference variant",
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
        },
        "summary": summary,
        "final_by_acid_type": by_acid,
        "variable_k_confusion": confusion,
        "paired_tests": paired,
        "per_seed_final": seed_rows,
        "seed_level_paired_tests": seed_paired,
        "validation": {
            "status": "PASS",
            "checkpoint_rows": len(rows),
            "expected_checkpoint_rows": expected_rows,
            "completed_shards": len(list(shard_directory.glob("*.json"))),
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "RUN_COMPLETE.json", payload["validation"])
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
