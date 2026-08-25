from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import itertools
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from benchmark_core import exact_mcnemar, holm_adjust
from experiment_utils import (
    ControlResult,
    exact_sign_flip_p,
    paired_seed_bootstrap,
    run_control_episode,
    summarize_control,
)
from io_utils import read_json, write_csv, write_json
from particle_inference import PF_VARIANTS
from task_generation import DEFAULT_SEEDS, generate_comparison_tasks


def run_job(job):
    seed, count, distribution, particles, variant, shard_directory, resume = job
    config = {
        "seed": seed,
        "tasks_per_seed": count,
        "distribution": distribution,
        "particles": particles,
        "variant": variant,
        "pka_prior_support": [1.5, 9.0],
    }
    shard_path = Path(shard_directory) / f"{variant}_seed{seed}.json"
    if resume and shard_path.exists():
        try:
            payload = read_json(shard_path)
            stored_rows = payload.get("rows", [])
            if payload.get("config") == config and len(stored_rows) == count:
                return [ControlResult(**row) for row in stored_rows]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    tasks = generate_comparison_tasks(seed, count, distribution)
    rows = []
    for index, task in enumerate(tasks, 1):
        result, _, _ = run_control_episode(
            task,
            variant,
            particles,
            seed * 1_000_003 + task.task_id,
        )
        rows.append(result)
        if index % 250 == 0:
            print(f"{variant} seed {seed}: {index}/{count}", flush=True)
    write_json(
        shard_path,
        {"config": config, "rows": [row.to_dict() for row in rows], "status": "COMPLETE"},
    )
    return rows


def per_seed_summary(results, seeds):
    rows = []
    for variant in PF_VARIANTS:
        for seed in seeds:
            subset = [row for row in results if row.method == variant and row.seed == seed]
            rows.append({"method": variant, "seed": seed, **summarize_control(subset)})
    return rows


def aggregate_summary(seed_rows):
    rows = []
    metric_names = [
        "success_percent",
        "successful_steps_mean",
        "overshoot_events_per_step_percent",
        "final_abs_error_mean_ph",
        "total_titrant_volume_mean_ml",
        "decision_time_median_ms",
        "update_time_mean_ms",
        "pair_count_accuracy_percent",
        "concentration_relative_error_median_percent",
    ]
    for variant in PF_VARIANTS:
        subset = [row for row in seed_rows if row["method"] == variant]
        output = {"method": variant, "seeds": len(subset)}
        for metric in metric_names:
            values = [float(row[metric]) for row in subset]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(output)
    return rows


def paired_tests(results):
    lookup = {(row.method, row.seed, row.task_id): row for row in results}
    rows = []
    for first, second in itertools.combinations(PF_VARIANTS, 2):
        keys = sorted(
            set((row.seed, row.task_id) for row in results if row.method == first)
            & set((row.seed, row.task_id) for row in results if row.method == second)
        )
        first_success = [lookup[(first, *key)].success for key in keys]
        second_success = [lookup[(second, *key)].success for key in keys]
        test = exact_mcnemar(first_success, second_success)
        rows.append(
            {
                "method_a": first,
                "method_b": second,
                "metric": "success",
                "matched_tasks": len(keys),
                **test,
            }
        )
        for metric, successful_only in (
            ("steps", True),
            ("final_abs_error", False),
            ("total_titrant_volume_ml", False),
        ):
            metric_keys = [
                key for key in keys
                if not successful_only
                or (lookup[(first, *key)].success and lookup[(second, *key)].success)
            ]
            if metric == "steps":
                first_values = np.asarray([lookup[(first, *key)].steps for key in metric_keys], dtype=float)
                second_values = np.asarray([lookup[(second, *key)].steps for key in metric_keys], dtype=float)
            elif metric == "final_abs_error":
                first_values = np.asarray([abs(lookup[(first, *key)].final_ph - lookup[(first, *key)].target_ph) for key in metric_keys])
                second_values = np.asarray([abs(lookup[(second, *key)].final_ph - lookup[(second, *key)].target_ph) for key in metric_keys])
            else:
                first_values = np.asarray([lookup[(first, *key)].acid_added_ml + lookup[(first, *key)].base_added_ml for key in metric_keys])
                second_values = np.asarray([lookup[(second, *key)].acid_added_ml + lookup[(second, *key)].base_added_ml for key in metric_keys])
            difference = second_values - first_values
            try:
                p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "metric": metric,
                    "matched_tasks": len(metric_keys),
                    "median_a": float(np.median(first_values)),
                    "median_b": float(np.median(second_values)),
                    "paired_difference_b_minus_a_median": float(np.median(difference)),
                    "fraction_b_lower_percent": 100.0 * float(np.mean(second_values < first_values)),
                    "wilcoxon_p_two_sided": p_value,
                }
            )
    for metric in sorted({str(row["metric"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["metric"] == metric]
        raw = []
        for index in indices:
            row = rows[index]
            value = row.get("p_value_exact_two_sided", row.get("wilcoxon_p_two_sided", np.nan))
            raw.append(float(value))
        finite = [(index, value) for index, value in zip(indices, raw) if np.isfinite(value)]
        adjusted = holm_adjust([value for _, value in finite])
        for (index, _), value in zip(finite, adjusted):
            rows[index]["holm_p_within_metric"] = value
    return rows


def seed_level_paired_tests(seed_rows):
    lookup = {(row["method"], int(row["seed"])): row for row in seed_rows}
    metrics = (
        "success_percent",
        "successful_steps_mean",
        "overshoot_events_per_step_percent",
        "final_abs_error_mean_ph",
        "total_titrant_volume_mean_ml",
        "decision_time_median_ms",
        "update_time_mean_ms",
        "pair_count_accuracy_percent",
        "concentration_relative_error_median_percent",
    )
    output = []
    for first, second in itertools.combinations(PF_VARIANTS, 2):
        seeds = sorted(
            set(seed for method, seed in lookup if method == first)
            & set(seed for method, seed in lookup if method == second)
        )
        for metric_index, metric in enumerate(metrics):
            differences = np.asarray(
                [float(lookup[(second, seed)][metric]) - float(lookup[(first, seed)][metric]) for seed in seeds],
                dtype=float,
            )
            finite = differences[np.isfinite(differences)]
            ci_low, ci_high = paired_seed_bootstrap(finite, seed=20260811 + metric_index)
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


def plot_summary(seed_rows, output_dir):
    metrics = [
        ("success_percent", "Success (%)"),
        ("successful_steps_mean", "Successful-run steps"),
        ("final_abs_error_mean_ph", "Final absolute error (pH)"),
        ("decision_time_median_ms", "Decision time (ms)"),
    ]
    labels = ["pKa only, K=3", "concentration + pKa, K=3", "concentration + pKa + K"]
    colors = ["#527A9A", "#D18C3C", "#3C8B67"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        groups = [
            np.asarray([float(row[metric]) for row in seed_rows if row["method"] == variant])
            for variant in PF_VARIANTS
        ]
        means = [float(np.mean(group)) for group in groups]
        sds = [float(np.std(group, ddof=1)) if len(group) > 1 else 0.0 for group in groups]
        x = np.arange(3)
        ax.bar(x, means, yerr=sds, color=colors, capsize=4, width=0.66)
        for position, group in zip(x, groups):
            ax.scatter(np.full(len(group), position), group, color="#263844", s=18, zorder=3)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
    fig.suptitle("Joint-parameter particle-filter comparison", weight="bold")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"pf_multiseed_control_summary.{extension}", dpi=260 if extension == "png" else None)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Five-seed closed-loop particle-filter comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--distribution", choices=["nominal", "variable_concentration"], default="nominal")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or base / "results" / f"pf_control_{args.distribution}_{args.tasks_per_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_directory = output_dir / "_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    jobs = [
        (seed, args.tasks_per_seed, args.distribution, args.particles, variant, str(shard_directory), resume)
        for variant in PF_VARIANTS
        for seed in args.seeds
    ]
    start = time.perf_counter()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            for job_rows in executor.map(run_job, jobs):
                results.extend(job_rows)
    else:
        for job in jobs:
            results.extend(run_job(job))
    elapsed = time.perf_counter() - start

    task_rows = [row.to_dict() for row in results]
    expected_conditions = {(variant, seed) for variant in PF_VARIANTS for seed in args.seeds}
    observed_conditions = {(row.method, row.seed) for row in results}
    if observed_conditions != expected_conditions:
        raise RuntimeError(
            f"Condition mismatch: missing={sorted(expected_conditions - observed_conditions)}, "
            f"extra={sorted(observed_conditions - expected_conditions)}"
        )
    for condition in sorted(expected_conditions):
        count = sum((row.method, row.seed) == condition for row in results)
        if count != args.tasks_per_seed:
            raise RuntimeError(f"{condition}: expected {args.tasks_per_seed} rows, found {count}")
    seed_rows = per_seed_summary(results, args.seeds)
    aggregate = aggregate_summary(seed_rows)
    paired = paired_tests(results)
    seed_paired = seed_level_paired_tests(seed_rows)
    write_csv(output_dir / "pf_control_per_task.csv", task_rows)
    write_csv(output_dir / "pf_control_per_seed.csv", seed_rows)
    write_csv(output_dir / "pf_control_aggregate.csv", aggregate)
    write_csv(output_dir / "pf_control_paired_mcnemar.csv", paired)
    write_csv(output_dir / "pf_control_seed_paired_tests.csv", seed_paired)
    plot_summary(seed_rows, output_dir)
    payload = {
        "settings": vars(args) | {"output_dir": str(output_dir), "elapsed_seconds": elapsed},
        "aggregate": aggregate,
        "paired_mcnemar": paired,
        "seed_level_paired_tests": seed_paired,
        "validation": {
            "status": "PASS",
            "conditions": len(observed_conditions),
            "task_rows": len(task_rows),
            "expected_task_rows": len(expected_conditions) * args.tasks_per_seed,
            "completed_shards": len(list(shard_directory.glob("*.json"))),
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "RUN_COMPLETE.json", payload["validation"])
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
