from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import argparse
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
    curve_metrics,
    exact_sign_flip_p,
    matched_pka_errors,
    paired_seed_bootstrap,
    replay_particle_filter,
    run_control_episode,
)
from io_utils import read_json, write_csv, write_json
from particle_inference import PF_VARIANTS
from pymc_inference import PYMC_VARIANTS, fit_pymc_variant
from task_generation import DEFAULT_SEEDS, generate_comparison_tasks


METHOD_PAIRS = (
    ("pf_pka_only_k3", "pymc_pka_only_k3"),
    ("pf_pka_conc_k3", "pymc_pka_conc_k3"),
    ("pf_pka_conc_variable_k", "pymc_pka_conc_variable_k"),
)


def result_row(task, method, backend, estimate, state, runtime_seconds, trajectory_steps, extra=None):
    row = {
        "seed": task.seed,
        "task_id": task.task_id,
        "backend": backend,
        "method": method,
        "acid_type": task.acid_type,
        "true_pair_count": len(task.pka_values),
        "estimated_pair_count": estimate.pair_count,
        "pair_count_correct": len(task.pka_values) == estimate.pair_count,
        "true_pkas": json.dumps(task.pka_values),
        "estimated_pkas": json.dumps(estimate.pka_values.tolist()),
        "pair_probabilities": json.dumps(estimate.pair_probabilities.tolist()),
        "true_concentration_m": task.analyte_conc_m,
        "estimated_concentration_m": estimate.concentration_m,
        "concentration_abs_error_m": abs(estimate.concentration_m - task.analyte_conc_m),
        "concentration_relative_error_percent": 100.0 * abs(estimate.concentration_m - task.analyte_conc_m) / task.analyte_conc_m,
        "trajectory_steps": trajectory_steps,
        "inference_runtime_seconds": runtime_seconds,
        **matched_pka_errors(task.pka_values, estimate),
        **curve_metrics(task, state, estimate),
    }
    if extra:
        row.update(extra)
    return row


def run_task_job(job):
    task, particles, draws, chains, shard_directory, resume = job
    config = {
        "seed": int(task.seed),
        "task_id": int(task.task_id),
        "particles": particles,
        "draws": draws,
        "chains": chains,
        "true_pair_count": len(task.pka_values),
        "true_concentration_m": float(task.analyte_conc_m),
        "pka_prior_support": [1.5, 9.0],
    }
    shard_path = Path(shard_directory) / f"seed{task.seed}_task{task.task_id}.json"
    expected_rows = len(METHOD_PAIRS) * 2
    if resume and shard_path.exists():
        try:
            payload = read_json(shard_path)
            rows = payload.get("rows", [])
            if payload.get("config") == config and len(rows) == expected_rows:
                return rows
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    task_seed = task.seed * 1_000_003 + task.task_id
    _, transitions, _ = run_control_episode(
        task,
        "pf_pka_only_k3",
        particles,
        task_seed,
        keep_trajectory=True,
    )
    if transitions:
        state = transitions[-1].after_state
    else:
        from chemistry_model import SolutionState
        state = SolutionState(task.initial_volume_ml, 0.0, 0.0)

    rows = []
    for variant_index, variant in enumerate(PF_VARIANTS):
        pf_start = time.perf_counter()
        estimate, _ = replay_particle_filter(
            task,
            transitions,
            variant,
            particles,
            task_seed + 10_000 * (variant_index + 1),
        )
        rows.append(
            result_row(
                task,
                variant,
                "particle_filter",
                estimate,
                state,
                time.perf_counter() - pf_start,
                len(transitions),
            )
        )

    for variant_index, variant in enumerate(PYMC_VARIANTS):
        fit = fit_pymc_variant(
            task,
            transitions,
            variant,
            draws,
            chains,
            task_seed + 100_000 * (variant_index + 1),
        )
        rows.append(
            result_row(
                task,
                variant,
                "pymc_smc",
                fit.estimate,
                state,
                fit.runtime_seconds,
                len(transitions),
                {
                    "log_evidence_k1": fit.log_evidence_by_k[0],
                    "log_evidence_k2": fit.log_evidence_by_k[1],
                    "log_evidence_k3": fit.log_evidence_by_k[2],
                    "smc_draws": fit.draws,
                    "smc_chains": fit.chains,
                },
            )
        )
    write_json(shard_path, {"config": config, "rows": rows, "status": "COMPLETE"})
    print(f"PyMC seed {task.seed}: task {task.task_id} complete", flush=True)
    return rows


def summarize(rows):
    output = []
    for method in [item for pair in METHOD_PAIRS for item in pair]:
        subset = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "backend": subset[0]["backend"] if subset else "",
                "tasks": len(subset),
                "pair_count_accuracy_percent": 100.0 * float(np.mean([row["pair_count_correct"] for row in subset])),
                "concentration_relative_error_median_percent": float(np.median([row["concentration_relative_error_percent"] for row in subset])),
                "pka_matched_mae_median": float(np.median([row["pka_matched_mae"] for row in subset])),
                "pka_penalized_mae_median": float(np.median([row["pka_penalized_mae"] for row in subset])),
                "local_rmse_0p10ml_median_ph": float(np.median([row["local_rmse_0p10ml_ph"] for row in subset])),
                "full_curve_rmse_median_ph": float(np.median([row["full_curve_rmse_0_33ml_ph"] for row in subset])),
                "runtime_median_seconds": float(np.median([row["inference_runtime_seconds"] for row in subset])),
            }
        )
    return output


def per_seed_summary(rows):
    output = []
    methods = [item for pair in METHOD_PAIRS for item in pair]
    for method in methods:
        for seed in sorted(set(int(row["seed"]) for row in rows)):
            subset = [row for row in rows if row["method"] == method and int(row["seed"]) == seed]
            if not subset:
                continue
            output.append(
                {
                    "method": method,
                    "seed": seed,
                    "tasks": len(subset),
                    "pair_count_accuracy_percent": 100.0 * float(np.mean([row["pair_count_correct"] for row in subset])),
                    "concentration_relative_error_median_percent": float(np.median([row["concentration_relative_error_percent"] for row in subset])),
                    "pka_matched_mae_median": float(np.median([row["pka_matched_mae"] for row in subset])),
                    "local_rmse_0p10ml_median_ph": float(np.median([row["local_rmse_0p10ml_ph"] for row in subset])),
                    "full_curve_rmse_median_ph": float(np.median([row["full_curve_rmse_0_33ml_ph"] for row in subset])),
                }
            )
    return output


def paired_backend_tests(rows):
    lookup = {(row["method"], int(row["seed"]), int(row["task_id"])): row for row in rows}
    output = []
    for pf_method, pymc_method in METHOD_PAIRS:
        keys = sorted(
            set((int(row["seed"]), int(row["task_id"])) for row in rows if row["method"] == pf_method)
            & set((int(row["seed"]), int(row["task_id"])) for row in rows if row["method"] == pymc_method)
        )
        pf_correct = [bool(lookup[(pf_method, *key)]["pair_count_correct"]) for key in keys]
        pymc_correct = [bool(lookup[(pymc_method, *key)]["pair_count_correct"]) for key in keys]
        output.append(
            {
                "pf_method": pf_method,
                "pymc_method": pymc_method,
                "metric": "pair_count_correct",
                "matched_tasks": len(keys),
                **exact_mcnemar(pf_correct, pymc_correct),
            }
        )
        for metric in (
            "concentration_relative_error_percent",
            "pka_penalized_mae",
            "local_rmse_0p10ml_ph",
            "full_curve_rmse_0_33ml_ph",
        ):
            pf_values = np.asarray([float(lookup[(pf_method, *key)][metric]) for key in keys])
            pymc_values = np.asarray([float(lookup[(pymc_method, *key)][metric]) for key in keys])
            difference = pymc_values - pf_values
            try:
                p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
            except ValueError:
                p_value = np.nan
            output.append(
                {
                    "pf_method": pf_method,
                    "pymc_method": pymc_method,
                    "metric": metric,
                    "matched_tasks": len(keys),
                    "pf_median": float(np.median(pf_values)),
                    "pymc_median": float(np.median(pymc_values)),
                    "paired_difference_pymc_minus_pf_median": float(np.median(difference)),
                    "fraction_pymc_lower_percent": 100.0 * float(np.mean(pymc_values < pf_values)),
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


def seed_level_backend_tests(seed_rows):
    lookup = {(row["method"], int(row["seed"])): row for row in seed_rows}
    metrics = (
        "pair_count_accuracy_percent",
        "concentration_relative_error_median_percent",
        "pka_matched_mae_median",
        "local_rmse_0p10ml_median_ph",
        "full_curve_rmse_median_ph",
    )
    output = []
    for pair_index, (pf_method, pymc_method) in enumerate(METHOD_PAIRS):
        seeds = sorted(
            set(seed for method, seed in lookup if method == pf_method)
            & set(seed for method, seed in lookup if method == pymc_method)
        )
        for metric_index, metric in enumerate(metrics):
            differences = np.asarray([
                float(lookup[(pymc_method, seed)][metric]) - float(lookup[(pf_method, seed)][metric])
                for seed in seeds
            ])
            finite = differences[np.isfinite(differences)]
            ci_low, ci_high = paired_seed_bootstrap(
                finite,
                seed=20262811 + pair_index * 100 + metric_index,
            )
            output.append(
                {
                    "pf_method": pf_method,
                    "pymc_method": pymc_method,
                    "metric": metric,
                    "independent_seeds": len(finite),
                    "difference_pymc_minus_pf_mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "difference_pymc_minus_pf_seed_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "exact_sign_flip_p_two_sided": exact_sign_flip_p(finite),
                }
            )
    return output


def plot_results(rows, output_dir):
    methods = [item for pair in METHOD_PAIRS for item in pair]
    short = ["PF pKa", "PyMC pKa", "PF C+pKa", "PyMC C+pKa", "PF C+pKa+K", "PyMC C+pKa+K"]
    colors = ["#5C83A1", "#A9BED0", "#D08A3A", "#E7BC88", "#388563", "#8BC1A8"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, metric, title in (
        (axes[0, 0], "full_curve_rmse_0_33ml_ph", "Full-curve RMSE"),
        (axes[0, 1], "local_rmse_0p10ml_ph", "Local-response RMSE"),
        (axes[1, 0], "concentration_relative_error_percent", "Concentration relative error (%)"),
        (axes[1, 1], "inference_runtime_seconds", "Inference runtime per task (s)"),
    ):
        groups = [[float(row[metric]) for row in rows if row["method"] == method] for method in methods]
        boxes = ax.boxplot(groups, tick_labels=short, showfliers=False, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        if "rmse" in metric or "runtime" in metric:
            ax.set_yscale("log")
    fig.suptitle("Particle filter versus PyMC SMC on matched trajectories", weight="bold")
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"pymc_comparison_summary.{extension}", dpi=260 if extension == "png" else None)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Matched fixed-trajectory PF versus PyMC SMC comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks-per-seed", type=int, default=5)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-initial-error-ph", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or base / "results" / f"pymc_comparison_{args.tasks_per_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_directory = output_dir / "_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    start_all = time.perf_counter()
    jobs = []
    for seed in args.seeds:
        tasks = generate_comparison_tasks(
            seed,
            args.tasks_per_seed,
            "variable_concentration",
            minimum_initial_error_ph=args.minimum_initial_error_ph,
        )
        jobs.extend((task, args.particles, args.draws, args.chains, str(shard_directory), resume) for task in tasks)

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            for task_rows in executor.map(run_task_job, jobs):
                rows.extend(task_rows)
    else:
        for job in jobs:
            rows.extend(run_task_job(job))

    elapsed = time.perf_counter() - start_all
    expected_rows = len(args.seeds) * args.tasks_per_seed * len(METHOD_PAIRS) * 2
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} PF/PyMC rows, found {len(rows)}")
    summary = summarize(rows)
    seed_summary = per_seed_summary(rows)
    paired = paired_backend_tests(rows)
    seed_paired = seed_level_backend_tests(seed_summary)
    write_csv(output_dir / "pymc_pf_per_task.csv", rows)
    write_csv(output_dir / "pymc_pf_summary.csv", summary)
    write_csv(output_dir / "pymc_pf_per_seed.csv", seed_summary)
    write_csv(output_dir / "pymc_pf_paired_tests.csv", paired)
    write_csv(output_dir / "pymc_pf_seed_paired_tests.csv", seed_paired)
    plot_results(rows, output_dir)
    payload = {
        "settings": vars(args) | {
            "distribution": "variable_concentration (log-uniform 0.03-0.25 M)",
            "trajectory": "same pKa-only K=3 controller trajectory for all PF and PyMC fits",
            "pymc_sampler": "pm.sample_smc; variable K enumerates K=1,2,3 and compares marginal likelihood",
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
        },
        "summary": summary,
        "paired_tests": paired,
        "seed_level_paired_tests": seed_paired,
        "validation": {
            "status": "PASS",
            "rows": len(rows),
            "expected_rows": expected_rows,
            "completed_task_shards": len(list(shard_directory.glob("*.json"))),
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "RUN_COMPLETE.json", payload["validation"])
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
