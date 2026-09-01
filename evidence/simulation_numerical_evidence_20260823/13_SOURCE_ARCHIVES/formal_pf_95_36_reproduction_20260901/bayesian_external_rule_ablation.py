from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

from chemistry_model import SolutionState, solve_ph_scalar
from particle_controllers import JointInferenceController
from task_distribution import generate_tasks, save_tasks


PF_VARIANT = "pf_pka_conc_variable_k"
ABLATION_PROTOCOL_VERSION = 2
SENSOR_RESOLUTION_PH = 0.01
POLICIES = ("hybrid_full", "hybrid_no_overshoot_cap", "posterior_direct")
POLICY_LABELS = {
    "hybrid_full": "PF + full dose rules",
    "hybrid_no_overshoot_cap": "PF without overshoot cap",
    "posterior_direct": "PF posterior-direct",
}
POLICY_DESCRIPTIONS = {
    "hybrid_full": (
        "Existing PF controller with the inherited pH-rate, uncertainty, buffering-response, "
        "required-volume, tanh dose shaping, and persistent overshoot-volume cap."
    ),
    "hybrid_no_overshoot_cap": (
        "Same controller and dose shaping, but the post-overshoot global candidate-volume cap is ignored."
    ),
    "posterior_direct": (
        "Primary titrant direction follows current versus target pH; volume is the posterior equilibrium "
        "model root (or the 10.00 mL boundary when the root is outside the action range). No overshoot cap "
        "or heuristic dose-shaping factors are used."
    ),
}


class NoOvershootCapController(JointInferenceController):
    def select_best_action(self):
        saved_threshold = self.overshoot_threshold
        self.overshoot_threshold = None
        try:
            return super().select_best_action()
        finally:
            self.overshoot_threshold = saved_threshold


class PosteriorDirectController(JointInferenceController):
    def select_best_action(self):
        current = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        reagent = "Dilute base 1" if current < self.target_ph else "Dilute acid 1"
        required = float(self.compute_required_volume())
        maximum = float(max(self.addition_volumes))
        if not np.isfinite(required) or required <= 0.0:
            required = maximum
        volume = float(np.clip(np.round(required / 0.01) * 0.01, self.min_addition_volume, maximum))
        return (reagent, volume), self.done


CONTROLLER_CLASSES = {
    "hybrid_full": JointInferenceController,
    "hybrid_no_overshoot_cap": NoOvershootCapController,
    "posterior_direct": PosteriorDirectController,
}


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initialize_controller(task, policy: str, particles: int, seed: int):
    seed = int(seed) % (2**32 - 1)
    np.random.seed(seed)
    controller_class = CONTROLLER_CLASSES[policy]
    controller = controller_class(PF_VARIANT, num_particles=particles, filter_seed=(seed + 17) % (2**32 - 1))
    controller.initialize_task(task)
    controller.base_added_moles = float(task.initial_base_moles)
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = float(task.initial_volume_ml)
    controller.previous_total_volume = float(task.initial_volume_ml)
    observed_initial_ph = float(np.round(task.initial_ph / SENSOR_RESOLUTION_PH) * SENSOR_RESOLUTION_PH)
    controller.current_ph = observed_initial_ph
    controller.previous_ph = observed_initial_ph
    controller.last_measured_ph = observed_initial_ph
    controller.prev_measured_ph = observed_initial_ph
    controller.last_action_volume = 0.0
    controller.done = False
    return controller


def run_episode(task, policy: str, particles: int, seed: int) -> dict:
    controller = initialize_controller(task, policy, particles, seed)
    overshoots = 0
    selection_seconds = 0.0
    update_seconds = 0.0
    threshold_activations = 0
    maximum_threshold_reduction = 0.0
    while not controller.done:
        # Secondary/dilute titrant switching is disabled for every ablation arm.
        controller.use_secondary_reagents = False
        started = time.perf_counter()
        action, _ = controller.select_best_action()
        selection_seconds += time.perf_counter() - started
        previous_threshold = controller.overshoot_threshold
        current_ph, _, done, info = controller.step(action, mode="Simulate")
        overshoots += int(bool(info.get("crossed_target", False)))
        if controller.overshoot_threshold is not None:
            threshold_activations += int(
                previous_threshold is None or controller.overshoot_threshold < previous_threshold
            )
            maximum_threshold_reduction = max(
                maximum_threshold_reduction,
                max(controller.addition_volumes) - float(controller.overshoot_threshold),
            )
        started = time.perf_counter()
        controller.update_posteriors(action, current_ph)
        update_seconds += time.perf_counter() - started
        if done:
            break
    measured_error = abs(controller.current_ph - controller.target_ph)
    final_state = SolutionState(
        total_volume_ml=float(controller.total_volume),
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles),
    )
    final_true_ph = solve_ph_scalar(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        final_state,
    )
    error = abs(final_true_ph - controller.target_ph)
    return {
        "task_seed": task.seed,
        "task_id": task.task_id,
        "policy": policy,
        "acid_type": task.acid_type,
        "true_pair_count": len(task.pka_values),
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "true_concentration_m": task.analyte_conc_m,
        "true_success": int(error <= 0.10),
        "strict_success": int(error <= 0.05),
        "severe_failure": int(error > 0.50),
        "measured_success": int(measured_error <= 0.10),
        "false_stop": int(measured_error <= 0.10 and error > 0.10),
        "steps": controller.steps_taken,
        "overshoots": overshoots,
        "total_volume_ml": controller.acid_volume + controller.base_volume,
        "final_abs_error": error,
        "final_true_ph": final_true_ph,
        "final_measured_ph": controller.current_ph,
        "selection_time_ms_total": 1000.0 * selection_seconds,
        "posterior_update_time_ms_total": 1000.0 * update_seconds,
        "controller_time_ms_per_step": 1000.0 * (selection_seconds + update_seconds) / max(1, controller.steps_taken),
        "overshoot_threshold_activations": threshold_activations,
        "maximum_threshold_reduction_ml": maximum_threshold_reduction,
    }


def run_task_payload(payload):
    task, particles, common_seed = payload
    return [run_episode(task, policy, particles, common_seed) for policy in POLICIES]


def summarize(rows) -> dict:
    rows = list(rows)
    successful_steps = [row["steps"] for row in rows if row["true_success"]]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean([row["true_success"] for row in rows])),
        "strict_success_rate_percent": 100.0 * float(np.mean([row["strict_success"] for row in rows])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([row["severe_failure"] for row in rows])),
        "false_stop_rate_percent": 100.0 * float(np.mean([row["false_stop"] for row in rows])),
        "steps_mean": float(np.mean([row["steps"] for row in rows])),
        "successful_steps_mean": float(np.mean(successful_steps)) if successful_steps else math.nan,
        "overshoots_mean": float(np.mean([row["overshoots"] for row in rows])),
        "total_volume_mean_ml": float(np.mean([row["total_volume_ml"] for row in rows])),
        "final_abs_error_mean": float(np.mean([row["final_abs_error"] for row in rows])),
        "controller_time_ms_per_step_mean": float(np.mean([row["controller_time_ms_per_step"] for row in rows])),
        "overshoot_threshold_activation_rate_percent": 100.0 * float(
            np.mean([row["overshoot_threshold_activations"] > 0 for row in rows])
        ),
    }


def exact_mcnemar(reference, comparison):
    reference_only = sum(bool(a) and not bool(b) for a, b in zip(reference, comparison))
    comparison_only = sum(not bool(a) and bool(b) for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(reference_only, discordant, 0.5).pvalue)
    return reference_only, comparison_only, p_value


def holm_adjust(rows, p_key="p_value", output_key="holm_adjusted_p"):
    indices = [index for index, row in enumerate(rows) if row.get(p_key) is not None]
    ordered = sorted(indices, key=lambda index: float(rows[index][p_key]))
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(rows[index][p_key]))
        running = max(running, adjusted)
        rows[index][output_key] = running


def paired_tests(task_rows, seeds):
    comparisons = (
        ("hybrid_full", "hybrid_no_overshoot_cap"),
        ("hybrid_full", "posterior_direct"),
        ("hybrid_no_overshoot_cap", "posterior_direct"),
    )
    success_rows = []
    continuous_rows = []
    for seed_scope in [*seeds, "pooled"]:
        subset = task_rows if seed_scope == "pooled" else [row for row in task_rows if row["benchmark_seed"] == seed_scope]
        lookup = {
            policy: {
                (row["task_seed"], row["task_id"]): row
                for row in subset if row["policy"] == policy
            }
            for policy in POLICIES
        }
        for reference_policy, comparison_policy in comparisons:
            keys = sorted(set(lookup[reference_policy]) & set(lookup[comparison_policy]))
            reference_success = [lookup[reference_policy][key]["true_success"] for key in keys]
            comparison_success = [lookup[comparison_policy][key]["true_success"] for key in keys]
            reference_only, comparison_only, p_value = exact_mcnemar(reference_success, comparison_success)
            success_rows.append(
                {
                    "seed_scope": seed_scope,
                    "comparison": f"{comparison_policy}_minus_{reference_policy}",
                    "paired_tasks": len(keys),
                    "reference_only_success": reference_only,
                    "comparison_only_success": comparison_only,
                    "success_difference_pp": 100.0 * (
                        float(np.mean(comparison_success)) - float(np.mean(reference_success))
                    ),
                    "p_value": p_value,
                }
            )
            if seed_scope != "pooled":
                continue
            for metric in ("steps", "overshoots", "total_volume_ml", "final_abs_error"):
                reference_values = np.asarray([lookup[reference_policy][key][metric] for key in keys], dtype=float)
                comparison_values = np.asarray([lookup[comparison_policy][key][metric] for key in keys], dtype=float)
                differences = comparison_values - reference_values
                if np.allclose(differences, 0.0):
                    statistic, continuous_p = 0.0, 1.0
                else:
                    result = wilcoxon(comparison_values, reference_values, zero_method="wilcox", method="auto")
                    statistic, continuous_p = float(result.statistic), float(result.pvalue)
                continuous_rows.append(
                    {
                        "comparison": f"{comparison_policy}_minus_{reference_policy}",
                        "metric": metric,
                        "paired_tasks": len(keys),
                        "mean_paired_difference": float(np.mean(differences)),
                        "median_paired_difference": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "p_value": continuous_p,
                    }
                )
    holm_adjust(success_rows)
    holm_adjust(continuous_rows)
    return success_rows, continuous_rows


def aggregate_summaries(seed_summaries):
    metrics = [key for key in seed_summaries[0] if key not in {"benchmark_seed", "policy"}]
    output = []
    for policy in POLICIES:
        subset = [row for row in seed_summaries if row["policy"] == policy]
        result = {"policy": policy, "runs": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            result[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
        output.append(result)
    return output


def plot_results(aggregate, output_dir):
    colors = ["#3D7A5C", "#557FA3", "#C98232"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    plot_specs = (
        ("success_rate_percent", "Success (%)"),
        ("successful_steps_mean", "Successful steps"),
        ("overshoots_mean", "Target crossings"),
    )
    for axis, (metric, label) in zip(axes, plot_specs):
        means = [row[f"{metric}_mean"] for row in aggregate]
        errors = [row[f"{metric}_sd"] or 0.0 for row in aggregate]
        axis.bar(range(len(POLICIES)), means, yerr=errors, capsize=4, color=colors)
        axis.set_xticks(range(len(POLICIES)), ["Full", "No cap", "Direct"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2, linestyle=":")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_dir / "bayesian_external_rule_ablation.png", dpi=260)
    fig.savefig(output_dir / "bayesian_external_rule_ablation.svg")
    plt.close(fig)


def format_value(mean, sd, digits=2):
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def write_report(output_dir, aggregate, success_tests, seeds, tasks_per_seed, particles):
    seed_word = "seed" if len(seeds) == 1 else "seeds"
    report = [
        "# Bayesian External-Rule Ablation",
        "",
        f"Matched evaluation used {len(seeds)} independently generated {seed_word}, {tasks_per_seed} tasks per seed, and {particles} particles.",
        "The titrant-direction rule, 0.01-10.00 mL action bounds, primary 0.1 M titrants, stopping tolerance, and maximum step count were common to all arms.",
        "Controllers observe pH at 0.01 resolution; outcome metrics use the unquantized true chemical pH.",
        "",
    ]
    for policy in POLICIES:
        report.append(f"- **{POLICY_LABELS[policy]}:** {POLICY_DESCRIPTIONS[policy]}")
    report.extend(
        [
            "",
            "| Policy | Success (%) | Strict (%) | Severe failure (%) | All steps | Successful steps | Crossings | Volume (mL) | Final error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        report.append(
            f"| {POLICY_LABELS[row['policy']]} | "
            f"{format_value(row['success_rate_percent_mean'], row['success_rate_percent_sd'])} | "
            f"{format_value(row['strict_success_rate_percent_mean'], row['strict_success_rate_percent_sd'])} | "
            f"{format_value(row['severe_failure_rate_percent_mean'], row['severe_failure_rate_percent_sd'])} | "
            f"{format_value(row['steps_mean_mean'], row['steps_mean_sd'])} | "
            f"{format_value(row['successful_steps_mean_mean'], row['successful_steps_mean_sd'])} | "
            f"{format_value(row['overshoots_mean_mean'], row['overshoots_mean_sd'])} | "
            f"{format_value(row['total_volume_mean_ml_mean'], row['total_volume_mean_ml_sd'])} | "
            f"{format_value(row['final_abs_error_mean_mean'], row['final_abs_error_mean_sd'], 4)} |"
        )
    report.extend(["", "## Pooled paired success tests", ""])
    for row in success_tests:
        if row["seed_scope"] == "pooled":
            report.append(
                f"- {row['comparison']}: {row['success_difference_pp']:+.2f} percentage points; "
                f"Holm-adjusted exact McNemar p = {row['holm_adjusted_p']:.4g}."
            )
    report.extend(
        [
            "",
            "Task-level results, per-seed summaries, paired tests, and the exact generated tasks are included in this directory.",
        ]
    )
    (output_dir / "RESULT_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Ablate inherited external dosing rules in the improved PF controller")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.particles < 60:
        raise ValueError("Variable-K inference requires at least 60 particles")
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": args.seeds,
        "tasks_per_seed": args.tasks_per_seed,
        "particles": args.particles,
        "workers": workers,
        "pf_variant": PF_VARIANT,
        "ablation_protocol_version": ABLATION_PROTOCOL_VERSION,
        "sensor_resolution_ph": SENSOR_RESOLUTION_PH,
        "action_volume_range_ml": [0.01, 10.0],
        "evaluation_uses_true_ph": True,
        "policies": list(POLICIES),
        "policy_descriptions": POLICY_DESCRIPTIONS,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError(
                f"Existing output uses a different configuration: {config_path}. "
                "Choose a new --output-dir for an independent run."
            )
        if not args.resume:
            raise FileExistsError(f"Output already exists; use --resume: {args.output_dir}")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty and has no RUN_CONFIG.json: {args.output_dir}")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows = []
    seed_summaries = []
    for benchmark_seed in args.seeds:
        result_path = args.output_dir / f"seed_{benchmark_seed}_task_results.csv"
        summary_path = args.output_dir / f"seed_{benchmark_seed}_summary.json"
        complete_path = args.output_dir / f"seed_{benchmark_seed}_COMPLETE.json"
        if args.resume and result_path.exists() and summary_path.exists() and complete_path.exists():
            with result_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for key in (
                    "task_seed", "task_id", "true_pair_count", "true_success", "strict_success",
                    "severe_failure", "measured_success", "false_stop", "steps", "overshoots",
                    "overshoot_threshold_activations",
                ):
                    row[key] = int(row[key])
                for key in (
                    "initial_ph", "target_ph", "true_concentration_m", "total_volume_ml",
                    "final_abs_error", "final_true_ph", "final_measured_ph", "selection_time_ms_total",
                    "posterior_update_time_ms_total", "controller_time_ms_per_step",
                    "maximum_threshold_reduction_ml",
                ):
                    row[key] = float(row[key])
                row["benchmark_seed"] = benchmark_seed
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
            all_rows.extend(rows)
            seed_summaries.extend(summaries)
            print(f"seed {benchmark_seed} already complete", flush=True)
            continue

        tasks = generate_tasks(1_000_000 + benchmark_seed, args.tasks_per_seed, f"rule_ablation_seed_{benchmark_seed}")
        save_tasks(args.output_dir / f"seed_{benchmark_seed}_tasks.jsonl", tasks)
        payloads = [
            (task, args.particles, benchmark_seed * 10_000_019 + task.task_id * 1009)
            for task in tasks
        ]
        rows = []
        if workers == 1:
            results = map(run_task_payload, payloads)
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            results = executor.map(run_task_payload, payloads, chunksize=2)
        try:
            for index, task_rows in enumerate(results, 1):
                for row in task_rows:
                    row["benchmark_seed"] = benchmark_seed
                rows.extend(task_rows)
                if index % 100 == 0 or index == len(tasks):
                    print(f"seed {benchmark_seed}: {index}/{len(tasks)} matched tasks", flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
        summaries = []
        for policy in POLICIES:
            policy_rows = [row for row in rows if row["policy"] == policy]
            summaries.append({"benchmark_seed": benchmark_seed, "policy": policy, **summarize(policy_rows)})
        write_csv(result_path, rows)
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        complete_path.write_text(json.dumps({"benchmark_seed": benchmark_seed, "rows": len(rows)}, indent=2), encoding="utf-8")
        all_rows.extend(rows)
        seed_summaries.extend(summaries)

    aggregate = aggregate_summaries(seed_summaries)
    success_tests, continuous_tests = paired_tests(all_rows, args.seeds)
    write_csv(args.output_dir / "all_task_results.csv", all_rows)
    write_csv(args.output_dir / "per_seed_summary.csv", seed_summaries)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    write_csv(args.output_dir / "paired_success_tests.csv", success_tests)
    write_csv(args.output_dir / "paired_continuous_tests.csv", continuous_tests)
    plot_results(aggregate, args.output_dir)
    write_report(args.output_dir, aggregate, success_tests, args.seeds, args.tasks_per_seed, args.particles)
    (args.output_dir / "ABLATION_COMPLETE.json").write_text(
        json.dumps({"config": config, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(f"Ablation complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
