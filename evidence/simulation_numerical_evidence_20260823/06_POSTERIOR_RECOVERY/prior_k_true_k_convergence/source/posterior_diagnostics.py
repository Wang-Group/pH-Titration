from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chemistry_model import SolutionState, response_curve, solve_ph_scalar
from particle_controllers import JointInferenceController
from task_distribution import ControlTask, generate_tasks, load_tasks, save_tasks


PROTOCOL_VERSION = 1
PF_VARIANT = "pf_pka_conc_variable_k"
POLICY = "hybrid_full"
DEFAULT_SEEDS = [101, 202, 303, 404, 555]
DEFAULT_CHECKPOINTS = [0, 1, 2, 3, 5, 8, 12]
CURVE_GRID_ML = np.linspace(-100.0, 100.0, 161)


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initialize_controller(task: ControlTask, particles: int, seed: int) -> JointInferenceController:
    seed = int(seed) % (2**32 - 1)
    np.random.seed(seed)
    controller = JointInferenceController(PF_VARIANT, num_particles=particles, filter_seed=(seed + 17) % (2**32 - 1))
    controller.initialize_task(task)
    controller.base_added_moles = float(task.initial_base_moles)
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = float(task.initial_volume_ml)
    controller.previous_total_volume = float(task.initial_volume_ml)
    observed_initial_ph = float(np.round(task.initial_ph, 2))
    controller.current_ph = observed_initial_ph
    controller.previous_ph = observed_initial_ph
    controller.last_measured_ph = observed_initial_ph
    controller.prev_measured_ph = observed_initial_ph
    controller.last_action_volume = 0.0
    controller.done = False
    return controller


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def snapshot_metrics(
    task: ControlTask,
    controller: JointInferenceController,
    benchmark_seed: int,
    checkpoint_type: str,
    observations: int,
    natural_stop_reason: str = "",
) -> dict:
    estimate = controller.posterior_estimate()
    initial_state = SolutionState(
        total_volume_ml=float(task.initial_volume_ml),
        base_moles=float(task.initial_base_moles),
        acid_moles=0.0,
    )
    true_curve = response_curve(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        initial_state,
        CURVE_GRID_ML,
    )
    fitted_curve = response_curve(
        estimate.concentration_m,
        estimate.pka_values,
        task.initial_volume_ml,
        initial_state,
        CURVE_GRID_ML,
    )
    residual = fitted_curve - true_curve
    centered = true_curve - float(np.mean(true_curve))
    denominator = float(np.sum(centered**2))
    curve_r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else math.nan

    true_pka = np.asarray(task.pka_values, dtype=float)
    pka_correct = int(estimate.pair_count == len(true_pka))
    if pka_correct:
        pka_residual = np.asarray(estimate.pka_values, dtype=float) - true_pka
        pka_mae = float(np.mean(np.abs(pka_residual)))
        pka_rmse = float(np.sqrt(np.mean(pka_residual**2)))
        pka_coverage = float(
            np.mean(
                np.abs(pka_residual)
                <= 1.96 * np.maximum(np.asarray(estimate.pka_sd, dtype=float), 1e-12)
            )
        )
    else:
        pka_mae = math.nan
        pka_rmse = math.nan
        pka_coverage = math.nan

    concentration_error = estimate.concentration_m - task.analyte_conc_m
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
    final_true_error = abs(final_true_ph - task.target_ph)
    return {
        "benchmark_seed": benchmark_seed,
        "task_seed": task.seed,
        "task_id": task.task_id,
        "checkpoint_type": checkpoint_type,
        "observations": observations,
        "acid_type": task.acid_type,
        "true_pair_count": len(task.pka_values),
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "true_concentration_m": task.analyte_conc_m,
        "estimated_concentration_m": estimate.concentration_m,
        "concentration_abs_error_m": abs(concentration_error),
        "concentration_relative_error_percent": 100.0 * abs(concentration_error) / task.analyte_conc_m,
        "concentration_95_covered": int(
            abs(concentration_error) <= 1.96 * max(estimate.concentration_sd_m, 1e-12)
        ),
        "estimated_pair_count": estimate.pair_count,
        "pair_count_correct": pka_correct,
        "true_pair_probability": float(estimate.pair_probabilities[len(true_pka) - 1]),
        "pka_mae_if_k_correct": pka_mae,
        "pka_rmse_if_k_correct": pka_rmse,
        "pka_95_coverage_if_k_correct": pka_coverage,
        "true_pka_json": json.dumps(true_pka.tolist()),
        "estimated_pka_json": json.dumps(np.asarray(estimate.pka_values, dtype=float).tolist()),
        "estimated_pka_sd_json": json.dumps(np.asarray(estimate.pka_sd, dtype=float).tolist()),
        "pair_probabilities_json": json.dumps(np.asarray(estimate.pair_probabilities, dtype=float).tolist()),
        "curve_rmse_ph": float(np.sqrt(np.mean(residual**2))),
        "curve_mae_ph": float(np.mean(np.abs(residual))),
        "curve_correlation": finite_correlation(true_curve, fitted_curve),
        "curve_r2": curve_r2,
        "curve_within_0p25_percent": 100.0 * float(np.mean(np.abs(residual) <= 0.25)),
        "control_true_success": int(final_true_error <= 0.10),
        "control_final_true_error": final_true_error,
        "control_final_true_ph": final_true_ph,
        "control_final_measured_ph": controller.current_ph,
        "natural_stop_reason": natural_stop_reason,
    }


def run_task(payload) -> list[dict]:
    task, particles, benchmark_seed, common_seed, checkpoints = payload
    fixed_horizon = max(checkpoints)
    controller = initialize_controller(task, particles, common_seed)
    rows = [snapshot_metrics(task, controller, benchmark_seed, "fixed_observation_count", 0)]
    natural_row = None
    while True:
        controller.use_secondary_reagents = False
        action, _ = controller.select_best_action()
        measured_ph, _, done, _ = controller.step(action, mode="Simulate")
        controller.update_posteriors(action, measured_ph)
        observations = int(controller.steps_taken)

        if observations in checkpoints:
            rows.append(
                snapshot_metrics(
                    task,
                    controller,
                    benchmark_seed,
                    "fixed_observation_count",
                    observations,
                )
            )

        if done and natural_row is None:
            measured_error = abs(controller.current_ph - controller.target_ph)
            stop_reason = "measured_success" if measured_error <= 0.10 else "max_steps"
            natural_row = snapshot_metrics(
                task,
                controller,
                benchmark_seed,
                "natural_control_end",
                observations,
                stop_reason,
            )

        if observations < fixed_horizon:
            if done:
                controller.done = False
            continue
        if natural_row is not None:
            break
        if done:
            break

    if natural_row is None:
        raise RuntimeError("Natural endpoint was not recorded")
    rows.append(natural_row)
    return rows


def mean_or_nan(values) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def median_or_nan(values) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if len(finite) else math.nan


def summarize(rows: list[dict]) -> dict:
    return {
        "tasks": len(rows),
        "observations_mean": mean_or_nan(row["observations"] for row in rows),
        "curve_rmse_ph_mean": mean_or_nan(row["curve_rmse_ph"] for row in rows),
        "curve_rmse_ph_median": median_or_nan(row["curve_rmse_ph"] for row in rows),
        "curve_mae_ph_mean": mean_or_nan(row["curve_mae_ph"] for row in rows),
        "curve_correlation_mean": mean_or_nan(row["curve_correlation"] for row in rows),
        "curve_r2_mean": mean_or_nan(row["curve_r2"] for row in rows),
        "curve_within_0p25_percent_mean": mean_or_nan(row["curve_within_0p25_percent"] for row in rows),
        "concentration_relative_error_percent_mean": mean_or_nan(
            row["concentration_relative_error_percent"] for row in rows
        ),
        "concentration_relative_error_percent_median": median_or_nan(
            row["concentration_relative_error_percent"] for row in rows
        ),
        "concentration_95_coverage_percent": 100.0 * mean_or_nan(
            row["concentration_95_covered"] for row in rows
        ),
        "pair_count_accuracy_percent": 100.0 * mean_or_nan(row["pair_count_correct"] for row in rows),
        "true_pair_probability_mean": mean_or_nan(row["true_pair_probability"] for row in rows),
        "pka_evaluable_tasks": sum(math.isfinite(float(row["pka_mae_if_k_correct"])) for row in rows),
        "pka_mae_if_k_correct_mean": mean_or_nan(row["pka_mae_if_k_correct"] for row in rows),
        "pka_rmse_if_k_correct_mean": mean_or_nan(row["pka_rmse_if_k_correct"] for row in rows),
        "pka_95_coverage_if_k_correct_percent": 100.0 * mean_or_nan(
            row["pka_95_coverage_if_k_correct"] for row in rows
        ),
        "control_success_rate_percent": 100.0 * mean_or_nan(row["control_true_success"] for row in rows),
        "control_final_true_error_mean": mean_or_nan(row["control_final_true_error"] for row in rows),
    }


def build_summaries(all_rows: list[dict], seeds: list[int], checkpoints: list[int]):
    per_seed = []
    for seed in seeds:
        seed_rows = [row for row in all_rows if int(row["benchmark_seed"]) == seed]
        for observations in checkpoints:
            subset = [
                row
                for row in seed_rows
                if row["checkpoint_type"] == "fixed_observation_count"
                and int(row["observations"]) == observations
            ]
            per_seed.append(
                {
                    "benchmark_seed": seed,
                    "checkpoint_type": "fixed_observation_count",
                    "observations": observations,
                    **summarize(subset),
                }
            )
        subset = [row for row in seed_rows if row["checkpoint_type"] == "natural_control_end"]
        per_seed.append(
            {
                "benchmark_seed": seed,
                "checkpoint_type": "natural_control_end",
                "observations": "natural_end",
                **summarize(subset),
            }
        )

    metric_fields = [
        key
        for key in per_seed[0]
        if key not in {"benchmark_seed", "checkpoint_type", "observations"}
    ]
    aggregate = []
    keys = [("fixed_observation_count", value) for value in checkpoints] + [
        ("natural_control_end", "natural_end")
    ]
    for checkpoint_type, observations in keys:
        subset = [
            row
            for row in per_seed
            if row["checkpoint_type"] == checkpoint_type and row["observations"] == observations
        ]
        output = {
            "checkpoint_type": checkpoint_type,
            "observations": observations,
            "runs": len(subset),
        }
        for metric in metric_fields:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            output[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            output[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
        aggregate.append(output)
    return per_seed, aggregate


def subgroup_summary(all_rows: list[dict]) -> list[dict]:
    output = []
    final_rows = [row for row in all_rows if row["checkpoint_type"] == "natural_control_end"]
    for field in ("direction", "true_pair_count", "pka_family", "difficulty"):
        for value in sorted({str(row[field]) for row in final_rows}):
            for seed in sorted({int(row["benchmark_seed"]) for row in final_rows}):
                subset = [
                    row
                    for row in final_rows
                    if str(row[field]) == value and int(row["benchmark_seed"]) == seed
                ]
                output.append(
                    {
                        "benchmark_seed": seed,
                        "subgroup": field,
                        "subgroup_value": value,
                        **summarize(subset),
                    }
                )
    return output


def plot_summaries(aggregate: list[dict], output_dir: Path) -> None:
    fixed = [row for row in aggregate if row["checkpoint_type"] == "fixed_observation_count"]
    x = np.asarray([int(row["observations"]) for row in fixed], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    specs = [
        ("curve_rmse_ph_mean", "Full-curve RMSE (pH)"),
        ("concentration_relative_error_percent_mean", "Concentration relative error (%)"),
        ("pair_count_accuracy_percent", "Pair-count accuracy (%)"),
        ("pka_mae_if_k_correct_mean", "pKa MAE when K is correct"),
    ]
    for axis, (metric, label) in zip(axes.flat, specs):
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in fixed])
        sd = np.asarray([float(row[f"{metric}_sd"] or 0.0) for row in fixed])
        axis.plot(x, mean, marker="o", color="#2F6F8F")
        axis.fill_between(x, mean - sd, mean + sd, color="#2F6F8F", alpha=0.16)
        axis.set_xlabel("PF observations")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2, linestyle=":")
    axes.flat[2].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_accuracy_by_observation_count.png", dpi=260)
    fig.savefig(output_dir / "posterior_accuracy_by_observation_count.svg")
    plt.close(fig)

    final = next(row for row in aggregate if row["checkpoint_type"] == "natural_control_end")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    final_specs = [
        ("curve_rmse_ph_mean", "Curve RMSE", "pH"),
        ("concentration_relative_error_percent_mean", "Concentration error", "%"),
        ("pair_count_accuracy_percent", "Pair-count accuracy", "%"),
        ("pka_mae_if_k_correct_mean", "pKa MAE (K correct)", "pKa"),
    ]
    for axis, (metric, title, unit) in zip(axes, final_specs):
        value = float(final[f"{metric}_mean"])
        error = float(final[f"{metric}_sd"] or 0.0)
        axis.bar([0], [value], yerr=[error], color="#3D7A5C", capsize=5)
        axis.set_xticks([0], ["Natural end"])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2, linestyle=":")
    axes[2].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_accuracy_at_natural_end.png", dpi=260)
    fig.savefig(output_dir / "posterior_accuracy_at_natural_end.svg")
    plt.close(fig)


def example_plots(all_rows: list[dict], all_tasks: dict[tuple[int, int], ControlTask], output_dir: Path) -> None:
    final_rows = [row for row in all_rows if row["checkpoint_type"] == "natural_control_end"]
    example_dir = output_dir / "curve_examples"
    example_dir.mkdir(exist_ok=True)
    for pair_count in (1, 2, 3):
        candidates = [row for row in final_rows if int(row["true_pair_count"]) == pair_count]
        if not candidates:
            continue
        median = float(np.median([float(row["curve_rmse_ph"]) for row in candidates]))
        chosen = min(candidates, key=lambda row: abs(float(row["curve_rmse_ph"]) - median))
        key = (int(chosen["task_seed"]), int(chosen["task_id"]))
        task = all_tasks[key]
        task_rows = [
            row
            for row in all_rows
            if int(row["task_seed"]) == key[0]
            and int(row["task_id"]) == key[1]
            and (
                row["checkpoint_type"] == "natural_control_end"
                or int(row["observations"]) in {0, 1, 3, 5, 8, 12}
            )
        ]
        available = {
            int(row["observations"])
            for row in task_rows
            if row["checkpoint_type"] == "fixed_observation_count"
        }
        preferred = [value for value in (0, 1, 3, 5, 8, 12) if value in available]
        if len(preferred) < min(6, len(available)):
            preferred = sorted(available)[:6]
        ordered = []
        for observations in preferred:
            ordered.append(
                next(
                    row
                    for row in task_rows
                    if row["checkpoint_type"] == "fixed_observation_count"
                    and int(row["observations"]) == observations
                )
            )
        ordered.append(next(row for row in task_rows if row["checkpoint_type"] == "natural_control_end"))
        initial_state = SolutionState(task.initial_volume_ml, task.initial_base_moles, 0.0)
        true_curve = response_curve(
            task.analyte_conc_m,
            task.pka_values,
            task.initial_volume_ml,
            initial_state,
            CURVE_GRID_ML,
        )
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), sharex=True, sharey=True)
        for axis, row in zip(axes.flat, ordered):
            estimate_pka = json.loads(row["estimated_pka_json"])
            fitted = response_curve(
                float(row["estimated_concentration_m"]),
                estimate_pka,
                task.initial_volume_ml,
                initial_state,
                CURVE_GRID_ML,
            )
            axis.plot(CURVE_GRID_ML, true_curve, color="#111111", linewidth=2.0, label="truth")
            axis.plot(CURVE_GRID_ML, fitted, color="#C98232", linewidth=1.6, label="PF fit")
            title = (
                "natural end"
                if row["checkpoint_type"] == "natural_control_end"
                else f"n={int(row['observations'])}"
            )
            axis.set_title(f"{title}; RMSE={float(row['curve_rmse_ph']):.3f}")
            axis.grid(alpha=0.18, linestyle=":")
        for axis in axes.flat[len(ordered):]:
            axis.axis("off")
        for axis in axes[-1, :3]:
            axis.set_xlabel("Signed titrant volume (mL)")
        for axis in axes[:, 0]:
            axis.set_ylabel("pH")
        axes.flat[0].legend(frameon=False)
        fig.suptitle(
            f"Representative K={pair_count} task: seed {task.seed}, task {task.task_id}",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(example_dir / f"representative_k{pair_count}.png", dpi=240)
        fig.savefig(example_dir / f"representative_k{pair_count}.svg")
        plt.close(fig)


def write_report(output_dir: Path, aggregate: list[dict], seeds: list[int], tasks_per_seed: int, particles: int) -> None:
    def formatted(row: dict, metric: str, digits: int) -> str:
        mean = float(row[f"{metric}_mean"])
        sd = row.get(f"{metric}_sd")
        if sd is None:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} +/- {float(sd):.{digits}f}"

    fixed = [row for row in aggregate if row["checkpoint_type"] == "fixed_observation_count"]
    final = next(row for row in aggregate if row["checkpoint_type"] == "natural_control_end")
    lines = [
        "# PF posterior accuracy and curve-similarity study",
        "",
        f"Evaluation used {len(seeds)} independent seeds, {tasks_per_seed} tasks per seed, and {particles} particles.",
        "The fixed-observation analysis continues the same hybrid-full PF controller to 12 observations even if it reaches the control target earlier. This keeps the task cohort identical at every observation count. Natural control endpoints are reported separately.",
        "Complete response curves are evaluated over signed primary-titrant additions from -100 to +100 mL relative to the initial chemical state.",
        "",
        "| Observations | Curve RMSE (pH) | Curve correlation | Concentration error (%) | K accuracy (%) | True-K probability | pKa MAE when K correct |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed:
        lines.append(
            f"| {row['observations']} | {formatted(row, 'curve_rmse_ph_mean', 4)} | "
            f"{formatted(row, 'curve_correlation_mean', 4)} | "
            f"{formatted(row, 'concentration_relative_error_percent_mean', 2)} | "
            f"{formatted(row, 'pair_count_accuracy_percent', 2)} | "
            f"{formatted(row, 'true_pair_probability_mean', 3)} | "
            f"{formatted(row, 'pka_mae_if_k_correct_mean', 4)} |"
        )
    lines.extend(
        [
            "",
            "## Natural control endpoint",
            "",
            f"Mean observations: {formatted(final, 'observations_mean', 2)}.",
            f"Curve RMSE: {formatted(final, 'curve_rmse_ph_mean', 4)} pH.",
            f"Concentration relative error: {formatted(final, 'concentration_relative_error_percent_mean', 2)}%.",
            f"Pair-count accuracy: {formatted(final, 'pair_count_accuracy_percent', 2)}%.",
            f"pKa MAE conditional on correct K: {formatted(final, 'pka_mae_if_k_correct_mean', 4)}.",
            f"Control success: {formatted(final, 'control_success_rate_percent', 2)}%.",
            "",
            "Task-level posterior estimates, seed-level summaries, final subgroups, plots, and exact generated tasks accompany this report.",
        ]
    )
    (output_dir / "POSTERIOR_DIAGNOSTIC_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PF curve fitting and posterior parameter accuracy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks-per-seed", type=int, default=300)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if checkpoints[0] != 0 or checkpoints[-1] < 1:
        raise ValueError("Checkpoints must include 0 and at least one positive observation count")
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "pf_variant": PF_VARIANT,
        "policy": POLICY,
        "seeds": args.seeds,
        "tasks_per_seed": args.tasks_per_seed,
        "particles": args.particles,
        "checkpoints": checkpoints,
        "curve_grid_ml": CURVE_GRID_ML.tolist(),
        "workers": workers,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("Existing posterior-diagnostic output uses a different configuration")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is non-empty and has no RUN_CONFIG.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows = []
    all_tasks = {}
    for benchmark_seed in args.seeds:
        task_path = args.output_dir / f"seed_{benchmark_seed}_tasks.jsonl"
        result_path = args.output_dir / f"seed_{benchmark_seed}_posterior_rows.csv"
        complete_path = args.output_dir / f"seed_{benchmark_seed}_COMPLETE.json"
        if args.resume and task_path.exists() and result_path.exists() and complete_path.exists():
            tasks = load_tasks(task_path)
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            print(f"posterior seed {benchmark_seed} already complete", flush=True)
        else:
            tasks = generate_tasks(
                2_000_000 + benchmark_seed,
                args.tasks_per_seed,
                f"posterior_diagnostic_seed_{benchmark_seed}",
            )
            save_tasks(task_path, tasks)
            payloads = [
                (
                    task,
                    args.particles,
                    benchmark_seed,
                    benchmark_seed * 30_000_049 + task.task_id * 1013,
                    checkpoints,
                )
                for task in tasks
            ]
            rows = []
            if workers == 1:
                results = map(run_task, payloads)
                executor = None
            else:
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
                results = executor.map(run_task, payloads, chunksize=2)
            try:
                for index, task_rows in enumerate(results, 1):
                    rows.extend(task_rows)
                    if index % 25 == 0 or index == len(tasks):
                        print(
                            f"posterior seed {benchmark_seed}: {index}/{len(tasks)} tasks",
                            flush=True,
                        )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)
            write_csv(result_path, rows)
            complete_path.write_text(
                json.dumps({"benchmark_seed": benchmark_seed, "tasks": len(tasks), "rows": len(rows)}, indent=2),
                encoding="utf-8",
            )
        all_rows.extend(rows)
        for task in tasks:
            all_tasks[(task.seed, task.task_id)] = task

    # Reload typed numeric values from the per-seed CSVs for resumed and fresh runs alike.
    typed_rows = []
    for benchmark_seed in args.seeds:
        with (args.output_dir / f"seed_{benchmark_seed}_posterior_rows.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                for key in (
                    "benchmark_seed", "task_seed", "task_id", "observations", "true_pair_count",
                    "estimated_pair_count", "pair_count_correct", "concentration_95_covered",
                    "control_true_success",
                ):
                    row[key] = int(row[key])
                for key in (
                    "true_concentration_m", "estimated_concentration_m", "concentration_abs_error_m",
                    "concentration_relative_error_percent", "true_pair_probability", "pka_mae_if_k_correct",
                    "pka_rmse_if_k_correct", "pka_95_coverage_if_k_correct", "curve_rmse_ph",
                    "curve_mae_ph", "curve_correlation", "curve_r2", "curve_within_0p25_percent",
                    "control_final_true_error", "control_final_true_ph", "control_final_measured_ph",
                ):
                    row[key] = float(row[key])
                typed_rows.append(row)
    per_seed, aggregate = build_summaries(typed_rows, args.seeds, checkpoints)
    write_csv(args.output_dir / "all_posterior_task_results.csv", typed_rows)
    write_csv(args.output_dir / "per_seed_posterior_summary.csv", per_seed)
    write_csv(args.output_dir / "aggregate_posterior_summary.csv", aggregate)
    write_csv(args.output_dir / "natural_end_subgroup_per_seed.csv", subgroup_summary(typed_rows))
    plot_summaries(aggregate, args.output_dir)
    example_plots(typed_rows, all_tasks, args.output_dir)
    write_report(args.output_dir, aggregate, args.seeds, args.tasks_per_seed, args.particles)
    (args.output_dir / "POSTERIOR_DIAGNOSTICS_COMPLETE.json").write_text(
        json.dumps({"config": config, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(f"Posterior diagnostics complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
