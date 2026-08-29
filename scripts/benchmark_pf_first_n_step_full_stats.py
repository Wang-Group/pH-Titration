from __future__ import annotations

"""PF closed-loop timing and outcome benchmark on the matched 100-task cohort.

For every executed dose, the measured interval is split into:

    observe() time + recommend() time = total decision time

The chemistry transition and sensor quantisation are intentionally outside the
timed interval.  A JSONL trajectory is also written for every task so that
closed-loop outcomes can be audited independently from timing summaries.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
PRIMARY = EVIDENCE / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
SEEDS = (101, 202, 303, 404, 555)
TASK_IDS = (
    1, 159, 317, 475, 632, 790, 948, 1106, 1264, 1422,
    1579, 1737, 1895, 2053, 2211, 2369, 2526, 2684, 2842, 3000,
)

SUCCESS_TOLERANCE_PH = 0.10
STRICT_TOLERANCE_PH = 0.05
SEVERE_FAILURE_TOLERANCE_PH = 0.50
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0
TITRANT_M = 0.1
MIN_ACTION_ML = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for benchmark_seed in SEEDS:
        path = PRIMARY / "tasks" / f"seed_{benchmark_seed}_tasks.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle if line.strip()]
        by_id = {int(row["task_id"]): row for row in payloads}
        for task_id in TASK_IDS:
            payload = by_id[task_id]
            task = dict(payload)
            task["benchmark_seed"] = benchmark_seed
            task["pka_values"] = tuple(float(value) for value in payload["pka_values"])
            task["initial_ph"] = float(payload["initial_ph"])
            task["measured_initial_ph"] = float(np.round(task["initial_ph"], 2))
            cases.append(task)
    if len(cases) != 100:
        raise RuntimeError(f"Expected 100 matched cases, found {len(cases)}")
    return cases


def distribution_stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "sd": math.nan,
            "iqr": math.nan,
            "p95": math.nan,
            "p99": math.nan,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def rollout(
    case: dict,
    particle_count: int,
    RobustPFController,
    SolutionState,
    solve_ph_scalar,
) -> tuple[dict, list[dict], dict]:
    """Run one complete trajectory and return task, step, and JSON records."""
    pf_seed = (int(case["seed"]) * 1_000_003 + int(case["task_id"])) % (2**32 - 1)
    controller = RobustPFController(
        particles=particle_count,
        seed=pf_seed,
        max_steps=MAX_STEPS,
        max_total_dose_ml=MAX_TOTAL_DOSE_ML,
    )
    controller.reset(
        case["measured_initial_ph"],
        float(case["target_ph"]),
        float(case["initial_volume_ml"]),
        float(case["initial_base_moles"]),
        0.0,
    )

    target_ph = float(case["target_ph"])
    true_ph = float(case["initial_ph"])
    total_volume_ml = float(case["initial_volume_ml"])
    base_moles = float(case["initial_base_moles"])
    acid_moles = 0.0
    total_added_ml = 0.0
    steps = 0
    overshoots = 0
    step_rows: list[dict] = []

    # The initial recommendation bootstraps the trajectory and is deliberately
    # excluded from observation-to-action timing, matching the original path.
    action = controller.recommend()
    initial_record = {
        "step_index": 0,
        "phase": "initial",
        "true_ph": true_ph,
        "measured_ph": float(case["measured_initial_ph"]),
        "target_ph": target_ph,
        "action_reagent": "" if action.reagent is None else str(action.reagent),
        "requested_volume_ml": float(action.volume_ml),
        "actual_volume_ml": 0.0,
        "observe_time_ms": None,
        "recommend_time_ms": None,
        "total_decision_time_ms": None,
        "next_action_stop": int(action.stop),
        "next_action_reason": str(action.reason),
    }
    trajectory: list[dict] = [initial_record]

    while (
        not controller.done
        and steps < MAX_STEPS
        and total_added_ml < MAX_TOTAL_DOSE_ML - 1e-12
    ):
        if action.stop:
            break

        before_true_ph = true_ph
        before_measured_ph = float(controller.current_ph)
        reagent = str(action.reagent)
        requested_volume_ml = float(action.volume_ml)
        actual_volume_ml = float(
            np.clip(requested_volume_ml, MIN_ACTION_ML, MAX_TOTAL_DOSE_ML - total_added_ml)
        )

        total_volume_ml += actual_volume_ml
        if reagent == "base":
            base_moles += TITRANT_M * actual_volume_ml / 1000.0
        else:
            acid_moles += TITRANT_M * actual_volume_ml / 1000.0

        true_ph = float(
            solve_ph_scalar(
                float(case["analyte_conc_m"]),
                case["pka_values"],
                float(case["initial_volume_ml"]),
                SolutionState(total_volume_ml, base_moles, acid_moles),
            )
        )
        measured_ph = float(np.round(np.clip(true_ph, 0.0, 14.0), 2))
        total_added_ml += actual_volume_ml
        steps += 1
        crossed = (before_true_ph - target_ph) * (true_ph - target_ph) < 0.0
        overshoots += int(crossed)

        # Split the same end-to-end interval into observe, recommend, and total.
        t0 = time.perf_counter_ns()
        controller.observe(
            measured_ph,
            actual_volume_ml=actual_volume_ml,
            reagent=reagent,
        )
        t1 = time.perf_counter_ns()
        action = controller.recommend()
        t2 = time.perf_counter_ns()
        observe_ms = (t1 - t0) / 1e6
        recommend_ms = (t2 - t1) / 1e6
        total_decision_ms = (t2 - t0) / 1e6

        step_record = {
            "method": f"pf_{particle_count}",
            "particle_count": particle_count,
            "benchmark_seed": int(case["benchmark_seed"]),
            "task_seed": int(case["seed"]),
            "task_id": int(case["task_id"]),
            "step_index": steps,
            "before_true_ph": before_true_ph,
            "before_measured_ph": before_measured_ph,
            "target_ph": target_ph,
            "action_reagent": reagent,
            "requested_volume_ml": requested_volume_ml,
            "actual_volume_ml": actual_volume_ml,
            "true_ph_after": true_ph,
            "measured_ph_after": measured_ph,
            "observe_time_ms": observe_ms,
            "recommend_time_ms": recommend_ms,
            "total_decision_time_ms": total_decision_ms,
            "next_action_stop": int(action.stop),
            "next_action_reason": str(action.reason),
            "next_action_reagent": "" if action.reagent is None else str(action.reagent),
            "next_action_volume_ml": float(action.volume_ml),
        }
        step_rows.append(step_record)
        trajectory.append({
            "step_index": steps,
            "phase": "post_action_observation",
            "true_ph": true_ph,
            "measured_ph": measured_ph,
            "target_ph": target_ph,
            "action_reagent": reagent,
            "requested_volume_ml": requested_volume_ml,
            "actual_volume_ml": actual_volume_ml,
            "observe_time_ms": observe_ms,
            "recommend_time_ms": recommend_ms,
            "total_decision_time_ms": total_decision_ms,
            "next_action_stop": int(action.stop),
            "next_action_reason": str(action.reason),
            "next_action_reagent": "" if action.reagent is None else str(action.reagent),
            "next_action_volume_ml": float(action.volume_ml),
        })

    if controller.done:
        stop_reason = str(controller.stop_reason)
    elif steps >= MAX_STEPS:
        stop_reason = "max_steps_external_guard"
    elif total_added_ml >= MAX_TOTAL_DOSE_ML - 1e-12:
        stop_reason = "dose_limit_external_guard"
    else:
        stop_reason = str(action.reason)

    final_true_ph = true_ph
    final_measured_ph = float(controller.current_ph)
    final_signed_error = final_true_ph - target_ph
    final_abs_error = abs(final_signed_error)
    final_measured_abs_error = abs(final_measured_ph - target_ph)
    true_success = final_abs_error <= SUCCESS_TOLERANCE_PH
    measured_success = final_measured_abs_error <= SUCCESS_TOLERANCE_PH
    task_record = {
        "method": f"pf_{particle_count}",
        "particle_count": particle_count,
        "benchmark_seed": int(case["benchmark_seed"]),
        "task_seed": int(case["seed"]),
        "task_id": int(case["task_id"]),
        "true_k": len(case["pka_values"]),
        "initial_ph": float(case["initial_ph"]),
        "measured_initial_ph": float(case["measured_initial_ph"]),
        "target_ph": target_ph,
        "steps": steps,
        "overshoots": overshoots,
        "total_volume_ml": total_added_ml,
        "final_true_ph": final_true_ph,
        "final_measured_ph": final_measured_ph,
        "final_signed_error_ph": final_signed_error,
        "final_abs_error_ph": final_abs_error,
        "final_measured_abs_error_ph": final_measured_abs_error,
        "true_success": int(true_success),
        "strict_success": int(final_abs_error <= STRICT_TOLERANCE_PH),
        "severe_failure": int(final_abs_error > SEVERE_FAILURE_TOLERANCE_PH),
        "measured_success": int(measured_success),
        "false_stop": int(measured_success and not true_success),
        "stop_reason": stop_reason,
        "trajectory_records": len(trajectory),
    }
    trajectory_record = {
        "method": f"pf_{particle_count}",
        "particle_count": particle_count,
        "benchmark_seed": int(case["benchmark_seed"]),
        "task_seed": int(case["seed"]),
        "task_id": int(case["task_id"]),
        "target_ph": target_ph,
        "stop_reason": stop_reason,
        "true_success": int(true_success),
        "final_true_ph": final_true_ph,
        "final_abs_error_ph": final_abs_error,
        "trajectory": trajectory,
    }
    return task_record, step_rows, trajectory_record


def summarize_closed_loop(rows: list[dict], particle_count: int) -> dict:
    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    summary = {
        "method": f"pf_{particle_count}",
        "particle_count": particle_count,
        "tasks": len(rows),
        "successful_tasks": int(sum(int(row["true_success"]) for row in rows)),
        "success_rate_percent": 100.0 * float(np.mean([int(row["true_success"]) for row in rows])),
        "strict_success_rate_percent": 100.0 * float(np.mean([int(row["strict_success"]) for row in rows])),
        "measured_success_rate_percent": 100.0 * float(np.mean([int(row["measured_success"]) for row in rows])),
        "false_stop_rate_percent": 100.0 * float(np.mean([int(row["false_stop"]) for row in rows])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([int(row["severe_failure"]) for row in rows])),
        "steps_mean": float(np.mean(vals("steps"))),
        "total_volume_mean_ml": float(np.mean(vals("total_volume_ml"))),
        "overshoots_mean": float(np.mean(vals("overshoots"))),
    }
    for label, key in (
        ("final_abs_error_ph", "final_abs_error_ph"),
        ("final_signed_error_ph", "final_signed_error_ph"),
    ):
        stats = distribution_stats(vals(key))
        for stat in ("mean", "median", "sd", "iqr", "p95", "p99"):
            summary[f"{label}_{stat}"] = stats[stat]
    return summary


def summarize_timing(
    step_rows: list[dict],
    particle_count: int,
    max_n: int,
) -> list[dict]:
    keys = [(int(row["benchmark_seed"]), int(row["task_id"])) for row in step_rows]
    counts: dict[tuple[int, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    output: list[dict] = []
    scopes: list[tuple[str, set[tuple[int, int]], list[dict]]] = []
    for n in range(1, max_n + 1):
        eligible = {key for key, count in counts.items() if count >= n}
        selected = [
            row
            for row in step_rows
            if (int(row["benchmark_seed"]), int(row["task_id"])) in eligible
            and int(row["step_index"]) <= n
        ]
        scopes.append((str(n), eligible, selected))

    # Include every recorded decision cycle, in addition to the first-n views.
    # This is the full-trajectory timing distribution for the configuration.
    scopes.append(("all", set(counts), list(step_rows)))

    for scope, eligible, selected in scopes:
        summary: dict[str, float | int] = {
            "method": f"pf_{particle_count}",
            "particle_count": particle_count,
            "n_first_steps": scope,
            "step_scope": "all_recorded_steps" if scope == "all" else "first_n_steps",
            "n_tasks_with_ge_n_steps": len(eligible),
            "n_step_measurements": len(selected),
        }
        for component in ("observe_time_ms", "recommend_time_ms", "total_decision_time_ms"):
            stats = distribution_stats([float(row[component]) for row in selected])
            prefix = component.removesuffix("_time_ms")
            for stat in ("mean", "median", "sd", "iqr", "p95", "p99"):
                summary[f"{prefix}_{stat}_ms"] = stats[stat]
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "pf_closed_loop_timing_100tasks",
    )
    parser.add_argument("--particles", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--warmup-tasks", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=6)
    args = parser.parse_args()
    if any(value < 60 for value in args.particles):
        raise ValueError("each particle count must be at least 60")
    if args.warmup_tasks < 0 or args.max_n < 1:
        raise ValueError("warmup-tasks must be nonnegative and max-n must be positive")

    sys.path.insert(0, str(ROOT))
    from controllers.chemistry_model import SolutionState, solve_ph_scalar
    from controllers.new_pf_controller import RobustPFController

    cases = load_cases()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict] = []
    step_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    closed_loop_rows: list[dict] = []
    timing_rows: list[dict] = []

    for particle_count in args.particles:
        for case in cases[: args.warmup_tasks]:
            rollout(case, int(particle_count), RobustPFController, SolutionState, solve_ph_scalar)

        method_tasks: list[dict] = []
        method_steps: list[dict] = []
        for index, case in enumerate(cases, 1):
            task, steps, trajectory = rollout(
                case, int(particle_count), RobustPFController, SolutionState, solve_ph_scalar
            )
            method_tasks.append(task)
            method_steps.extend(steps)
            trajectory_rows.append(trajectory)
            if index % 20 == 0 or index == len(cases):
                print(f"PF {particle_count}: {index}/{len(cases)} cases", flush=True)
        task_rows.extend(method_tasks)
        step_rows.extend(method_steps)
        closed_loop_rows.append(summarize_closed_loop(method_tasks, int(particle_count)))
        timing_rows.extend(summarize_timing(method_steps, int(particle_count), args.max_n))

    write_csv(output / "task_results.csv", task_rows)
    write_csv(output / "per_step_timing.csv", step_rows)
    write_csv(output / "closed_loop_summary.csv", closed_loop_rows)
    write_csv(output / "timing_first_n_summary.csv", timing_rows)
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for record in trajectory_rows:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

    config = {
        "study_id": "pf_first_n_step_timing_and_closed_loop_full_stats_20260829",
        "status": "complete",
        "benchmark_seeds": list(SEEDS),
        "selected_task_ids_per_seed": list(TASK_IDS),
        "cases_per_particle_count": len(cases),
        "particle_counts": [int(value) for value in args.particles],
        "warmup_tasks_per_particle_count": args.warmup_tasks,
        "max_n_first_steps": args.max_n,
        "timing_scope": "post-dose measured pH observation to next controller action",
        "timed_operations": "controller.observe(), controller.recommend()",
        "excluded_from_timing": [
            "controller construction and reset", "bootstrap recommend()", "chemical transition",
            "solve_ph_scalar", "sensor quantisation", "dose clipping", "file I/O",
        ],
        "success_definition": "true final absolute pH error <= 0.10",
        "strict_success_definition": "true final absolute pH error <= 0.05",
        "false_stop_definition": "measured final pH error <= 0.10 while true final pH error > 0.10",
        "timing_statistics": "pooled step-level mean, median, sample SD, IQR (P75-P25), P95, P99",
        "n_tasks_with_ge_n_steps_definition": "only tasks with at least n measured decision cycles are included",
        "titrant_concentration_m": TITRANT_M,
        "max_steps": MAX_STEPS,
        "max_total_dose_ml": MAX_TOTAL_DOSE_ML,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "cpu",
        },
        "source_hashes": {
            "worker": sha256(Path(__file__).resolve()),
            "pf_controller": sha256(ROOT / "controllers" / "new_pf_controller.py"),
            "chemistry_model": sha256(ROOT / "controllers" / "chemistry_model.py"),
        },
    }
    (output / "RUN_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    readme = """# PF first-n timing and closed-loop statistics

`per_step_timing.csv` records separate `observe_time_ms`, `recommend_time_ms`,
and `total_decision_time_ms` values for every post-dose observation.
`trajectories.jsonl` contains the complete initial state and action/observation
trajectory for every task. `task_results.csv` records final true and measured
pH, signed and absolute errors, success flags, stop reason, dose, and steps.

Success uses the unrounded final true pH with absolute-error tolerance 0.10 pH;
strict success uses 0.05 pH. Timing summaries are pooled over eligible task
steps and report mean, median, sample standard deviation, IQR, P95, and P99.
The timing summary includes first-n rows for n=1..6 plus an `all` row covering
all recorded decision cycles in each configuration.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== Closed-loop summary ===")
    print(json.dumps(closed_loop_rows, indent=2), flush=True)
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
