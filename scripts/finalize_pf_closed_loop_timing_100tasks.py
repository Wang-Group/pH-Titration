from __future__ import annotations

"""Audit and regenerate derived PF closed-loop timing summaries.

The released task, step, and trajectory files are primary observations. This
script recomputes every derived table without rerunning the timing benchmark.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("pf_1000", "pf_10000", "pf_100000")
PARTICLES = {"pf_1000": 1_000, "pf_10000": 10_000, "pf_100000": 100_000}
SEEDS = (101, 202, 303, 404, 555)
TASK_IDS = (
    1,
    159,
    317,
    475,
    632,
    790,
    948,
    1106,
    1264,
    1422,
    1579,
    1737,
    1895,
    2053,
    2211,
    2369,
    2526,
    2684,
    2842,
    3000,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def stats(values: list[float]) -> dict[str, float | int]:
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


def closed_loop_summary(method: str, rows: list[dict[str, str]]) -> dict:
    particle_count = PARTICLES[method]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    result = {
        "method": method,
        "particle_count": particle_count,
        "tasks": len(rows),
        "successful_tasks": sum(int(row["true_success"]) for row in rows),
        "success_rate_percent": 100.0 * np.mean([int(row["true_success"]) for row in rows]),
        "strict_success_rate_percent": 100.0 * np.mean([int(row["strict_success"]) for row in rows]),
        "measured_success_rate_percent": 100.0 * np.mean([int(row["measured_success"]) for row in rows]),
        "false_stop_rate_percent": 100.0 * np.mean([int(row["false_stop"]) for row in rows]),
        "severe_failure_rate_percent": 100.0 * np.mean([int(row["severe_failure"]) for row in rows]),
        "steps_mean": float(np.mean(values("steps"))),
        "total_volume_mean_ml": float(np.mean(values("total_volume_ml"))),
        "overshoots_mean": float(np.mean(values("overshoots"))),
    }
    for label in ("final_abs_error_ph", "final_signed_error_ph"):
        summary = stats(values(label))
        for stat in ("mean", "median", "sd", "iqr", "p95", "p99"):
            result[f"{label}_{stat}"] = summary[stat]
    return result


def timing_summary(method: str, rows: list[dict[str, str]], max_n: int = 6) -> list[dict]:
    particle_count = PARTICLES[method]
    counts: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["benchmark_seed"]), int(row["task_id"]))
        counts[key] = counts.get(key, 0) + 1

    scopes: list[tuple[str, str, set[tuple[int, int]], list[dict[str, str]]]] = []
    for n in range(1, max_n + 1):
        eligible = {key for key, count in counts.items() if count >= n}
        selected = [
            row
            for row in rows
            if (int(row["benchmark_seed"]), int(row["task_id"])) in eligible
            and int(row["step_index"]) <= n
        ]
        scopes.append((str(n), "first_n_steps", eligible, selected))
    scopes.append(("all", "all_recorded_steps", set(counts), list(rows)))

    output: list[dict] = []
    for n_label, scope_label, eligible, selected in scopes:
        result: dict[str, float | int | str] = {
            "method": method,
            "particle_count": particle_count,
            "n_first_steps": n_label,
            "step_scope": scope_label,
            "n_tasks_with_ge_n_steps": len(eligible),
            "n_step_measurements": len(selected),
        }
        for key in ("observe_time_ms", "recommend_time_ms", "total_decision_time_ms"):
            summary = stats([float(row[key]) for row in selected])
            prefix = key.removesuffix("_time_ms")
            for stat in ("mean", "median", "sd", "iqr", "p95", "p99"):
                result[f"{prefix}_{stat}_ms"] = summary[stat]
        output.append(result)
    return output


def validate_task_and_trajectory_rows(
    method: str,
    tasks: list[dict[str, str]],
    steps: list[dict[str, str]],
    trajectory_path: Path,
) -> None:
    expected = {(seed, task_id) for seed in SEEDS for task_id in TASK_IDS}
    observed = {(int(row["benchmark_seed"]), int(row["task_id"])) for row in tasks}
    if observed != expected or len(tasks) != 100:
        raise RuntimeError(f"{method}: task selection does not match the locked cohort")
    if any(row["method"] != method for row in tasks + steps):
        raise RuntimeError(f"{method}: method label mismatch")

    trajectories = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(trajectories) != 100:
        raise RuntimeError(f"{method}: expected 100 trajectories")
    by_key = {
        (int(row["benchmark_seed"]), int(row["task_id"])): row for row in trajectories
    }
    if set(by_key) != expected:
        raise RuntimeError(f"{method}: trajectory selection mismatch")
    task_by_key = {(int(row["benchmark_seed"]), int(row["task_id"])): row for row in tasks}
    for key, trajectory in by_key.items():
        task = task_by_key[key]
        if len(trajectory["trajectory"]) != int(task["trajectory_records"]):
            raise RuntimeError(f"{method}: trajectory length mismatch for {key}")
        if abs(float(trajectory["final_true_ph"]) - float(task["final_true_ph"])) > 1e-12:
            raise RuntimeError(f"{method}: endpoint mismatch for {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "evidence"
            / "simulation_numerical_evidence_20260823"
            / "17_PF_CLOSED_LOOP_TIMING_100TASKS"
        ),
    )
    args = parser.parse_args()
    block = args.block.resolve()
    results = block / "results"
    repository = Path(__file__).resolve().parents[1]

    closed_rows: list[dict] = []
    all_timing_rows: list[dict] = []
    for method in METHODS:
        method_root = results / method
        tasks = read_csv(method_root / "task_results.csv")
        steps = read_csv(method_root / "per_step_timing.csv")
        validate_task_and_trajectory_rows(method, tasks, steps, method_root / "trajectories.jsonl")
        closed = closed_loop_summary(method, tasks)
        timing = timing_summary(method, steps)
        write_csv(method_root / "closed_loop_summary.csv", [closed])
        write_csv(method_root / "timing_first_n_summary.csv", timing)
        closed_rows.append(closed)
        all_timing_rows.extend(timing)

    write_csv(results / "PF_CLOSED_LOOP_OUTCOME_SUMMARY.csv", closed_rows)
    all_rows = [row for row in all_timing_rows if row["n_first_steps"] == "all"]
    write_csv(results / "PF_CLOSED_LOOP_TIMING_SUMMARY.csv", all_rows)

    block16_results = (
        repository
        / "evidence"
        / "simulation_numerical_evidence_20260823"
        / "16_MATCHED_TIMING_RECOVERY_100TASKS"
        / "results"
    )
    publication_rows: list[dict] = []
    for method in ("imitation", "ppo", "pymc"):
        raw = read_csv(block16_results / method / "raw.csv")
        summary = stats([float(row["wall_ns"]) / 1e6 for row in raw])
        publication_rows.append(
            {
                "method": method,
                "protocol_block": "16_MATCHED_TIMING_RECOVERY_100TASKS",
                "timing_scope": "matched_single_step_call",
                "tasks": 100,
                "timed_observations": int(summary["n"]),
                "cpu_affinity_control": "yes",
                **{f"{key}_ms": summary[key] for key in ("mean", "median", "sd", "iqr", "p95", "p99")},
            }
        )
    for row in all_rows:
        publication_rows.append(
            {
                "method": row["method"],
                "protocol_block": "17_PF_CLOSED_LOOP_TIMING_100TASKS",
                "timing_scope": "all_recorded_closed_loop_steps",
                "tasks": 100,
                "timed_observations": int(row["n_step_measurements"]),
                "cpu_affinity_control": "no",
                **{
                    f"{key}_ms": row[f"total_decision_{key}_ms"]
                    for key in ("mean", "median", "sd", "iqr", "p95", "p99")
                },
            }
        )
    write_csv(results / "PUBLICATION_TIMING_SCOPE_SUMMARY.csv", publication_rows)

    audit = {
        "status": "PASS",
        "primary_raw_files": [
            "task_results.csv",
            "per_step_timing.csv",
            "trajectories.jsonl",
        ],
        "task_cases_per_particle_count": 100,
        "step_measurements": {
            row["method"]: int(row["n_step_measurements"]) for row in all_rows
        },
        "success_rate_percent": {
            row["method"]: float(row["success_rate_percent"]) for row in closed_rows
        },
        "median_observation_to_action_ms": {
            row["method"]: float(row["total_decision_median_ms"]) for row in all_rows
        },
        "publication_timing_scope_summary_regenerated": True,
    }
    (results / "RELEASE_VALIDATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
