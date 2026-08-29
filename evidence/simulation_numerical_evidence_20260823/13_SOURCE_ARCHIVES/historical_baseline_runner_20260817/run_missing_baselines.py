from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from baseline_controllers import (
    PRESPECIFIED_PID,
    FixedGainPIDController,
    PIDConfig,
    SimpleRuleController,
    rollout_baseline,
)
from task_distribution import load_tasks


BENCHMARK_SEEDS = (101, 202, 303, 404, 555)
METHODS = ("prespecified_pid", "simple_rule")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_controller(method: str, selected_pid: dict | None = None):
    if method == "prespecified_pid":
        return FixedGainPIDController(PRESPECIFIED_PID)
    if method == "selected_pid":
        if selected_pid is None:
            raise ValueError("selected_pid parameters are required")
        return FixedGainPIDController(PIDConfig.from_mapping(selected_pid))
    if method == "simple_rule":
        return SimpleRuleController()
    raise ValueError(method)


def run_job(payload: tuple) -> dict:
    method, benchmark_seed, task, selected_pid = payload
    metrics = rollout_baseline(task, build_controller(method, selected_pid))
    return {
        "benchmark_seed": benchmark_seed,
        "method": method,
        "task_seed": task.seed,
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        **metrics,
    }


def summarize(rows: list[dict], seed: int, method: str) -> dict:
    subset = [row for row in rows if row["benchmark_seed"] == seed and row["method"] == method]
    successes = [row for row in subset if row["true_success"]]
    return {
        "benchmark_seed": seed,
        "method": method,
        "tasks": len(subset),
        "success_rate_percent": 100.0 * float(np.mean([row["true_success"] for row in subset])),
        "strict_success_rate_percent": 100.0 * float(np.mean([row["strict_success"] for row in subset])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([row["severe_failure"] for row in subset])),
        "false_stop_rate_percent": 100.0 * float(np.mean([row["false_stop"] for row in subset])),
        "successful_steps_mean": (
            float(np.mean([row["steps"] for row in successes])) if successes else math.nan
        ),
        "overshoots_mean": float(np.mean([row["overshoots"] for row in subset])),
        "total_volume_mean_ml": float(np.mean([row["total_volume_ml"] for row in subset])),
        "final_abs_error_mean": float(np.mean([row["final_abs_error"] for row in subset])),
    }


def aggregate(per_seed: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in per_seed}):
        subset = [row for row in per_seed if row["method"] == method]
        result = {"method": method, "seed_runs": len(subset)}
        for metric in subset[0]:
            if metric in {"benchmark_seed", "method"}:
                continue
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            result[f"{metric}_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
            )
        output.append(result)
    return output


def load_pf_rows(reference_dir: Path, tasks_by_seed: dict[int, list]) -> list[dict]:
    output = []
    for seed, tasks in tasks_by_seed.items():
        rows = [
            row
            for row in read_csv(reference_dir / f"seed_{seed}_task_results.csv")
            if row.get("policy") == "hybrid_full"
        ]
        by_key = {(int(row["task_seed"]), int(row["task_id"])): row for row in rows}
        for task in tasks:
            source = by_key[(task.seed, task.task_id)]
            output.append(
                {
                    "benchmark_seed": seed,
                    "method": "pf_teacher",
                    "task_seed": task.seed,
                    "task_id": task.task_id,
                    "true_success": int(source["true_success"]),
                }
            )
    return output


def load_matched_neural_rows(results_dir: Path) -> list[dict]:
    if not results_dir.is_dir():
        return []
    output = []
    for seed in BENCHMARK_SEEDS:
        path = results_dir / f"seed_{seed}_task_results.csv"
        if not path.is_file():
            return []
        for row in read_csv(path):
            if row.get("method") in {"imitation", "ppo"}:
                output.append(
                    {
                        "benchmark_seed": seed,
                        "method": row["method"],
                        "task_seed": int(row["task_seed"]),
                        "task_id": int(row["task_id"]),
                        "true_success": int(row["true_success"]),
                    }
                )
    return output


def exact_mcnemar(reference: list[int], comparison: list[int]) -> tuple[int, int, float]:
    reference_only = sum(a == 1 and b == 0 for a, b in zip(reference, comparison))
    comparison_only = sum(a == 0 and b == 1 for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(reference_only, discordant, 0.5).pvalue
    )
    return reference_only, comparison_only, p_value


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value"]))
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(order) - rank) * float(rows[index]["p_value"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def paired_tests(rows: list[dict]) -> list[dict]:
    output = []
    scopes: list[int | str] = list(BENCHMARK_SEEDS) + ["pooled"]
    for scope in scopes:
        scoped = rows if scope == "pooled" else [row for row in rows if row["benchmark_seed"] == scope]
        methods = sorted({row["method"] for row in scoped})
        lookup = {
            method: {
                (int(row["task_seed"]), int(row["task_id"])): int(row["true_success"])
                for row in scoped
                if row["method"] == method
            }
            for method in methods
        }
        scope_rows = []
        for reference in methods:
            for comparison in methods:
                if comparison <= reference:
                    continue
                keys = sorted(set(lookup[reference]) & set(lookup[comparison]))
                if not keys:
                    continue
                ref = [lookup[reference][key] for key in keys]
                cmp = [lookup[comparison][key] for key in keys]
                ref_only, cmp_only, p_value = exact_mcnemar(ref, cmp)
                scope_rows.append(
                    {
                        "scope": scope,
                        "comparison": f"{comparison}_minus_{reference}",
                        "paired_tasks": len(keys),
                        "reference_only_success": ref_only,
                        "comparison_only_success": cmp_only,
                        "success_difference_pp": 100.0 * (float(np.mean(cmp)) - float(np.mean(ref))),
                        "p_value": p_value,
                    }
                )
        holm(scope_rows)
        output.extend(scope_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Current matched PID and simple-rule baselines")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selected-pid-json", type=Path, default=None)
    parser.add_argument("--matched-results-dir", type=Path, default=None)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    output = (args.output_dir or package / "results_missing_baselines").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selected_pid = None
    methods = list(METHODS)
    if args.selected_pid_json is not None:
        payload = json.loads(args.selected_pid_json.read_text(encoding="utf-8"))
        selected_pid = payload.get("parameters", payload)
        methods.append("selected_pid")

    tasks_by_seed = {
        seed: load_tasks(package / "tasks" / f"seed_{seed}_tasks.jsonl")[: args.tasks_per_seed]
        for seed in BENCHMARK_SEEDS
    }
    jobs = [
        (method, seed, task, selected_pid)
        for seed, tasks in tasks_by_seed.items()
        for task in tasks
        for method in methods
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(run_job, jobs, chunksize=20))
    else:
        rows = [run_job(job) for job in jobs]

    per_seed = [summarize(rows, seed, method) for seed in BENCHMARK_SEEDS for method in methods]
    aggregate_rows = aggregate(per_seed)
    comparison_rows = [
        {
            "benchmark_seed": row["benchmark_seed"],
            "method": row["method"],
            "task_seed": row["task_seed"],
            "task_id": row["task_id"],
            "true_success": row["true_success"],
        }
        for row in rows
    ]
    comparison_rows.extend(load_pf_rows(package / "pf_reference", tasks_by_seed))
    matched_dir = args.matched_results_dir or package / "results_main_matched"
    comparison_rows.extend(load_matched_neural_rows(matched_dir.resolve()))
    tests = paired_tests(comparison_rows)

    write_csv(output / "all_baseline_task_results.csv", rows)
    write_csv(output / "per_seed_summary.csv", per_seed)
    write_csv(output / "aggregate_summary.csv", aggregate_rows)
    write_csv(output / "paired_success_tests.csv", tests)
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "benchmark_seeds": list(BENCHMARK_SEEDS),
                "tasks_per_seed": args.tasks_per_seed,
                "methods": methods,
                "workers": args.workers,
                "prespecified_pid": PRESPECIFIED_PID.__dict__,
                "selected_pid": selected_pid,
                "neural_results_included": bool(load_matched_neural_rows(matched_dir.resolve())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
