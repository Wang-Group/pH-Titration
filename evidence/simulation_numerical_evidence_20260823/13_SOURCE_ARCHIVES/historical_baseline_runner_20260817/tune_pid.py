from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baseline_controllers import PRESPECIFIED_PID, FixedGainPIDController, PIDConfig, rollout_baseline
from task_distribution import generate_tasks, save_tasks


PARAMETER_RANGES = {
    "kp": (0.05, 0.80, "log"),
    "ki": (0.001, 0.080, "log"),
    "kd": (0.005, 0.200, "log"),
    "integral_limit": (5.0, 30.0, "linear"),
    "overshoot_decay": (0.0, 0.9, "linear"),
}


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


def latin_hypercube_candidates(count: int, seed: int) -> list[dict[str, float]]:
    if count < 1:
        return []
    rng = np.random.default_rng(seed)
    names = list(PARAMETER_RANGES)
    unit = np.empty((count, len(names)), dtype=float)
    for column in range(len(names)):
        unit[:, column] = (rng.permutation(count) + rng.random(count)) / count
    candidates = []
    for row in unit:
        values: dict[str, float] = {}
        for name, fraction in zip(names, row):
            low, high, scale = PARAMETER_RANGES[name]
            if scale == "log":
                values[name] = float(np.exp(np.log(low) + fraction * (np.log(high) - np.log(low))))
            else:
                values[name] = float(low + fraction * (high - low))
        values["minimum_output_ml"] = 0.01
        values["maximum_output_ml"] = 3.00
        candidates.append(values)
    return candidates


def evaluate_candidate(payload: tuple[int, dict[str, float], list]) -> tuple[dict, list[dict]]:
    candidate_id, parameters, tasks = payload
    controller = FixedGainPIDController(PIDConfig.from_mapping(parameters))
    rows = []
    for task in tasks:
        metrics = rollout_baseline(task, controller)
        rows.append(
            {
                "candidate_id": candidate_id,
                "task_seed": task.seed,
                "task_id": task.task_id,
                "true_success": metrics["true_success"],
                "steps": metrics["steps"],
                "overshoots": metrics["overshoots"],
                "final_abs_error": metrics["final_abs_error"],
                "total_volume_ml": metrics["total_volume_ml"],
            }
        )
    successful = [row for row in rows if row["true_success"]]
    summary = {
        "candidate_id": candidate_id,
        **parameters,
        "validation_tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean([row["true_success"] for row in rows])),
        "successful_steps_mean": (
            float(np.mean([row["steps"] for row in successful])) if successful else math.inf
        ),
        "overshoots_mean": float(np.mean([row["overshoots"] for row in rows])),
        "final_abs_error_mean": float(np.mean([row["final_abs_error"] for row in rows])),
        "total_volume_mean_ml": float(np.mean([row["total_volume_ml"] for row in rows])),
    }
    return summary, rows


def ranking_key(row: dict) -> tuple:
    return (
        -float(row["success_rate_percent"]),
        float(row["successful_steps_mean"]),
        float(row["overshoots_mean"]),
        float(row["final_abs_error_mean"]),
        int(row["candidate_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a fixed-gain PID on an independent validation set")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_pid_validation")
    parser.add_argument("--validation-seed", type=int, default=7_100_001)
    parser.add_argument("--validation-tasks", type=int, default=500)
    parser.add_argument("--lhs-candidates", type=int, default=120)
    parser.add_argument("--candidate-seed", type=int, default=7_100_002)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(
        args.validation_seed,
        args.validation_tasks,
        "pid_independent_validation",
    )
    save_tasks(output / "validation_tasks.jsonl", tasks)

    candidates = [PRESPECIFIED_PID.__dict__.copy()]
    candidates.extend(latin_hypercube_candidates(args.lhs_candidates, args.candidate_seed))
    jobs = [(index, parameters, tasks) for index, parameters in enumerate(candidates)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(evaluate_candidate, jobs, chunksize=1))
    else:
        evaluated = [evaluate_candidate(job) for job in jobs]

    summaries = [item[0] for item in evaluated]
    task_rows = [row for item in evaluated for row in item[1]]
    ranked = sorted(summaries, key=ranking_key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    selected = ranked[0]
    parameter_names = set(PIDConfig.__dataclass_fields__)
    selected_parameters = {name: float(selected[name]) for name in parameter_names}
    selected_payload = {
        "selection_protocol": "independent_validation_only",
        "ranking_order": [
            "success_rate_percent descending",
            "successful_steps_mean ascending",
            "overshoots_mean ascending",
            "final_abs_error_mean ascending",
            "candidate_id ascending",
        ],
        "validation_seed": args.validation_seed,
        "validation_tasks": args.validation_tasks,
        "candidate_seed": args.candidate_seed,
        "lhs_candidates": args.lhs_candidates,
        "prespecified_candidate_included": True,
        "selected_candidate_id": int(selected["candidate_id"]),
        "parameters": selected_parameters,
        "validation_metrics": {
            key: selected[key]
            for key in (
                "success_rate_percent",
                "successful_steps_mean",
                "overshoots_mean",
                "final_abs_error_mean",
                "total_volume_mean_ml",
            )
        },
        "locked_benchmark_used_for_selection": False,
    }
    write_csv(output / "candidate_summary.csv", ranked)
    write_csv(output / "all_candidate_task_results.csv", task_rows)
    (output / "selected_pid.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "validation_seed": args.validation_seed,
                "validation_tasks": args.validation_tasks,
                "lhs_candidates": args.lhs_candidates,
                "candidate_seed": args.candidate_seed,
                "workers": args.workers,
                "parameter_ranges": PARAMETER_RANGES,
                "fixed_output_bounds_ml": [0.01, 3.00],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(selected_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
