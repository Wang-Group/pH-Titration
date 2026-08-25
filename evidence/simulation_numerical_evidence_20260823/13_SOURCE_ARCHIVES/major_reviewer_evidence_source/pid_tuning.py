from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark_core import StressScenario, generate_tasks, portable_settings, summarize_results
from multiseed_benchmark import AdaptivePIDController, execute_rule_task, load_module


ORIGINAL = {
    "kp": 0.32,
    "ki": 0.012,
    "kd": 0.08,
    "integral_limit": 12.0,
    "overshoot_decay": 0.10,
}

_PID_MODULE = None
_PID_TRAIN_TASKS = None


def sample_candidate(rng: np.random.Generator) -> dict[str, float]:
    return {
        "kp": float(np.exp(rng.uniform(np.log(0.05), np.log(1.2)))),
        "ki": 0.0 if rng.random() < 0.15 else float(np.exp(rng.uniform(np.log(0.001), np.log(0.08)))),
        "kd": 0.0 if rng.random() < 0.15 else float(np.exp(rng.uniform(np.log(0.005), np.log(0.5)))),
        "integral_limit": float(rng.uniform(4.0, 30.0)),
        "overshoot_decay": float(rng.uniform(0.0, 0.8)),
    }


def evaluate(module, tasks, params: dict[str, float], label: str) -> tuple[dict, list]:
    results = []
    for task in tasks:
        controller = AdaptivePIDController(
            params["kp"],
            params["ki"],
            params["kd"],
            params["integral_limit"],
            params["overshoot_decay"],
        )
        results.append(execute_rule_task(module, task, label, controller))
    return summarize_results(results), results


def initialize_pid_worker(source: str, train_seed: int, train_count: int) -> None:
    global _PID_MODULE, _PID_TRAIN_TASKS
    _PID_MODULE = load_module(Path(source), "major_review_pid_worker")
    _PID_TRAIN_TASKS = generate_tasks(train_seed, train_count, StressScenario("nominal")) if train_count else None


def evaluate_candidate_job(job: tuple[int, dict[str, float]]) -> dict:
    trial, params = job
    if _PID_MODULE is None or _PID_TRAIN_TASKS is None:
        raise RuntimeError("PID worker was not initialized.")
    summary, _ = evaluate(_PID_MODULE, _PID_TRAIN_TASKS, params, "pid")
    return {"trial": trial, **params, **summary}


def evaluate_held_out_job(job: tuple[int, int, str, dict[str, float]]) -> dict:
    seed, task_count, name, params = job
    if _PID_MODULE is None:
        raise RuntimeError("PID worker was not initialized.")
    tasks = generate_tasks(seed, task_count, StressScenario("nominal"))
    summary, _ = evaluate(_PID_MODULE, tasks, params, name)
    return {"seed": seed, "method": name, **params, **summary}


def ranking_key(summary: dict) -> tuple[float, float, float]:
    steps = float(summary["successful_steps_mean"])
    if math.isnan(steps):
        steps = 1e9
    return (
        float(summary["success_rate_percent"]),
        -steps,
        -float(summary["overshoot_rate_percent"]),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    repo = base.parent / "repo_reviewcopy"
    bundled_source = base / "inputs" / "bayesian_controller.py"
    parser = argparse.ArgumentParser(description="Held-out random-search tuning for the adaptive PID baseline.")
    parser.add_argument("--search-seed", type=int, default=20260721)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--train-task-seed", type=int, default=9001)
    parser.add_argument("--train-tasks", type=int, default=500)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--evaluation-tasks-per-seed", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--bayesian-source",
        type=Path,
        default=bundled_source if bundled_source.exists() else repo / "main_code3_modularcopy" / "bayesian_and_data" / "03_evaluate_bayesian_final.py",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "pid_tuning")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_module(args.bayesian_source.resolve(), "major_review_pid_tuning")
    nominal = StressScenario("nominal")
    train_tasks = generate_tasks(args.train_task_seed, args.train_tasks, nominal)
    rng = np.random.default_rng(args.search_seed)
    candidates = [ORIGINAL] + [sample_candidate(rng) for _ in range(args.trials)]
    search_rows: list[dict] = []

    if args.workers > 1:
        jobs = list(enumerate(candidates))
        worker_count = min(args.workers, len(jobs))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_pid_worker,
            initargs=(str(args.bayesian_source.resolve()), args.train_task_seed, args.train_tasks),
        ) as executor:
            for row in executor.map(evaluate_candidate_job, jobs):
                search_rows.append(row)
                print(
                    f"trial {row['trial']}/{len(candidates)-1}: success={row['success_rate_percent']:.2f}, "
                    f"steps={row['successful_steps_mean']:.2f}, overshoot={row['overshoot_rate_percent']:.2f}"
                )
    else:
        for trial, params in enumerate(candidates):
            summary, _ = evaluate(module, train_tasks, params, "pid")
            row = {"trial": trial, **params, **summary}
            search_rows.append(row)
            print(
                f"trial {trial}/{len(candidates)-1}: success={summary['success_rate_percent']:.2f}, "
                f"steps={summary['successful_steps_mean']:.2f}, overshoot={summary['overshoot_rate_percent']:.2f}"
            )

    best_row = max(search_rows, key=ranking_key)
    best = {key: float(best_row[key]) for key in ORIGINAL}
    write_csv(args.output_dir / "search_results.csv", search_rows)
    (args.output_dir / "selected_pid_parameters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    evaluation_rows: list[dict] = []
    evaluation_jobs = [
        (seed, args.evaluation_tasks_per_seed, name, params)
        for seed in args.evaluation_seeds
        for name, params in (("original_pid", ORIGINAL), ("tuned_pid", best))
    ]
    if args.workers > 1:
        worker_count = min(args.workers, len(evaluation_jobs))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_pid_worker,
            initargs=(str(args.bayesian_source.resolve()), args.train_task_seed, 0),
        ) as executor:
            evaluation_rows.extend(executor.map(evaluate_held_out_job, evaluation_jobs))
    else:
        for job in evaluation_jobs:
            seed, task_count, name, params = job
            tasks = generate_tasks(seed, task_count, nominal)
            summary, _ = evaluate(module, tasks, params, name)
            evaluation_rows.append({"seed": seed, "method": name, **params, **summary})
    for row in evaluation_rows:
        print(f"evaluation seed {row['seed']}, {row['method']}: {row['success_rate_percent']:.2f}%")
    write_csv(args.output_dir / "held_out_evaluation.csv", evaluation_rows)
    (args.output_dir / "settings.json").write_text(
        json.dumps(portable_settings(vars(args), base), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
