from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from posterior_diagnostics import initialize_controller, snapshot_metrics, write_csv
from task_distribution import load_tasks


PROTOCOL_VERSION = 1
DEFAULT_SEEDS = (101, 202, 303, 404, 555)
DEFAULT_PRIOR_K = (1, 2, 3)
DEFAULT_CHECKPOINTS = tuple(range(13))


def prior_probabilities(prior_k: int, strength: float) -> np.ndarray:
    if prior_k not in (1, 2, 3):
        raise ValueError("prior_k must be 1, 2, or 3")
    if not 1.0 / 3.0 < strength < 1.0:
        raise ValueError("prior strength must be between 1/3 and 1")
    remainder = (1.0 - strength) / 2.0
    probabilities = np.full(3, remainder, dtype=float)
    probabilities[prior_k - 1] = strength
    return probabilities


def apply_k_prior(controller, prior_k: int, strength: float) -> np.ndarray:
    probabilities = prior_probabilities(prior_k, strength)
    inference = controller.inference
    if not hasattr(inference, "model_log_probabilities"):
        raise TypeError("The selected controller is not a variable-K particle filter")
    inference.model_log_probabilities = np.log(probabilities)
    controller._refresh_summary()
    return probabilities


def enriched_snapshot(
    task,
    controller,
    benchmark_seed: int,
    observations: int,
    prior_k: int,
    probabilities: np.ndarray,
) -> dict:
    row = snapshot_metrics(
        task,
        controller,
        benchmark_seed,
        "fixed_observation_count",
        observations,
    )
    estimate = controller.posterior_estimate()
    return {
        "prior_k": prior_k,
        "prior_probability": float(probabilities[prior_k - 1]),
        "prior_probabilities_json": json.dumps(probabilities.tolist()),
        "estimated_concentration_sd_m": float(estimate.concentration_sd_m),
        **row,
    }


def run_task(payload) -> list[dict]:
    task, particles, benchmark_seed, common_seed, checkpoints, prior_k, strength = payload
    controller = initialize_controller(task, particles, common_seed)
    probabilities = apply_k_prior(controller, prior_k, strength)
    rows = [
        enriched_snapshot(
            task,
            controller,
            benchmark_seed,
            0,
            prior_k,
            probabilities,
        )
    ]
    fixed_horizon = max(checkpoints)
    while int(controller.steps_taken) < fixed_horizon:
        controller.use_secondary_reagents = False
        action, _ = controller.select_best_action()
        measured_ph, _, done, _ = controller.step(action, mode="Simulate")
        controller.update_posteriors(action, measured_ph)
        observations = int(controller.steps_taken)
        if observations in checkpoints:
            rows.append(
                enriched_snapshot(
                    task,
                    controller,
                    benchmark_seed,
                    observations,
                    prior_k,
                    probabilities,
                )
            )
        if done and observations < fixed_horizon:
            controller.done = False
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired PF posterior diagnostics under K=1, K=2, and K=3 biased priors"
    )
    parser.add_argument("--source-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--prior-k", nargs="+", type=int, default=list(DEFAULT_PRIOR_K))
    parser.add_argument("--prior-strength", type=float, default=0.80)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(set(args.checkpoints))
    if checkpoints != list(DEFAULT_CHECKPOINTS):
        raise ValueError(f"This protocol requires checkpoints {DEFAULT_CHECKPOINTS}")
    prior_values = sorted(set(args.prior_k))
    if any(value not in (1, 2, 3) for value in prior_values):
        raise ValueError("Prior K values must be drawn from 1, 2, and 3")
    workers = args.workers or min(12, max(1, (os.cpu_count() or 2) - 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_counts = {}
    for seed in args.seeds:
        task_path = args.source_task_dir / f"seed_{seed}_tasks.jsonl"
        if not task_path.is_file():
            raise FileNotFoundError(task_path)
        task_counts[str(seed)] = len(load_tasks(task_path))
    if len(set(task_counts.values())) != 1:
        raise RuntimeError(f"Unbalanced source task counts: {task_counts}")

    config = {
        "protocol_version": PROTOCOL_VERSION,
        "source_task_dir": str(args.source_task_dir.resolve()),
        "seeds": args.seeds,
        "tasks_per_seed": next(iter(task_counts.values())),
        "prior_k": prior_values,
        "prior_strength": args.prior_strength,
        "nonpreferred_prior_probability": (1.0 - args.prior_strength) / 2.0,
        "particles": args.particles,
        "checkpoints": checkpoints,
        "workers": workers,
        "common_random_filter_particles_across_priors": True,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("Existing output uses a different configuration")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is non-empty and lacks RUN_CONFIG.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows = []
    completed = []
    for prior_k in prior_values:
        for benchmark_seed in args.seeds:
            shard_name = f"prior_k{prior_k}_seed_{benchmark_seed}"
            shard_path = args.output_dir / f"{shard_name}_posterior_rows.csv"
            marker_path = args.output_dir / f"{shard_name}_COMPLETE.json"
            if args.resume and shard_path.is_file() and marker_path.is_file():
                rows = read_csv(shard_path)
                print(f"{shard_name} already complete", flush=True)
            else:
                tasks = load_tasks(
                    args.source_task_dir / f"seed_{benchmark_seed}_tasks.jsonl"
                )
                payloads = [
                    (
                        task,
                        args.particles,
                        benchmark_seed,
                        benchmark_seed * 30_000_049 + task.task_id * 1013,
                        checkpoints,
                        prior_k,
                        args.prior_strength,
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
                            print(f"{shard_name}: {index}/{len(tasks)} tasks", flush=True)
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=False)
                write_csv(shard_path, rows)
                marker_path.write_text(
                    json.dumps(
                        {
                            "prior_k": prior_k,
                            "benchmark_seed": benchmark_seed,
                            "tasks": len(tasks),
                            "rows": len(rows),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            all_rows.extend(rows)
            completed.append(shard_name)

    write_csv(args.output_dir / "all_prior_k_posterior_rows.csv", all_rows)
    expected_tasks = len(prior_values) * len(args.seeds) * next(iter(task_counts.values()))
    expected_rows = expected_tasks * len(checkpoints)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(all_rows)}")
    completion = {
        "status": "complete",
        "config": config,
        "completed_shards": completed,
        "task_prior_combinations": expected_tasks,
        "rows": len(all_rows),
    }
    (args.output_dir / "PRIOR_K_GRID_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
