from __future__ import annotations

import argparse
import csv
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_core import NeuralVolumePolicy, generate_tasks
from challenge_common import SCENARIOS, load_bayesian_module
from evaluate_candidates import (
    LoadedCandidate,
    run_candidate,
    run_common_bayesian,
    run_original_bayesian,
    run_reference,
)


TRAIN_SEEDS = [101, 202, 303, 404, 555]
EVAL_SEEDS = [7101, 7202, 7303, 7404, 7555]

ALL_CANDIDATES = [
    "ppo_nominal",
    "ppo_robust",
    "a2c_robust",
    "ppo_history_robust",
    "sac_history_robust",
    "ppo_residual_robust",
    "ppo_filtered_robust",
    "ppo_conservative_robust",
    "td3_filtered_robust",
]

METHOD_SETS = {
    "all": [
        "bayesian_common",
        "imitation",
        "submitted_rl",
        "ppo_reference",
        *ALL_CANDIDATES,
    ],
    "core": [
        "bayesian_common",
        "imitation",
        "submitted_rl",
        "ppo_reference",
        "ppo_history_robust",
        "ppo_residual_robust",
        "ppo_filtered_robust",
        "sac_history_robust",
        "td3_filtered_robust",
    ],
    "winner": [
        "bayesian_common",
        "imitation",
        "submitted_rl",
        "ppo_reference",
        "sac_history_robust",
        "td3_filtered_robust",
    ],
}


def stable_scenario_offset(name: str) -> int:
    return int(zlib.crc32(name.encode("utf-8")) % 1_000_000)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        raise RuntimeError(f"No rows were produced for {path.name}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def expected_methods(methods: list[str], scenario: str, include_native_bayesian: bool) -> list[str]:
    output = []
    for method in methods:
        output.append("ppo_residual_imitation" if method == "ppo_residual_robust" else method)
    if scenario == "nominal" and include_native_bayesian:
        output.append("bayesian_original")
    return sorted(set(output))


def shard_is_complete(path: Path, task_count: int, methods: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    expected_rows = task_count * len(methods)
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle) - 1
    return row_count == expected_rows


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    train_seed = int(job["train_seed"])
    eval_seed = int(job["eval_seed"])
    scenario_name = str(job["scenario"])
    task_count = int(job["task_count"])
    task_seed = int(job["task_seed"])
    methods = list(job["methods"])
    device = torch.device(str(job["device"]))
    output_path = Path(job["output_path"])

    candidate_names = [name for name in ALL_CANDIDATES if name in methods]
    candidates = {
        name: LoadedCandidate(Path(job["candidate_dir"]) / f"{name}_seed{train_seed}.pth", device)
        for name in candidate_names
    }

    needs_imitation = "imitation" in methods or bool(candidate_names)
    imitation = NeuralVolumePolicy(Path(job["imitation_weights"]), str(device)) if needs_imitation else None
    submitted = (
        NeuralVolumePolicy(Path(job["submitted_rl_weights"]), str(device))
        if "submitted_rl" in methods
        else None
    )
    ppo_reference = (
        NeuralVolumePolicy(Path(job["ppo_reference_dir"]) / f"ppo_full_seed{train_seed}.pth", str(device))
        if "ppo_reference" in methods
        else None
    )
    module = (
        load_bayesian_module(Path(job["bayesian_source"]))
        if "bayesian_common" in methods or (scenario_name == "nominal" and job["include_native_bayesian"])
        else None
    )

    scenario = SCENARIOS[scenario_name]
    scenario_task_seed = task_seed + stable_scenario_offset(scenario_name)
    tasks = generate_tasks(scenario_task_seed, task_count, scenario)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        rng_seed = (
            eval_seed * 1_000_003
            + stable_scenario_offset(scenario_name) * 97
            + int(task.task_id)
        )
        produced: list[dict[str, Any]] = []
        if "bayesian_common" in methods:
            produced.append(
                run_common_bayesian(
                    module,
                    task,
                    scenario,
                    int(job["particles"]),
                    rng_seed,
                    train_seed,
                    eval_seed,
                )
            )
        if scenario_name == "nominal" and bool(job["include_native_bayesian"]):
            produced.append(
                run_original_bayesian(
                    module,
                    task,
                    int(job["particles"]),
                    rng_seed,
                    train_seed,
                    eval_seed,
                )
            )
        if "imitation" in methods:
            produced.append(run_reference(imitation, task, scenario, "imitation", rng_seed, train_seed, eval_seed))
        if "submitted_rl" in methods:
            produced.append(run_reference(submitted, task, scenario, "submitted_rl", rng_seed, train_seed, eval_seed))
        if "ppo_reference" in methods:
            produced.append(run_reference(ppo_reference, task, scenario, "ppo_reference", rng_seed, train_seed, eval_seed))
        for name, candidate in candidates.items():
            method = "ppo_residual_imitation" if candidate.residual else name
            produced.append(run_candidate(candidate, imitation, task, scenario, method, rng_seed, train_seed, eval_seed))

        for row in produced:
            row["design"] = str(job["design"])
            row["fixed_task_seed"] = scenario_task_seed
            row["replicate_id"] = f"train{train_seed}_eval{eval_seed}"
            rows.append(row)
        if index % 250 == 0 or index == task_count:
            print(
                f"[{scenario_name}] train={train_seed}, eval={eval_seed}: "
                f"{index}/{task_count} tasks",
                flush=True,
            )

    write_csv_atomic(output_path, rows)
    return {
        "scenario": scenario_name,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "rows": len(rows),
        "path": str(output_path),
    }


def merge_shards(shard_paths: list[Path], output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    with temporary.open("w", encoding="utf-8", newline="") as target:
        for shard in sorted(shard_paths):
            with shard.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    os.replace(temporary, output_path)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fixed-3000 paired and crossed-seed confirmatory evaluation."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", choices=["paired", "cross"], default="paired")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--task-count", type=int, default=3000)
    parser.add_argument("--task-seed", type=int, default=20260724)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=EVAL_SEEDS)
    parser.add_argument("--method-set", choices=sorted(METHOD_SETS), default="all")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--include-native-bayesian", action="store_true")
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-dir", type=Path, default=base / "candidate_models")
    parser.add_argument("--imitation-weights", type=Path, default=base / "models" / "imitation.pth")
    parser.add_argument("--submitted-rl-weights", type=Path, default=base / "models" / "reinforcement.pth")
    parser.add_argument("--ppo-reference-dir", type=Path, default=base / "models" / "ppo_reference")
    parser.add_argument("--bayesian-source", type=Path, default=base / "inputs" / "bayesian_controller.py")
    parser.add_argument("--analysis-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task_count <= 0:
        raise SystemExit("--task-count must be positive")
    if args.design == "paired" and len(args.train_seeds) != len(args.eval_seeds):
        raise SystemExit("Paired design requires equal numbers of train and evaluation seeds.")
    if args.device != "cpu" and args.workers > 1:
        raise SystemExit("Multiple workers are supported only for CPU evaluation.")

    methods = list(args.methods or METHOD_SETS[args.method_set])
    unknown = sorted(set(methods) - set(METHOD_SETS["all"]))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    seed_pairs = (
        list(zip(args.train_seeds, args.eval_seeds))
        if args.design == "paired"
        else [(train, evaluate) for train in args.train_seeds for evaluate in args.eval_seeds]
    )

    settings = {
        "design": args.design,
        "scenarios": args.scenarios,
        "task_count": args.task_count,
        "task_seed": args.task_seed,
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "seed_pairs": [[int(train), int(evaluate)] for train, evaluate in seed_pairs],
        "methods_requested": methods,
        "include_native_bayesian": args.include_native_bayesian,
        "particles": args.particles,
        "bootstrap_iterations": args.bootstrap_iterations,
        "workers": args.workers,
        "device": args.device,
    }
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        scientific_keys = [
            "design",
            "scenarios",
            "task_count",
            "task_seed",
            "train_seeds",
            "eval_seeds",
            "seed_pairs",
            "methods_requested",
            "include_native_bayesian",
            "particles",
            "device",
        ]
        changed = [key for key in scientific_keys if existing.get(key) != settings.get(key)]
        if changed:
            raise SystemExit(
                "The output directory contains a different experimental design "
                f"({', '.join(changed)} changed). Use a new output directory."
            )
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    jobs: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    for scenario in args.scenarios:
        actual_methods = expected_methods(methods, scenario, args.include_native_bayesian)
        for train_seed, eval_seed in seed_pairs:
            shard = shard_dir / f"{scenario}__train{train_seed}__eval{eval_seed}.csv"
            shard_paths.append(shard)
            if shard_is_complete(shard, args.task_count, actual_methods):
                print(f"SKIP complete shard: {shard.name}")
                continue
            jobs.append(
                {
                    "design": args.design,
                    "scenario": scenario,
                    "task_count": args.task_count,
                    "task_seed": args.task_seed,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "methods": methods,
                    "include_native_bayesian": args.include_native_bayesian,
                    "particles": args.particles,
                    "device": args.device,
                    "candidate_dir": str(args.candidate_dir.resolve()),
                    "imitation_weights": str(args.imitation_weights.resolve()),
                    "submitted_rl_weights": str(args.submitted_rl_weights.resolve()),
                    "ppo_reference_dir": str(args.ppo_reference_dir.resolve()),
                    "bayesian_source": str(args.bayesian_source.resolve()),
                    "output_path": str(shard),
                }
            )

    if not args.analysis_only and jobs:
        workers = max(1, min(args.workers, len(jobs)))
        print(f"Running {len(jobs)} shards with {workers} worker(s).")
        if workers == 1:
            for job in jobs:
                print(run_job(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_job, job) for job in jobs]
                for future in as_completed(futures):
                    print(f"DONE {future.result()}", flush=True)

    incomplete = [path for path in shard_paths if not path.exists()]
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} shard(s) are incomplete. Re-run the same command to resume."
        )

    merge_shards(shard_paths, output_dir / "per_task_results.csv")
    from analyze_fixed3000 import analyze_directory

    analyze_directory(output_dir, int(args.bootstrap_iterations))
    (output_dir / "RUN_COMPLETE.txt").write_text("fixed3000 confirmatory run complete\n", encoding="ascii")
    print(f"Complete: {output_dir}")


if __name__ == "__main__":
    main()
