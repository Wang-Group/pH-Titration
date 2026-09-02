from __future__ import annotations

"""Evaluate PF-distilled imitation/PPO policies on three locked benchmarks.

Each policy is evaluated deterministically on the same five 3,000-task manifests
for each truth domain.  The runner writes task-level gzip CSV shards, summaries
at benchmark-seed and policy level, and a family-level PPO robustness summary.
Existing shards are reused so interrupted runs can be resumed safely.
"""

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


BENCHMARK_SEEDS = (101, 202, 303, 404, 555)
TRAINING_SEEDS = (101, 202, 303, 404, 555)
FAMILIES = ("F2", "F3", "F4", "F5", "F6")
DOMAINS = ("sequential_k123", "fixed_two_independent", "independent_j123")
TASKS_PER_SEED = 3000
EVALUATION_SEED_OFFSET = 30_000_001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def task_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise RuntimeError(f"{path}: expected {limit} tasks, found {len(rows)}")
    return rows


def configure_imports(formal_source: Path, staging_root: Path) -> None:
    for path in (formal_source.resolve(), staging_root.resolve()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def checkpoint_actor(path: Path, formal_source: Path, staging_root: Path):
    configure_imports(formal_source, staging_root)
    import torch
    from models import StateNormalizer, VolumeActor

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = VolumeActor().cpu()
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    normalizer = StateNormalizer(
        np.asarray(payload["state_mean"], dtype=np.float32),
        np.asarray(payload["state_std"], dtype=np.float32),
    )
    return actor, normalizer, payload.get("metadata", {})


def rollout(actor, normalizer, task: dict[str, Any], domain: str) -> dict[str, Any]:
    import torch
    from policy_family_environment import GenericControlEnvironment, actor_volume

    seed = (
        EVALUATION_SEED_OFFSET
        + int(task["seed"]) * 1_000_003
        + int(task["task_id"])
    ) % (2**32 - 1)
    env = GenericControlEnvironment(
        SimpleNamespace(**task), np.random.default_rng(seed), domain=None
    )
    actor.eval()
    while not env.done:
        env.step(actor_volume(actor, normalizer, env.state(), torch.device("cpu")))
    true_count = int(task.get("component_count", len(task["pka_values"])))
    return {
        "task_seed": int(task["seed"]),
        "task_id": int(task["task_id"]),
        "acid_type": task["acid_type"],
        "difficulty": task["difficulty"],
        "direction": task["direction"],
        "pka_family": task["pka_family"],
        "true_component_count": true_count,
        "initial_ph": float(task["initial_ph"]),
        "target_ph": float(task["target_ph"]),
        **env.metrics(),
    }


def atomic_write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(job["checkpoint"])
    actor, normalizer, metadata = checkpoint_actor(
        checkpoint, Path(job["formal_source"]), Path(job["staging_root"])
    )
    tasks = task_rows(Path(job["manifest"]), int(job["tasks_per_seed"]))
    rows = [rollout(actor, normalizer, task, str(job["domain"])) for task in tasks]
    output = Path(job["output"])
    atomic_write_gzip_csv(output, rows)
    return {
        "family": job["family"],
        "policy": job["policy"],
        "training_seed": job["training_seed"],
        "domain": job["domain"],
        "benchmark_seed": job["benchmark_seed"],
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean([r["true_success"] for r in rows])),
        "checkpoint_source": metadata.get("source", "imitation"),
    }


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize_task_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    successful = [row for row in rows if int(row["true_success"])]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * mean(rows, "true_success"),
        "strict_success_rate_percent": 100.0 * mean(rows, "strict_success"),
        "severe_failure_rate_percent": 100.0 * mean(rows, "severe_failure"),
        "false_stop_rate_percent": 100.0 * mean(rows, "false_stop"),
        "successful_additions_mean": mean(successful, "steps") if successful else math.nan,
        "additions_per_task_mean": mean(rows, "steps"),
        "overshoots_per_task_mean": mean(rows, "overshoots"),
        "total_volume_mean_ml": mean(rows, "total_volume_ml"),
        "final_abs_error_mean": mean(rows, "final_abs_error"),
    }


def sample_stats(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate(output: Path, jobs: list[dict[str, Any]]) -> None:
    cells: list[dict[str, Any]] = []
    for job in jobs:
        rows = read_gzip_csv(Path(job["output"]))
        if len(rows) != int(job["tasks_per_seed"]):
            raise RuntimeError(f"Incomplete shard: {job['output']} ({len(rows)} rows)")
        cells.append(
            {
                "family": job["family"],
                "policy": job["policy"],
                "training_seed": job["training_seed"],
                "domain": job["domain"],
                "benchmark_seed": job["benchmark_seed"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                **summarize_task_rows(rows),
            }
        )
    cells.sort(key=lambda row: (row["domain"], row["family"], row["policy"], row["benchmark_seed"]))
    atomic_write_csv(output / "per_evaluation_cell_summary.csv", cells)

    metric_fields = (
        "success_rate_percent",
        "strict_success_rate_percent",
        "severe_failure_rate_percent",
        "false_stop_rate_percent",
        "successful_additions_mean",
        "additions_per_task_mean",
        "overshoots_per_task_mean",
        "total_volume_mean_ml",
        "final_abs_error_mean",
    )
    policy_rows: list[dict[str, Any]] = []
    policy_keys = sorted({(r["domain"], r["family"], r["policy"], r["training_seed"]) for r in cells})
    for domain, family, policy, training_seed in policy_keys:
        selected = [
            r for r in cells
            if r["domain"] == domain and r["family"] == family
            and r["policy"] == policy and r["training_seed"] == training_seed
        ]
        entry: dict[str, Any] = {
            "domain": domain,
            "family": family,
            "policy": policy,
            "training_seed": training_seed,
            "benchmark_sets": len(selected),
            "tasks_total": sum(int(r["tasks"]) for r in selected),
        }
        for field in metric_fields:
            entry[field + "_mean"], entry[field + "_sample_sd"] = sample_stats(
                float(r[field]) for r in selected
            )
        policy_rows.append(entry)
    atomic_write_csv(output / "per_policy_summary.csv", policy_rows)

    family_rows: list[dict[str, Any]] = []
    for domain in sorted({str(r["domain"]) for r in policy_rows}):
        for family in sorted({str(r["family"]) for r in policy_rows if r["domain"] == domain}):
            imitation = next(
                r for r in policy_rows
                if r["domain"] == domain and r["family"] == family and r["policy"] == "imitation"
            )
            ppo = [
                r for r in policy_rows
                if r["domain"] == domain and r["family"] == family and r["policy"].startswith("ppo_seed_")
            ]
            entry: dict[str, Any] = {
                "domain": domain,
                "family": family,
                "imitation_success_percent_mean": imitation["success_rate_percent_mean"],
                "imitation_success_percent_benchmark_sd": imitation["success_rate_percent_sample_sd"],
            }
            for field in metric_fields:
                values = [float(r[field + "_mean"]) for r in ppo]
                entry["ppo_" + field + "_training_seed_mean"], entry[
                    "ppo_" + field + "_training_seed_sample_sd"
                ] = sample_stats(values)
            success_values = [float(r["success_rate_percent_mean"]) for r in ppo]
            entry["ppo_success_training_seed_min"] = min(success_values)
            entry["ppo_success_training_seed_max"] = max(success_values)
            entry["ppo_minus_imitation_success_percentage_points_mean"] = (
                float(entry["ppo_success_rate_percent_training_seed_mean"])
                - float(imitation["success_rate_percent_mean"])
            )
            entry["ppo_training_seeds_above_imitation"] = sum(
                value > float(imitation["success_rate_percent_mean"]) for value in success_values
            )
            family_rows.append(entry)
    atomic_write_csv(output / "family_method_summary.csv", family_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--families-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--tasks-per-seed", type=int, default=TASKS_PER_SEED)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    policies: list[dict[str, Any]] = []
    for family in args.families:
        imitation = args.families_root / family / "imitation" / "imitation_best.pth"
        policies.append({"family": family, "policy": "imitation", "training_seed": "", "checkpoint": imitation})
        for seed in TRAINING_SEEDS:
            checkpoint = (
                args.families_root / family / "ppo_individual" / f"seed_{seed}_run"
                / f"seed_{seed}" / "best_ppo.pth"
            )
            policies.append(
                {"family": family, "policy": f"ppo_seed_{seed}", "training_seed": seed, "checkpoint": checkpoint}
            )
    for policy in policies:
        if not policy["checkpoint"].is_file():
            raise FileNotFoundError(policy["checkpoint"])
        policy["checkpoint_sha256"] = sha256(policy["checkpoint"])

    dataset_dirs = {
        "sequential_k123": args.datasets_root / "sequential_k123" / "tasks",
        "fixed_two_independent": args.datasets_root / "fixed_two_independent" / "tasks",
        "independent_j123": args.datasets_root / "independent_j123" / "tasks",
    }
    manifests: list[dict[str, Any]] = []
    for domain in args.domains:
        for seed in BENCHMARK_SEEDS:
            manifest = dataset_dirs[domain] / f"seed_{seed}_tasks.jsonl"
            count = count_jsonl(manifest)
            if count < args.tasks_per_seed:
                raise RuntimeError(f"{manifest}: only {count} tasks")
            manifests.append(
                {"domain": domain, "benchmark_seed": seed, "manifest": manifest,
                 "tasks_available": count, "sha256": sha256(manifest)}
            )

    jobs: list[dict[str, Any]] = []
    for manifest in manifests:
        for policy in policies:
            output = (
                args.output / "task_results" / manifest["domain"] / policy["family"]
                / policy["policy"] / f"benchmark_seed_{manifest['benchmark_seed']}.csv.gz"
            )
            jobs.append(
                {
                    **policy,
                    **manifest,
                    "formal_source": str(args.formal_source.resolve()),
                    "staging_root": str(args.staging_root.resolve()),
                    "manifest": str(manifest["manifest"].resolve()),
                    "output": str(output.resolve()),
                    "tasks_per_seed": args.tasks_per_seed,
                }
            )

    config = {
        "study": "locked policy family x truth domain factorial",
        "families": args.families,
        "domains": args.domains,
        "benchmark_seeds": list(BENCHMARK_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "tasks_per_seed": args.tasks_per_seed,
        "evaluation_seed_offset": EVALUATION_SEED_OFFSET,
        "action_selection": "deterministic argmax",
        "environment_randomization": "disabled",
        "formal_source": str(args.formal_source.resolve()),
        "source_sha256": {
            name: sha256(args.formal_source / name)
            for name in ("chemistry_model.py", "models.py", "policy_evaluation.py")
        },
        "environment_source": str((args.staging_root / "policy_family_environment.py").resolve()),
        "environment_sha256": sha256(args.staging_root / "policy_family_environment.py"),
        "manifests": [
            {k: (str(v.resolve()) if k == "manifest" else v) for k, v in row.items()}
            for row in manifests
        ],
        "checkpoints": [
            {k: (str(v.resolve()) if k == "checkpoint" else v) for k, v in row.items()}
            for row in policies
        ],
        "workers": args.workers,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    config_path = args.output / "RUN_CONFIG.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        # Runtime worker count is not scientific configuration.
        for payload in (prior, config):
            payload.pop("workers", None)
        if prior != config:
            raise RuntimeError("Existing output configuration differs from requested run")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    pending = [job for job in jobs if not Path(job["output"]).exists()]
    print(f"Policy factorial: {len(pending)}/{len(jobs)} cells pending; workers={args.workers}", flush=True)
    completed = len(jobs) - len(pending)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_jobs = {executor.submit(run_cell, job): job for job in pending}
            for future in concurrent.futures.as_completed(future_jobs):
                report = future.result()
                completed += 1
                print(
                    f"{completed}/{len(jobs)} {report['domain']} {report['family']} "
                    f"{report['policy']} benchmark={report['benchmark_seed']} "
                    f"success={report['success_rate_percent']:.2f}%",
                    flush=True,
                )

    aggregate(args.output, jobs)
    (args.output / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "evaluation_cells": len(jobs),
                "task_policy_outcomes": len(jobs) * args.tasks_per_seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Locked policy factorial complete", flush=True)


if __name__ == "__main__":
    main()
