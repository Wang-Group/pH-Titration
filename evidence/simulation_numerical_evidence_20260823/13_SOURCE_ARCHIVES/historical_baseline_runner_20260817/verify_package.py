from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from run_matched_evaluation import EXPECTED_CHECKPOINT_SHA256, load_pf_reference
from run_unified_timing import load_actor, run_episode
from task_distribution import load_tasks


BENCHMARK_SEEDS = (101, 202, 303, 404, 555)
REQUIRED_ROOT_FILES = (
    "README_CN.md",
    "requirements.txt",
    "baseline_controllers.py",
    "tune_pid.py",
    "run_missing_baselines.py",
    "run_unified_timing.py",
    "run_matched_evaluation.py",
    "chemistry_model.py",
    "control_environment.py",
    "task_distribution.py",
    "models.py",
    "controller_api.py",
    "particle_inference.py",
    "new_pf_controller.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(package: Path) -> int:
    manifest = package / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = package / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def count_pf_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(row.get("policy") == "hybrid_full" for row in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the missing-evidence reproduction package")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    package = args.package_dir.resolve()

    for relative in REQUIRED_ROOT_FILES:
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(path)

    task_counts = {}
    pf_counts = {}
    tasks_by_seed = {}
    for seed in BENCHMARK_SEEDS:
        task_path = package / "tasks" / f"seed_{seed}_tasks.jsonl"
        pf_path = package / "pf_reference" / f"seed_{seed}_task_results.csv"
        tasks = load_tasks(task_path)
        if len(tasks) != 3000:
            raise RuntimeError(f"Expected 3000 tasks in {task_path}, found {len(tasks)}")
        pf_count = count_pf_rows(pf_path)
        if pf_count != 3000:
            raise RuntimeError(f"Expected 3000 hybrid_full PF rows in {pf_path}, found {pf_count}")
        load_pf_reference(pf_path, tasks[:2], seed)
        task_counts[str(seed)] = len(tasks)
        pf_counts[str(seed)] = pf_count
        tasks_by_seed[seed] = tasks

    device = torch.device("cpu")
    checkpoints = {
        "imitation": package / "models" / "imitation_best.pth",
        "ppo": package / "models" / "ppo_seed_303.pth",
    }
    actors = {}
    checkpoint_hashes = {}
    for method, path in checkpoints.items():
        actual = sha256(path)
        if actual != EXPECTED_CHECKPOINT_SHA256[method]:
            raise RuntimeError(f"Unexpected {method} checkpoint hash: {actual}")
        checkpoint_hashes[method] = actual
        actors[method] = load_actor(path, device)

    smoke_task = tasks_by_seed[101][0]
    smoke = {
        method: run_episode(method, smoke_task, package, device, actors, None)
        for method in ("prespecified_pid", "simple_rule", "imitation", "ppo", "pf_1000")
    }
    manifest_files = 0 if args.skip_checksums else verify_manifest(package)
    report = {
        "status": "PASS",
        "package": str(package),
        "task_counts": task_counts,
        "pf_hybrid_full_counts": pf_counts,
        "checkpoint_sha256": checkpoint_hashes,
        "manifest_files_checked": manifest_files,
        "smoke_task": {"seed": smoke_task.seed, "task_id": smoke_task.task_id},
        "smoke_methods": {
            method: {"steps": row["steps"], "completed": True}
            for method, row in smoke.items()
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
