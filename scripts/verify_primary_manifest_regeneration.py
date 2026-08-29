"""Regenerate and hash-check the five locked primary benchmark manifests.

This is intentionally separate from the fast package contract check because
the chemical-equilibrium oracle makes generation of 15,000 tasks take several
minutes on a CPU.  It verifies that the archived generator reproduces the
authoritative JSONL inputs byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path


SEEDS = (101, 202, 303, 404, 555)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence" / "simulation_numerical_evidence_20260823"
    formal = evidence / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
    source = evidence / "13_SOURCE_ARCHIVES" / "primary_locked_benchmark_source"
    parser.add_argument("--output", type=Path, default=formal / "PRIMARY_MANIFEST_REGENERATION_AUDIT.json")
    args = parser.parse_args()

    sys.path.insert(0, str(source))
    task_distribution = importlib.import_module("task_distribution")
    checks = []
    with tempfile.TemporaryDirectory(prefix="primary_manifest_regen_") as temporary:
        temporary_root = Path(temporary)
        for benchmark_seed in SEEDS:
            expected_path = formal / "tasks" / f"seed_{benchmark_seed}_tasks.jsonl"
            regenerated_path = temporary_root / f"seed_{benchmark_seed}_tasks.jsonl"
            tasks = task_distribution.generate_tasks(
                1_000_000 + benchmark_seed,
                3000,
                f"rule_ablation_seed_{benchmark_seed}",
            )
            task_distribution.save_tasks(regenerated_path, tasks)
            expected_hash = sha256(expected_path)
            regenerated_hash = sha256(regenerated_path)
            checks.append(
                {
                    "benchmark_seed": benchmark_seed,
                    "tasks": len(tasks),
                    "expected_sha256": expected_hash,
                    "regenerated_sha256": regenerated_hash,
                    "match": expected_hash == regenerated_hash,
                }
            )

    report = {
        "status": "PASS" if all(item["match"] for item in checks) else "FAIL",
        "generator": "13_SOURCE_ARCHIVES/primary_locked_benchmark_source/task_distribution.py",
        "protocol": "seed=1,000,000+benchmark_seed; 3,000 tasks; rule_ablation_seed_<seed>",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
