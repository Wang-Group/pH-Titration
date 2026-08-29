"""Audit and summarize the locked primary benchmark package.

This command validates task/result alignment and writes the manuscript-style
summary from the released task-level files. The locked manifests and the
recovered benchmark source are both included in the evidence archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from summarize_primary_benchmark import read_rows, summarize


ROOT = Path(__file__).resolve().parents[1]
FORMAL = (
    ROOT
    / "evidence"
    / "simulation_numerical_evidence_20260823"
    / "01_PRIMARY_5x3000_BENCHMARK"
    / "formal_matched_evaluation"
)
SEEDS = (101, 202, 303, 404, 555)
PRIMARY_METHODS = {"pf_teacher", "imitation", "ppo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            # The task seed is namespaced as benchmark seed + 1,000,000.
            keys.add((int(payload["seed"]) - 1_000_000, int(payload["task_id"])))
    return keys


def audit(formal: Path) -> dict[str, object]:
    task_root = formal / "tasks"
    task_keys: set[tuple[int, int]] = set()
    manifest_counts: dict[str, int] = {}
    manifest_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = task_root / f"seed_{seed}_tasks.jsonl"
        keys = manifest_keys(path)
        if len(keys) != 3000:
            raise ValueError(f"{path.name} contains {len(keys)} unique tasks, expected 3000")
        if task_keys & keys:
            raise ValueError(f"Duplicate task keys found in {path.name}")
        task_keys |= keys
        manifest_counts[str(seed)] = len(keys)
        manifest_hashes[path.name] = sha256(path)

    results_path = formal / "all_task_results.csv"
    rows = read_rows(results_path)
    result_counts = Counter(
        (int(row["benchmark_seed"]), int(row["task_id"])) for row in rows
    )
    result_keys = set(result_counts)
    methods = {row["method"] for row in rows}
    expected_keys = task_keys
    if len(rows) != 45000:
        raise ValueError(f"Primary result rows={len(rows)}, expected 45000")
    if result_keys != expected_keys:
        raise ValueError("Primary result task keys do not match locked manifests")
    if methods != PRIMARY_METHODS:
        raise ValueError(f"Primary methods={sorted(methods)}, expected {sorted(PRIMARY_METHODS)}")
    invalid_counts = {key: count for key, count in result_counts.items() if count != 3}
    if invalid_counts:
        raise ValueError(f"Tasks without exactly three primary method rows: {list(invalid_counts)[:3]}")

    replay_path = formal / "rule_baseline_replay" / "all_rule_baseline_results.csv"
    replay_rows = read_rows(replay_path) if replay_path.is_file() else []
    all_rows = rows + replay_rows
    summary_methods = ["pf_teacher", "imitation", "ppo", "simple_rule", "prespecified_pid", "tuned_pid"]
    summaries = [
        summarize(all_rows, method)
        for method in summary_methods
        if any(row["method"] == method for row in all_rows)
    ]
    return {
        "status": "PASS",
        "audit_scope": "locked_manifest_and_task_level_results",
        "seed_regeneration_claim": "source_archived_with_locked_manifests",
        "seed_regeneration_note": "The locked manifests, task generator, environment, and evaluation source are archived together.",
        "manifest_counts": manifest_counts,
        "manifest_sha256": manifest_hashes,
        "unique_tasks": len(task_keys),
        "primary_result_rows": len(rows),
        "primary_methods": sorted(methods),
        "replay_result_rows": len(replay_rows),
        "summaries": summaries,
        "primary_results_sha256": sha256(results_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=FORMAL)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    formal = args.formal_dir.resolve()
    summary_output = args.summary_output or formal / "REGENERATED_PRIMARY_SUMMARY.csv"
    audit_output = args.audit_output or formal / "PRIMARY_REPRODUCTION_AUDIT.json"

    report = audit(formal)
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        summaries = report["summaries"]
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    audit_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": str(summary_output), "audit": str(audit_output)}, indent=2))


if __name__ == "__main__":
    main()
