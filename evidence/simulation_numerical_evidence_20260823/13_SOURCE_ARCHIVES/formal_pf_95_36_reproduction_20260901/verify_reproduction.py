from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEEDS = (101, 202, 303, 404, 555)
TIMING_FIELDS = {
    "selection_time_ms_total",
    "posterior_update_time_ms_total",
    "controller_time_ms_per_step",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_task_results(actual: Path, expected: Path) -> list[str]:
    actual_rows = read_csv(actual)
    expected_rows = read_csv(expected)
    problems: list[str] = []
    if len(actual_rows) != len(expected_rows):
        return [
            f"row-count mismatch for {actual.name}: "
            f"observed {len(actual_rows)}, expected {len(expected_rows)}"
        ]

    fields = [name for name in expected_rows[0] if name not in TIMING_FIELDS]
    for index, (observed, reference) in enumerate(
        zip(actual_rows, expected_rows, strict=True), start=2
    ):
        for field in fields:
            if observed.get(field) != reference.get(field):
                problems.append(
                    f"{actual.name}:{index} {field}: "
                    f"observed {observed.get(field)!r}, expected {reference.get(field)!r}"
                )
                if len(problems) >= 20:
                    return problems
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a new formal PF run against the locked repository evidence"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent
    evidence_root = source_dir.parents[1]
    formal_dir = (
        evidence_root
        / "01_PRIMARY_5x3000_BENCHMARK"
        / "formal_matched_evaluation"
    )
    expected_tasks = formal_dir / "tasks"
    expected_results = formal_dir / "pf_reference"
    problems: list[str] = []

    for seed in SEEDS:
        task_name = f"seed_{seed}_tasks.jsonl"
        actual_task = args.run_dir / task_name
        expected_task = expected_tasks / task_name
        if not actual_task.exists():
            problems.append(f"missing generated task manifest: {actual_task}")
        elif sha256(actual_task) != sha256(expected_task):
            problems.append(f"task-manifest hash mismatch: {task_name}")

        result_name = f"seed_{seed}_task_results.csv"
        actual_result = args.run_dir / result_name
        expected_result = expected_results / result_name
        if not actual_result.exists():
            problems.append(f"missing task result: {actual_result}")
        else:
            problems.extend(compare_task_results(actual_result, expected_result))

    aggregate_path = args.run_dir / "aggregate_summary.csv"
    if not aggregate_path.exists():
        problems.append(f"missing aggregate summary: {aggregate_path}")
    else:
        aggregate = {row["policy"]: row for row in read_csv(aggregate_path)}
        hybrid = aggregate.get("hybrid_full")
        if hybrid is None:
            problems.append("aggregate summary has no hybrid_full row")
        elif float(hybrid["success_rate_percent_mean"]) != 95.36:
            problems.append(
                "hybrid_full success mean mismatch: "
                f"observed {hybrid['success_rate_percent_mean']}, expected 95.36"
            )

    report = {
        "status": "PASS" if not problems else "FAIL",
        "seeds": list(SEEDS),
        "tasks_per_seed": 3000,
        "task_manifest_check": "byte-for-byte SHA-256",
        "task_result_check": "all fields except machine-dependent timing fields",
        "excluded_timing_fields": sorted(TIMING_FIELDS),
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
