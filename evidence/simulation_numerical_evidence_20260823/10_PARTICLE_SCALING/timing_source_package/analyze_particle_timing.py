from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path


PARTICLES = [1000, 10000, 100000]
METRICS = ("update_posteriors_ms", "select_best_action_ms", "decision_cycle_ms")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"particle_count", "task_seed", "task_id", "cases"}:
                row[key] = int(value)
            elif key not in {"acid_type"}:
                row[key] = float(value)
    return rows


def summarize(directory: Path, expected_cases: int) -> dict:
    path = directory / "particle_count_timing_summary.csv"
    rows = read_rows(path)
    raw_rows = read_rows(directory / "particle_count_timing_per_task.csv")
    errors = directory / "particle_count_timing_errors.csv"
    problems = []
    if len(rows) != 3 or [row["particle_count"] for row in rows] != PARTICLES:
        problems.append("summary does not contain the three prespecified particle counts")
    if any(int(row["cases"]) != expected_cases for row in rows):
        problems.append(f"expected {expected_cases} cases per particle count")
    raw_counts = {particle: sum(row["particle_count"] == particle for row in raw_rows) for particle in PARTICLES}
    if len(raw_rows) != expected_cases * len(PARTICLES) or any(count != expected_cases for count in raw_counts.values()):
        problems.append("raw per-task rows do not match the expected particle-count allocation")
    if errors.exists() and errors.stat().st_size > 0:
        problems.append("error CSV is present")
    for row in rows:
        for key in row:
            if key.endswith(("_mean", "_median", "_sd", "_p95")) and not math.isfinite(float(row[key])):
                problems.append(f"non-finite value in {key}")
    baseline = float(rows[0]["decision_cycle_ms_median"])
    for row in rows:
        row["decision_cycle_median_ratio"] = float(row["decision_cycle_ms_median"]) / baseline
    return {
        "rows": rows,
        "expected_cases": expected_cases,
        "raw_rows": len(raw_rows),
        "raw_particle_counts": raw_counts,
        "errors_file": errors.exists(),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    default = summarize(root / "particle_count_timing_results", 5)
    extended = summarize(root / "particle_count_timing_results_extended_20260806", 15)
    all_problems = default["problems"] + extended["problems"]
    required = [
        root / "RUN_COMPLETE.txt",
        root / "RUN_COMPLETE_EXTENDED.txt",
        root / "particle_count_timing_benchmark_executed.ipynb",
        root / "particle_count_timing_benchmark_extended_executed.ipynb",
        root / "PARTICLE_TIMING_ANALYSIS_CN_20260806.md",
        root / "REVIEWER_RESPONSE_PARTICLE_TIMING_EN_20260806.md",
        root / "particle_count_timing_run.log",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        all_problems.append("missing or empty required files: " + ", ".join(missing))
    extended_cycle = [float(row["decision_cycle_ms_median"]) for row in extended["rows"]]
    extended_update = [float(row["update_posteriors_ms_median"]) for row in extended["rows"]]
    def log_slope(values: list[float]) -> float:
        x = [math.log10(value) for value in PARTICLES]
        y = [math.log10(value) for value in values]
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum((a - x_mean) ** 2 for a in x)

    payload = {
        "status": "PASS" if not all_problems else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "default": default,
        "extended": extended,
        "extended_log10_scaling_exponent": {
            "update_posteriors_median": log_slope(extended_update),
            "decision_cycle_median": log_slope(extended_cycle),
        },
        "missing_required_files": missing,
        "notes": [
            "Timing measures the current controller's posterior update, action selection, and their cycle only.",
            "Results are single-machine wall-clock measurements and are not a hardware-independent real-time guarantee.",
        ],
    }
    (root / "PARTICLE_TIMING_VALIDATION_20260806.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if all_problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
