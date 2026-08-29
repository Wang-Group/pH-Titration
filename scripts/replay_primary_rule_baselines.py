"""Validate the recovered simple-rule and PID baseline outputs.

The original runner and its task-level outputs are archived under
``evidence/.../13_SOURCE_ARCHIVES/historical_baseline_runner_20260817``.
This command checks those 45,000 rows and regenerates the comparison table;
it does not overwrite the recovered output with a different implementation.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL = (
    ROOT
    / "evidence"
    / "simulation_numerical_evidence_20260823"
    / "01_PRIMARY_5x3000_BENCHMARK"
    / "formal_matched_evaluation"
)
DEFAULT_OUTPUT = FORMAL / "rule_baseline_replay"
SEEDS = (101, 202, 303, 404, 555)
METHODS = ("simple_rule", "prespecified_pid", "tuned_pid")
REPORTED = {
    "simple_rule": {"success_rate_percent": 77.28, "successful_steps_mean": 16.11, "final_abs_error_mean": 0.1106},
    "prespecified_pid": {"success_rate_percent": 84.59, "successful_steps_mean": 17.14, "final_abs_error_mean": 0.2214},
    "tuned_pid": {"success_rate_percent": 92.44, "successful_steps_mean": 14.75, "final_abs_error_mean": 0.1504},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], method: str) -> dict[str, float | str]:
    by_seed: dict[int, list[dict[str, str]]] = {seed: [] for seed in SEEDS}
    for row in rows:
        if row["method"] == method:
            by_seed[int(row["benchmark_seed"])].append(row)
    seed_values = []
    for seed in SEEDS:
        seed_rows = by_seed[seed]
        if len(seed_rows) != 3000:
            raise ValueError(f"{method} seed {seed} has {len(seed_rows)} rows, expected 3000")
        successes = [row for row in seed_rows if int(row["true_success"])]
        seed_values.append(
            {
                "success_rate_percent": 100.0 * len(successes) / len(seed_rows),
                "successful_steps_mean": statistics.mean(float(row["steps"]) for row in successes),
                "final_abs_error_mean": statistics.mean(float(row["final_abs_error"]) for row in seed_rows),
            }
        )
    return {
        "method": method,
        "success_rate_percent": statistics.mean(row["success_rate_percent"] for row in seed_values),
        "success_rate_percent_sd": statistics.stdev(row["success_rate_percent"] for row in seed_values),
        "successful_steps_mean": statistics.mean(row["successful_steps_mean"] for row in seed_values),
        "successful_steps_mean_sd": statistics.stdev(row["successful_steps_mean"] for row in seed_values),
        "final_abs_error_mean": statistics.mean(row["final_abs_error_mean"] for row in seed_values),
        "final_abs_error_mean_sd": statistics.stdev(row["final_abs_error_mean"] for row in seed_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    rows = read_rows(output_dir / "all_rule_baseline_results.csv")
    if len(rows) != 45000:
        raise SystemExit(f"Expected 45,000 baseline rows, found {len(rows)}")
    methods = {row["method"] for row in rows}
    if methods != set(METHODS):
        raise SystemExit(f"Unexpected baseline methods: {sorted(methods)}")

    summaries = [summarize(rows, method) for method in METHODS]
    comparison = []
    for summary in summaries:
        method = str(summary["method"])
        for metric in ("success_rate_percent", "successful_steps_mean", "final_abs_error_mean"):
            observed = float(summary[metric])
            reported = REPORTED[method][metric]
            comparison.append(
                {
                    "method": method,
                    "metric": metric,
                    "observed_mean": observed,
                    "observed_sample_sd": float(summary[f"{metric}_sd"]),
                    "reported_mean": reported,
                    "observed_minus_reported": observed - reported,
                    "status": "matches_rounded_reported_value" if abs(observed - reported) < 0.01 else "check",
                }
            )
    write_csv(output_dir / "comparison_to_reported.csv", comparison)
    for summary in summaries:
        print(
            f"{summary['method']}: {summary['success_rate_percent']:.2f} +/- "
            f"{summary['success_rate_percent_sd']:.2f}% success"
        )


if __name__ == "__main__":
    main()
