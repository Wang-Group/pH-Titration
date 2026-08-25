"""Build a primary-benchmark summary from locked task-level CSV files."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], method: str) -> dict[str, float | str]:
    selected = [row for row in rows if row["method"] == method]
    by_seed: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_seed.setdefault(row["benchmark_seed"], []).append(row)
    seed_metrics: list[dict[str, float]] = []
    for seed_rows in by_seed.values():
        successful = [row for row in seed_rows if int(row["true_success"])]
        seed_metrics.append(
            {
                "success_percent": 100.0 * len(successful) / len(seed_rows),
                "successful_steps": statistics.mean(
                    [int(row["steps"]) for row in successful]
                )
                if successful
                else float("nan"),
                "overshoots": statistics.mean(
                    [float(row["overshoots"]) for row in seed_rows]
                ),
                "final_abs_error": statistics.mean(
                    [float(row["final_abs_error"]) for row in seed_rows]
                ),
            }
        )
    return {
        "method": method,
        "success_percent": statistics.mean(m["success_percent"] for m in seed_metrics),
        "success_percent_sd": statistics.stdev(m["success_percent"] for m in seed_metrics),
        "successful_steps": statistics.mean(m["successful_steps"] for m in seed_metrics),
        "successful_steps_sd": statistics.stdev(m["successful_steps"] for m in seed_metrics),
        "overshoots": statistics.mean(m["overshoots"] for m in seed_metrics),
        "overshoots_sd": statistics.stdev(m["overshoots"] for m in seed_metrics),
        "final_abs_error": statistics.mean(m["final_abs_error"] for m in seed_metrics),
        "final_abs_error_sd": statistics.stdev(m["final_abs_error"] for m in seed_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=FORMAL)
    parser.add_argument("--output", type=Path, default=FORMAL / "REGENERATED_PRIMARY_SUMMARY.csv")
    args = parser.parse_args()

    rows = read_rows(args.formal_dir / "all_task_results.csv")
    replay_path = args.formal_dir / "rule_baseline_replay" / "all_rule_baseline_results.csv"
    if replay_path.is_file():
        rows.extend(read_rows(replay_path))
    methods = ["pf_teacher", "imitation", "ppo", "simple_rule", "prespecified_pid", "tuned_pid"]
    summaries = [summarize(rows, method) for method in methods if any(r["method"] == method for r in rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
