from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_full"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    curves = read_csv(RESULTS / "learning_curves.csv")
    aggregate = read_csv(RESULTS / "aggregate_summary.csv")
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in curves:
        grouped[(row["algorithm"], row["initialization"], int(row["seed"]))].append(row)

    endpoint_by_seed: list[dict[str, object]] = []
    for (algorithm, initialization, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["environment_steps"]))
        start = ordered[0]
        final = ordered[-1]
        endpoint_by_seed.append(
            {
                "algorithm": algorithm,
                "initialization": initialization,
                "seed": seed,
                "start_success_rate_percent": float(start["success_rate_percent"]),
                "final_success_rate_percent": float(final["success_rate_percent"]),
                "change_success_percentage_points": float(final["success_rate_percent"]) - float(start["success_rate_percent"]),
                "start_strict_success_rate_percent": float(start["strict_success_rate_percent"]),
                "final_strict_success_rate_percent": float(final["strict_success_rate_percent"]),
                "start_severe_failure_rate_percent": float(start["severe_failure_rate_percent"]),
                "final_severe_failure_rate_percent": float(final["severe_failure_rate_percent"]),
                "final_environment_steps": int(final["environment_steps"]),
            }
        )
    write_csv(RESULTS / "learning_endpoint_by_seed.csv", endpoint_by_seed)

    condition_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in endpoint_by_seed:
        condition_groups[(str(row["algorithm"]), str(row["initialization"]))].append(row)
    endpoint_summary: list[dict[str, object]] = []
    for (algorithm, initialization), rows in sorted(condition_groups.items()):
        starts = [float(row["start_success_rate_percent"]) for row in rows]
        finals = [float(row["final_success_rate_percent"]) for row in rows]
        changes = [float(row["change_success_percentage_points"]) for row in rows]
        endpoint_summary.append(
            {
                "algorithm": algorithm,
                "initialization": initialization,
                "training_seeds": len(rows),
                "untrained_success_rate_percent_mean": float(np.mean(starts)),
                "untrained_success_rate_percent_sd": sample_sd(starts),
                "final_success_rate_percent_mean": float(np.mean(finals)),
                "final_success_rate_percent_sd": sample_sd(finals),
                "training_change_percentage_points_mean": float(np.mean(changes)),
                "training_change_percentage_points_sd": sample_sd(changes),
                "seeds_improved": sum(change > 0 for change in changes),
                "seeds_unchanged": sum(math.isclose(change, 0.0, abs_tol=1e-12) for change in changes),
                "seeds_degraded": sum(change < 0 for change in changes),
            }
        )
    write_csv(RESULTS / "learning_endpoint_summary.csv", endpoint_summary)

    def best(field: str, lower: bool = False) -> dict[str, str]:
        return min(aggregate, key=lambda row: float(row[field])) if lower else max(aggregate, key=lambda row: float(row[field]))

    highlights = [
        ("Highest mean success", best("success_rate_percent_mean"), "success_rate_percent_mean", "success_rate_percent_sd"),
        ("Highest strict success", best("strict_success_rate_percent_mean"), "strict_success_rate_percent_mean", "strict_success_rate_percent_sd"),
        ("Lowest severe-failure rate", best("severe_failure_rate_percent_mean", lower=True), "severe_failure_rate_percent_mean", "severe_failure_rate_percent_sd"),
        ("Fewest mean steps", best("steps_mean_mean", lower=True), "steps_mean_mean", "steps_mean_sd"),
        ("Lowest final absolute error", best("final_abs_error_mean_mean", lower=True), "final_abs_error_mean_mean", "final_abs_error_mean_sd"),
        ("Lowest mean total volume", best("total_volume_mean_ml_mean", lower=True), "total_volume_mean_ml_mean", "total_volume_mean_ml_sd"),
    ]
    lines = [
        "# Performance highlights",
        "",
        "All values are mean +/- sample SD across five independent training seeds.",
        "",
        "| Metric | Best condition | Value |",
        "|---|---|---:|",
    ]
    for label, row, mean_field, sd_field in highlights:
        lines.append(
            f"| **{label}** | **{row['algorithm'].upper()} / {row['initialization']}** | "
            f"**{float(row[mean_field]):.2f} +/- {float(row[sd_field]):.2f}** |"
        )
    lines.extend(
        [
            "",
            "The lowest-volume condition is not the best overall controller: REINFORCE/imitation uses less liquid on average but has materially lower success and higher severe-failure rates. PPO/imitation is the strongest balanced result.",
            "",
            "For randomly initialized actors, A2C achieved the highest final mean success (53.00 +/- 6.91%) and the largest mean gain over the untrained random actor (+21.62 percentage points).",
            "",
        ]
    )
    (RESULTS / "PERFORMANCE_HIGHLIGHTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
