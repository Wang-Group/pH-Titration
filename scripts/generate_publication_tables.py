"""Generate manuscript-ready tables from the locked simulation CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


SEEDS = (101, 202, 303, 404, 555)
METHOD_LABELS = {
    "pf_teacher": "PF controller",
    "imitation": "Imitation",
    "ppo": "PPO",
    "simple_rule": "Simple rule",
    "prespecified_pid": "Prespecified PID",
    "tuned_pid": "Tuned PID",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_summary(rows: list[dict[str, str]], method: str, seed: int) -> dict[str, float]:
    selected = [row for row in rows if row["method"] == method and int(row["benchmark_seed"]) == seed]
    if not selected:
        raise ValueError(f"No rows found for {method} and benchmark seed {seed}")
    successful = [row for row in selected if int(row["true_success"])]
    return {
        "success": 100.0 * len(successful) / len(selected),
        "successful_steps": statistics.mean(float(row["steps"]) for row in successful)
        if successful else math.nan,
        "overshoots": statistics.mean(float(row["overshoots"]) for row in selected),
        "final_error": statistics.mean(float(row["final_abs_error"]) for row in selected),
    }


def aggregate(rows: list[dict[str, str]], method: str) -> dict[str, float | str]:
    values = [seed_summary(rows, method, seed) for seed in SEEDS]
    output: dict[str, float | str] = {"method": method, "label": METHOD_LABELS[method]}
    for metric in ("success", "successful_steps", "overshoots", "final_error"):
        series = [item[metric] for item in values if math.isfinite(item[metric])]
        output[f"{metric}_mean"] = statistics.mean(series)
        output[f"{metric}_sd"] = statistics.stdev(series)
    return output


def fmt(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    formal = root / "evidence" / "simulation_numerical_evidence_20260823" / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
    parser.add_argument("--formal-dir", type=Path, default=formal)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = (args.output_dir or args.formal_dir / "publication_tables").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.formal_dir / "all_task_results.csv")
    baseline_path = args.formal_dir / "rule_baseline_replay" / "all_rule_baseline_results.csv"
    if baseline_path.is_file():
        rows.extend(read_rows(baseline_path))
    methods = [method for method in METHOD_LABELS if any(row["method"] == method for row in rows)]
    summaries = [aggregate(rows, method) for method in methods]

    csv_path = output_dir / "primary_controller_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    md_lines = [
        "# Primary controller comparison",
        "",
        "Five locked benchmark sets, 3,000 matched tasks per set; values are mean +/- sample SD across the five set-level summaries.",
        "",
        "| Controller | Success (%) | Successful steps | Overshoots/task | Final absolute error (pH) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md_lines.append(
            f"| {row['label']} | {fmt(float(row['success_mean']), float(row['success_sd']))} | "
            f"{fmt(float(row['successful_steps_mean']), float(row['successful_steps_sd']))} | "
            f"{fmt(float(row['overshoots_mean']), float(row['overshoots_sd']))} | "
            f"{fmt(float(row['final_error_mean']), float(row['final_error_sd']), 4)} |"
        )
    (output_dir / "primary_controller_comparison.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'primary_controller_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
