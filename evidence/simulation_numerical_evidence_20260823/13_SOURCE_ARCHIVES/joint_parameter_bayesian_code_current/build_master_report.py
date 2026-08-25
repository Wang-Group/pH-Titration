from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows, columns):
    if not rows:
        return "No results found."
    best = {}
    for key, _, direction in columns:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values and direction == "max":
            best[key] = max(values)
        elif values and direction == "min":
            best[key] = min(values)
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                formatted = f"{value:.5g}"
                if key in best and abs(value - best[key]) <= 1e-10 * max(1.0, abs(best[key])):
                    formatted = f"**{formatted}**"
                value = formatted
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    nominal = load_json(run_dir / "pf_control_nominal" / "summary.json")
    variable = load_json(run_dir / "pf_control_variable_concentration" / "summary.json")
    curve = load_json(run_dir / "pf_curve_recovery" / "summary.json")
    pymc = load_json(run_dir / "pymc_comparison" / "summary.json")

    def protocol(payload):
        settings = (payload or {}).get("settings", {})
        seeds = settings.get("seeds", [])
        tasks = settings.get("tasks_per_seed", "?")
        noun = "task" if tasks == 1 else "tasks"
        return f"{len(seeds)} independent seeds; {tasks} {noun} per seed."

    lines = [
        "# Joint Bayesian inference comparison: master result index",
        "",
        "This file is generated after the selected run profile finishes.",
        "",
        "## Nominal 0.1 M closed-loop control",
        "",
        protocol(nominal),
        "",
        markdown_table(
            (nominal or {}).get("aggregate", []),
            [
                ("method", "Method", None),
                ("success_percent_mean", "Success (%)", "max"),
                ("success_percent_seed_sd", "Seed SD", "min"),
                ("successful_steps_mean_mean", "Successful steps", "min"),
                ("final_abs_error_mean_ph_mean", "Final error", "min"),
                ("decision_time_median_ms_mean", "Decision ms", "min"),
            ],
        ),
        "",
        "## Variable-concentration closed-loop control",
        "",
        protocol(variable),
        "",
        markdown_table(
            (variable or {}).get("aggregate", []),
            [
                ("method", "Method", None),
                ("success_percent_mean", "Success (%)", "max"),
                ("success_percent_seed_sd", "Seed SD", "min"),
                ("successful_steps_mean_mean", "Successful steps", "min"),
                ("concentration_relative_error_median_percent_mean", "Concentration error (%)", "min"),
            ],
        ),
        "",
        "## Fixed-trajectory particle-filter curve recovery",
        "",
        protocol(curve),
        "",
        markdown_table(
            [row for row in (curve or {}).get("summary", []) if row.get("checkpoint") == "final"],
            [
                ("method", "Method", None),
                ("pair_count_accuracy_percent", "K accuracy (%)", "max"),
                ("concentration_relative_error_percent_median", "Concentration error (%)", "min"),
                ("local_rmse_0p10ml_ph_median", "Local RMSE", "min"),
                ("full_curve_rmse_0_33ml_ph_median", "Full RMSE", "min"),
            ],
        ),
        "",
        "## Particle filter versus PyMC SMC",
        "",
        protocol(pymc),
        "",
        "This small matched-trajectory section is a backend cross-check, not a replacement for the large PF benchmark.",
        "",
        markdown_table(
            (pymc or {}).get("summary", []),
            [
                ("method", "Method", None),
                ("pair_count_accuracy_percent", "K accuracy (%)", "max"),
                ("concentration_relative_error_median_percent", "Concentration error (%)", "min"),
                ("pka_matched_mae_median", "Matched pKa MAE", "min"),
                ("local_rmse_0p10ml_median_ph", "Local RMSE", "min"),
                ("full_curve_rmse_median_ph", "Full RMSE", "min"),
                ("runtime_median_seconds", "Runtime (s)", "min"),
            ],
        ),
        "",
        "Detailed task-level CSV files, figures, settings, and paired tests are in the four subdirectories.",
    ]
    (run_dir / "MASTER_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
