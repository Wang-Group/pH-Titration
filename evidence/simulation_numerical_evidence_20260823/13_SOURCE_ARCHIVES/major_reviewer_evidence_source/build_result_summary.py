from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, str]], columns: list[str], limit: int | None = None) -> list[str]:
    if not rows:
        return ["Not available.", ""]
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    return lines


def add_section(
    lines: list[str],
    title: str,
    path: Path,
    run_dir: Path,
    columns: list[str],
    limit: int | None = None,
) -> None:
    source = path.relative_to(run_dir).as_posix()
    lines.extend([f"## {title}", "", f"Source: `{source}`", ""])
    lines.extend(markdown_table(read_csv(path), columns, limit))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact index of all reviewer-analysis results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    lines = [
        "# Major-reviewer analysis result index",
        "",
        "This file is generated automatically after the one-click run. Task-level CSV files remain in their corresponding subdirectories for statistical auditing.",
        "",
    ]

    pid_json = run_dir / "pid_tuning" / "selected_pid_parameters.json"
    lines.extend(["## Tuned PID parameters", ""])
    if pid_json.exists():
        lines.append("```json")
        lines.append(json.dumps(json.loads(pid_json.read_text(encoding="utf-8")), indent=2))
        lines.extend(["```", ""])
    else:
        lines.extend(["Not available.", ""])

    add_section(
        lines,
        "Multi-seed benchmark",
        run_dir / "multiseed" / "aggregate_summary.csv",
        run_dir,
        ["method", "seeds", "success_rate_percent_mean", "success_rate_percent_seed_sd", "successful_steps_mean_mean", "overshoot_rate_percent_mean"],
    )
    add_section(
        lines,
        "Pooled paired tests",
        run_dir / "multiseed" / "mcnemar_tests.csv",
        run_dir,
        ["scope", "method_a", "method_b", "matched_tasks", "a_success_b_fail", "a_fail_b_success", "p_value_holm"],
        20,
    )
    add_section(
        lines,
        "RL versus imitation stress regimes",
        run_dir / "rl_il_stress" / "paired_tests.csv",
        run_dir,
        ["scenario", "imitation_success_percent", "rl_success_percent", "rl_minus_imitation_pp", "p_value_holm_across_scenarios", "predeclared_clear_rl_gain"],
    )
    add_section(
        lines,
        "pKa-reference sensitivity",
        run_dir / "bayesian_reference" / "aggregate_summary.csv",
        run_dir,
        ["condition", "success_mean", "success_seed_sd", "steps_mean", "overshoot_mean", "pka_error_mean"],
    )
    add_section(
        lines,
        "Observation-noise robustness",
        run_dir / "bayesian_noise" / "aggregate_summary.csv",
        run_dir,
        ["condition", "success_mean", "success_seed_sd", "steps_mean", "false_stop_mean", "pka_error_mean"],
    )
    add_section(
        lines,
        "RL algorithm comparison",
        run_dir / "rl_algorithms" / "final_results.csv",
        run_dir,
        ["seed", "algorithm", "reward_variant", "environment_steps", "success_rate_percent", "successful_steps_mean", "overshoot_rate_percent"],
    )
    add_section(
        lines,
        "Reward ablation",
        run_dir / "rl_rewards" / "final_results.csv",
        run_dir,
        ["seed", "algorithm", "reward_variant", "environment_steps", "success_rate_percent", "successful_steps_mean", "overshoot_rate_percent"],
    )

    output = run_dir / "RESULT_SUMMARY.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.relative_to(Path.cwd()).as_posix())


if __name__ == "__main__":
    main()
