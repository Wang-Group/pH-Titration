from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "bayesian_original": "Bayesian (original)",
    "bayesian_common": "Bayesian (common env)",
    "bayesian_residual_ppo": "Bayesian + transferred residual PPO",
    "bayesian_common_residual_ppo": "Bayesian + transferred residual PPO",
    "imitation": "Imitation",
    "submitted_rl": "Submitted RL",
    "ppo_reference": "Previous PPO",
    "ppo_nominal": "PPO nominal",
    "ppo_robust": "PPO robust",
    "a2c_robust": "A2C robust",
    "ppo_history_robust": "PPO history robust",
    "sac_history_robust": "SAC history robust",
    "ppo_residual_robust": "PPO residual robust",
    "ppo_residual_imitation": "Imitation + residual PPO",
    "ppo_filtered_robust": "PPO filtered robust",
    "ppo_conservative_robust": "PPO conservative robust",
    "td3_filtered_robust": "TD3 filtered robust",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean(values) -> float:
    values = [float(x) for x in values if not math.isnan(float(x))]
    return float(np.mean(values)) if values else math.nan


def write_learning_plot(training_dir: Path, output_dir: Path) -> None:
    rows = read_csv(training_dir / "learning_curves.csv")
    if not rows:
        return
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["candidate"]][int(float(row["environment_steps"]))].append(f(row, "success_rate_percent"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for candidate, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [mean(points[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=DISPLAY.get(candidate, candidate))
    ax.set_xlabel("Environment interactions")
    ax.set_ylabel("Validation success (%)")
    ax.set_title("Predeclared RL candidate learning curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=220)
    plt.close(fig)


def write_heatmap(aggregate: list[dict], output_dir: Path) -> None:
    scenarios = sorted({row["scenario"] for row in aggregate})
    preferred = [
        "bayesian_original", "bayesian_common", "bayesian_residual_ppo", "bayesian_common_residual_ppo",
        "imitation", "submitted_rl", "ppo_reference", "ppo_nominal", "ppo_robust", "a2c_robust",
        "ppo_history_robust", "sac_history_robust", "ppo_residual_imitation", "ppo_filtered_robust",
        "ppo_conservative_robust", "td3_filtered_robust",
    ]
    methods = [name for name in preferred if any(row["method"] == name for row in aggregate)]
    matrix = np.full((len(scenarios), len(methods)), np.nan)
    lookup = {(row["scenario"], row["method"]): f(row, "success_rate_percent_mean") for row in aggregate}
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            matrix[i, j] = lookup.get((scenario, method), np.nan)
    fig_height = max(6.0, 0.34 * len(scenarios) + 2.5)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.9 * len(methods)), fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(len(methods)), [DISPLAY.get(x, x) for x in methods], rotation=45, ha="right")
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_title("Success rate across predeclared evaluation scenarios")
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < 72 else "black")
    fig.colorbar(image, ax=ax, label="Success (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_success_heatmap.png", dpi=220)
    plt.close(fig)


def write_pareto(aggregate: list[dict], output_dir: Path) -> None:
    rows = [row for row in aggregate if row["scenario"] == "nominal"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for row in rows:
        success = f(row, "success_rate_percent_mean")
        overshoot = f(row, "overshoot_rate_percent_mean")
        volume = f(row, "total_added_ml_mean_mean")
        size = 40 + 8 * min(20.0, max(0.0, volume))
        ax.scatter(overshoot, success, s=size, alpha=0.75, label=DISPLAY.get(row["method"], row["method"]))
    ax.set_xlabel("Overshoot rate (%) - lower is better")
    ax.set_ylabel("Success rate (%) - higher is better")
    ax.set_title("Nominal success/overshoot trade-off (marker size reflects added volume)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "nominal_pareto.png", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the RL-versus-Bayesian challenge report.")
    parser.add_argument("--training-dir", type=Path, default=base / "results_challenge" / "training")
    parser.add_argument("--evaluation-dir", type=Path, default=base / "results_challenge" / "evaluation")
    parser.add_argument("--output-dir", type=Path, default=base / "results_challenge")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = read_csv(args.training_dir / "validation_results.csv")
    aggregate = read_csv(args.evaluation_dir / "aggregate_summary.csv")
    tests = read_csv(args.evaluation_dir / "paired_tests.csv")
    if not aggregate or not tests:
        raise SystemExit("Evaluation results are incomplete; aggregate_summary.csv and paired_tests.csv are required.")

    nominal = sorted(
        [row for row in aggregate if row["scenario"] == "nominal"],
        key=lambda row: f(row, "success_rate_percent_mean"),
        reverse=True,
    )
    stress_by_method = defaultdict(list)
    for row in aggregate:
        if row["scenario"] != "nominal":
            stress_by_method[row["method"]].append(f(row, "success_rate_percent_mean"))
    stress_summary = sorted(
        [
            {
                "method": method,
                "scenario_count": len(values),
                "mean_success": mean(values),
                "worst_success": min(values) if values else math.nan,
            }
            for method, values in stress_by_method.items()
        ],
        key=lambda row: row["mean_success"],
        reverse=True,
    )
    clear_wins = [row for row in tests if row.get("clear_success_win", "False").lower() == "true"]
    tradeoff_wins = [row for row in tests if row.get("multiobjective_tradeoff_win", "False").lower() == "true"]
    strict_wins = [row for row in tests if f(row, "strict_success_005_difference_pp") >= 1.0 and f(row, "strict_success_005_p_value_holm") < 0.05]
    severe_failure_wins = [row for row in tests if f(row, "severe_failure_050_difference_pp") <= -1.0 and f(row, "severe_failure_050_p_value_holm") < 0.05]

    lines = [
        "# RL versus Bayesian Challenge Report",
        "",
        "This report evaluates all predeclared candidates and scenarios. A method is called a clear success-rate winner only when it improves success by at least 1.0 percentage point and the paired McNemar p value remains below 0.05 after Holm correction. A separate trade-off criterion uses a seed-clustered paired-bootstrap lower confidence bound above -0.5 percentage points, at least a 10% significant improvement in steps, overshoots, or added volume, and no material final-error harm.",
        "",
        "## Nominal held-out benchmark",
        "",
        "| Method | Success mean +/- seed SD (%) | Strict +/-0.05 (%) | Severe >0.50 (%) | Steps | Overshoot (%) | Added volume (mL) | Final error CVaR95 | Decision P95 (ms/task) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in nominal:
        lines.append(
            f"| {DISPLAY.get(row['method'], row['method'])} | {f(row, 'success_rate_percent_mean'):.2f} +/- {f(row, 'success_rate_percent_seed_sd'):.2f} | "
            f"{f(row, 'strict_success_005_percent_mean'):.2f} | {f(row, 'severe_failure_050_percent_mean'):.2f} | "
            f"{f(row, 'successful_steps_mean_mean'):.2f} | {f(row, 'overshoot_rate_percent_mean'):.2f} | "
            f"{f(row, 'total_added_ml_mean_mean'):.2f} | {f(row, 'final_abs_error_cvar95_mean'):.4f} | {f(row, 'decision_time_ms_p95_mean'):.2f} |"
        )
    lines.extend([
        "",
        "## Stress-panel summary",
        "",
        "| Method | Scenarios | Mean success (%) | Worst-scenario success (%) |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in stress_summary:
        lines.append(f"| {DISPLAY.get(row['method'], row['method'])} | {row['scenario_count']} | {row['mean_success']:.2f} | {row['worst_success']:.2f} |")
    lines.extend(["", "## Predeclared decisions", ""])
    if clear_wins:
        lines.append("Clear success-rate wins over the relevant Bayesian baseline:")
        lines.append("")
        for row in clear_wins:
            lines.append(f"- {DISPLAY.get(row['method'], row['method'])} in {row['scenario']}: {f(row, 'success_difference_pp'):+.2f} pp, Holm p={f(row, 'success_p_value_holm'):.4g}.")
    else:
        lines.append("No candidate met the predeclared clear success-rate win criterion.")
    lines.append("")
    if tradeoff_wins:
        lines.append("Success-noninferior multi-objective trade-off wins:")
        lines.append("")
        for row in tradeoff_wins:
            lines.append(f"- {DISPLAY.get(row['method'], row['method'])} in {row['scenario']}: success difference {f(row, 'success_difference_pp'):+.2f} pp (seed-clustered 95% CI {f(row, 'success_difference_cluster_ci95_low'):+.2f} to {f(row, 'success_difference_cluster_ci95_high'):+.2f}).")
    else:
        lines.append("No candidate met the predeclared success-noninferior multi-objective trade-off criterion.")
    lines.extend(["", "## Tail-risk and strict-tolerance checks", ""])
    if strict_wins:
        lines.append("Methods with a significant >=1.0 pp gain at the stricter +/-0.05 pH tolerance:")
        lines.append("")
        for row in strict_wins:
            lines.append(f"- {DISPLAY.get(row['method'], row['method'])} in {row['scenario']}: {f(row, 'strict_success_005_difference_pp'):+.2f} pp, Holm p={f(row, 'strict_success_005_p_value_holm'):.4g}.")
    else:
        lines.append("No method met the strict-tolerance gain criterion.")
    lines.append("")
    if severe_failure_wins:
        lines.append("Methods reducing severe failures (>0.50 pH error) by >=1.0 pp with Holm p < 0.05:")
        lines.append("")
        for row in severe_failure_wins:
            lines.append(f"- {DISPLAY.get(row['method'], row['method'])} in {row['scenario']}: {f(row, 'severe_failure_050_difference_pp'):+.2f} pp, Holm p={f(row, 'severe_failure_050_p_value_holm'):.4g}.")
    else:
        lines.append("No method met the severe-failure reduction criterion.")
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "A win in one predeclared mismatch regime supports a regime-specific claim only. It does not establish universal RL superiority. If no success-rate win is found but a trade-off win is found, the defensible claim is improved overshoot, intervention count, volume, or latency at noninferior success, not higher accuracy.",
    ])
    (args.output_dir / "CHALLENGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "clear_success_wins": clear_wins,
        "multiobjective_tradeoff_wins": tradeoff_wins,
        "strict_tolerance_wins": strict_wins,
        "severe_failure_wins": severe_failure_wins,
        "nominal_ranking": [{"method": row["method"], "success": f(row, "success_rate_percent_mean")} for row in nominal],
        "stress_ranking": stress_summary,
        "validation_rows": validation,
    }
    (args.output_dir / "DECISION_SUMMARY.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_learning_plot(args.training_dir, args.output_dir)
    write_heatmap(aggregate, args.output_dir)
    write_pareto(aggregate, args.output_dir)


if __name__ == "__main__":
    main()
