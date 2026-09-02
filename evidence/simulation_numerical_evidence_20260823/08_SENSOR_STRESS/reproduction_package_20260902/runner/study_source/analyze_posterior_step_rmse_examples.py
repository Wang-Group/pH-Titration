from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from chemistry_model import SolutionState, response_curve
from task_distribution import load_tasks


CURVE_GRID_ML = np.linspace(-100.0, 100.0, 161)
EXAMPLE_QUANTILES = (0.10, 0.50, 0.90)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "median": math.nan,
                "q25": math.nan, "q75": math.nan}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def natural_rows(posterior_dir: Path) -> list[dict]:
    source = posterior_dir / "all_posterior_task_results.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing task-level posterior output: {source}")
    rows = []
    for row in read_csv(source):
        if row.get("checkpoint_type") != "natural_control_end":
            continue
        rows.append({
            "benchmark_seed": int(row["benchmark_seed"]),
            "task_seed": int(row["task_seed"]),
            "task_id": int(row["task_id"]),
            "observations": int(row["observations"]),
            "true_pair_count": int(row["true_pair_count"]),
            "estimated_pair_count": int(row["estimated_pair_count"]),
            "pair_count_correct": int(row["pair_count_correct"]),
            "true_concentration_m": float(row["true_concentration_m"]),
            "estimated_concentration_m": float(row["estimated_concentration_m"]),
            "true_pka_json": row["true_pka_json"],
            "estimated_pka_json": row["estimated_pka_json"],
            "curve_rmse_ph": float(row["curve_rmse_ph"]),
            "curve_correlation": float(row["curve_correlation"]),
        })
    if not rows:
        raise ValueError("No natural_control_end rows found")
    return rows


def grouped_step_summary(rows: list[dict]) -> list[dict]:
    output = []
    steps = sorted({row["observations"] for row in rows})
    for observations in steps:
        for label, expected in (("all", None), ("K_correct", 1), ("K_incorrect", 0)):
            subset = [
                row for row in rows
                if row["observations"] == observations
                and (expected is None or row["pair_count_correct"] == expected)
            ]
            if not subset:
                continue
            stats = summary(row["curve_rmse_ph"] for row in subset)
            output.append({
                "observations": observations,
                "group": label,
                "tasks": stats["n"],
                "curve_rmse_mean": stats["mean"],
                "curve_rmse_sd": stats["sd"],
                "curve_rmse_median": stats["median"],
                "curve_rmse_q25": stats["q25"],
                "curve_rmse_q75": stats["q75"],
            })
    return output


def correlation(rows: list[dict]) -> dict:
    output = {}
    for label, expected in (("all", None), ("K_correct", 1), ("K_incorrect", 0)):
        subset = [row for row in rows if expected is None or row["pair_count_correct"] == expected]
        if len(subset) < 3 or len({row["observations"] for row in subset}) < 2:
            rho, p_value = math.nan, math.nan
        else:
            rho, p_value = spearmanr(
                [row["observations"] for row in subset],
                [row["curve_rmse_ph"] for row in subset],
            )
        output[label] = {"tasks": len(subset), "spearman_rho": float(rho), "p_value": float(p_value)}
    return output


def plot_step_relationship(path: Path, rows: list[dict], grouped: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    rng = np.random.default_rng(20260814)
    for label, expected, color in (("K correct", 1, "#1f77b4"), ("K incorrect", 0, "#d62728")):
        subset = [row for row in rows if row["pair_count_correct"] == expected]
        x = np.asarray([row["observations"] for row in subset], dtype=float)
        y = np.asarray([row["curve_rmse_ph"] for row in subset], dtype=float)
        axis.scatter(x + rng.normal(0.0, 0.055, size=x.size), y, s=12, alpha=0.20,
                     color=color, edgecolors="none", label=label)
        medians = [row for row in grouped if row["group"] == ("K_correct" if expected else "K_incorrect")]
        axis.plot([row["observations"] for row in medians],
                  [row["curve_rmse_median"] for row in medians],
                  color=color, linewidth=2.2, marker="o")
    axis.set_xlabel("Number of titration observations")
    axis.set_ylabel("Full-curve RMSE (pH)")
    axis.set_yscale("log")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def select_examples(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["curve_rmse_ph"])
    values = np.asarray([row["curve_rmse_ph"] for row in ordered], dtype=float)
    selected = []
    used = set()
    for quantile in EXAMPLE_QUANTILES:
        target = float(np.quantile(values, quantile))
        candidates = sorted(ordered, key=lambda row: abs(row["curve_rmse_ph"] - target))
        chosen = next(
            row for row in candidates
            if (row["benchmark_seed"], row["task_seed"], row["task_id"]) not in used
        )
        used.add((chosen["benchmark_seed"], chosen["task_seed"], chosen["task_id"]))
        selected.append({**chosen, "requested_rmse_quantile": quantile})
    return selected


def load_task(posterior_dir: Path, row: dict):
    path = posterior_dir / f"seed_{row['benchmark_seed']}_tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing exact generated tasks: {path}")
    tasks = load_tasks(path)
    matches = [task for task in tasks if task.seed == row["task_seed"] and task.task_id == row["task_id"]]
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely locate task for row {row}")
    return matches[0]


def plot_examples(path: Path, posterior_dir: Path, examples: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(15.0, 4.4), sharey=True, constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]
    labels = ("Low RMSE", "Median RMSE", "High RMSE")
    for axis, label, row in zip(axes, labels, examples):
        task = load_task(posterior_dir, row)
        initial_state = SolutionState(
            total_volume_ml=float(task.initial_volume_ml),
            base_moles=float(task.initial_base_moles),
            acid_moles=0.0,
        )
        true_curve = response_curve(
            task.analyte_conc_m, task.pka_values, task.initial_volume_ml,
            initial_state, CURVE_GRID_ML,
        )
        fitted_curve = response_curve(
            row["estimated_concentration_m"], json.loads(row["estimated_pka_json"]),
            task.initial_volume_ml, initial_state, CURVE_GRID_ML,
        )
        axis.plot(CURVE_GRID_ML, true_curve, color="#111111", linewidth=2.2, label="Ground truth")
        axis.plot(CURVE_GRID_ML, fitted_curve, color="#1f77b4", linewidth=2.0,
                  linestyle="--", label="PF posterior")
        axis.set_title(
            f"{label}\nRMSE={row['curve_rmse_ph']:.3f}; steps={row['observations']}\n"
            f"K true/pred={row['true_pair_count']}/{row['estimated_pair_count']}"
        )
        axis.set_xlabel("Signed titrant volume (mL)")
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("pH")
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def write_report(path: Path, rows: list[dict], correlations: dict, examples: list[dict]) -> None:
    lines = [
        "# Posterior fit versus titration steps",
        "",
        "Natural control endpoints are used, so the observation count is the number of pH observations accumulated before stopping.",
        "The scatter plot shows task-level full-curve RMSE, with separate median trajectories for correct and incorrect K classification.",
        "A statistical association does not imply that extra steps alone cause better fitting because difficult tasks can require more steps.",
        "",
        "## Spearman associations",
        "",
    ]
    for label in ("all", "K_correct", "K_incorrect"):
        item = correlations[label]
        lines.append(
            f"- {label}: n={item['tasks']}, rho={item['spearman_rho']:.4f}, p={item['p_value']:.4g}."
        )
    lines.extend(["", "## Representative full-curve examples", ""])
    for row in examples:
        lines.append(
            f"- Quantile {row['requested_rmse_quantile']:.2f}: RMSE={row['curve_rmse_ph']:.4f} pH, "
            f"observations={row['observations']}, K true/pred={row['true_pair_count']}/{row['estimated_pair_count']}, "
            f"benchmark seed={row['benchmark_seed']}, task={row['task_id']}."
        )
    lines.extend([
        "",
        "The examples were selected by RMSE quantile, not by visual preference, and use the same -100 to +100 mL grid used to calculate the reported full-curve RMSE.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate posterior curve RMSE to titration steps and plot representative cases")
    parser.add_argument("--posterior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = natural_rows(args.posterior_dir)
    grouped = grouped_step_summary(rows)
    correlations = correlation(rows)
    examples = select_examples(rows)
    write_csv(args.output_dir / "step_rmse_summary.csv", grouped)
    write_csv(args.output_dir / "selected_rmse_examples.csv", examples)
    (args.output_dir / "step_rmse_correlations.json").write_text(
        json.dumps(correlations, indent=2), encoding="utf-8"
    )
    plot_step_relationship(args.output_dir / "step_vs_curve_rmse.png", rows, grouped)
    plot_examples(args.output_dir / "representative_rmse_curves.png", args.posterior_dir, examples)
    write_report(args.output_dir / "POSTERIOR_STEP_RMSE_SUMMARY.md", rows, correlations, examples)
    print(f"Wrote step/RMSE evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
