from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def summary(values) -> dict[str, float | int]:
    array = finite(values)
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


def symmetric_nearest_pka_distance(true_values, estimated_values) -> float:
    """A descriptive set-distance for wrong-K cases; not a one-to-one MAE."""
    true = np.asarray(true_values, dtype=float)
    estimated = np.asarray(estimated_values, dtype=float)
    if true.size == 0 or estimated.size == 0:
        return math.nan
    forward = np.mean(np.min(np.abs(true[:, None] - estimated[None, :]), axis=1))
    reverse = np.mean(np.min(np.abs(estimated[:, None] - true[None, :]), axis=1))
    return float(0.5 * (forward + reverse))


def parse_array(row: dict[str, str], field: str) -> list[float]:
    value = row.get(field, "")
    if not value:
        return []
    parsed = json.loads(value)
    return [float(item) for item in parsed]


def build_rows(rows: list[dict[str, str]], checkpoint: str) -> list[dict]:
    selected = [row for row in rows if row.get("checkpoint_type") == checkpoint]
    if not selected:
        raise ValueError(f"No rows with checkpoint_type={checkpoint!r}")
    output = []
    for row in selected:
        true = parse_array(row, "true_pka_json")
        estimated = parse_array(row, "estimated_pka_json")
        k_correct = str(row.get("pair_count_correct", "0")) == "1"
        direct_mae = float(row["pka_mae_if_k_correct"]) if k_correct and row.get("pka_mae_if_k_correct") else math.nan
        direct_rmse = float(row["pka_rmse_if_k_correct"]) if k_correct and row.get("pka_rmse_if_k_correct") else math.nan
        output.append({
            "benchmark_seed": int(row["benchmark_seed"]),
            "task_id": int(row["task_id"]),
            "observations": int(row["observations"]),
            "true_pair_count": int(row["true_pair_count"]),
            "estimated_pair_count": int(row["estimated_pair_count"]),
            "k_classification": "correct" if k_correct else "incorrect",
            "curve_rmse_ph": float(row["curve_rmse_ph"]),
            "curve_mae_ph": float(row["curve_mae_ph"]),
            "curve_correlation": float(row["curve_correlation"]),
            "pka_mae_direct": direct_mae,
            "pka_rmse_direct": direct_rmse,
            "pka_symmetric_nearest_distance": symmetric_nearest_pka_distance(true, estimated),
        })
    return output


def aggregate(rows: list[dict]) -> list[dict]:
    groups = sorted({(row["observations"], row["k_classification"]) for row in rows})
    result = []
    for observations, classification in groups:
        subset = [row for row in rows if row["observations"] == observations and row["k_classification"] == classification]
        curve = summary(row["curve_rmse_ph"] for row in subset)
        curve_mae = summary(row["curve_mae_ph"] for row in subset)
        corr = summary(row["curve_correlation"] for row in subset)
        direct_mae = summary(row["pka_mae_direct"] for row in subset)
        direct_rmse = summary(row["pka_rmse_direct"] for row in subset)
        set_distance = summary(row["pka_symmetric_nearest_distance"] for row in subset)
        result.append({
            "observations": observations,
            "k_classification": classification,
            "tasks": len(subset),
            "curve_rmse_mean": curve["mean"],
            "curve_rmse_sd": curve["sd"],
            "curve_rmse_median": curve["median"],
            "curve_rmse_q25": curve["q25"],
            "curve_rmse_q75": curve["q75"],
            "curve_mae_mean": curve_mae["mean"],
            "curve_correlation_mean": corr["mean"],
            "pka_direct_mae_tasks": direct_mae["n"],
            "pka_direct_mae_mean": direct_mae["mean"],
            "pka_direct_mae_sd": direct_mae["sd"],
            "pka_direct_rmse_mean": direct_rmse["mean"],
            "pka_symmetric_nearest_distance_mean": set_distance["mean"],
            "pka_symmetric_nearest_distance_sd": set_distance["sd"],
        })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], checkpoint: str, source: Path) -> None:
    natural = rows
    correct = [row for row in natural if row["k_classification"] == "correct"]
    incorrect = [row for row in natural if row["k_classification"] == "incorrect"]
    def fmt(values, field):
        stats = summary(row[field] for row in values)
        if stats["n"] == 0:
            return "n=0"
        return f"n={stats['n']}; mean={stats['mean']:.4g}; SD={stats['sd']:.4g}; median={stats['median']:.4g}"
    lines = [
        "# K-stratified posterior fit diagnostics",
        "",
        f"Source: `{source}`",
        f"Checkpoint: `{checkpoint}`",
        "",
        "The classification-correct and classification-incorrect groups are separated before summarizing fit quality.",
        "For correct K, pKa error is the direct one-to-one MAE/RMSE already defined by the posterior diagnostic.",
        "For incorrect K, direct pKa MAE is not reported because the vectors have different dimensions. Instead, the report gives a symmetric nearest-neighbor pKa distance as a descriptive diagnostic only.",
        "Curve RMSE is comparable in both groups because every posterior is evaluated on the same full titration grid.",
        "",
        "## Natural-end comparison",
        "",
        f"Correct K: curve RMSE ({fmt(correct, 'curve_rmse_ph')}); direct pKa MAE ({fmt(correct, 'pka_mae_direct')}); symmetric nearest-neighbor pKa distance ({fmt(correct, 'pka_symmetric_nearest_distance')}).",
        f"Incorrect K: curve RMSE ({fmt(incorrect, 'curve_rmse_ph')}); direct pKa MAE is undefined by design; symmetric nearest-neighbor pKa distance ({fmt(incorrect, 'pka_symmetric_nearest_distance')}).",
        "",
        "The appropriate conclusion is therefore whether the posterior response curve remains useful when K is misclassified, while exact pKa interpretation is restricted to the K-correct subset.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    observations = sorted({row["observations"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for classification, color in (("correct", "#1f77b4"), ("incorrect", "#d62728")):
        subset = [row for row in rows if row["k_classification"] == classification]
        means = []
        errors = []
        for obs in observations:
            stats = summary(row["curve_rmse_ph"] for row in subset if row["observations"] == obs)
            means.append(stats["mean"])
            errors.append(stats["sd"])
        axes[0].errorbar(observations, means, yerr=errors, marker="o", capsize=3, label=classification, color=color)
        nearest = []
        nearest_err = []
        for obs in observations:
            stats = summary(row["pka_symmetric_nearest_distance"] for row in subset if row["observations"] == obs)
            nearest.append(stats["mean"])
            nearest_err.append(stats["sd"])
        axes[1].errorbar(observations, nearest, yerr=nearest_err, marker="o", capsize=3, label=classification, color=color)
    axes[0].set_xlabel("Observations")
    axes[0].set_ylabel("Full-curve RMSE (pH)")
    axes[1].set_xlabel("Observations")
    axes[1].set_ylabel("Symmetric nearest-neighbor pKa distance")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify posterior pKa and full-curve fit by K classification")
    parser.add_argument("--posterior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-type", default="natural_control_end")
    args = parser.parse_args()
    source = args.posterior_dir / "all_posterior_task_results.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing task-level posterior output: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_rows(source), args.checkpoint_type)
    aggregates = aggregate(rows)
    write_csv(args.output_dir / "k_stratified_fit_task_rows.csv", rows)
    write_csv(args.output_dir / "k_stratified_fit_summary.csv", aggregates)
    write_report(args.output_dir / "K_STRATIFIED_FIT_SUMMARY.md", rows, args.checkpoint_type, source)
    plot(args.output_dir / "k_stratified_fit.png", rows)
    print(f"Wrote K-stratified diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
