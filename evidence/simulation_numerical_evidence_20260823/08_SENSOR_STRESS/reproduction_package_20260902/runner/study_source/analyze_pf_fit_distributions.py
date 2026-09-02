from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, kruskal, spearmanr, t, wilcoxon


PROTOCOL_VERSION = 1
OBSERVATIONS = [0, 1, 2, 3, 5, 8, 12]
TRANSITIONS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (0, 12)]
COLORS = ["#355F8A", "#3E7C78", "#64A061", "#A3AF4B", "#D39A36", "#C86B3C", "#A6414B"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def distribution(values) -> dict:
    values = finite(values)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "p05": math.nan,
            "q1": math.nan,
            "median": math.nan,
            "q3": math.nan,
            "p95": math.nan,
        }
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
        "p05": float(quantiles[0]),
        "q1": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q3": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def holm_adjust(rows: list[dict], field: str = "p_value") -> None:
    indices = [index for index, row in enumerate(rows) if math.isfinite(float(row[field]))]
    ordered = sorted(indices, key=lambda index: float(rows[index][field]))
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(rows[index][field]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def fixed_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in all_rows if row["checkpoint_type"] == "fixed_observation_count"]


def natural_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in all_rows if row["checkpoint_type"] == "natural_control_end"]


def key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["benchmark_seed"]), int(row["task_id"])


def rmse_distributions(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    task_summary = []
    per_seed = []
    for observations in OBSERVATIONS:
        subset = [row for row in rows if int(row["observations"]) == observations]
        values = finite(float(row["curve_rmse_ph"]) for row in subset)
        result = {"observations": observations, **distribution(values)}
        for threshold in (0.25, 0.50, 1.00, 2.00, 3.00):
            result[f"within_{str(threshold).replace('.', 'p')}_percent"] = (
                100.0 * float(np.mean(values <= threshold))
            )
        task_summary.append(result)
        for seed in sorted({int(row["benchmark_seed"]) for row in subset}):
            seed_values = [
                float(row["curve_rmse_ph"])
                for row in subset
                if int(row["benchmark_seed"]) == seed
            ]
            per_seed.append(
                {
                    "observations": observations,
                    "benchmark_seed": seed,
                    **distribution(seed_values),
                }
            )
    return task_summary, per_seed


def paired_changes(rows: list[dict[str, str]]) -> tuple[list[dict], dict[int, dict[tuple[int, int], float]]]:
    lookup: dict[int, dict[tuple[int, int], float]] = {observations: {} for observations in OBSERVATIONS}
    for row in rows:
        lookup[int(row["observations"])][key(row)] = float(row["curve_rmse_ph"])
    output = []
    for start, end in TRANSITIONS:
        keys = sorted(set(lookup[start]) & set(lookup[end]))
        before = np.asarray([lookup[start][task_key] for task_key in keys], dtype=float)
        after = np.asarray([lookup[end][task_key] for task_key in keys], dtype=float)
        changes = after - before
        improved = int(np.sum(changes < 0.0))
        worsened = int(np.sum(changes > 0.0))
        nonzero = improved + worsened
        wilcoxon_result = wilcoxon(changes, zero_method="wilcox", method="auto")
        sign_p = (
            1.0
            if nonzero == 0
            else float(binomtest(improved, nonzero, 0.5, alternative="greater").pvalue)
        )
        change_stats = distribution(changes)
        output.append(
            {
                "start_observations": start,
                "end_observations": end,
                "paired_tasks": len(keys),
                "before_mean": float(np.mean(before)),
                "after_mean": float(np.mean(after)),
                "mean_change_after_minus_before": change_stats["mean"],
                "change_sd": change_stats["sd"],
                "change_p05": change_stats["p05"],
                "change_q1": change_stats["q1"],
                "change_median": change_stats["median"],
                "change_q3": change_stats["q3"],
                "change_p95": change_stats["p95"],
                "improved_tasks": improved,
                "worsened_tasks": worsened,
                "improved_percent": 100.0 * improved / len(keys),
                "paired_wilcoxon_statistic": float(wilcoxon_result.statistic),
                "p_value": float(wilcoxon_result.pvalue),
                "one_sided_improvement_sign_p": sign_p,
            }
        )
    holm_adjust(output)
    return output, lookup


def exact_seed_sign_flip_p(values: list[float]) -> float:
    """One-sided exact randomization p-value for a negative across-seed change."""
    observed = float(np.mean(values))
    magnitudes = np.abs(np.asarray(values, dtype=float))
    null_means = [
        float(np.mean(magnitudes * np.asarray(signs, dtype=float)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null_means) <= observed + 1e-15))


def seed_paired_changes(
    lookup: dict[int, dict[tuple[int, int], float]],
) -> tuple[list[dict], list[dict]]:
    per_seed = []
    summary = []
    for start, end in TRANSITIONS:
        transition_rows = []
        for seed in sorted({task_key[0] for task_key in lookup[start]}):
            keys = sorted(
                task_key
                for task_key in set(lookup[start]) & set(lookup[end])
                if task_key[0] == seed
            )
            before = np.asarray([lookup[start][task_key] for task_key in keys], dtype=float)
            after = np.asarray([lookup[end][task_key] for task_key in keys], dtype=float)
            changes = after - before
            row = {
                "start_observations": start,
                "end_observations": end,
                "benchmark_seed": seed,
                "paired_tasks": len(keys),
                "before_mean": float(np.mean(before)),
                "after_mean": float(np.mean(after)),
                "mean_change_after_minus_before": float(np.mean(changes)),
                "median_change_after_minus_before": float(np.median(changes)),
                "improved_tasks": int(np.sum(changes < 0.0)),
                "worsened_tasks": int(np.sum(changes > 0.0)),
                "improved_percent": 100.0 * float(np.mean(changes < 0.0)),
            }
            per_seed.append(row)
            transition_rows.append(row)
        seed_changes = np.asarray(
            [float(row["mean_change_after_minus_before"]) for row in transition_rows],
            dtype=float,
        )
        seed_count = len(seed_changes)
        seed_mean = float(np.mean(seed_changes))
        seed_sd = float(np.std(seed_changes, ddof=1)) if seed_count > 1 else math.nan
        margin = (
            float(t.ppf(0.975, seed_count - 1) * seed_sd / math.sqrt(seed_count))
            if seed_count > 1
            else math.nan
        )
        summary.append(
            {
                "start_observations": start,
                "end_observations": end,
                "benchmark_seeds": seed_count,
                "paired_tasks_total": int(sum(int(row["paired_tasks"]) for row in transition_rows)),
                "mean_of_seed_mean_changes": seed_mean,
                "seed_change_sd": seed_sd,
                "seed_t95_ci_low": seed_mean - margin,
                "seed_t95_ci_high": seed_mean + margin,
                "minimum_seed_mean_change": float(np.min(seed_changes)),
                "maximum_seed_mean_change": float(np.max(seed_changes)),
                "seeds_with_lower_mean_rmse": int(np.sum(seed_changes < 0.0)),
                "exact_one_sided_seed_sign_flip_p": exact_seed_sign_flip_p(seed_changes.tolist()),
                "note": "Inference unit is the benchmark seed; negative change means lower RMSE.",
            }
        )
    return per_seed, summary


def parameter_distributions(rows: list[dict[str, str]]) -> list[dict]:
    output = []
    for observations in OBSERVATIONS:
        subset = [row for row in rows if int(row["observations"]) == observations]
        for metric in (
            "concentration_relative_error_percent",
            "pka_mae_if_k_correct",
            "curve_correlation",
        ):
            values = [float(row[metric]) for row in subset]
            output.append(
                {
                    "observations": observations,
                    "metric": metric,
                    **distribution(values),
                }
            )
        output.append(
            {
                "observations": observations,
                "metric": "pair_count_correct",
                **distribution(float(row["pair_count_correct"]) for row in subset),
                "accuracy_percent": 100.0
                * float(np.mean([int(row["pair_count_correct"]) for row in subset])),
            }
        )
    return output


def subgroup_distributions(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    definitions = {
        "direction": ["acid", "base"],
        "difficulty": ["near", "medium", "far"],
        "true_pair_count": ["1", "2", "3"],
        "pka_family": ["single", "separated", "overlapping"],
    }
    output = []
    tests = []
    seeds = sorted({int(row["benchmark_seed"]) for row in rows})
    for field, levels in definitions.items():
        groups = []
        available_levels = [level for level in levels if any(row[field] == level for row in rows)]
        for level in available_levels:
            subset = [row for row in rows if row[field] == level]
            values = [float(row["curve_rmse_ph"]) for row in subset]
            seed_means = []
            for seed in seeds:
                seed_values = [
                    float(row["curve_rmse_ph"])
                    for row in subset
                    if int(row["benchmark_seed"]) == seed
                ]
                if seed_values:
                    seed_means.append(float(np.mean(seed_values)))
            groups.append(finite(values))
            output.append(
                {
                    "subgroup_field": field,
                    "subgroup_level": level,
                    **distribution(values),
                    "seed_mean_mean": float(np.mean(seed_means)),
                    "seed_mean_sd": (
                        float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else math.nan
                    ),
                }
            )
        result = kruskal(*groups) if len(groups) > 1 else None
        tests.append(
            {
                "subgroup_field": field,
                "levels": "|".join(available_levels),
                "test": "kruskal_wallis_task_level_descriptive",
                "statistic": float(result.statistic) if result is not None else math.nan,
                "p_value": float(result.pvalue) if result is not None else math.nan,
                "note": "Descriptive heterogeneity test; tasks are nested within five benchmark seeds.",
            }
        )
    holm_adjust(tests)
    return output, tests


def control_association(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], dict]:
    rmse = np.asarray([float(row["curve_rmse_ph"]) for row in rows], dtype=float)
    error = np.asarray([float(row["control_final_true_error"]) for row in rows], dtype=float)
    success = np.asarray([int(row["control_true_success"]) for row in rows], dtype=int)
    rho = spearmanr(rmse, error)
    quantile_edges = np.quantile(rmse, np.linspace(0.0, 1.0, 6))
    quintiles = []
    for index in range(5):
        mask = (rmse >= quantile_edges[index]) & (
            (rmse <= quantile_edges[index + 1]) if index == 4 else (rmse < quantile_edges[index + 1])
        )
        count = int(np.sum(mask))
        quintiles.append(
            {
                "rmse_quintile": index + 1,
                "rmse_lower": float(quantile_edges[index]),
                "rmse_upper": float(quantile_edges[index + 1]),
                "tasks": count,
                "control_success_percent": (
                    100.0 * float(np.mean(success[mask])) if count else math.nan
                ),
                "control_final_error_mean": float(np.mean(error[mask])) if count else math.nan,
                "control_final_error_median": (
                    float(np.median(error[mask])) if count else math.nan
                ),
            }
        )
    success_groups = []
    for value, label in ((1, "control_success"), (0, "control_failure")):
        subset = [row for row in rows if int(row["control_true_success"]) == value]
        success_groups.append(
            {
                "outcome": label,
                **distribution(float(row["curve_rmse_ph"]) for row in subset),
            }
        )
    overall = {
        "tasks": len(rows),
        "spearman_rmse_vs_final_error_rho": float(rho.statistic),
        "spearman_p_value": float(rho.pvalue),
        "control_success_percent": 100.0 * float(np.mean(success)),
    }
    return quintiles, success_groups, overall


def plot_rmse_distribution(rows: list[dict[str, str]], summary: list[dict], output_dir: Path) -> None:
    data = [
        finite(
            float(row["curve_rmse_ph"])
            for row in rows
            if int(row["observations"]) == observations
        )
        for observations in OBSERVATIONS
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    parts = ax.violinplot(data, positions=np.arange(len(OBSERVATIONS)), widths=0.82, showextrema=False)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.55)
    for position, values, color in zip(range(len(data)), data, COLORS):
        q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        ax.vlines(position, q05, q95, color="#222222", linewidth=1.1)
        ax.vlines(position, q25, q75, color="#222222", linewidth=5.0)
        ax.scatter(position, median, color="white", edgecolor="#222222", s=38, zorder=4)
        ax.scatter(position, np.mean(values), color=color, edgecolor="white", marker="D", s=34, zorder=4)
    ax.set_xticks(range(len(OBSERVATIONS)), OBSERVATIONS)
    ax.set_xlabel("PF observations")
    ax.set_ylabel("Full-curve RMSE (pH)")
    ax.set_title("New PF fit-error distribution across repeated observations")
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    ax.text(
        0.99,
        0.98,
        "white circle: median\ncolored diamond: mean\nthick line: IQR\nthin line: 5th-95th percentile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "pf_rmse_distribution_by_observation.png", dpi=260)
    fig.savefig(output_dir / "pf_rmse_distribution_by_observation.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    thresholds = [0.25, 0.50, 1.00, 2.00]
    threshold_colors = ["#2C6EAA", "#4C956C", "#D68C45", "#9A4F7A"]
    for threshold, color in zip(thresholds, threshold_colors):
        percentages = [row[f"within_{str(threshold).replace('.', 'p')}_percent"] for row in summary]
        ax.plot(OBSERVATIONS, percentages, marker="o", linewidth=2.0, color=color, label=f"RMSE <= {threshold:.2f} pH")
    ax.set_xlabel("PF observations")
    ax.set_ylabel("Tasks meeting threshold (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fraction of tasks achieving full-curve fit thresholds")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "pf_rmse_threshold_attainment.png", dpi=260)
    fig.savefig(output_dir / "pf_rmse_threshold_attainment.svg")
    plt.close(fig)


def plot_paired_changes(changes: list[dict], output_dir: Path) -> None:
    labels = [f"{row['start_observations']}->{row['end_observations']}" for row in changes]
    means = np.asarray([float(row["mean_change_after_minus_before"]) for row in changes])
    medians = np.asarray([float(row["change_median"]) for row in changes])
    improved = np.asarray([float(row["improved_percent"]) for row in changes])
    positions = np.arange(len(changes))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = ["#3678A8" if value < 0 else "#C3473C" for value in means]
    axes[0].bar(positions, means, color=colors, width=0.72, label="mean change")
    axes[0].scatter(positions, medians, color="#111111", marker="D", s=28, label="median change")
    axes[0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_xticks(positions, labels, rotation=25)
    axes[0].set_ylabel("RMSE change (after - before, pH)")
    axes[0].set_title("Strictly paired within-task fit changes")
    axes[0].grid(axis="y", alpha=0.2, linestyle=":")
    axes[0].legend(frameon=False)
    axes[1].bar(positions, improved, color="#4C956C", width=0.72)
    axes[1].axhline(50.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(positions, labels, rotation=25)
    axes[1].set_ylabel("Tasks with lower RMSE after update (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("How often an additional observation helps")
    axes[1].grid(axis="y", alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_dir / "pf_paired_rmse_changes.png", dpi=260)
    fig.savefig(output_dir / "pf_paired_rmse_changes.svg")
    plt.close(fig)


def plot_parameter_distributions(summary: list[dict], output_dir: Path) -> None:
    by_metric = defaultdict(dict)
    for row in summary:
        by_metric[row["metric"]][int(row["observations"])] = row
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    panels = [
        ("concentration_relative_error_percent", "Concentration relative error (%)", True),
        ("pair_count_correct", "Pair-count accuracy (%)", False),
        ("pka_mae_if_k_correct", "pKa MAE when K is correct", False),
        ("curve_correlation", "Full-curve correlation", False),
    ]
    for ax, (metric, ylabel, log_scale) in zip(axes.flat, panels):
        rows = [by_metric[metric][observations] for observations in OBSERVATIONS]
        if metric == "pair_count_correct":
            values = [float(row["accuracy_percent"]) for row in rows]
            ax.plot(OBSERVATIONS, values, color="#355F8A", marker="o", linewidth=2.0)
            ax.set_ylim(0, 100)
        else:
            median = np.asarray([float(row["median"]) for row in rows])
            q1 = np.asarray([float(row["q1"]) for row in rows])
            q3 = np.asarray([float(row["q3"]) for row in rows])
            p05 = np.asarray([float(row["p05"]) for row in rows])
            p95 = np.asarray([float(row["p95"]) for row in rows])
            ax.fill_between(OBSERVATIONS, p05, p95, color="#9EC1D4", alpha=0.23, label="5th-95th")
            ax.fill_between(OBSERVATIONS, q1, q3, color="#4F8CAA", alpha=0.32, label="IQR")
            ax.plot(OBSERVATIONS, median, color="#244F6A", marker="o", linewidth=2.0, label="median")
            if log_scale:
                ax.set_yscale("log")
        ax.set_xlabel("PF observations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linestyle=":")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Posterior-accuracy distributions, not only means", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pf_parameter_distribution_by_observation.png", dpi=260)
    fig.savefig(output_dir / "pf_parameter_distribution_by_observation.svg")
    plt.close(fig)


def plot_subgroups(rows: list[dict[str, str]], output_dir: Path) -> None:
    panels = [
        ("direction", ["acid", "base"], "Direction"),
        ("difficulty", ["near", "medium", "far"], "Target distance"),
        ("true_pair_count", ["1", "2", "3"], "True pair count K"),
        ("pka_family", ["single", "separated", "overlapping"], "pKa family"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    for ax, (field, levels, title) in zip(axes.flat, panels):
        available_levels = [level for level in levels if any(row[field] == level for row in rows)]
        data = [
            finite(float(row["curve_rmse_ph"]) for row in rows if row[field] == level)
            for level in available_levels
        ]
        box = ax.boxplot(
            data,
            tick_labels=available_levels,
            showfliers=False,
            patch_artist=True,
            widths=0.62,
        )
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        ax.set_title(title)
        ax.set_ylabel("Natural-end full-curve RMSE (pH)")
        ax.grid(axis="y", alpha=0.2, linestyle=":")
    fig.suptitle("Where the new PF curve fit is easier or harder", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pf_natural_end_rmse_subgroups.png", dpi=260)
    fig.savefig(output_dir / "pf_natural_end_rmse_subgroups.svg")
    plt.close(fig)


def plot_control_association(
    rows: list[dict[str, str]], quintiles: list[dict], output_dir: Path
) -> None:
    rmse = np.asarray([float(row["curve_rmse_ph"]) for row in rows], dtype=float)
    error = np.asarray([float(row["control_final_true_error"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].hexbin(rmse, error, gridsize=35, mincnt=1, cmap="viridis", bins="log")
    axes[0].axhline(0.10, color="#C3473C", linestyle="--", linewidth=1.3, label="success tolerance")
    axes[0].set_xlabel("Natural-end full-curve RMSE (pH)")
    axes[0].set_ylabel("Final control error (pH)")
    axes[0].set_title("Global curve fit versus local control endpoint")
    axes[0].legend(frameon=False)
    positions = np.arange(1, 6)
    success = [float(row["control_success_percent"]) for row in quintiles]
    axes[1].bar(positions, success, color=["#355F8A", "#3E7C78", "#64A061", "#D39A36", "#A6414B"])
    axes[1].set_xticks(positions, ["Q1\nbest fit", "Q2", "Q3", "Q4", "Q5\nworst fit"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Control success (%)")
    axes[1].set_title("Control success by curve-RMSE quintile")
    axes[1].grid(axis="y", alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_dir / "pf_fit_vs_control_association.png", dpi=260)
    fig.savefig(output_dir / "pf_fit_vs_control_association.svg")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _write_report_legacy_mojibake(
    output_dir: Path,
    rmse_summary: list[dict],
    changes: list[dict],
    parameter_summary: list[dict],
    subgroups: list[dict],
    association: dict,
) -> None:
    first = next(row for row in rmse_summary if int(row["observations"]) == 0)
    one = next(row for row in rmse_summary if int(row["observations"]) == 1)
    five = next(row for row in rmse_summary if int(row["observations"]) == 5)
    final = next(row for row in rmse_summary if int(row["observations"]) == 12)
    total_change = next(
        row for row in changes if int(row["start_observations"]) == 0 and int(row["end_observations"]) == 12
    )
    concentration_0 = next(
        row
        for row in parameter_summary
        if row["metric"] == "concentration_relative_error_percent" and int(row["observations"]) == 0
    )
    concentration_12 = next(
        row
        for row in parameter_summary
        if row["metric"] == "concentration_relative_error_percent" and int(row["observations"]) == 12
    )
    pka_0 = next(
        row
        for row in parameter_summary
        if row["metric"] == "pka_mae_if_k_correct" and int(row["observations"]) == 0
    )
    pka_12 = next(
        row
        for row in parameter_summary
        if row["metric"] == "pka_mae_if_k_correct" and int(row["observations"]) == 12
    )

    def subgroup(field: str, level: str) -> dict | None:
        return next(
            (
                row
                for row in subgroups
                if row["subgroup_field"] == field and row["subgroup_level"] == level
            ),
            None,
        )

    lines = [
        "# 新 PF 拟合误差统计分布与随实验步数变化",
        "",
        "本分析直接复用正式后验诊断中的 5 个独立种子、每种子 300 个任务、每任务 1000 个粒子。固定观测次数 0、1、2、3、5、8、12 使用同一批 1500 个任务，因此步数间变化是严格任务内配对，而不是不同任务均值的松散比较。",
        "",
        "## 1. RMSE 分布随观测次数的变化",
        "",
        "| 观测次数 | 均值 +/- SD | 中位数 [IQR] | 5%-95% | RMSE <= 0.5 (%) | RMSE <= 1.0 (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rmse_summary:
        lines.append(
            f"| {row['observations']} | {fmt(row['mean'], 4)} +/- {fmt(row['sd'], 4)} | "
            f"{fmt(row['median'], 4)} [{fmt(row['q1'], 4)}, {fmt(row['q3'], 4)}] | "
            f"{fmt(row['p05'], 4)}-{fmt(row['p95'], 4)} | "
            f"{fmt(row['within_0p5_percent'], 2)} | {fmt(row['within_1p0_percent'], 2)} |"
        )
    lines.extend(
        [
            "",
            f"均值 RMSE 从先验的 {fmt(first['mean'], 4)} 降至 1 次观测的 {fmt(one['mean'], 4)}，"
            f"到 5 次为 {fmt(five['mean'], 4)}，12 次为 {fmt(final['mean'], 4)} pH。"
            f"中位数从 {fmt(first['median'], 4)} 降至 {fmt(final['median'], 4)}。最明显的总体改善发生在第一次观测；正式标准档在约 5 次后进入明显平台期。",
            "",
            f"从 0 到 12 次的严格任务内配对差为 {fmt(total_change['mean_change_after_minus_before'], 4)} pH（负值表示改善），"
            f"中位差 {fmt(total_change['change_median'], 4)}；{fmt(total_change['improved_percent'], 1)}% 的任务改善，"
            f"配对 Wilcoxon p={float(total_change['p_value']):.3g}。因此总体分布显著改善，但不能声称每个任务都单调收敛，约 40% 的任务最终 RMSE 反而更高。",
            "",
            "## 2. 后验参数分布",
            "",
            f"浓度相对误差的中位数从 {fmt(concentration_0['median'], 2)}% 降至 {fmt(concentration_12['median'], 2)}%，"
            f"但 12 次观测时 IQR 仍为 {fmt(concentration_12['q1'], 2)}%-{fmt(concentration_12['q3'], 2)}%，"
            "分布右尾很长，均值容易被少量大误差任务拉高。",
            "",
            f"仅在 K 预测正确时，pKa MAE 中位数从 {fmt(pka_0['median'], 3)} 降至 {fmt(pka_12['median'], 3)}；"
            "与此同时 K 准确率仅从约 42% 增至约 46%。因此 pKa 条件误差改善不能脱离 K 准确率单独解释。",
            "",
            "## 3. 自然实验结束时的困难子群",
            "",
            f"酸向任务 RMSE 为 {fmt(subgroup('direction', 'acid')['mean'], 3)} +/- {fmt(subgroup('direction', 'acid')['sd'], 3)}，"
            f"明显高于碱向任务的 {fmt(subgroup('direction', 'base')['mean'], 3)} +/- {fmt(subgroup('direction', 'base')['sd'], 3)}。",
            f"near/medium/far 任务均值分别为 {fmt(subgroup('difficulty', 'near')['mean'], 3)}、"
            f"{fmt(subgroup('difficulty', 'medium')['mean'], 3)}、{fmt(subgroup('difficulty', 'far')['mean'], 3)} pH。"
            "较远目标提供了更大的曲线扰动和更多辨识信息，因此全曲线拟合反而更容易。",
            f"重叠 pKa 任务 RMSE 为 {fmt(subgroup('pka_family', 'overlapping')['mean'], 3)} pH，"
            f"高于 separated 的 {fmt(subgroup('pka_family', 'separated')['mean'], 3)} 和单质子 single 的 {fmt(subgroup('pka_family', 'single')['mean'], 3)}。",
            "",
            "## 4. 全曲线拟合与控制是否等价",
            "",
            f"自然结束点的全曲线 RMSE 与最终控制误差仅呈弱正相关：Spearman rho={fmt(association['spearman_rmse_vs_final_error_rho'], 3)}，"
            f"p={float(association['spearman_p_value']):.3g}。统计显著主要来自 1500 个任务的样本量，效应强度较弱。",
            "",
            "最低 RMSE 五分位的控制成功率并不是最高。原因是全曲线 RMSE 评价 -100 至 +100 mL 的全局形状，而控制只要求局部达到目标 pH；平坦或局部信息不足的任务可以有较低全局 RMSE，却仍较难精确停止。",
            "",
            "## 5. 可用于论文的结论",
            "",
            "1. 新 PF 的拟合误差在总体分布上随观测显著下降，主要收益集中在前 1-5 次观测，之后边际收益很小。",
            "2. 分布很宽且任务内并非单调改善，因此只报告均值会掩盖约 40% 任务没有净改善的事实。",
            "3. 酸向、near 目标和重叠 pKa 是当前 PF 拟合的主要困难区域，可作为下一轮模型改进和定向采样的重点。",
            "4. 控制成功、全曲线拟合和参数恢复是三个不同终点，必须分别报告，不能互相替代。",
        ]
    )
    (output_dir / "PF_FIT_DISTRIBUTION_ANALYSIS_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_report(
    output_dir: Path,
    rmse_summary: list[dict],
    changes: list[dict],
    seed_change_summary: list[dict],
    parameter_summary: list[dict],
    subgroups: list[dict],
    association: dict,
    benchmark_seed_count: int,
    tasks_per_seed: int,
    particle_count: int,
) -> None:
    final_observations = OBSERVATIONS[-1]
    first = next(row for row in rmse_summary if int(row["observations"]) == 0)
    one = next(row for row in rmse_summary if int(row["observations"]) == 1)
    intermediate_observations = 5 if 5 in OBSERVATIONS else OBSERVATIONS[-1]
    intermediate = next(
        row for row in rmse_summary if int(row["observations"]) == intermediate_observations
    )
    final = next(
        row for row in rmse_summary if int(row["observations"]) == final_observations
    )
    total_change = next(
        row
        for row in changes
        if int(row["start_observations"]) == 0
        and int(row["end_observations"]) == final_observations
    )
    seed_total_change = next(
        row
        for row in seed_change_summary
        if int(row["start_observations"]) == 0
        and int(row["end_observations"]) == final_observations
    )

    def parameter(metric: str, observations: int) -> dict:
        return next(
            row
            for row in parameter_summary
            if row["metric"] == metric and int(row["observations"]) == observations
        )

    def subgroup(field: str, level: str) -> dict | None:
        return next(
            (
                row
                for row in subgroups
                if row["subgroup_field"] == field and row["subgroup_level"] == level
            ),
            None,
        )

    concentration_0 = parameter("concentration_relative_error_percent", 0)
    concentration_final = parameter(
        "concentration_relative_error_percent", final_observations
    )
    pka_0 = parameter("pka_mae_if_k_correct", 0)
    pka_final = parameter("pka_mae_if_k_correct", final_observations)
    pair_count_0 = parameter("pair_count_correct", 0)
    pair_count_final = parameter("pair_count_correct", final_observations)
    subgroup_lines = []
    acid = subgroup("direction", "acid")
    base = subgroup("direction", "base")
    if acid is not None and base is not None:
        subgroup_lines.append(
            f"酸向任务 RMSE 为 {fmt(acid['mean'], 3)} +/- {fmt(acid['sd'], 3)}，"
            f"碱向任务为 {fmt(base['mean'], 3)} +/- {fmt(base['sd'], 3)}。"
        )
    difficulty_rows = [
        (level, subgroup("difficulty", level)) for level in ("near", "medium", "far")
    ]
    difficulty_rows = [(level, row) for level, row in difficulty_rows if row is not None]
    if difficulty_rows:
        subgroup_lines.append(
            "目标距离子群均值为 "
            + "、".join(f"{level} {fmt(row['mean'], 3)}" for level, row in difficulty_rows)
            + " pH。较远目标通常提供更大的曲线扰动和更多辨识信息。"
        )
    family_rows = [
        (level, subgroup("pka_family", level))
        for level in ("single", "separated", "overlapping")
    ]
    family_rows = [(level, row) for level, row in family_rows if row is not None]
    if family_rows:
        subgroup_lines.append(
            "pKa family 子群均值为 "
            + "、".join(f"{level} {fmt(row['mean'], 3)}" for level, row in family_rows)
            + " pH。"
        )

    lines = [
        "# 新 PF 拟合误差统计分布与随实验步数变化",
        "",
        f"本分析直接复用后验诊断中的 {benchmark_seed_count} 个独立随机种子、每种子 {tasks_per_seed} 个任务、每任务 {particle_count} 个粒子。固定观测次数 "
        f"{','.join(str(value) for value in OBSERVATIONS)} 使用同一批 {benchmark_seed_count * tasks_per_seed} 个任务，因此步数间变化是严格任务内配对，而不是不同任务均值的松散比较。",
        "",
        "## 1. RMSE 分布随观测次数的变化",
        "",
        "| 观测次数 | 均值 +/- SD | 中位数 [IQR] | 5%-95% | RMSE <= 0.5 (%) | RMSE <= 1.0 (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rmse_summary:
        lines.append(
            f"| {row['observations']} | {fmt(row['mean'], 4)} +/- {fmt(row['sd'], 4)} | "
            f"{fmt(row['median'], 4)} [{fmt(row['q1'], 4)}, {fmt(row['q3'], 4)}] | "
            f"{fmt(row['p05'], 4)}-{fmt(row['p95'], 4)} | "
            f"{fmt(row['within_0p5_percent'], 2)} | {fmt(row['within_1p0_percent'], 2)} |"
        )
    lines.extend(
        [
            "",
            f"均值 RMSE 从先验的 {fmt(first['mean'], 4)} 降至 1 次观测的 {fmt(one['mean'], 4)}，"
            f"到 {intermediate_observations} 次为 {fmt(intermediate['mean'], 4)}，{final_observations} 次为 {fmt(final['mean'], 4)} pH。"
            f"中位数从 {fmt(first['median'], 4)} 降至 {fmt(final['median'], 4)}。最明显的总体改善发生在第一次观测，5 次以后进入明显平台期。",
            "",
            f"从 0 到 {final_observations} 次的严格任务内配对差为 {fmt(total_change['mean_change_after_minus_before'], 4)} pH（负值表示改善），"
            f"中位差 {fmt(total_change['change_median'], 4)}，{fmt(total_change['improved_percent'], 1)}% 的任务改善，"
            f"任务层面配对 Wilcoxon p={float(total_change['p_value']):.3g}。因此总体分布显著改善，但不能声称每个任务都单调收敛，"
            f"{100.0 - float(total_change['improved_percent']):.1f}% 的任务没有获得严格的最终 RMSE 改善。",
            "",
            f"{benchmark_seed_count} 个随机种子的 0 到 {final_observations} 次均值变化中，"
            f"{int(seed_total_change['seeds_with_lower_mean_rmse'])} 个为负；种子均值变化为 {fmt(seed_total_change['mean_of_seed_mean_changes'], 4)} +/- "
            f"{fmt(seed_total_change['seed_change_sd'], 4)} pH，跨种子 t 区间为 [{fmt(seed_total_change['seed_t95_ci_low'], 4)}, "
            f"{fmt(seed_total_change['seed_t95_ci_high'], 4)}]，单侧精确符号翻转 p={float(seed_total_change['exact_one_sided_seed_sign_flip_p']):.5f}。"
            "任务层面检验回答同一模拟任务是否改善，种子层面检验回答改善方向能否跨独立基准种子复现，两者不混用。",
            "",
            "## 2. 后验参数分布",
            "",
            f"浓度相对误差的中位数从 {fmt(concentration_0['median'], 2)}% 降至 {fmt(concentration_final['median'], 2)}%，"
            f"但 {final_observations} 次观测时 IQR 仍为 {fmt(concentration_final['q1'], 2)}%-{fmt(concentration_final['q3'], 2)}%。"
            "分布右尾很长，均值容易被少量大误差任务拉高。",
            "",
            f"仅在 K 预测正确时，pKa MAE 中位数从 {fmt(pka_0['median'], 3)} 降至 {fmt(pka_final['median'], 3)}。"
            f"与此同时 K 准确率仅从 {fmt(pair_count_0['accuracy_percent'], 2)}% 增至 {fmt(pair_count_final['accuracy_percent'], 2)}%。"
            "因此 pKa 条件误差改善不能脱离 K 准确率单独解释。",
            "",
            "## 3. 自然实验结束时的困难子群",
            "",
            *subgroup_lines,
            "",
            "## 4. 全曲线拟合与控制是否等价",
            "",
            f"自然结束点的全曲线 RMSE 与最终控制误差仅呈弱正相关：Spearman rho={fmt(association['spearman_rmse_vs_final_error_rho'], 3)}，"
            f"p={float(association['spearman_p_value']):.3g}。统计显著性需结合 {int(association['tasks'])} 个任务的样本量解释，效应强度较弱。",
            "",
            "最佳 RMSE 五分位的控制成功率并不是最高。原因是全曲线 RMSE 评价 -100 至 +100 mL 的全局形状，而控制只要求局部达到目标 pH；平坦或局部信息不足的任务可以有较低全局 RMSE，却仍较难精确停止。该结果是任务难度混杂下的关联，不应作因果解释。",
            "",
            "## 5. 可用于论文的结论",
            "",
            "1. 新 PF 的拟合误差在总体分布上随观测显著下降，主要收益集中在前 1-5 次观测，之后边际收益很小。",
            f"2. {benchmark_seed_count} 个独立种子中有 {int(seed_total_change['seeds_with_lower_mean_rmse'])} 个平均 RMSE 改善，但任务分布很宽且任务内并非单调改善，只报告均值会掩盖部分任务没有净改善的事实。",
            "3. 酸向、near 目标和重叠 pKa 是当前 PF 拟合的主要困难区域，可作为下一轮模型改进和定向采样的重点。",
            "4. 控制成功、全曲线拟合和参数恢复是三个不同终点，必须分别报告，不能互相替代。",
        ]
    )
    (output_dir / "PF_FIT_DISTRIBUTION_ANALYSIS_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributional statistics for new-PF curve fitting")
    parser.add_argument("--posterior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv(args.posterior_dir / "all_posterior_task_results.csv")
    fixed = fixed_rows(all_rows)
    natural = natural_rows(all_rows)
    global OBSERVATIONS, TRANSITIONS, COLORS
    OBSERVATIONS = sorted({int(row["observations"]) for row in fixed})
    if not OBSERVATIONS or OBSERVATIONS[0] != 0:
        raise RuntimeError(f"Fixed checkpoints must start at observation 0, got {OBSERVATIONS}")
    TRANSITIONS = list(zip(OBSERVATIONS[:-1], OBSERVATIONS[1:])) + [
        (OBSERVATIONS[0], OBSERVATIONS[-1])
    ]
    color_map = plt.get_cmap("viridis")
    COLORS = [color_map(value) for value in np.linspace(0.20, 0.85, len(OBSERVATIONS))]
    seeds = sorted({int(row["benchmark_seed"]) for row in natural})
    if not seeds:
        raise RuntimeError("No natural-control-end rows were found")
    task_counts = {
        seed: len({int(row["task_id"]) for row in natural if int(row["benchmark_seed"]) == seed})
        for seed in seeds
    }
    if len(set(task_counts.values())) != 1:
        raise RuntimeError(f"Unequal natural-end task counts across seeds: {task_counts}")
    tasks_per_seed = next(iter(task_counts.values()))
    expected_natural = len(seeds) * tasks_per_seed
    expected_fixed = expected_natural * len(OBSERVATIONS)
    if len(fixed) != expected_fixed or len(natural) != expected_natural:
        raise RuntimeError(
            f"Expected {expected_fixed} fixed and {expected_natural} natural rows, "
            f"got {len(fixed)} and {len(natural)}"
        )
    for observations in OBSERVATIONS:
        checkpoint_rows = [row for row in fixed if int(row["observations"]) == observations]
        if len(checkpoint_rows) != expected_natural:
            raise RuntimeError(
                f"Checkpoint {observations} has {len(checkpoint_rows)} rows, expected {expected_natural}"
            )
    posterior_completion = json.loads(
        (args.posterior_dir / "POSTERIOR_DIAGNOSTICS_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    particle_count = int(
        posterior_completion.get("particles")
        or posterior_completion.get("config", {}).get("particles", 0)
    )

    rmse_summary, per_seed = rmse_distributions(fixed)
    changes, lookup = paired_changes(fixed)
    per_seed_changes, seed_change_summary = seed_paired_changes(lookup)
    parameter_summary = parameter_distributions(fixed)
    subgroups, subgroup_tests = subgroup_distributions(natural)
    quintiles, success_groups, association = control_association(natural)
    write_csv(args.output_dir / "rmse_distribution_by_observation.csv", rmse_summary)
    write_csv(args.output_dir / "rmse_per_seed_distribution.csv", per_seed)
    write_csv(args.output_dir / "paired_rmse_change_tests.csv", changes)
    write_csv(args.output_dir / "rmse_change_by_seed.csv", per_seed_changes)
    write_csv(args.output_dir / "rmse_change_seed_summary.csv", seed_change_summary)
    write_csv(args.output_dir / "posterior_parameter_distribution.csv", parameter_summary)
    write_csv(args.output_dir / "natural_end_rmse_subgroups.csv", subgroups)
    write_csv(args.output_dir / "natural_end_subgroup_tests.csv", subgroup_tests)
    write_csv(args.output_dir / "fit_control_rmse_quintiles.csv", quintiles)
    write_csv(args.output_dir / "fit_control_success_groups.csv", success_groups)
    (args.output_dir / "fit_control_association.json").write_text(
        json.dumps(association, indent=2), encoding="utf-8"
    )
    plot_rmse_distribution(fixed, rmse_summary, args.output_dir)
    plot_paired_changes(changes, args.output_dir)
    plot_parameter_distributions(parameter_summary, args.output_dir)
    plot_subgroups(natural, args.output_dir)
    plot_control_association(natural, quintiles, args.output_dir)
    write_report(
        args.output_dir,
        rmse_summary,
        changes,
        seed_change_summary,
        parameter_summary,
        subgroups,
        association,
        len(seeds),
        tasks_per_seed,
        particle_count,
    )
    completion = {
        "protocol_version": PROTOCOL_VERSION,
        "source_posterior_dir": str(args.posterior_dir.resolve()),
        "fixed_task_rows": len(fixed),
        "natural_end_task_rows": len(natural),
        "benchmark_seeds": seeds,
        "tasks_per_seed": tasks_per_seed,
        "particles": particle_count,
        "tasks_per_observation": expected_natural,
        "observations": OBSERVATIONS,
        "rmse_distribution_rows": len(rmse_summary),
        "per_seed_distribution_rows": len(per_seed),
        "paired_change_rows": len(changes),
        "per_seed_change_rows": len(per_seed_changes),
        "seed_change_summary_rows": len(seed_change_summary),
        "parameter_distribution_rows": len(parameter_summary),
        "subgroup_distribution_rows": len(subgroups),
        "subgroup_test_rows": len(subgroup_tests),
        "control_quintile_rows": len(quintiles),
        "control_success_group_rows": len(success_groups),
        "key_conclusions": {
            "final_observations": OBSERVATIONS[-1],
            "rmse_mean_observation_0": rmse_summary[0]["mean"],
            "rmse_mean_final_observation": rmse_summary[-1]["mean"],
            "rmse_median_observation_0": rmse_summary[0]["median"],
            "rmse_median_final_observation": rmse_summary[-1]["median"],
            "paired_improved_percent_0_to_final": next(
                row["improved_percent"]
                for row in changes
                if row["start_observations"] == 0
                and row["end_observations"] == OBSERVATIONS[-1]
            ),
            **association,
        },
    }
    (args.output_dir / "PF_FIT_DISTRIBUTION_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
