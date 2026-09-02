from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, wilcoxon

from chemistry_model import SolutionState, response_curve
from task_distribution import load_tasks


CURVE_GRID_ML = np.linspace(-100.0, 100.0, 161)
EXPECTED_STEPS = tuple(range(13))
EXAMPLE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
TRUE_COLOR = "#111111"
PF_COLOR = "#0072B2"
K_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def descriptive(values) -> dict[str, float | int]:
    array = finite(values)
    if array.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
        }
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    se = sd / math.sqrt(array.size)
    half_width = float(t.ppf(0.975, array.size - 1) * se) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def parse_fixed_rows(source: Path) -> list[dict]:
    rows = []
    for raw in read_csv(source):
        if raw.get("checkpoint_type") != "fixed_observation_count":
            continue
        row = {
            "benchmark_seed": int(raw["benchmark_seed"]),
            "task_seed": int(raw["task_seed"]),
            "task_id": int(raw["task_id"]),
            "observations": int(raw["observations"]),
            "true_pair_count": int(raw["true_pair_count"]),
            "estimated_pair_count": int(raw["estimated_pair_count"]),
            "pair_count_correct": int(raw["pair_count_correct"]),
            "true_concentration_m": float(raw["true_concentration_m"]),
            "estimated_concentration_m": float(raw["estimated_concentration_m"]),
            "concentration_relative_error_percent": float(
                raw["concentration_relative_error_percent"]
            ),
            "true_pka_json": raw["true_pka_json"],
            "estimated_pka_json": raw["estimated_pka_json"],
            "curve_rmse_ph": float(raw["curve_rmse_ph"]),
            "curve_correlation": float(raw["curve_correlation"]),
            "pka_mae_if_k_correct": (
                float(raw["pka_mae_if_k_correct"])
                if raw.get("pka_mae_if_k_correct")
                else math.nan
            ),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No fixed-observation rows in {source}")
    return rows


def validate_rows(rows: list[dict]) -> None:
    keys = [
        (row["benchmark_seed"], row["task_seed"], row["task_id"], row["observations"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate task/checkpoint rows detected")
    steps = tuple(sorted({row["observations"] for row in rows}))
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"Expected steps {EXPECTED_STEPS}, found {steps}")
    counts = {step: sum(row["observations"] == step for row in rows) for step in steps}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Unbalanced step counts: {counts}")
    task_steps: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for row in rows:
        task_steps[(row["benchmark_seed"], row["task_seed"], row["task_id"])].add(
            row["observations"]
        )
    incomplete = [key for key, value in task_steps.items() if value != set(EXPECTED_STEPS)]
    if incomplete:
        raise RuntimeError(f"Incomplete step history for {len(incomplete)} tasks")


def summarize_step(rows: list[dict], step: int) -> dict:
    subset = [row for row in rows if row["observations"] == step]
    stats = descriptive(row["curve_rmse_ph"] for row in subset)
    seed_means = []
    for seed in sorted({row["benchmark_seed"] for row in subset}):
        seed_values = [
            row["curve_rmse_ph"] for row in subset if row["benchmark_seed"] == seed
        ]
        seed_means.append(float(np.mean(seed_values)))
    seed_stats = descriptive(seed_means)
    initial_by_task = {
        (row["benchmark_seed"], row["task_seed"], row["task_id"]): row["curve_rmse_ph"]
        for row in rows
        if row["observations"] == 0
    }
    changes = []
    for row in subset:
        key = (row["benchmark_seed"], row["task_seed"], row["task_id"])
        changes.append(initial_by_task[key] - row["curve_rmse_ph"])
    improved = sum(value > 0 for value in changes)
    worsened = sum(value < 0 for value in changes)
    equal = len(changes) - improved - worsened
    return {
        "observations": step,
        "tasks": stats["n"],
        "curve_rmse_mean": stats["mean"],
        "curve_rmse_sd": stats["sd"],
        "curve_rmse_se": stats["se"],
        "curve_rmse_task_ci95_low": stats["ci95_low"],
        "curve_rmse_task_ci95_high": stats["ci95_high"],
        "curve_rmse_median": stats["median"],
        "curve_rmse_q25": stats["q25"],
        "curve_rmse_q75": stats["q75"],
        "independent_seed_means": seed_stats["n"],
        "seed_mean_rmse": seed_stats["mean"],
        "seed_mean_sd": seed_stats["sd"],
        "seed_mean_ci95_low": seed_stats["ci95_low"],
        "seed_mean_ci95_high": seed_stats["ci95_high"],
        "mean_delta_vs_step0_positive_is_improvement": float(np.mean(changes)),
        "median_delta_vs_step0_positive_is_improvement": float(np.median(changes)),
        "improved_vs_step0_tasks": improved,
        "improved_vs_step0_percent": 100.0 * improved / len(changes),
        "worsened_vs_step0_tasks": worsened,
        "worsened_vs_step0_percent": 100.0 * worsened / len(changes),
        "equal_vs_step0_tasks": equal,
    }


def overall_summary(rows: list[dict]) -> list[dict]:
    return [summarize_step(rows, step) for step in EXPECTED_STEPS]


def holm_adjust(p_values: list[float]) -> list[float]:
    result = [math.nan] * len(p_values)
    finite_indices = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_values[index])
        running = max(running, adjusted)
        result[index] = running
    return result


def paired_tests(rows: list[dict]) -> list[dict]:
    by_task: dict[tuple[int, int, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (row["benchmark_seed"], row["task_seed"], row["task_id"])
        by_task[key][row["observations"]] = row["curve_rmse_ph"]
    output = []
    for step in EXPECTED_STEPS[1:]:
        for reference, label in ((step - 1, "previous_step"), (0, "step_0")):
            before = np.asarray([values[reference] for values in by_task.values()], dtype=float)
            after = np.asarray([values[step] for values in by_task.values()], dtype=float)
            delta = before - after
            try:
                statistic, p_value = wilcoxon(
                    before,
                    after,
                    zero_method="pratt",
                    alternative="two-sided",
                    method="approx",
                )
            except ValueError:
                statistic, p_value = 0.0, 1.0
            improved = int(np.sum(delta > 0.0))
            worsened = int(np.sum(delta < 0.0))
            equal = int(delta.size - improved - worsened)
            output.append({
                "comparison_family": label,
                "reference_step": reference,
                "current_step": step,
                "paired_tasks": int(delta.size),
                "mean_rmse_delta_positive_is_improvement": float(np.mean(delta)),
                "median_rmse_delta_positive_is_improvement": float(np.median(delta)),
                "improved_tasks": improved,
                "improved_percent": 100.0 * improved / delta.size,
                "worsened_tasks": worsened,
                "worsened_percent": 100.0 * worsened / delta.size,
                "equal_tasks": equal,
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_two_sided": float(p_value),
            })
    adjusted = holm_adjust([row["wilcoxon_p_two_sided"] for row in output])
    for row, value in zip(output, adjusted):
        row["holm_adjusted_p_across_all_24_tests"] = value
        row["significant_after_holm_0p05"] = int(value < 0.05)
    return output


def k_summary(rows: list[dict]) -> list[dict]:
    output = []
    for true_k in (1, 2, 3):
        for step in EXPECTED_STEPS:
            subset = [
                row
                for row in rows
                if row["true_pair_count"] == true_k and row["observations"] == step
            ]
            rmse = descriptive(row["curve_rmse_ph"] for row in subset)
            corr = descriptive(row["curve_correlation"] for row in subset)
            concentration = descriptive(
                row["concentration_relative_error_percent"] for row in subset
            )
            pka = descriptive(row["pka_mae_if_k_correct"] for row in subset)
            correct = sum(row["pair_count_correct"] for row in subset)
            seed_means = []
            for seed in sorted({row["benchmark_seed"] for row in subset}):
                seed_means.append(float(np.mean([
                    row["curve_rmse_ph"]
                    for row in subset
                    if row["benchmark_seed"] == seed
                ])))
            seed_stats = descriptive(seed_means)
            output.append({
                "true_k": true_k,
                "observations": step,
                "tasks": len(subset),
                "curve_rmse_mean": rmse["mean"],
                "curve_rmse_sd": rmse["sd"],
                "curve_rmse_median": rmse["median"],
                "curve_rmse_q25": rmse["q25"],
                "curve_rmse_q75": rmse["q75"],
                "seed_mean_rmse": seed_stats["mean"],
                "seed_mean_ci95_low": seed_stats["ci95_low"],
                "seed_mean_ci95_high": seed_stats["ci95_high"],
                "curve_correlation_mean": corr["mean"],
                "curve_correlation_sd": corr["sd"],
                "k_correct_tasks": correct,
                "k_accuracy_percent": 100.0 * correct / len(subset),
                "concentration_relative_error_percent_mean": concentration["mean"],
                "concentration_relative_error_percent_median": concentration["median"],
                "pka_mae_if_k_correct_tasks": pka["n"],
                "pka_mae_if_k_correct_mean": pka["mean"],
                "pka_mae_if_k_correct_median": pka["median"],
            })
    return output


def task_key(row: dict) -> tuple[int, int, int]:
    return row["benchmark_seed"], row["task_seed"], row["task_id"]


def histories(rows: list[dict]) -> dict[tuple[int, int, int], list[dict]]:
    output: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        output[task_key(row)].append(row)
    for values in output.values():
        values.sort(key=lambda row: row["observations"])
    return dict(output)


def task_trend_rows(all_histories: dict[tuple[int, int, int], list[dict]]) -> list[dict]:
    output = []
    for key, history in sorted(all_histories.items()):
        values = np.asarray([row["curve_rmse_ph"] for row in history], dtype=float)
        adjacent_improvements = values[:-1] - values[1:]
        improving = int(np.sum(adjacent_improvements > 0.0))
        worsening = int(np.sum(adjacent_improvements < 0.0))
        equal = int(adjacent_improvements.size - improving - worsening)
        final_delta = float(values[0] - values[-1])
        output.append({
            "benchmark_seed": key[0],
            "task_seed": key[1],
            "task_id": key[2],
            "true_k": history[0]["true_pair_count"],
            "final_predicted_k": history[-1]["estimated_pair_count"],
            "final_k_correct": history[-1]["pair_count_correct"],
            "step0_curve_rmse_ph": values[0],
            "step12_curve_rmse_ph": values[-1],
            "rmse_delta_step0_minus_step12_positive_is_improvement": final_delta,
            "net_improved_by_step12": int(final_delta > 0.0),
            "adjacent_improving_updates": improving,
            "adjacent_worsening_updates": worsening,
            "adjacent_equal_updates": equal,
            "all_12_updates_nonincreasing": int(worsening == 0),
        })
    return output


def select_k_representatives(rows: list[dict]) -> list[dict]:
    final_rows = [row for row in rows if row["observations"] == 12]
    selected = []
    for true_k in (1, 2, 3):
        group = [row for row in final_rows if row["true_pair_count"] == true_k]
        target = float(np.median([row["curve_rmse_ph"] for row in group]))
        preferred = [row for row in group if row["pair_count_correct"] == 1]
        candidates = preferred or group
        chosen = min(candidates, key=lambda row: abs(row["curve_rmse_ph"] - target))
        selected.append({
            **chosen,
            "selection_rule": "closest_to_true_k_group_median_preferring_correct_final_k",
            "target_group_median_rmse": target,
        })
    return selected


def select_rmse_quantile_examples(rows: list[dict]) -> list[dict]:
    final_rows = sorted(
        [row for row in rows if row["observations"] == 12],
        key=lambda row: row["curve_rmse_ph"],
    )
    values = np.asarray([row["curve_rmse_ph"] for row in final_rows], dtype=float)
    selected = []
    used: set[tuple[int, int, int]] = set()
    for quantile in EXAMPLE_QUANTILES:
        target = float(np.quantile(values, quantile))
        candidates = sorted(final_rows, key=lambda row: abs(row["curve_rmse_ph"] - target))
        chosen = next(row for row in candidates if task_key(row) not in used)
        used.add(task_key(chosen))
        selected.append({
            **chosen,
            "requested_final_rmse_quantile": quantile,
            "target_quantile_rmse": target,
            "selection_rule": "closest_observed_final_rmse_to_prespecified_quantile",
        })
    return selected


def load_task_lookup(posterior_dir: Path) -> dict[tuple[int, int, int], object]:
    lookup = {}
    for path in sorted(posterior_dir.glob("seed_*_tasks.jsonl")):
        benchmark_seed = int(path.stem.split("_")[1])
        for task in load_tasks(path):
            key = benchmark_seed, task.seed, task.task_id
            if key in lookup:
                raise RuntimeError(f"Duplicate task in JSONL files: {key}")
            lookup[key] = task
    return lookup


def curves(task, row: dict) -> tuple[np.ndarray, np.ndarray]:
    initial_state = SolutionState(
        total_volume_ml=float(task.initial_volume_ml),
        base_moles=float(task.initial_base_moles),
        acid_moles=0.0,
    )
    truth = response_curve(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        initial_state,
        CURVE_GRID_ML,
    )
    fitted = response_curve(
        row["estimated_concentration_m"],
        json.loads(row["estimated_pka_json"]),
        task.initial_volume_ml,
        initial_state,
        CURVE_GRID_ML,
    )
    return truth, fitted


def plot_overall_trajectory(path: Path, summary_rows: list[dict]) -> None:
    steps = np.asarray([row["observations"] for row in summary_rows])
    mean = np.asarray([row["curve_rmse_mean"] for row in summary_rows])
    low = np.asarray([row["seed_mean_ci95_low"] for row in summary_rows])
    high = np.asarray([row["seed_mean_ci95_high"] for row in summary_rows])
    median = np.asarray([row["curve_rmse_median"] for row in summary_rows])
    q25 = np.asarray([row["curve_rmse_q25"] for row in summary_rows])
    q75 = np.asarray([row["curve_rmse_q75"] for row in summary_rows])
    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    axis.fill_between(steps, low, high, color=PF_COLOR, alpha=0.20,
                      label="95% CI across 5 seed means")
    axis.plot(steps, mean, color=PF_COLOR, linewidth=2.5, marker="o", label="Mean RMSE")
    axis.fill_between(steps, q25, q75, color="#E69F00", alpha=0.16,
                      label="Task-level IQR")
    axis.plot(steps, median, color="#D55E00", linewidth=2.0, marker="s",
              label="Median RMSE")
    axis.set_xticks(steps)
    axis.set_xlabel("Number of titration observations")
    axis.set_ylabel("Full-curve RMSE (pH)")
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_distribution(path: Path, rows: list[dict]) -> None:
    data = [
        [row["curve_rmse_ph"] for row in rows if row["observations"] == step]
        for step in EXPECTED_STEPS
    ]
    fig, axis = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    box = axis.boxplot(data, positions=EXPECTED_STEPS, widths=0.62, showfliers=False,
                       patch_artist=True, medianprops={"color": TRUE_COLOR, "linewidth": 1.5})
    for patch in box["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.45)
    means = [float(np.mean(values)) for values in data]
    axis.plot(EXPECTED_STEPS, means, color="#D55E00", linewidth=2.1, marker="o",
              label="Mean")
    axis.set_xticks(EXPECTED_STEPS)
    axis.set_xlabel("Number of titration observations")
    axis.set_ylabel("Full-curve RMSE (pH)")
    axis.grid(axis="y", alpha=0.23)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_k_trajectory(path: Path, summary_rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for true_k in (1, 2, 3):
        subset = [row for row in summary_rows if row["true_k"] == true_k]
        steps = np.asarray([row["observations"] for row in subset])
        mean = np.asarray([row["curve_rmse_mean"] for row in subset])
        low = np.asarray([row["seed_mean_ci95_low"] for row in subset])
        high = np.asarray([row["seed_mean_ci95_high"] for row in subset])
        axis.fill_between(steps, low, high, color=K_COLORS[true_k], alpha=0.12)
        axis.plot(steps, mean, color=K_COLORS[true_k], linewidth=2.3, marker="o",
                  label=f"True K={true_k}")
    axis.set_xticks(EXPECTED_STEPS)
    axis.set_xlabel("Number of titration observations")
    axis.set_ylabel("Mean full-curve RMSE (pH)")
    axis.grid(alpha=0.23)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_final_k_comparison(
    path: Path,
    representatives: list[dict],
    task_lookup: dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharey=True, constrained_layout=True)
    for axis, row in zip(axes, representatives):
        task = task_lookup[task_key(row)]
        truth, fitted = curves(task, row)
        axis.plot(CURVE_GRID_ML, truth, color=TRUE_COLOR, linewidth=2.4, label="Ground truth")
        axis.plot(CURVE_GRID_ML, fitted, color=PF_COLOR, linewidth=2.1, linestyle="--",
                  label="PF posterior")
        axis.set_title(
            f"True K={row['true_pair_count']}; predicted K={row['estimated_pair_count']}\n"
            f"Step 12 RMSE={row['curve_rmse_ph']:.3f} pH"
        )
        axis.set_xlabel("Signed titrant volume (mL)")
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("pH")
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_stepwise_history(
    path: Path,
    selected_row: dict,
    history: list[dict],
    task_lookup: dict,
    heading: str,
) -> None:
    task = task_lookup[task_key(selected_row)]
    fig, axes = plt.subplots(4, 4, figsize=(14.2, 12.2), sharex=True, sharey=True,
                             constrained_layout=True)
    flat = list(axes.flat)
    for axis, row in zip(flat, history):
        truth, fitted = curves(task, row)
        axis.plot(CURVE_GRID_ML, truth, color=TRUE_COLOR, linewidth=1.9, label="Ground truth")
        axis.plot(CURVE_GRID_ML, fitted, color=PF_COLOR, linewidth=1.7, linestyle="--",
                  label="PF posterior")
        axis.set_title(
            f"Step {row['observations']}: RMSE={row['curve_rmse_ph']:.3f}\n"
            f"K true/pred={row['true_pair_count']}/{row['estimated_pair_count']}",
            fontsize=9.5,
        )
        axis.grid(alpha=0.18)
    for axis in flat[len(history):]:
        axis.axis("off")
    for row_index in range(4):
        axes[row_index, 0].set_ylabel("pH")
    for axis in axes[-1, :]:
        axis.set_xlabel("Signed volume (mL)")
    flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(heading, fontsize=14)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def build_selection_rows(selected: list[dict], all_histories: dict, kind: str) -> list[dict]:
    output = []
    for row in selected:
        history = all_histories[task_key(row)]
        base = {
            "selection_kind": kind,
            "benchmark_seed": row["benchmark_seed"],
            "task_seed": row["task_seed"],
            "task_id": row["task_id"],
            "true_k": row["true_pair_count"],
            "final_predicted_k": row["estimated_pair_count"],
            "final_k_correct": row["pair_count_correct"],
            "final_curve_rmse_ph": row["curve_rmse_ph"],
            "selection_rule": row["selection_rule"],
        }
        if "requested_final_rmse_quantile" in row:
            base["requested_final_rmse_quantile"] = row["requested_final_rmse_quantile"]
            base["selection_target_rmse"] = row["target_quantile_rmse"]
        else:
            base["requested_final_rmse_quantile"] = ""
            base["selection_target_rmse"] = row["target_group_median_rmse"]
        for step_row in history:
            base[f"step_{step_row['observations']}_rmse"] = step_row["curve_rmse_ph"]
        output.append(base)
    return output


def fmt_p(value: float) -> str:
    return f"{value:.3g}" if value >= 0.001 else f"{value:.2e}"


def write_report(
    path: Path,
    overall: list[dict],
    tests: list[dict],
    k_rows: list[dict],
    k_selected: list[dict],
    quantile_selected: list[dict],
    trend_rows: list[dict],
) -> None:
    first = overall[0]
    final = overall[-1]
    relative = 100.0 * (first["curve_rmse_mean"] - final["curve_rmse_mean"]) / first["curve_rmse_mean"]
    adjacent = [row for row in tests if row["comparison_family"] == "previous_step"]
    significant_better = [
        row for row in adjacent
        if row["significant_after_holm_0p05"] and row["mean_rmse_delta_positive_is_improvement"] > 0
    ]
    adjacent_mean_increases = [
        row for row in adjacent if row["mean_rmse_delta_positive_is_improvement"] < 0
    ]
    total_tasks = int(first["tasks"])
    monotonic_tasks = sum(row["all_12_updates_nonincreasing"] for row in trend_rows)
    improving_update_counts = np.asarray(
        [row["adjacent_improving_updates"] for row in trend_rows], dtype=float
    )
    lines = [
        "# 不同 K 与逐步后验滴定曲线分析",
        "",
        "## 数据与方法",
        "",
        "本分析使用 5 个独立随机种子（101、202、303、404、555），每个种子 300 个任务，共 1500 个任务。每个任务均保存从先验 step 0 到 step 12 的固定观测数后验，因此所有相邻步比较均为同一任务内的配对比较。完整滴定曲线统一在 -100 至 +100 mL 的 161 点网格上计算 RMSE。",
        "",
        "均值的不确定性同时保留任务级标准差和 5 个独立种子均值的 t 型 95% CI；主轨迹图采用后者。相邻步及 step 0 对照采用双侧配对 Wilcoxon 检验，24 个检验统一进行 Holm 校正。",
        "",
        "## 主要结论",
        "",
        f"- 全体任务平均完整曲线 RMSE 从 step 0 的 **{first['curve_rmse_mean']:.4f} pH** 降至 step 12 的 **{final['curve_rmse_mean']:.4f} pH**，相对降低 **{relative:.2f}%**。step 12 的 5 种子均值 95% CI 为 {final['seed_mean_ci95_low']:.4f}–{final['seed_mean_ci95_high']:.4f} pH。",
        f"- step 12 相比先验有 {final['improved_vs_step0_tasks']}/{total_tasks} 个任务（{final['improved_vs_step0_percent']:.2f}%）RMSE 更低，仍有 {final['worsened_vs_step0_tasks']}/{total_tasks} 个任务（{final['worsened_vs_step0_percent']:.2f}%）更高。",
        f"- 只有 {monotonic_tasks}/{total_tasks} 个任务（{100.0 * monotonic_tasks / total_tasks:.2f}%）在全部 12 次更新中从不反弹。每个任务 12 次相邻更新中，RMSE 下降次数的中位数为 {np.median(improving_update_counts):.0f} 次，四分位区间为 {np.percentile(improving_update_counts, 25):.0f}–{np.percentile(improving_update_counts, 75):.0f} 次。",
        f"- 12 个相邻步比较中，有 {len(significant_better)} 个表现为均值下降且 Holm 校正后显著。出现总体均值暂时上升的相邻更新有 {len(adjacent_mean_increases)} 个：{', '.join('step ' + str(row['reference_step']) + '→' + str(row['current_step']) for row in adjacent_mean_increases) if adjacent_mean_increases else '无'}。",
        "- 因此可以说总体分布随观测增加而改善并逐渐进入平台，但不能声称每个任务的每一步 RMSE 都严格下降；单步观测会改变模型阶数和参数权重，个别任务可出现暂时反弹。",
        "",
        "## 按真实 K 分层",
        "",
        "| 真实 K | 任务数 | step 0 RMSE | step 12 RMSE | 相对变化 | step 12 K 准确率 | step 12 曲线相关系数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for true_k in (1, 2, 3):
        start = next(row for row in k_rows if row["true_k"] == true_k and row["observations"] == 0)
        end = next(row for row in k_rows if row["true_k"] == true_k and row["observations"] == 12)
        change = 100.0 * (start["curve_rmse_mean"] - end["curve_rmse_mean"]) / start["curve_rmse_mean"]
        lines.append(
            f"| {true_k} | {end['tasks']} | {start['curve_rmse_mean']:.4f} | {end['curve_rmse_mean']:.4f} | {change:+.2f}% | {end['k_accuracy_percent']:.2f}% | {end['curve_correlation_mean']:.4f} |"
        )
    lines.extend([
        "",
        "曲线拟合与 K 分类是两个不同问题。即使最终 K 判断错误，后验曲线仍可能在有限剂量区间内接近真值；因此 K 准确率不能代替完整曲线 RMSE，反过来也不能由较低曲线 RMSE 推断分子模型阶数已经准确恢复。",
        "",
        "## 代表任务与误差分位案例",
        "",
        "K=1/2/3 的代表任务按各 K 组 step 12 RMSE 中位数附近自动选择，并优先选取最终 K 判断正确的任务。另按全体 step 12 RMSE 的预设 5%、25%、50%、75%、95% 分位选择五个案例，不根据图形外观人工挑选。每个案例均展示 step 0–12 的完整真实曲线与 PF 后验曲线。",
        "",
        "### K 代表任务",
        "",
    ])
    for row in k_selected:
        lines.append(
            f"- K={row['true_pair_count']}：seed={row['benchmark_seed']}，task={row['task_id']}，最终预测 K={row['estimated_pair_count']}，最终 RMSE={row['curve_rmse_ph']:.4f} pH。"
        )
    lines.extend(["", "### 最终 RMSE 分位案例", ""])
    for row in quantile_selected:
        lines.append(
            f"- {100 * row['requested_final_rmse_quantile']:.0f}% 分位：seed={row['benchmark_seed']}，task={row['task_id']}，真实/预测 K={row['true_pair_count']}/{row['estimated_pair_count']}，最终 RMSE={row['curve_rmse_ph']:.4f} pH。"
        )
    lines.extend([
        "",
        "## 统计边界",
        "",
        "本轮保存的是每一步的后验点估计，而不是每个任务每一步的完整联合粒子云。报告中的 95% CI 是跨独立随机种子均值的不确定性，不是单条滴定曲线的贝叶斯可信带。若用边际 pKa 标准差拼接曲线带，会忽略浓度与 pKa、各 pKa 之间的联合相关性，因此本分析没有绘制可能误导的逐曲线 95% 后验带。",
        "",
        "所有数值可在 `rmse_by_step_overall.csv`、`paired_rmse_step_tests.csv`、`rmse_by_step_and_true_k.csv`、`task_level_rmse_trends.csv` 以及两张选择表中逐项复核。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze stepwise PF curve RMSE overall, by true K, and for fixed examples"
    )
    parser.add_argument("--posterior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.posterior_dir / "all_posterior_task_results.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)

    rows = parse_fixed_rows(source)
    validate_rows(rows)
    all_histories = histories(rows)
    trend_rows = task_trend_rows(all_histories)
    task_lookup = load_task_lookup(args.posterior_dir)
    missing_tasks = sorted(set(all_histories) - set(task_lookup))
    if missing_tasks:
        raise RuntimeError(f"Missing {len(missing_tasks)} exact tasks from JSONL files")

    overall = overall_summary(rows)
    tests = paired_tests(rows)
    by_k = k_summary(rows)
    k_selected = select_k_representatives(rows)
    quantile_selected = select_rmse_quantile_examples(rows)

    write_csv(args.output_dir / "rmse_by_step_overall.csv", overall)
    write_csv(args.output_dir / "paired_rmse_step_tests.csv", tests)
    write_csv(args.output_dir / "rmse_by_step_and_true_k.csv", by_k)
    write_csv(args.output_dir / "task_level_rmse_trends.csv", trend_rows)
    write_csv(
        args.output_dir / "selected_k_representative_tasks.csv",
        build_selection_rows(k_selected, all_histories, "true_k_group_median"),
    )
    write_csv(
        args.output_dir / "selected_final_rmse_examples.csv",
        build_selection_rows(quantile_selected, all_histories, "final_rmse_quantile"),
    )

    plot_overall_trajectory(figures / "rmse_trajectory_overall.png", overall)
    plot_distribution(figures / "rmse_distribution_by_step.png", rows)
    plot_k_trajectory(figures / "rmse_trajectory_by_true_k.png", by_k)
    plot_final_k_comparison(
        figures / "final_curve_comparison_by_true_k.png", k_selected, task_lookup
    )
    for row in k_selected:
        plot_stepwise_history(
            figures / f"k{row['true_pair_count']}_representative_stepwise_curves.png",
            row,
            all_histories[task_key(row)],
            task_lookup,
            f"Representative true K={row['true_pair_count']} task: seed {row['benchmark_seed']}, task {row['task_id']}",
        )
    labels = ("q05", "q25", "q50", "q75", "q95")
    for label, row in zip(labels, quantile_selected):
        plot_stepwise_history(
            figures / f"final_rmse_{label}_stepwise_curves.png",
            row,
            all_histories[task_key(row)],
            task_lookup,
            f"Final RMSE {100 * row['requested_final_rmse_quantile']:.0f}th-percentile example: seed {row['benchmark_seed']}, task {row['task_id']}",
        )

    write_report(
        args.output_dir / "STEPWISE_K_CURVE_ANALYSIS_CN.md",
        overall,
        tests,
        by_k,
        k_selected,
        quantile_selected,
        trend_rows,
    )
    completion = {
        "status": "complete",
        "source": str(source.resolve()),
        "fixed_observation_rows": len(rows),
        "unique_tasks": len(all_histories),
        "steps": list(EXPECTED_STEPS),
        "independent_seeds": sorted({row["benchmark_seed"] for row in rows}),
        "curve_grid_ml": CURVE_GRID_ML.tolist(),
        "selection_quantiles": list(EXAMPLE_QUANTILES),
    }
    (args.output_dir / "STEPWISE_K_ANALYSIS_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
