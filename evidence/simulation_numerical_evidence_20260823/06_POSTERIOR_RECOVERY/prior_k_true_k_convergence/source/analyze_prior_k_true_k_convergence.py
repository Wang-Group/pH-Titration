from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from chemistry_model import SolutionState, response_curve, solve_ph_particles
from posterior_diagnostics import initialize_controller
from posterior_prior_k_grid import apply_k_prior
from task_distribution import load_tasks


CURVE_GRID_ML = np.linspace(-100.0, 100.0, 161)
STEPS = tuple(range(13))
CURVE_STEPS = (0, 4, 8, 12)
TRUE_COLOR = "#111111"
PF_COLOR = "#0072B2"
STEP_COLORS = {0: "#CC79A7", 4: "#E69F00", 8: "#009E73", 12: "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_rows(path: Path) -> list[dict]:
    output = []
    for raw in read_csv(path):
        output.append({
            "prior_k": int(raw["prior_k"]),
            "benchmark_seed": int(raw["benchmark_seed"]),
            "task_seed": int(raw["task_seed"]),
            "task_id": int(raw["task_id"]),
            "observations": int(raw["observations"]),
            "true_k": int(raw["true_pair_count"]),
            "estimated_k": int(raw["estimated_pair_count"]),
            "k_correct": int(raw["pair_count_correct"]),
            "pair_probabilities": np.asarray(
                json.loads(raw["pair_probabilities_json"]), dtype=float
            ),
            "estimated_concentration_m": float(raw["estimated_concentration_m"]),
            "estimated_pka": np.asarray(json.loads(raw["estimated_pka_json"]), dtype=float),
            "curve_rmse_ph": float(raw["curve_rmse_ph"]),
            "curve_correlation": float(raw["curve_correlation"]),
        })
    return output


def task_key(row: dict) -> tuple[int, int, int, int]:
    return row["prior_k"], row["benchmark_seed"], row["task_seed"], row["task_id"]


def validate(rows: list[dict]) -> None:
    keys = [(task_key(row), row["observations"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate prior/task/step keys")
    histories = defaultdict(set)
    for row in rows:
        histories[task_key(row)].add(row["observations"])
    incomplete = [key for key, values in histories.items() if values != set(STEPS)]
    if incomplete:
        raise RuntimeError(f"Incomplete histories: {len(incomplete)}")
    expected = 3 * 5 * 300 * len(STEPS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    at_zero = [row for row in rows if row["observations"] == 0]
    mismatched = [row for row in at_zero if row["estimated_k"] != row["prior_k"]]
    if mismatched:
        raise RuntimeError(f"Step-0 K does not match declared prior for {len(mismatched)} rows")


def make_histories(rows: list[dict]) -> dict[tuple[int, int, int, int], list[dict]]:
    output = defaultdict(list)
    for row in rows:
        output[task_key(row)].append(row)
    for history in output.values():
        history.sort(key=lambda row: row["observations"])
    return dict(output)


def mean_ci95(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan,
                "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    half = float(t.ppf(0.975, array.size - 1) * sd / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {"n": int(array.size), "mean": mean, "sd": sd,
            "ci95_low": mean - half, "ci95_high": mean + half}


def step_summary(rows: list[dict]) -> list[dict]:
    output = []
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            for step in STEPS:
                subset = [
                    row for row in rows
                    if row["prior_k"] == prior_k
                    and row["true_k"] == true_k
                    and row["observations"] == step
                ]
                seed_rmse = []
                seed_accuracy = []
                seed_true_probability = []
                for seed in sorted({row["benchmark_seed"] for row in subset}):
                    seed_rows = [row for row in subset if row["benchmark_seed"] == seed]
                    seed_rmse.append(float(np.mean([row["curve_rmse_ph"] for row in seed_rows])))
                    seed_accuracy.append(100.0 * float(np.mean([row["k_correct"] for row in seed_rows])))
                    seed_true_probability.append(float(np.mean([
                        row["pair_probabilities"][true_k - 1] for row in seed_rows
                    ])))
                rmse = mean_ci95(seed_rmse)
                accuracy = mean_ci95(seed_accuracy)
                probability = mean_ci95(seed_true_probability)
                task_rmse = np.asarray([row["curve_rmse_ph"] for row in subset], dtype=float)
                output.append({
                    "prior_k": prior_k,
                    "true_k": true_k,
                    "observations": step,
                    "tasks": len(subset),
                    "seed_means": rmse["n"],
                    "curve_rmse_mean_of_seed_means": rmse["mean"],
                    "curve_rmse_seed_sd": rmse["sd"],
                    "curve_rmse_seed_ci95_low": rmse["ci95_low"],
                    "curve_rmse_seed_ci95_high": rmse["ci95_high"],
                    "curve_rmse_task_median": float(np.median(task_rmse)),
                    "curve_rmse_task_q25": float(np.percentile(task_rmse, 25)),
                    "curve_rmse_task_q75": float(np.percentile(task_rmse, 75)),
                    "k_accuracy_percent_mean_of_seed_means": accuracy["mean"],
                    "k_accuracy_seed_ci95_low": max(0.0, accuracy["ci95_low"]),
                    "k_accuracy_seed_ci95_high": min(100.0, accuracy["ci95_high"]),
                    "true_k_probability_mean": probability["mean"],
                    "true_k_probability_seed_ci95_low": max(0.0, probability["ci95_low"]),
                    "true_k_probability_seed_ci95_high": min(1.0, probability["ci95_high"]),
                })
    return output


def stable_correction_step(history: list[dict]) -> int | None:
    true_k = history[0]["true_k"]
    for step in STEPS[1:]:
        if all(row["estimated_k"] == true_k for row in history[step:]):
            return step
    return None


def transition_summary(histories: dict) -> list[dict]:
    output = []
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            group = [
                history for history in histories.values()
                if history[0]["prior_k"] == prior_k and history[0]["true_k"] == true_k
            ]
            ever = [
                history for history in group
                if any(row["estimated_k"] == true_k for row in history[1:])
            ]
            stable_steps = [] if prior_k == true_k else [
                value for value in (stable_correction_step(history) for history in group)
                if value is not None
            ]
            final_correct = sum(history[-1]["estimated_k"] == true_k for history in group)
            retained = sum(
                all(row["estimated_k"] == true_k for row in history) for history in group
            )
            output.append({
                "prior_k": prior_k,
                "true_k": true_k,
                "tasks": len(group),
                "prior_is_correct": int(prior_k == true_k),
                "ever_correct_after_step0_tasks": len(ever),
                "ever_correct_after_step0_percent": 100.0 * len(ever) / len(group),
                "stable_correction_tasks": len(stable_steps),
                "stable_correction_percent": (
                    100.0 * len(stable_steps) / len(group)
                    if prior_k != true_k else math.nan
                ),
                "stable_correction_step_median": (
                    float(np.median(stable_steps)) if stable_steps else math.nan
                ),
                "stable_correction_step_q25": (
                    float(np.percentile(stable_steps, 25)) if stable_steps else math.nan
                ),
                "stable_correction_step_q75": (
                    float(np.percentile(stable_steps, 75)) if stable_steps else math.nan
                ),
                "final_k_correct_tasks": final_correct,
                "final_k_accuracy_percent": 100.0 * final_correct / len(group),
                "correct_k_retained_all_steps_tasks": retained,
                "correct_k_retained_all_steps_percent": 100.0 * retained / len(group),
            })
    return output


def load_task_lookup(task_dir: Path) -> dict[tuple[int, int, int], object]:
    lookup = {}
    for path in sorted(task_dir.glob("seed_*_tasks.jsonl")):
        benchmark_seed = int(path.stem.split("_")[1])
        for task in load_tasks(path):
            lookup[(benchmark_seed, task.seed, task.task_id)] = task
    return lookup


def initial_state(task) -> SolutionState:
    return SolutionState(float(task.initial_volume_ml), float(task.initial_base_moles), 0.0)


def task_curve(task, row: dict) -> np.ndarray:
    return response_curve(
        row["estimated_concentration_m"],
        row["estimated_pka"],
        task.initial_volume_ml,
        initial_state(task),
        CURVE_GRID_ML,
    )


def true_curve(task) -> np.ndarray:
    return response_curve(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        initial_state(task),
        CURVE_GRID_ML,
    )


def curve_mean_ci95(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(curves, axis=0)
    sd = np.std(curves, axis=0, ddof=1)
    half = t.ppf(0.975, curves.shape[0] - 1) * sd / math.sqrt(curves.shape[0])
    return mean, mean - half, mean + half


def plot_rmse_grid(path: Path, summary: list[dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.5), sharex=True, constrained_layout=True)
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            axis = axes[prior_k - 1, true_k - 1]
            subset = [row for row in summary if row["prior_k"] == prior_k and row["true_k"] == true_k]
            steps = np.asarray([row["observations"] for row in subset])
            mean = np.asarray([row["curve_rmse_mean_of_seed_means"] for row in subset])
            low = np.asarray([row["curve_rmse_seed_ci95_low"] for row in subset])
            high = np.asarray([row["curve_rmse_seed_ci95_high"] for row in subset])
            axis.fill_between(steps, low, high, color=PF_COLOR, alpha=0.20,
                              label="95% CI across seed means")
            axis.plot(steps, mean, color=PF_COLOR, linewidth=2.2, marker="o", label="Mean RMSE")
            axis.set_title(f"Prior K={prior_k}, true K={true_k}; n={subset[0]['tasks']}")
            axis.set_xticks(STEPS)
            axis.grid(alpha=0.20)
            if true_k == 1:
                axis.set_ylabel("Full-curve RMSE (pH)")
            if prior_k == 3:
                axis.set_xlabel("Observations")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_accuracy_grid(path: Path, summary: list[dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            axis = axes[prior_k - 1, true_k - 1]
            subset = [row for row in summary if row["prior_k"] == prior_k and row["true_k"] == true_k]
            steps = np.asarray([row["observations"] for row in subset])
            mean = np.asarray([row["k_accuracy_percent_mean_of_seed_means"] for row in subset])
            low = np.asarray([row["k_accuracy_seed_ci95_low"] for row in subset])
            high = np.asarray([row["k_accuracy_seed_ci95_high"] for row in subset])
            axis.fill_between(steps, low, high, color="#009E73", alpha=0.20,
                              label="95% CI across seed means")
            axis.plot(steps, mean, color="#009E73", linewidth=2.2, marker="o",
                      label="K accuracy")
            axis.set_title(f"Prior K={prior_k}, true K={true_k}")
            axis.set_xticks(STEPS)
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.20)
            if true_k == 1:
                axis.set_ylabel("Correct K (%)")
            if prior_k == 3:
                axis.set_xlabel("Observations")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_category_mean_curves(path: Path, rows: list[dict], task_lookup: dict) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            axis = axes[prior_k - 1, true_k - 1]
            category = [
                row for row in rows
                if row["prior_k"] == prior_k and row["true_k"] == true_k
            ]
            base_rows = [row for row in category if row["observations"] == 0]
            truths = []
            for row in base_rows:
                task = task_lookup[(row["benchmark_seed"], row["task_seed"], row["task_id"])]
                truths.append(true_curve(task))
            true_mean, true_low, true_high = curve_mean_ci95(np.asarray(truths))
            axis.fill_between(CURVE_GRID_ML, true_low, true_high, color="#777777", alpha=0.13)
            axis.plot(CURVE_GRID_ML, true_mean, color=TRUE_COLOR, linewidth=2.4, label="Mean truth")
            for step in CURVE_STEPS:
                step_rows = [row for row in category if row["observations"] == step]
                curves = []
                for row in step_rows:
                    task = task_lookup[(row["benchmark_seed"], row["task_seed"], row["task_id"])]
                    curves.append(task_curve(task, row))
                mean, low, high = curve_mean_ci95(np.asarray(curves))
                color = STEP_COLORS[step]
                axis.fill_between(CURVE_GRID_ML, low, high, color=color, alpha=0.055)
                axis.plot(CURVE_GRID_ML, mean, color=color, linewidth=1.6,
                          label=f"Posterior step {step}")
            axis.set_title(f"Prior K={prior_k}, true K={true_k}; n={len(base_rows)}")
            axis.grid(alpha=0.18)
            if true_k == 1:
                axis.set_ylabel("Mean pH")
            if prior_k == 3:
                axis.set_xlabel("Signed titrant volume (mL)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def select_category_examples(histories: dict) -> list[dict]:
    selected = []
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            group = [
                history for history in histories.values()
                if history[0]["prior_k"] == prior_k and history[0]["true_k"] == true_k
            ]
            target = float(np.median([history[-1]["curve_rmse_ph"] for history in group]))
            chosen = min(group, key=lambda history: abs(history[-1]["curve_rmse_ph"] - target))
            row = chosen[-1]
            selected.append({
                "selection_type": "category_final_rmse_median",
                "prior_k": prior_k,
                "true_k": true_k,
                "benchmark_seed": row["benchmark_seed"],
                "task_seed": row["task_seed"],
                "task_id": row["task_id"],
                "target_category_median_final_rmse": target,
                "final_rmse": row["curve_rmse_ph"],
                "final_estimated_k": row["estimated_k"],
                "stable_correction_step": stable_correction_step(chosen),
            })
    return selected


def select_wrong_to_correct_examples(histories: dict) -> list[dict]:
    selected = []
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            if prior_k == true_k:
                continue
            candidates = []
            for history in histories.values():
                if history[0]["prior_k"] != prior_k or history[0]["true_k"] != true_k:
                    continue
                correction = stable_correction_step(history)
                if correction is not None:
                    candidates.append((history, correction))
            if not candidates:
                continue
            target_step = float(np.median([value for _, value in candidates]))
            nearest = [item for item in candidates if abs(item[1] - target_step) == min(
                abs(value - target_step) for _, value in candidates
            )]
            target_rmse = float(np.median([history[-1]["curve_rmse_ph"] for history, _ in nearest]))
            history, correction = min(
                nearest, key=lambda item: abs(item[0][-1]["curve_rmse_ph"] - target_rmse)
            )
            row = history[-1]
            selected.append({
                "selection_type": "stable_wrong_to_correct_median_transition",
                "prior_k": prior_k,
                "true_k": true_k,
                "benchmark_seed": row["benchmark_seed"],
                "task_seed": row["task_seed"],
                "task_id": row["task_id"],
                "stable_correction_step": correction,
                "target_median_correction_step": target_step,
                "final_rmse": row["curve_rmse_ph"],
                "final_estimated_k": row["estimated_k"],
            })
    return selected


def sample_posterior_curves(controller, task, draws: int, rng: np.random.Generator) -> np.ndarray:
    inference = controller.inference
    model_probabilities = np.asarray(inference.model_probabilities, dtype=float)
    sampled_k = rng.choice(np.asarray([1, 2, 3]), size=draws, p=model_probabilities)
    curves = np.empty((draws, len(CURVE_GRID_ML)), dtype=float)
    for k in (1, 2, 3):
        positions = np.flatnonzero(sampled_k == k)
        if positions.size == 0:
            continue
        bank = inference.banks[k]
        indices = rng.choice(
            bank.particle_count,
            size=positions.size,
            replace=True,
            p=bank.weights,
        )
        concentrations = bank.concentrations_m[indices]
        pka_matrix = bank.pka_particles[indices]
        grid_count = len(CURVE_GRID_ML)
        base_ml = np.maximum(0.0, CURVE_GRID_ML)
        acid_ml = np.maximum(0.0, -CURVE_GRID_ML)
        expanded_concentrations = np.repeat(concentrations, grid_count)
        expanded_pka = np.repeat(pka_matrix, grid_count, axis=0)
        state = SolutionState(
            total_volume_ml=np.tile(float(task.initial_volume_ml) + base_ml + acid_ml, positions.size),
            base_moles=np.tile(float(task.initial_base_moles) + 0.1 * base_ml / 1000.0, positions.size),
            acid_moles=np.tile(0.1 * acid_ml / 1000.0, positions.size),
        )
        curves[positions, :] = solve_ph_particles(
            expanded_concentrations,
            expanded_pka,
            task.initial_volume_ml,
            state,
        ).reshape(positions.size, grid_count)
    return curves


def rerun_with_intervals(task, prior_k: int, benchmark_seed: int, particles: int,
                         strength: float, draws: int) -> list[dict]:
    common_seed = benchmark_seed * 30_000_049 + task.task_id * 1013
    controller = initialize_controller(task, particles, common_seed)
    apply_k_prior(controller, prior_k, strength)
    rng = np.random.default_rng(common_seed + prior_k * 1_000_003 + 20260814)
    output = []

    def capture(step: int) -> None:
        estimate = controller.posterior_estimate()
        sampled = sample_posterior_curves(controller, task, draws, rng)
        fitted = response_curve(
            estimate.concentration_m,
            estimate.pka_values,
            task.initial_volume_ml,
            initial_state(task),
            CURVE_GRID_ML,
        )
        truth = true_curve(task)
        output.append({
            "step": step,
            "estimated_k": estimate.pair_count,
            "pair_probabilities": estimate.pair_probabilities.copy(),
            "fitted": fitted,
            "truth": truth,
            "lower": np.percentile(sampled, 2.5, axis=0),
            "upper": np.percentile(sampled, 97.5, axis=0),
            "rmse": float(np.sqrt(np.mean((fitted - truth) ** 2))),
        })

    capture(0)
    while int(controller.steps_taken) < 12:
        controller.use_secondary_reagents = False
        action, _ = controller.select_best_action()
        measured_ph, _, done, _ = controller.step(action, mode="Simulate")
        controller.update_posteriors(action, measured_ph)
        capture(int(controller.steps_taken))
        if done and int(controller.steps_taken) < 12:
            controller.done = False
    return output


def selected_key(row: dict) -> tuple[int, int, int, int]:
    return row["prior_k"], row["benchmark_seed"], row["task_seed"], row["task_id"]


def plot_interval_history(path: Path, selected: dict, interval_rows: list[dict], heading: str) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14.4, 12.3), sharex=True, sharey=True,
                             constrained_layout=True)
    flat = list(axes.flat)
    true_k = selected["true_k"]
    for axis, row in zip(flat, interval_rows):
        axis.fill_between(CURVE_GRID_ML, row["lower"], row["upper"], color="#56B4E9",
                          alpha=0.30, label="95% PF posterior interval")
        axis.plot(CURVE_GRID_ML, row["truth"], color=TRUE_COLOR, linewidth=2.0,
                  label="Ground truth")
        axis.plot(CURVE_GRID_ML, row["fitted"], color=PF_COLOR, linewidth=1.7,
                  linestyle="--", label="PF posterior mean")
        probability = row["pair_probabilities"][true_k - 1]
        axis.set_title(
            f"Step {row['step']}: RMSE={row['rmse']:.3f}\n"
            f"Khat={row['estimated_k']}; P(true K)={probability:.2f}",
            fontsize=9.2,
        )
        axis.grid(alpha=0.18)
    for axis in flat[len(interval_rows):]:
        axis.axis("off")
    for row_index in range(4):
        axes[row_index, 0].set_ylabel("pH")
    for axis in axes[-1, :]:
        axis.set_xlabel("Signed volume (mL)")
    flat[0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(heading, fontsize=14)
    fig.savefig(path, dpi=245)
    plt.close(fig)


def plot_correction_overview(path: Path, selected_rows: list[dict], interval_cache: dict) -> None:
    if not selected_rows:
        return
    fig, axes = plt.subplots(len(selected_rows), 3, figsize=(12.8, 3.25 * len(selected_rows)),
                             sharex=True, sharey=True, constrained_layout=True)
    if len(selected_rows) == 1:
        axes = np.asarray([axes])
    for row_index, selected in enumerate(selected_rows):
        interval_rows = interval_cache[selected_key(selected)]
        correction = int(selected["stable_correction_step"])
        shown = [interval_rows[0], interval_rows[correction], interval_rows[-1]]
        labels = ("Prior", "Stable correction", "Step 12")
        for column_index, (snapshot, label) in enumerate(zip(shown, labels)):
            axis = axes[row_index, column_index]
            axis.fill_between(CURVE_GRID_ML, snapshot["lower"], snapshot["upper"],
                              color="#56B4E9", alpha=0.30,
                              label="95% PF posterior interval")
            axis.plot(CURVE_GRID_ML, snapshot["truth"], color=TRUE_COLOR, linewidth=2.0,
                      label="Ground truth")
            axis.plot(CURVE_GRID_ML, snapshot["fitted"], color=PF_COLOR, linewidth=1.7,
                      linestyle="--", label="PF posterior mean")
            axis.set_title(
                f"{label}, step {snapshot['step']}\n"
                f"Khat={snapshot['estimated_k']}; RMSE={snapshot['rmse']:.3f}", fontsize=9.5
            )
            axis.grid(alpha=0.18)
            if column_index == 0:
                axis.set_ylabel(
                    f"Prior K={selected['prior_k']} to true K={selected['true_k']}\npH"
                )
            if row_index == len(selected_rows) - 1:
                axis.set_xlabel("Signed volume (mL)")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Examples with stable correction from an incorrect K prior", fontsize=14)
    fig.savefig(path, dpi=245)
    plt.close(fig)


def write_report(path: Path, summary: list[dict], transitions: list[dict],
                 corrections: list[dict], prior_strength: float, draws: int) -> None:
    lines = [
        "# 先验 K × 真实 K 的后验曲线收敛分析",
        "",
        "## 协议",
        "",
        f"三种先验分别设置为 P(K_prior)={prior_strength:.2f}，另外两个 K 各 {(1.0-prior_strength)/2.0:.2f}。三种先验复用完全相同的 5×300 个任务和 PF 随机粒子种子，因此属于配对比较。每个任务记录 step 0–12。",
        "",
        "九宫格 RMSE 和 K 准确率中的 95% CI 基于五个独立 benchmark seed 的均值。九宫格平均滴定曲线的阴影是跨任务平均曲线的 95% CI。具体任务图重新运行相同任务并直接从变量 K 粒子滤波器的联合模型概率、浓度粒子和 pKa 粒子中抽样，因此阴影为逐任务 95% PF 后验曲线区间。",
        "",
        "## step 12 分类结果",
        "",
        "| 先验 K | 真实 K | 任务数 | step 0 RMSE | step 12 RMSE | step 12 K 准确率 | 稳定纠正比例 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prior_k in (1, 2, 3):
        for true_k in (1, 2, 3):
            start = next(row for row in summary if row["prior_k"] == prior_k and row["true_k"] == true_k and row["observations"] == 0)
            end = next(row for row in summary if row["prior_k"] == prior_k and row["true_k"] == true_k and row["observations"] == 12)
            transition = next(row for row in transitions if row["prior_k"] == prior_k and row["true_k"] == true_k)
            correction_text = (
                f"{transition['stable_correction_percent']:.2f}%"
                if prior_k != true_k
                else f"保持正确 {transition['correct_k_retained_all_steps_percent']:.2f}%"
            )
            lines.append(
                f"| {prior_k} | {true_k} | {end['tasks']} | {start['curve_rmse_mean_of_seed_means']:.4f} | {end['curve_rmse_mean_of_seed_means']:.4f} | {end['k_accuracy_percent_mean_of_seed_means']:.2f}% | {correction_text} |"
            )
    lines.extend([
        "",
        "## 错误先验更新到正确 K 的例子",
        "",
    ])
    if corrections:
        for row in corrections:
            lines.append(
                f"- Kprior={row['prior_k']} → Ktrue={row['true_k']}：seed={row['benchmark_seed']}，task={row['task_id']}，从 step {row['stable_correction_step']} 起保持正确，最终 RMSE={row['final_rmse']:.4f} pH。"
            )
    else:
        lines.append("- 本协议中没有发现错误先验后稳定更新到正确 K 的任务。")
    lines.extend([
        "",
        "## 区间解释",
        "",
        f"每个具体任务、每个 step 使用 {draws} 条联合 PF 后验曲线抽样计算 2.5%–97.5% 分位。它应称为 95% 后验可信区间，而不是重复实验均值的频率学置信区间。两类区间均在图例和表述中分开标注。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze posterior convergence by biased prior K and true K")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--prior-strength", type=float, default=0.80)
    parser.add_argument("--posterior-draws", type=int, default=200)
    args = parser.parse_args()

    source = args.result_dir / "all_prior_k_posterior_rows.csv"
    marker = args.result_dir / "PRIOR_K_GRID_COMPLETE.json"
    if not source.is_file() or not marker.is_file():
        raise FileNotFoundError("The prior-K grid run is not complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    example_dir = figures / "category_examples"
    correction_dir = figures / "wrong_to_correct_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_rows(source)
    validate(rows)
    histories = make_histories(rows)
    task_lookup = load_task_lookup(args.source_task_dir)
    summary = step_summary(rows)
    transitions = transition_summary(histories)
    category_selected = select_category_examples(histories)
    correction_selected = select_wrong_to_correct_examples(histories)

    write_csv(args.output_dir / "prior_k_true_k_step_summary.csv", summary)
    write_csv(args.output_dir / "prior_k_true_k_transition_summary.csv", transitions)
    write_csv(args.output_dir / "selected_category_examples.csv", category_selected)
    if correction_selected:
        write_csv(args.output_dir / "selected_wrong_to_correct_examples.csv", correction_selected)

    plot_rmse_grid(figures / "rmse_convergence_by_prior_k_and_true_k.png", summary)
    plot_accuracy_grid(figures / "k_accuracy_by_prior_k_and_true_k.png", summary)
    plot_category_mean_curves(
        figures / "mean_curve_convergence_by_prior_k_and_true_k.png", rows, task_lookup
    )

    unique_selected = {}
    for selected in category_selected + correction_selected:
        unique_selected[selected_key(selected)] = selected
    interval_cache = {}
    for index, (key, selected) in enumerate(sorted(unique_selected.items()), 1):
        task = task_lookup[(selected["benchmark_seed"], selected["task_seed"], selected["task_id"])]
        interval_cache[key] = rerun_with_intervals(
            task,
            selected["prior_k"],
            selected["benchmark_seed"],
            args.particles,
            args.prior_strength,
            args.posterior_draws,
        )
        print(f"posterior intervals: {index}/{len(unique_selected)} examples", flush=True)

    for selected in category_selected:
        plot_interval_history(
            example_dir / f"prior_k{selected['prior_k']}_true_k{selected['true_k']}_representative.png",
            selected,
            interval_cache[selected_key(selected)],
            f"Representative category task: prior K={selected['prior_k']}, true K={selected['true_k']}, seed {selected['benchmark_seed']}, task {selected['task_id']}",
        )
    for selected in correction_selected:
        plot_interval_history(
            correction_dir / f"prior_k{selected['prior_k']}_to_true_k{selected['true_k']}_corrected.png",
            selected,
            interval_cache[selected_key(selected)],
            f"Stable K correction: prior K={selected['prior_k']} to true K={selected['true_k']}, seed {selected['benchmark_seed']}, task {selected['task_id']}",
        )
    plot_correction_overview(
        figures / "wrong_to_correct_examples_overview.png",
        correction_selected,
        interval_cache,
    )
    write_report(
        args.output_dir / "PRIOR_K_TRUE_K_CONVERGENCE_ANALYSIS_CN.md",
        summary,
        transitions,
        correction_selected,
        args.prior_strength,
        args.posterior_draws,
    )
    completion = {
        "status": "complete",
        "source_rows": len(rows),
        "task_prior_combinations": len(histories),
        "category_examples": len(category_selected),
        "wrong_to_correct_examples": len(correction_selected),
        "posterior_draws_per_task_step": args.posterior_draws,
        "curve_grid_points": len(CURVE_GRID_ML),
    }
    (args.output_dir / "PRIOR_K_TRUE_K_ANALYSIS_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
