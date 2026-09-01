from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_reviewer_analysis import (  # noqa: E402
    BayesianParticleEnv,
    FIGURES_DIR,
    INITIAL_ACID_VOL,
    SUCCESS_THRESHOLD,
    TABLES_DIR,
    ensure_dirs,
    load_experiment_conditions,
    solve_pH,
)


STEP_BUDGET = 8
OUTPUT_PREFIX = "q1b_3000sim"


@dataclass
class MatchResult:
    used_indices: tuple[int, ...]
    mae: float
    rmse: float
    max_abs_error: float
    within_0p5_frac: float
    within_1p0_frac: float


def optimal_pka_match(inferred: np.ndarray, truth: list[float]) -> MatchResult:
    inferred = np.asarray(inferred, dtype=float)
    truth_sorted = np.sort(np.asarray(truth, dtype=float))
    best: MatchResult | None = None

    for combo in itertools.combinations(range(len(inferred)), len(truth_sorted)):
        candidate = np.sort(inferred[list(combo)])
        diffs = np.abs(candidate - truth_sorted)
        mae = float(np.mean(diffs))
        rmse = float(np.sqrt(np.mean(np.square(diffs))))
        result = MatchResult(
            used_indices=tuple(combo),
            mae=mae,
            rmse=rmse,
            max_abs_error=float(np.max(diffs)),
            within_0p5_frac=float(np.mean(diffs <= 0.5)),
            within_1p0_frac=float(np.mean(diffs <= 1.0)),
        )
        if best is None or result.mae < best.mae or (math.isclose(result.mae, best.mae) and result.rmse < best.rmse):
            best = result

    if best is None:
        raise RuntimeError("Failed to compute pKa matching result.")
    return best


def serialize_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def predict_next_ph(env: BayesianParticleEnv, action: tuple[str, float]) -> float:
    reagent, volume = action
    volume = float(volume)
    added_moles = env.reagents[reagent] * (volume / 1000.0)
    acid_added_moles = env.acid_added_moles
    base_added_moles = env.base_added_moles
    acid_volume = env.acid_volume
    base_volume = env.base_volume

    if "acid" in reagent.lower():
        acid_added_moles += added_moles
        acid_volume += volume
    else:
        base_added_moles += added_moles
        base_volume += volume

    total_volume_l = (INITIAL_ACID_VOL + acid_volume + base_volume) / 1000.0
    n_analyte = (INITIAL_ACID_VOL / 1000.0) * 0.1
    c_A = n_analyte / total_volume_l
    c_Na = base_added_moles / total_volume_l
    c_HCl = acid_added_moles / total_volume_l
    return float(solve_pH(c_A, c_Na, c_HCl, env.get_effective_pka_array().tolist()))


def snapshot_metrics(
    env: BayesianParticleEnv,
    condition: dict[str, object],
    experiment: int,
    step: int,
    previous_prediction_abs_error: float | None = None,
) -> dict[str, object]:
    true_pkas = [float(value) for value in condition["acid_params"]]
    posterior_pkas = np.asarray(env.pKa_list, dtype=float)
    posterior_stds = np.asarray(env.pKa_std, dtype=float)
    posterior_moles = np.asarray(env.buffer_total_moles, dtype=float)
    effective_pkas = np.asarray(env.get_effective_pka_array(), dtype=float)

    raw_match = optimal_pka_match(posterior_pkas, true_pkas)
    effective_match = optimal_pka_match(effective_pkas, true_pkas)
    unused_indices = tuple(index for index in range(len(posterior_pkas)) if index not in raw_match.used_indices)
    total_moles = float(np.sum(np.abs(posterior_moles)))
    unused_mole_fraction = (
        float(np.sum(np.abs(posterior_moles[list(unused_indices)])) / total_moles)
        if unused_indices and total_moles > 0
        else 0.0
    )

    return {
        "experiment": experiment,
        "acid_type": str(condition["acid_type"]).lower(),
        "true_buffer_count": len(true_pkas),
        "step": step,
        "current_pH": float(env.current_ph),
        "target_pH": float(env.target_ph),
        "abs_pH_error": float(abs(env.current_ph - env.target_ph)),
        "posterior_mean_std": float(np.mean(posterior_stds)),
        "posterior_max_std": float(np.max(posterior_stds)),
        "matched_pka_mae": raw_match.mae,
        "matched_pka_rmse": raw_match.rmse,
        "matched_pka_max_abs_error": raw_match.max_abs_error,
        "matched_within_0p5_frac": raw_match.within_0p5_frac,
        "matched_within_1p0_frac": raw_match.within_1p0_frac,
        "all_true_pkas_within_0p5": float(raw_match.within_0p5_frac == 1.0),
        "all_true_pkas_within_1p0": float(raw_match.within_1p0_frac == 1.0),
        "effective_pka_mae": effective_match.mae,
        "unused_slot_mole_fraction": unused_mole_fraction,
        "success_if_stop_now": float(abs(env.current_ph - env.target_ph) <= SUCCESS_THRESHOLD),
        "previous_prediction_abs_error": previous_prediction_abs_error,
        "posterior_pkas": serialize_array(posterior_pkas),
        "posterior_stds": serialize_array(posterior_stds),
        "posterior_moles": serialize_array(posterior_moles),
        "effective_pkas": serialize_array(effective_pkas),
        "true_pkas": serialize_array(np.asarray(true_pkas, dtype=float)),
        "matched_slot_indices": str(list(raw_match.used_indices)),
        "unused_slot_indices": str(list(unused_indices)),
    }


def run_analysis(limit: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    conditions = load_experiment_conditions()
    if limit is not None:
        conditions = conditions[:limit]

    np.random.seed(seed)
    trajectory_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []

    started = time.time()
    for index, condition in enumerate(conditions, start=1):
        trajectory_part, experiment_part, transition_part = run_single_experiment(index, condition, seed + index)
        trajectory_rows.extend(trajectory_part)
        experiment_rows.append(experiment_part)
        transition_rows.extend(transition_part)

        if index % 100 == 0 or index == len(conditions):
            elapsed = time.time() - started
            print(f"Completed {index}/{len(conditions)} experiments in {elapsed:.1f}s")

    return pd.DataFrame(trajectory_rows), pd.DataFrame(experiment_rows), pd.DataFrame(transition_rows)


def run_analysis_parallel(limit: int | None, seed: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    conditions = load_experiment_conditions()
    if limit is not None:
        conditions = conditions[:limit]

    trajectory_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    started = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_single_experiment, index, condition, seed + index)
            for index, condition in enumerate(conditions, start=1)
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            trajectory_part, experiment_part, transition_part = future.result()
            trajectory_rows.extend(trajectory_part)
            experiment_rows.append(experiment_part)
            transition_rows.extend(transition_part)
            if completed % 100 == 0 or completed == len(futures):
                elapsed = time.time() - started
                print(f"Completed {completed}/{len(futures)} experiments in {elapsed:.1f}s")

    return pd.DataFrame(trajectory_rows), pd.DataFrame(experiment_rows), pd.DataFrame(transition_rows)


def run_single_experiment(
    index: int,
    condition: dict[str, object],
    local_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    np.random.seed(local_seed)
    env = BayesianParticleEnv(num_particles=1000)
    env.initialize(
        str(condition["acid_type"]),
        list(condition["acid_params"]),
        float(condition["initial_ph"]),
        float(condition["target_ph"]),
    )

    trajectory_rows: list[dict[str, object]] = [snapshot_metrics(env, condition, experiment=index, step=0)]
    transition_rows: list[dict[str, object]] = []

    action, _ = env.select_best_action()
    while not env.done:
        predicted_next = predict_next_ph(env, action)
        pre_step_match = optimal_pka_match(np.asarray(env.pKa_list, dtype=float), list(condition["acid_params"]))
        current_ph, _, done, _ = env.step(action, mode="Simulate")
        prediction_abs_error = abs(predicted_next - current_ph)
        transition_rows.append(
            {
                "experiment": index,
                "acid_type": str(condition["acid_type"]).lower(),
                "true_buffer_count": len(condition["acid_params"]),
                "source_step": int(env.steps_taken),
                "predicted_next_pH": float(predicted_next),
                "observed_next_pH": float(current_ph),
                "prediction_abs_error": float(prediction_abs_error),
                "posterior_mean_std_before": float(np.mean(env.pKa_std)),
                "matched_pka_mae_before": float(pre_step_match.mae),
            }
        )
        env.update_posteriors(action, current_ph)
        trajectory_rows.append(
            snapshot_metrics(
                env,
                condition,
                experiment=index,
                step=env.steps_taken,
                previous_prediction_abs_error=float(prediction_abs_error),
            )
        )
        if done:
            break
        action, _ = env.select_best_action()

    final_snapshot = trajectory_rows[-1]
    experiment_row = {
        "experiment": index,
        "acid_type": str(condition["acid_type"]).lower(),
        "true_buffer_count": len(condition["acid_params"]),
        "steps_taken": int(env.steps_taken),
        "success": bool(abs(env.current_ph - env.target_ph) <= SUCCESS_THRESHOLD),
        "final_pH": float(env.current_ph),
        "target_pH": float(env.target_ph),
        "final_abs_pH_error": float(abs(env.current_ph - env.target_ph)),
        "final_matched_pka_mae": float(final_snapshot["matched_pka_mae"]),
        "final_posterior_mean_std": float(final_snapshot["posterior_mean_std"]),
        "final_unused_slot_mole_fraction": float(final_snapshot["unused_slot_mole_fraction"]),
    }
    return trajectory_rows, experiment_row, transition_rows


def stage_frame(trajectory: pd.DataFrame, max_step: int | None) -> pd.DataFrame:
    if max_step is None:
        stage = trajectory.sort_values(["experiment", "step"]).groupby("experiment", as_index=False).tail(1)
        stage = stage.copy()
        stage["stage"] = "final"
        return stage

    stage = trajectory.loc[trajectory["step"] <= max_step].sort_values(["experiment", "step"]).groupby("experiment", as_index=False).tail(1)
    stage = stage.copy()
    stage["stage"] = f"up_to_{max_step}_steps"
    return stage


def summarize_stages(trajectory: pd.DataFrame) -> pd.DataFrame:
    initial = trajectory.loc[trajectory["step"] == 0].copy()
    initial["stage"] = "initial"
    step_budget = stage_frame(trajectory, STEP_BUDGET)
    final = stage_frame(trajectory, None)
    combined = pd.concat([initial, step_budget, final], ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    for acid_type, subset in combined.groupby("acid_type"):
        for stage_name, stage_subset in subset.groupby("stage"):
            summary_rows.append(build_summary_row(stage_subset, acid_type, stage_name))
    for stage_name, stage_subset in combined.groupby("stage"):
        summary_rows.append(build_summary_row(stage_subset, "overall", stage_name))

    return pd.DataFrame(summary_rows)


def build_summary_row(frame: pd.DataFrame, acid_type: str, stage_name: str) -> dict[str, object]:
    valid_unused = frame.loc[frame["true_buffer_count"] < 3, "unused_slot_mole_fraction"]
    return {
        "acid_type": acid_type,
        "stage": stage_name,
        "experiments": int(len(frame)),
        "mean_matched_pka_mae": round(float(frame["matched_pka_mae"].mean()), 4),
        "median_matched_pka_mae": round(float(frame["matched_pka_mae"].median()), 4),
        "mean_effective_pka_mae": round(float(frame["effective_pka_mae"].mean()), 4),
        "mean_posterior_std": round(float(frame["posterior_mean_std"].mean()), 4),
        "median_posterior_std": round(float(frame["posterior_mean_std"].median()), 4),
        "mean_abs_pH_error": round(float(frame["abs_pH_error"].mean()), 4),
        "all_true_pkas_within_0p5_pct": round(float(frame["all_true_pkas_within_0p5"].mean() * 100), 2),
        "all_true_pkas_within_1p0_pct": round(float(frame["all_true_pkas_within_1p0"].mean() * 100), 2),
        "matched_within_0p5_frac_pct": round(float(frame["matched_within_0p5_frac"].mean() * 100), 2),
        "matched_within_1p0_frac_pct": round(float(frame["matched_within_1p0_frac"].mean() * 100), 2),
        "mean_unused_slot_mole_fraction_if_true_count_lt3": round(float(valid_unused.mean()), 4) if not valid_unused.empty else np.nan,
        "unused_slot_fraction_lt_0p1_pct_if_true_count_lt3": round(float((valid_unused < 0.1).mean() * 100), 2) if not valid_unused.empty else np.nan,
    }


def build_per_experiment_stage_table(trajectory: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    initial = trajectory.loc[trajectory["step"] == 0].copy()
    initial["stage"] = "initial"
    step_budget = stage_frame(trajectory, STEP_BUDGET)
    final = stage_frame(trajectory, None)
    stage_table = pd.concat([initial, step_budget, final], ignore_index=True)
    merged = stage_table.merge(experiments, on=["experiment", "acid_type", "true_buffer_count"], how="left")
    return merged.sort_values(["experiment", "stage"])


def build_correlations(stage_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage_name, subset in stage_table.groupby("stage"):
        rows.append(
            {
                "stage": stage_name,
                "corr_posterior_std_vs_pka_mae": round(float(subset["posterior_mean_std"].corr(subset["matched_pka_mae"])), 4),
                "corr_posterior_std_vs_abs_pH_error": round(float(subset["posterior_mean_std"].corr(subset["abs_pH_error"])), 4),
                "corr_pka_mae_vs_abs_pH_error": round(float(subset["matched_pka_mae"].corr(subset["abs_pH_error"])), 4),
            }
        )
    return pd.DataFrame(rows)


def build_species_count_diagnostic(stage_table: pd.DataFrame) -> pd.DataFrame:
    subset = stage_table.loc[stage_table["true_buffer_count"] < 3].copy()
    rows = []
    for (acid_type, stage_name), frame in subset.groupby(["acid_type", "stage"]):
        rows.append(
            {
                "acid_type": acid_type,
                "stage": stage_name,
                "experiments": int(len(frame)),
                "mean_unused_slot_mole_fraction": round(float(frame["unused_slot_mole_fraction"].mean()), 4),
                "median_unused_slot_mole_fraction": round(float(frame["unused_slot_mole_fraction"].median()), 4),
                "unused_slot_fraction_lt_0p1_pct": round(float((frame["unused_slot_mole_fraction"] < 0.1).mean() * 100), 2),
                "unused_slot_fraction_lt_0p2_pct": round(float((frame["unused_slot_mole_fraction"] < 0.2).mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    transitions = transitions.copy()
    transitions["stage"] = np.where(transitions["source_step"] <= STEP_BUDGET, f"within_{STEP_BUDGET}_steps", "after_step_budget")
    rows = []
    for acid_type, acid_subset in transitions.groupby("acid_type"):
        rows.append(build_prediction_row(acid_subset, acid_type, "all_steps"))
        rows.append(build_prediction_row(acid_subset.loc[acid_subset["source_step"] <= STEP_BUDGET], acid_type, f"within_{STEP_BUDGET}_steps"))
    rows.append(build_prediction_row(transitions, "overall", "all_steps"))
    rows.append(build_prediction_row(transitions.loc[transitions["source_step"] <= STEP_BUDGET], "overall", f"within_{STEP_BUDGET}_steps"))
    return pd.DataFrame(rows)


def build_prediction_row(frame: pd.DataFrame, acid_type: str, stage_name: str) -> dict[str, object]:
    if frame.empty:
        return {
            "acid_type": acid_type,
            "stage": stage_name,
            "transitions": 0,
            "mean_prediction_abs_error": np.nan,
            "median_prediction_abs_error": np.nan,
            "prediction_error_le_0p1_pct": np.nan,
            "prediction_error_le_0p2_pct": np.nan,
        }
    return {
        "acid_type": acid_type,
        "stage": stage_name,
        "transitions": int(len(frame)),
        "mean_prediction_abs_error": round(float(frame["prediction_abs_error"].mean()), 4),
        "median_prediction_abs_error": round(float(frame["prediction_abs_error"].median()), 4),
        "prediction_error_le_0p1_pct": round(float((frame["prediction_abs_error"] <= 0.1).mean() * 100), 2),
        "prediction_error_le_0p2_pct": round(float((frame["prediction_abs_error"] <= 0.2).mean() * 100), 2),
    }


def plot_stage_metric(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    acid_order = ["monoprotic", "diprotic", "triprotic", "overall"]
    stage_order = ["initial", f"up_to_{STEP_BUDGET}_steps", "final"]
    color_map = {
        "initial": "#b56576",
        f"up_to_{STEP_BUDGET}_steps": "#355070",
        "final": "#6d597a",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(acid_order))
    width = 0.23
    offsets = [-width, 0.0, width]

    for offset, stage_name in zip(offsets, stage_order):
        values = []
        for acid_type in acid_order:
            row = summary.loc[(summary["acid_type"] == acid_type) & (summary["stage"] == stage_name)]
            values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=stage_name.replace("_", " "), color=color_map[stage_name])

    ax.set_xticks(x)
    ax.set_xticklabels(acid_order)
    ax.set_ylabel(ylabel)
    ax.set_title("Bayesian interpretability metric across 3000 simulated tasks")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=220)
    plt.close(fig)


def write_note(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    species_diag: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    def fetch(acid_type: str, stage_name: str, metric: str) -> float:
        row = summary.loc[(summary["acid_type"] == acid_type) & (summary["stage"] == stage_name), metric]
        return float(row.iloc[0])

    lines = [
        "# Q1b Quantitative Analysis",
        "",
        f"- Dataset: {int(summary.loc[(summary['acid_type'] == 'overall') & (summary['stage'] == 'final'), 'experiments'].iloc[0])} simulated tasks from `experiment_summary.csv`.",
        f"- Step-budget view: metrics at `min(final step, {STEP_BUDGET})` to answer the reviewer's question about whether roughly eight titration measurements are sufficient.",
        "",
        "## Headline observations",
        "",
        f"- Overall matched posterior pKa MAE changes from {fetch('overall', 'initial', 'mean_matched_pka_mae'):.3f} initially to {fetch('overall', f'up_to_{STEP_BUDGET}_steps', 'mean_matched_pka_mae'):.3f} by step <= {STEP_BUDGET}, and {fetch('overall', 'final', 'mean_matched_pka_mae'):.3f} at the final step.",
        f"- By step <= {STEP_BUDGET}, the fraction of tasks for which all true pKa values are recovered within +-1.0 pH units is {fetch('overall', f'up_to_{STEP_BUDGET}_steps', 'all_true_pkas_within_1p0_pct'):.2f}%; at the final step it is {fetch('overall', 'final', 'all_true_pkas_within_1p0_pct'):.2f}%.",
        f"- The mean posterior pKa standard deviation is {fetch('overall', 'initial', 'mean_posterior_std'):.3f} initially, {fetch('overall', f'up_to_{STEP_BUDGET}_steps', 'mean_posterior_std'):.3f} by step <= {STEP_BUDGET}, and {fetch('overall', 'final', 'mean_posterior_std'):.3f} at the final step.",
        "",
        "## Predictive usefulness of the latent state",
        "",
        "- To quantify whether the Bayesian latent state remains chemically meaningful even when exact pKa recovery is imperfect, I also measured the one-step pH prediction error made by the current posterior model before each titration action.",
        "",
        prediction_summary.to_markdown(index=False),
        "",
        "## Species-count caveat",
        "",
        "- The current Bayesian controller keeps exactly three latent buffer slots throughout the episode. It does not maintain an explicit posterior over the number of species.",
        "- As a diagnostic, I tracked the inferred mole fraction assigned to unmatched latent slots for monoprotic and diprotic tasks after optimal pKa matching. Large residual mass in unmatched slots means the model is not cleanly deactivating extra components.",
        "",
        "## Correlations",
        "",
        correlations.to_markdown(index=False),
        "",
        "## Extra-slot diagnostic",
        "",
        species_diag.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quantitative 3000-task interpretability audit for Reviewer 1b.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of experiments for smoke testing.")
    parser.add_argument("--seed", type=int, default=20260629, help="Random seed for posterior sampling reproducibility.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes. Use 1 for serial execution.")
    args = parser.parse_args()

    if args.workers > 1:
        trajectory, experiments, transitions = run_analysis_parallel(limit=args.limit, seed=args.seed, workers=args.workers)
    else:
        trajectory, experiments, transitions = run_analysis(limit=args.limit, seed=args.seed)
    summary = summarize_stages(trajectory)
    stage_table = build_per_experiment_stage_table(trajectory, experiments)
    correlations = build_correlations(stage_table)
    species_diag = build_species_count_diagnostic(stage_table)
    prediction_summary = build_prediction_summary(transitions)

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    trajectory_path = TABLES_DIR / f"{OUTPUT_PREFIX}_trajectory{suffix}.csv"
    experiment_path = TABLES_DIR / f"{OUTPUT_PREFIX}_experiment_summary{suffix}.csv"
    transition_path = TABLES_DIR / f"{OUTPUT_PREFIX}_transition_metrics{suffix}.csv"
    summary_path = TABLES_DIR / f"{OUTPUT_PREFIX}_stage_summary{suffix}.csv"
    stage_path = TABLES_DIR / f"{OUTPUT_PREFIX}_per_experiment_stage_metrics{suffix}.csv"
    corr_path = TABLES_DIR / f"{OUTPUT_PREFIX}_correlations{suffix}.csv"
    species_path = TABLES_DIR / f"{OUTPUT_PREFIX}_species_count_diagnostic{suffix}.csv"
    prediction_path = TABLES_DIR / f"{OUTPUT_PREFIX}_prediction_summary{suffix}.csv"
    note_path = TABLES_DIR / f"{OUTPUT_PREFIX}_interpretability_note{suffix}.md"

    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")
    experiments.to_csv(experiment_path, index=False, encoding="utf-8-sig")
    transitions.to_csv(transition_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    stage_table.to_csv(stage_path, index=False, encoding="utf-8-sig")
    correlations.to_csv(corr_path, index=False, encoding="utf-8-sig")
    species_diag.to_csv(species_path, index=False, encoding="utf-8-sig")
    prediction_summary.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    write_note(summary, correlations, species_diag, prediction_summary, note_path)

    if args.limit is None:
        plot_stage_metric(summary, "mean_matched_pka_mae", "Matched posterior pKa MAE", f"{OUTPUT_PREFIX}_pka_mae.png")
        plot_stage_metric(summary, "mean_posterior_std", "Mean posterior pKa std", f"{OUTPUT_PREFIX}_posterior_std.png")

    print(f"Wrote trajectory table to {trajectory_path}")
    print(f"Wrote stage summary to {summary_path}")
    print(f"Wrote note to {note_path}")


if __name__ == "__main__":
    main()
