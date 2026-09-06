from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
STUDY_DIR = ROOT / "study_source"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from chemistry_model import SolutionState, response_curve
from particle_controllers import JointInferenceController
from task_distribution import ControlTask, generate_tasks, save_tasks


PF_VARIANT = "pf_pka_conc_variable_k"
CHECKPOINTS = (0, 1, 2, 3, 5, 8, 12)
WINDOWS_ML = (0.10, 0.50, 1.00)


def initialize_controller(task: ControlTask, particles: int, seed: int) -> JointInferenceController:
    seed = int(seed) % (2**32 - 1)
    np.random.seed(seed)
    controller = JointInferenceController(PF_VARIANT, num_particles=particles, filter_seed=(seed + 17) % (2**32 - 1))
    controller.initialize_task(task)
    controller.base_added_moles = float(task.initial_base_moles)
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = float(task.initial_volume_ml)
    controller.previous_total_volume = float(task.initial_volume_ml)
    controller.current_ph = float(task.initial_ph)
    controller.previous_ph = float(task.initial_ph)
    controller.last_measured_ph = float(task.initial_ph)
    controller.prev_measured_ph = float(task.initial_ph)
    controller.last_action_volume = 0.0
    controller.done = False
    return controller


def snapshot(task: ControlTask, controller: JointInferenceController, observations: int, checkpoint_type: str) -> dict:
    estimate = controller.posterior_estimate()
    state = SolutionState(
        total_volume_ml=float(controller.total_volume),
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles),
    )
    row = {
        "task_seed": task.seed,
        "task_id": task.task_id,
        "checkpoint_type": checkpoint_type,
        "observations": observations,
        "direction": task.direction,
        "difficulty": task.difficulty,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "current_true_ph": float(controller.simulate_observed_ph()),
        "target_ph": task.target_ph,
        "estimated_concentration_m": estimate.concentration_m,
        "estimated_pair_count": estimate.pair_count,
        "true_pka_json": json.dumps(task.pka_values),
        "estimated_pka_json": json.dumps(estimate.pka_values.tolist()),
    }
    for window in WINDOWS_ML:
        grid = np.linspace(-window, window, 81)
        true_curve = response_curve(
            task.analyte_conc_m,
            task.pka_values,
            task.initial_volume_ml,
            state,
            grid,
        )
        fitted_curve = response_curve(
            estimate.concentration_m,
            estimate.pka_values,
            task.initial_volume_ml,
            state,
            grid,
        )
        # Anchor both responses at the current observed state. The diagnostic
        # evaluates dose-response shape, not an intercept error already known
        # from the current pH measurement.
        center = len(grid) // 2
        true_delta = true_curve - true_curve[center]
        fitted_delta = fitted_curve - fitted_curve[center]
        residual = fitted_delta - true_delta
        suffix = str(window).replace(".", "p")
        row[f"local_rmse_{suffix}_ml"] = float(np.sqrt(np.mean(residual**2)))
        row[f"local_mae_{suffix}_ml"] = float(np.mean(np.abs(residual)))
        row[f"local_max_abs_{suffix}_ml"] = float(np.max(np.abs(residual)))
    return row


def run_task(payload) -> list[dict]:
    task, particles, benchmark_seed = payload
    common_seed = benchmark_seed * 30_000_049 + task.task_id * 1013
    controller = initialize_controller(task, particles, common_seed)
    rows = [snapshot(task, controller, 0, "fixed_observation_count")]
    natural_row = None
    last_valid_row = rows[0]
    for observation in range(1, max(CHECKPOINTS) + 1):
        if controller.steps_taken >= controller.max_steps:
            break
        action, _ = controller.select_best_action()
        measured_ph, _, _, _ = controller.step(action, mode="Simulate")
        if controller.steps_taken == 0:
            break
        controller.update_posteriors(action, measured_ph)
        current_row = snapshot(task, controller, observation, "fixed_observation_count")
        last_valid_row = current_row
        if natural_row is None and abs(controller.current_ph - controller.target_ph) <= 0.10:
            natural_row = {**current_row, "checkpoint_type": "natural_control_end"}
        controller.done = False
        if observation in CHECKPOINTS:
            rows.append(current_row)
    if natural_row is None:
        natural_row = {**last_valid_row, "checkpoint_type": "natural_control_end"}
    rows.append(natural_row)
    return rows


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


def summarize(all_rows: list[dict], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    per_seed = []
    keys = [("fixed_observation_count", value) for value in CHECKPOINTS] + [("natural_control_end", "natural")]
    metrics = [f"local_rmse_{str(window).replace('.', 'p')}_ml" for window in WINDOWS_ML]
    for benchmark_seed in seeds:
        seed_rows = [row for row in all_rows if row["benchmark_seed"] == benchmark_seed]
        for checkpoint_type, observation in keys:
            subset = [
                row for row in seed_rows
                if row["checkpoint_type"] == checkpoint_type
                and (observation == "natural" or row["observations"] == observation)
            ]
            result = {
                "benchmark_seed": benchmark_seed,
                "checkpoint_type": checkpoint_type,
                "observations": observation,
                "tasks": len(subset),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in subset], dtype=float)
                result[f"{metric}_mean"] = float(np.mean(values))
                result[f"{metric}_median"] = float(np.median(values))
                result[f"{metric}_le_0p05_percent"] = 100.0 * float(np.mean(values <= 0.05))
                result[f"{metric}_le_0p10_percent"] = 100.0 * float(np.mean(values <= 0.10))
            per_seed.append(result)
    aggregate = []
    for checkpoint_type, observation in keys:
        subset = [
            row for row in per_seed
            if row["checkpoint_type"] == checkpoint_type and row["observations"] == observation
        ]
        result = {"checkpoint_type": checkpoint_type, "observations": observation, "seed_runs": len(subset)}
        for metric in subset[0]:
            if metric in {"benchmark_seed", "checkpoint_type", "observations"}:
                continue
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        aggregate.append(result)
    return per_seed, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Control-point local response diagnostics for the new PF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--tasks-per-seed", type=int, default=300)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    for benchmark_seed in args.seeds:
        tasks = generate_tasks(4_000_000 + benchmark_seed, args.tasks_per_seed, f"local_response_seed_{benchmark_seed}")
        save_tasks(output / f"seed_{benchmark_seed}_tasks.jsonl", tasks)
        payloads = [(task, args.particles, benchmark_seed) for task in tasks]
        if workers == 1:
            results = map(run_task, payloads)
        else:
            import concurrent.futures

            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            results = executor.map(run_task, payloads, chunksize=2)
        try:
            for index, task_rows in enumerate(results, 1):
                for row in task_rows:
                    row["benchmark_seed"] = benchmark_seed
                all_rows.extend(task_rows)
                if index % 50 == 0 or index == len(tasks):
                    print(f"local-response seed {benchmark_seed}: {index}/{len(tasks)}", flush=True)
        finally:
            if workers != 1:
                executor.shutdown(wait=True)
    per_seed, aggregate = summarize(all_rows, args.seeds)
    write_csv(output / "all_local_response_rows.csv", all_rows)
    write_csv(output / "per_seed_summary.csv", per_seed)
    write_csv(output / "aggregate_summary.csv", aggregate)
    (output / "LOCAL_RESPONSE_COMPLETE.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "tasks_per_seed": args.tasks_per_seed,
                "particles": args.particles,
                "windows_ml": WINDOWS_ML,
                "anchoring": "truth and PF curves are anchored at the current decision point",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
