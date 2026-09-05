from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parent
CONTROLLER_DIR = ROOT / "controllers_release"
STUDY_DIR = ROOT / "study_source"
for path in (STUDY_DIR, CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chemistry_model import SolutionState, solve_ph_scalar
from controller_api import MAX_ACTION_VOLUME_ML, MIN_ACTION_VOLUME_ML, ControllerAction
from new_pf_controller import RobustPFController
from task_distribution import ControlTask, generate_tasks, save_tasks


VARIANTS = (
    "full",
    "no_ph_rate_bonus",
    "no_uncertainty_factor",
    "no_buffering_factor",
    "no_required_volume_term",
    "linear_clip_instead_of_tanh",
)
SUCCESS_TOLERANCE = 0.10
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0


@dataclass(frozen=True)
class RuleFlags:
    ph_rate_bonus: bool = True
    uncertainty_factor: bool = True
    buffering_factor: bool = True
    required_volume_term: bool = True
    tanh_mapping: bool = True


FLAGS = {
    "full": RuleFlags(),
    "no_ph_rate_bonus": RuleFlags(ph_rate_bonus=False),
    "no_uncertainty_factor": RuleFlags(uncertainty_factor=False),
    "no_buffering_factor": RuleFlags(buffering_factor=False),
    "no_required_volume_term": RuleFlags(required_volume_term=False),
    "linear_clip_instead_of_tanh": RuleFlags(tanh_mapping=False),
}


class RuleAblationPFController(RobustPFController):
    def __init__(self, *args, rule_flags: RuleFlags, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rule_flags = rule_flags

    def status(self) -> dict:
        status = super().status()
        status["rule_flags"] = self.rule_flags.__dict__
        return status

    def recommend(self) -> ControllerAction:
        self._require_initialized()
        self._stop_if_needed()
        if self.done:
            from controller_api import stop_action

            return stop_action(self.stop_reason, self.status())
        if self.pending_action is not None:
            raise RuntimeError("The previous action has not been acknowledged by observe()")

        if self.overshoot_occurred and self.overshoot_reagent is not None:
            reagent = "acid" if self.overshoot_reagent == "base" else "base"
            self.overshoot_occurred = False
            self.overshoot_reagent = None
        else:
            reagent = "base" if self.current_ph < self.target_ph else "acid"

        candidate_volumes = np.asarray(
            [round(MIN_ACTION_VOLUME_ML * index, 2) for index in range(1, 1001)],
            dtype=float,
        )
        if self.overshoot_threshold_ml is not None:
            filtered = candidate_volumes[candidate_volumes <= self.overshoot_threshold_ml]
            if len(filtered):
                candidate_volumes = filtered

        estimate = self.posterior_estimate()
        error = abs(self.current_ph - self.target_ph)
        ph_change = abs(self.current_ph - self.previous_ph)
        bonus = (
            1.0 + 0.5 * (1.0 - min(ph_change, 1.0))
            if self.rule_flags.ph_rate_bonus
            else 1.0
        )
        average_uncertainty = float(np.mean(estimate.pka_sd))
        uncertainty = (
            1.0 - 0.1 * min(average_uncertainty, 1.0)
            if self.rule_flags.uncertainty_factor
            else 1.0
        )
        buffering = (
            float(np.clip(1.0 + 0.1 * (self._legacy_buffer_mean - 0.5), 0.95, 1.05))
            if self.rule_flags.buffering_factor
            else 1.0
        )
        alpha = 0.2 * bonus * uncertainty * buffering
        required_volume = float(self._compute_required_volume())
        combined = error + (0.1 * required_volume if self.rule_flags.required_volume_term else 0.0)
        signal = alpha * combined
        mapped = float(np.tanh(signal)) if self.rule_flags.tanh_mapping else float(np.clip(signal, 0.0, 1.0))
        ideal_volume = MIN_ACTION_VOLUME_ML + (MAX_ACTION_VOLUME_ML - MIN_ACTION_VOLUME_ML) * mapped
        volume = float(candidate_volumes[np.argmin(np.abs(candidate_volumes - ideal_volume))])
        action = ControllerAction(
            stop=False,
            reagent=reagent,
            volume_ml=round(volume, 2),
            titrant_concentration_m=self.titrant_concentration_m,
            diagnostics={
                "variant_flags": self.rule_flags.__dict__,
                "required_volume_ml": required_volume,
                "ideal_volume_ml": ideal_volume,
                "estimated_pair_count": estimate.pair_count,
            },
        )
        self.pending_action = action
        return action


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_episode(task: ControlTask, variant: str, particles: int, seed: int) -> dict:
    controller = RuleAblationPFController(
        particles=particles,
        seed=seed,
        max_steps=MAX_STEPS,
        max_total_dose_ml=MAX_TOTAL_DOSE_ML,
        rule_flags=FLAGS[variant],
    )
    controller.reset(
        task.initial_ph,
        task.target_ph,
        task.initial_volume_ml,
        task.initial_base_moles,
        0.0,
    )
    true_ph = float(task.initial_ph)
    base_moles = float(task.initial_base_moles)
    acid_moles = 0.0
    total_volume_ml = float(task.initial_volume_ml)
    crossings = 0
    controller_ms = 0.0
    while not controller.status()["done"]:
        started = time.perf_counter()
        action = controller.recommend()
        controller_ms += (time.perf_counter() - started) * 1000.0
        if action.stop:
            break
        previous_ph = true_ph
        actual = float(action.volume_ml)
        total_volume_ml += actual
        if action.reagent == "base":
            base_moles += 0.1 * actual / 1000.0
        else:
            acid_moles += 0.1 * actual / 1000.0
        true_ph = solve_ph_scalar(
            task.analyte_conc_m,
            task.pka_values,
            task.initial_volume_ml,
            SolutionState(total_volume_ml, base_moles, acid_moles),
        )
        measured = float(np.round(true_ph, 2))
        crossings += int((previous_ph - task.target_ph) * (true_ph - task.target_ph) < 0.0)
        started = time.perf_counter()
        controller.observe(measured, actual, action.reagent)
        controller_ms += (time.perf_counter() - started) * 1000.0
    error = abs(true_ph - task.target_ph)
    status = controller.status()
    return {
        "variant": variant,
        "task_id": task.task_id,
        "task_seed": task.seed,
        "direction": task.direction,
        "difficulty": task.difficulty,
        "true_pair_count": len(task.pka_values),
        "success": int(error <= 0.10),
        "strict_success": int(error <= 0.05),
        "severe_failure": int(error > 0.50),
        "steps": int(status["steps"]),
        "crossings": crossings,
        "total_volume_ml": float(status["total_added_volume_ml"]),
        "final_abs_error": error,
        "controller_ms_per_step": controller_ms / max(1, int(status["steps"])),
    }


def run_task(payload) -> list[dict]:
    task, particles, benchmark_seed = payload
    common_seed = benchmark_seed * 10_000_019 + task.task_id * 1009
    return [run_episode(task, variant, particles, common_seed) for variant in VARIANTS]


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["success"]]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean([row["success"] for row in rows])),
        "strict_success_rate_percent": 100.0 * float(np.mean([row["strict_success"] for row in rows])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([row["severe_failure"] for row in rows])),
        "successful_steps_mean": float(np.mean([row["steps"] for row in successful])) if successful else math.nan,
        "crossings_mean": float(np.mean([row["crossings"] for row in rows])),
        "total_volume_mean_ml": float(np.mean([row["total_volume_ml"] for row in rows])),
        "final_abs_error_mean": float(np.mean([row["final_abs_error"] for row in rows])),
        "controller_ms_per_step_mean": float(np.mean([row["controller_ms_per_step"] for row in rows])),
    }


def exact_mcnemar(reference: list[int], comparison: list[int]) -> tuple[int, int, float]:
    reference_only = sum(a == 1 and b == 0 for a, b in zip(reference, comparison))
    comparison_only = sum(a == 0 and b == 1 for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(reference_only, comparison_only), discordant, 0.5).pvalue)
    return reference_only, comparison_only, p_value


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["raw_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * float(rows[index]["raw_p"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description="One-factor internal dose-rule ablation for the released new PF")
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
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    all_rows = []
    per_seed = []
    for benchmark_seed in args.seeds:
        tasks = generate_tasks(
            5_000_000 + benchmark_seed,
            args.tasks_per_seed,
            f"pf_internal_ablation_{benchmark_seed}",
        )
        save_tasks(output / f"seed_{benchmark_seed}_tasks.jsonl", tasks)
        payloads = [(task, args.particles, benchmark_seed) for task in tasks]
        if workers == 1:
            results = map(run_task, payloads)
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            results = executor.map(run_task, payloads, chunksize=2)
        rows = []
        try:
            for index, task_rows in enumerate(results, 1):
                for row in task_rows:
                    row["benchmark_seed"] = benchmark_seed
                rows.extend(task_rows)
                if index % 50 == 0 or index == len(tasks):
                    print(f"internal-ablation seed {benchmark_seed}: {index}/{len(tasks)}", flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        all_rows.extend(rows)
        for variant in VARIANTS:
            per_seed.append(
                {
                    "benchmark_seed": benchmark_seed,
                    "variant": variant,
                    **summarize([row for row in rows if row["variant"] == variant]),
                }
            )

    aggregate = []
    for variant in VARIANTS:
        subset = [row for row in per_seed if row["variant"] == variant]
        result = {"variant": variant, "seed_runs": len(subset)}
        for metric in subset[0]:
            if metric in {"benchmark_seed", "variant"}:
                continue
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            result[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
        aggregate.append(result)

    lookup = {
        variant: {
            (row["benchmark_seed"], row["task_id"]): row
            for row in all_rows
            if row["variant"] == variant
        }
        for variant in VARIANTS
    }
    tests = []
    continuous = []
    for variant in VARIANTS[1:]:
        keys = sorted(set(lookup["full"]) & set(lookup[variant]))
        full_success = [lookup["full"][key]["success"] for key in keys]
        variant_success = [lookup[variant][key]["success"] for key in keys]
        full_only, variant_only, p_value = exact_mcnemar(full_success, variant_success)
        tests.append(
            {
                "comparison": f"{variant}_minus_full",
                "paired_tasks": len(keys),
                "full_only_success": full_only,
                "variant_only_success": variant_only,
                "success_difference_pp": 100.0 * (np.mean(variant_success) - np.mean(full_success)),
                "raw_p": p_value,
            }
        )
        for metric in ("steps", "crossings", "total_volume_ml", "final_abs_error"):
            full_values = np.asarray([lookup["full"][key][metric] for key in keys], dtype=float)
            variant_values = np.asarray([lookup[variant][key][metric] for key in keys], dtype=float)
            differences = variant_values - full_values
            try:
                paired_p = float(wilcoxon(variant_values, full_values, zero_method="zsplit").pvalue)
            except ValueError:
                paired_p = 1.0
            continuous.append(
                {
                    "comparison": f"{variant}_minus_full",
                    "metric": metric,
                    "paired_tasks": len(keys),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "raw_p": paired_p,
                }
            )
    holm(tests)
    holm(continuous)
    write_csv(output / "all_task_results.csv", all_rows)
    write_csv(output / "per_seed_summary.csv", per_seed)
    write_csv(output / "aggregate_summary.csv", aggregate)
    write_csv(output / "paired_success_tests.csv", tests)
    write_csv(output / "paired_continuous_tests.csv", continuous)
    (output / "INTERNAL_RULE_ABLATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "tasks_per_seed": args.tasks_per_seed,
                "particles": args.particles,
                "variants": VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
