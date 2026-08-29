from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmark_core import StressScenario, Task, exact_mcnemar, generate_tasks, holm_adjust, portable_settings


@dataclass(frozen=True)
class Condition:
    name: str
    reference_strategy: str
    measurement_noise_sd: float
    likelihood_sigma: float


REFERENCE_CONDITIONS = [
    Condition("nominal_self_reference", "nominal", 0.0, 0.01),
    Condition("fixed_generic_3_4p5_6", "generic", 0.0, 0.01),
    Condition("fixed_generic_minus_1", "generic_minus_1", 0.0, 0.01),
    Condition("fixed_generic_plus_1", "generic_plus_1", 0.0, 0.01),
    Condition("fixed_generic_plus_2", "generic_plus_2", 0.0, 0.01),
    Condition("random_wide_0_14", "random_wide", 0.0, 0.01),
    Condition("oracle_solution_pka_reference", "oracle", 0.0, 0.01),
]

NOISE_CONDITIONS = [
    Condition("noise_0_like_0p01", "nominal", 0.00, 0.01),
    Condition("noise_0p01_like_0p01", "nominal", 0.01, 0.01),
    Condition("noise_0p03_like_0p01", "nominal", 0.03, 0.01),
    Condition("noise_0p03_like_0p03", "nominal", 0.03, 0.03),
    Condition("noise_0p05_like_0p01", "nominal", 0.05, 0.01),
    Condition("noise_0p05_like_0p05", "nominal", 0.05, 0.05),
    Condition("noise_0p10_like_0p01", "nominal", 0.10, 0.01),
    Condition("noise_0p10_like_0p10", "nominal", 0.10, 0.10),
]


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("major_review_bayesian_robustness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_reference(env, task: Task, strategy: str, rng: np.random.Generator) -> None:
    if strategy == "nominal":
        return
    if strategy == "generic":
        reference = np.array([3.0, 4.5, 6.0])
    elif strategy == "generic_minus_1":
        reference = np.array([2.0, 3.5, 5.0])
    elif strategy == "generic_plus_1":
        reference = np.array([4.0, 5.5, 7.0])
    elif strategy == "generic_plus_2":
        reference = np.array([5.0, 6.5, 8.0])
    elif strategy == "random_wide":
        reference = rng.uniform(0.0, 14.0, size=3)
    elif strategy == "oracle":
        reference = np.array([3.0, 4.5, 6.0])
        count = min(3, len(task.pka_values))
        reference[:count] = np.asarray(task.pka_values[:count], dtype=float)
    else:
        raise ValueError(f"Unknown reference strategy: {strategy}")
    env.ref_pKa = reference.astype(float)


def bind_noise_and_likelihood(
    env,
    module: ModuleType,
    noise_sd: float,
    likelihood_sigma: float,
    rng: np.random.Generator,
) -> None:
    original_simulate = env.simulate_observed_ph
    env.last_true_ph = float(env.current_ph)

    def noisy_simulate(self):
        true_ph = float(original_simulate())
        self.last_true_ph = true_ph
        noisy = true_ph + float(rng.normal(0.0, noise_sd))
        return round(float(np.clip(noisy, 0.0, 14.0)), 2)

    def update_with_sigma(self, action, observed_ph):
        del action
        sampled_pka = np.random.normal(self.pKa_list, self.pKa_std, size=(self.num_particles, self.num_buffers))
        sampled_total = np.random.normal(
            self.buffer_total_moles,
            self.buffer_total_std,
            size=(self.num_particles, self.num_buffers),
        )
        effective_pka = self.get_effective_pka_matrix(sampled_pka)
        total_volume_l = (module.TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        analyte_moles = module.TITRATED_VOLUME / 1000.0 * module.ANALYTE_CONC
        c_analyte = analyte_moles / total_volume_l
        c_na = self.base_added_moles / total_volume_l
        c_hcl = self.acid_added_moles / total_volume_l
        predicted = module.solve_pH_batch(c_analyte, c_na, c_hcl, effective_pka)
        sigma = max(float(likelihood_sigma), 1e-6)
        log_weights = -0.5 * ((observed_ph - predicted) / sigma) ** 2
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights) + 1e-12
        weights /= weights.sum()
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        resampled_pka = sampled_pka[indices]
        resampled_total = sampled_total[indices]
        self.pKa_list = resampled_pka.mean(axis=0)
        self.pKa_std = resampled_pka.std(axis=0) + 1e-3
        self.buffer_total_moles = resampled_total.mean(axis=0)
        self.buffer_total_std = resampled_total.std(axis=0) + 1e-3

    env.simulate_observed_ph = MethodType(noisy_simulate, env)
    env.update_posteriors = MethodType(update_with_sigma, env)
    env._true_ph_function = original_simulate


def posterior_match_error(true_pkas: tuple[float, ...], slots: np.ndarray) -> float:
    truth = np.asarray(true_pkas, dtype=float)
    costs = np.abs(truth[:, None] - np.asarray(slots, dtype=float)[None, :])
    true_indices, slot_indices = linear_sum_assignment(costs)
    return float(costs[true_indices, slot_indices].max())


def run_task(module: ModuleType, task: Task, condition: Condition, particles: int, common_seed: int) -> dict:
    np.random.seed(common_seed)
    auxiliary_rng = np.random.default_rng(common_seed + 17)
    env = module.PHAdjustmentEnv(num_particles=particles)
    env.initialize(task.acid_type, list(task.pka_values), task.initial_ph, task.target_ph, module.MAX_STEPS)
    set_reference(env, task, condition.reference_strategy, auxiliary_rng)
    bind_noise_and_likelihood(
        env,
        module,
        condition.measurement_noise_sd,
        condition.likelihood_sigma,
        auxiliary_rng,
    )

    overshoots = 0
    if abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD:
        env.done = True
    if not env.done:
        action, _ = env.select_best_action()
        while not env.done:
            measured_ph, _, done, info = env.step(action, mode="Simulate")
            env.update_posteriors(action, measured_ph)
            overshoots += int(bool(info.get("crossed_target", False)))
            if done:
                break
            action, _ = env.select_best_action()

    final_true_ph = float(env._true_ph_function())
    measured_success = abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD
    true_success = abs(final_true_ph - env.target_ph) <= module.SUCCESS_THRESHOLD
    return {
        "condition": condition.name,
        "seed": task.seed,
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "pka_values": json.dumps(task.pka_values),
        "target_ph": task.target_ph,
        "steps": env.steps_taken,
        "final_measured_ph": float(env.current_ph),
        "final_true_ph": final_true_ph,
        "measured_success": measured_success,
        "true_success": true_success,
        "false_stop": measured_success and not true_success,
        "overshoots": overshoots,
        "posterior_slots": json.dumps(env.pKa_list.tolist()),
        "max_pka_match_error": posterior_match_error(task.pka_values, env.pKa_list),
    }


def run_condition_seed(job: tuple[str, int, int, Condition, int]) -> list[dict]:
    source, seed, task_count, condition, particles = job
    module = load_module(Path(source))
    tasks = generate_tasks(seed, task_count, StressScenario("nominal"))
    rows: list[dict] = []
    for index, task in enumerate(tasks, 1):
        common_seed = seed * 1_000_003 + task.task_id
        rows.append(run_task(module, task, condition, particles, common_seed))
        if index % 100 == 0:
            print(f"{condition.name}, seed {seed}: {index}/{len(tasks)}")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for condition in sorted({row["condition"] for row in rows}):
        seed_rows: list[dict] = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            subset = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            if not subset:
                continue
            successes = [row for row in subset if row["true_success"]]
            total_steps = sum(int(row["steps"]) for row in subset)
            seed_rows.append(
                {
                    "success": 100.0 * len(successes) / len(subset),
                    "steps": statistics.mean(int(row["steps"]) for row in successes) if successes else math.nan,
                    "overshoot": 100.0 * sum(int(row["overshoots"]) for row in subset) / total_steps if total_steps else 0.0,
                    "false_stop": 100.0 * sum(bool(row["false_stop"]) for row in subset) / len(subset),
                    "pka_error": statistics.mean(float(row["max_pka_match_error"]) for row in subset),
                }
            )
        aggregate: dict[str, float | int | str] = {"condition": condition, "seeds": len(seed_rows)}
        for metric in ("success", "steps", "overshoot", "false_stop", "pka_error"):
            values = [float(item[metric]) for item in seed_rows if not math.isnan(float(item[metric]))]
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            aggregate[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(aggregate)
    return output


def paired_tests(rows: list[dict], baseline: str) -> list[dict]:
    baseline_map = {(int(row["seed"]), int(row["task_id"])): bool(row["true_success"]) for row in rows if row["condition"] == baseline}
    tests: list[dict] = []
    for condition in sorted({row["condition"] for row in rows if row["condition"] != baseline}):
        condition_map = {(int(row["seed"]), int(row["task_id"])): bool(row["true_success"]) for row in rows if row["condition"] == condition}
        keys = sorted(set(baseline_map) & set(condition_map))
        tests.append(
            {
                "baseline": baseline,
                "condition": condition,
                "matched_tasks": len(keys),
                **exact_mcnemar([baseline_map[key] for key in keys], [condition_map[key] for key in keys]),
            }
        )
    adjusted = holm_adjust([float(row["p_value_exact_two_sided"]) for row in tests])
    for row, value in zip(tests, adjusted):
        row["p_value_holm"] = value
    return tests


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    repo = base.parent / "repo_reviewcopy"
    bundled_source = base / "inputs" / "bayesian_controller.py"
    parser = argparse.ArgumentParser(description="Bayesian pKa-reference and observation-noise sensitivity analysis.")
    parser.add_argument("mode", choices=["reference", "noise"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--tasks-per-seed", type=int, default=1000)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--bayesian-source",
        type=Path,
        default=bundled_source if bundled_source.exists() else repo / "main_code3_modularcopy" / "bayesian_and_data" / "03_evaluate_bayesian_final.py",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path(__file__).resolve().parent
    output_dir = args.output_dir or base / "results" / f"bayesian_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_module(args.bayesian_source.resolve())
    conditions = REFERENCE_CONDITIONS if args.mode == "reference" else NOISE_CONDITIONS
    baseline = conditions[0].name
    rows: list[dict] = []

    if args.workers > 1:
        del module
        jobs = [
            (str(args.bayesian_source.resolve()), seed, args.tasks_per_seed, condition, args.particles)
            for seed in args.seeds
            for condition in conditions
        ]
        worker_count = min(args.workers, len(jobs))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for batch in executor.map(run_condition_seed, jobs):
                rows.extend(batch)
    else:
        for seed in args.seeds:
            tasks = generate_tasks(seed, args.tasks_per_seed, StressScenario("nominal"))
            for condition in conditions:
                for index, task in enumerate(tasks, 1):
                    common_seed = seed * 1_000_003 + task.task_id
                    rows.append(run_task(module, task, condition, args.particles, common_seed))
                    if index % 100 == 0:
                        print(f"{condition.name}, seed {seed}: {index}/{len(tasks)}")

    summary = summarize(rows)
    tests = paired_tests(rows, baseline)
    write_csv(output_dir / "per_task_results.csv", rows)
    write_csv(output_dir / "aggregate_summary.csv", summary)
    write_csv(output_dir / "paired_tests.csv", tests)
    (output_dir / "settings.json").write_text(
        json.dumps(portable_settings(vars(args), base), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
