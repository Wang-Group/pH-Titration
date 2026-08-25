from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np

from benchmark_core import (
    EpisodeResult,
    NeuralVolumePolicy,
    StressScenario,
    Task,
    exact_mcnemar,
    generate_tasks,
    holm_adjust,
    load_tasks_csv,
    portable_settings,
    run_neural_policy,
    summarize_results,
)


DEFAULT_SEEDS = [101, 202, 303, 404, 555]


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpertRuleController:
    def reset(self, current_ph: float, target_ph: float) -> None:
        self.last_volume_equivalent: float | None = None
        self.last_overshot = False
        self.net_equivalent_ml = 0.0
        self.lower_bracket = 0.0 if current_ph <= target_ph else None
        self.upper_bracket = 0.0 if current_ph >= target_ph else None

    @staticmethod
    def bucket(error: float) -> float:
        if error > 6.0:
            return 3.00
        if error > 4.0:
            return 2.50
        if error > 2.0:
            return 2.00
        if error > 1.0:
            return 1.00
        if error > 0.5:
            return 0.50
        if error > 0.2:
            return 0.20
        return 0.05

    def volume(self, env) -> float:
        error = abs(env.target_ph - env.current_ph)
        has_bracket = self.lower_bracket is not None and self.upper_bracket is not None
        if has_bracket:
            target_net = (self.lower_bracket + self.upper_bracket) / 2.0
            equivalent = max(0.01, abs(target_net - self.net_equivalent_ml))
        else:
            equivalent = self.bucket(error)
            if self.last_overshot and self.last_volume_equivalent is not None:
                equivalent = max(0.01, self.last_volume_equivalent * 0.5)
        if error <= 0.30:
            equivalent = min(equivalent, 0.10)
        if error <= 0.15:
            equivalent = min(equivalent, 0.03)
        return equivalent

    def observe(self, env, reagent: str, volume: float, crossed: bool) -> None:
        scale = env.reagents[reagent] / env.reagents["Dilute base 1"]
        equivalent = volume * scale
        self.net_equivalent_ml += equivalent if "base" in reagent.lower() else -equivalent
        if env.current_ph < env.target_ph:
            self.lower_bracket = self.net_equivalent_ml if self.lower_bracket is None else max(self.lower_bracket, self.net_equivalent_ml)
        elif env.current_ph > env.target_ph:
            self.upper_bracket = self.net_equivalent_ml if self.upper_bracket is None else min(self.upper_bracket, self.net_equivalent_ml)
        self.last_volume_equivalent = equivalent
        self.last_overshot = crossed


class AdaptivePIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float, overshoot_decay: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.overshoot_decay = overshoot_decay

    def reset(self, current_ph: float, target_ph: float) -> None:
        del current_ph, target_ph
        self.integral = 0.0
        self.previous_error: float | None = None

    def volume(self, env) -> float:
        error = env.target_ph - env.current_ph
        if self.previous_error is not None and error * self.previous_error < 0:
            self.integral *= self.overshoot_decay
        self.integral = float(np.clip(self.integral + error, -self.integral_limit, self.integral_limit))
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error
        return abs(self.kp * error + self.ki * self.integral + self.kd * derivative)

    def observe(self, env, reagent: str, volume: float, crossed: bool) -> None:
        del env, reagent, volume, crossed


def choose_reagent(env) -> str:
    suffix = "2" if env.use_secondary_reagents else "1"
    return f"Dilute base {suffix}" if env.current_ph < env.target_ph else f"Dilute acid {suffix}"


def execute_rule_task(module: ModuleType, task: Task, method: str, controller) -> EpisodeResult:
    env = module.PHAdjustmentEnv(num_particles=1)
    env.initialize(task.acid_type, list(task.pka_values), task.initial_ph, task.target_ph, module.MAX_STEPS)
    if abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD:
        env.done = True
    controller.reset(env.current_ph, env.target_ph)
    overshoots = 0
    while not env.done:
        reagent = choose_reagent(env)
        equivalent = max(0.01, controller.volume(env))
        scale = env.reagents[reagent] / env.reagents["Dilute base 1"]
        maximum = 9.99
        volume = round(float(np.clip(equivalent / scale, 0.01, maximum)) / 0.01) * 0.01
        _, _, _, info = env.step((reagent, volume), mode="Simulate")
        if abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD:
            env.done = True
        crossed = bool(info.get("crossed_target", False))
        overshoots += int(crossed)
        controller.observe(env, reagent, volume, crossed)
    success = abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD
    return EpisodeResult(
        seed=task.seed,
        task_id=task.task_id,
        scenario="nominal",
        method=method,
        acid_type=task.acid_type,
        pka_values=json.dumps(task.pka_values),
        initial_ph=task.initial_ph,
        target_ph=task.target_ph,
        final_true_ph=float(env.current_ph),
        final_measured_ph=float(env.current_ph),
        steps=int(env.steps_taken),
        true_success=success,
        measured_success=success,
        overshoots=overshoots,
        acid_added_ml=float(env.acid_volume),
        base_added_ml=float(env.base_volume),
    )


def execute_bayesian_task(module: ModuleType, task: Task, particles: int, rng_seed: int) -> EpisodeResult:
    np.random.seed(rng_seed)
    env = module.PHAdjustmentEnv(num_particles=particles)
    env.initialize(task.acid_type, list(task.pka_values), task.initial_ph, task.target_ph, module.MAX_STEPS)
    if abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD:
        env.done = True
    overshoots = 0
    if not env.done:
        action, _ = env.select_best_action()
        while not env.done:
            current_ph, _, done, info = env.step(action, mode="Simulate")
            env.update_posteriors(action, current_ph)
            if abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD:
                env.done = True
                done = True
            overshoots += int(bool(info.get("crossed_target", False)))
            if done:
                break
            action, _ = env.select_best_action()
    success = abs(env.current_ph - env.target_ph) <= module.SUCCESS_THRESHOLD
    return EpisodeResult(
        seed=task.seed,
        task_id=task.task_id,
        scenario="nominal",
        method="bayesian",
        acid_type=task.acid_type,
        pka_values=json.dumps(task.pka_values),
        initial_ph=task.initial_ph,
        target_ph=task.target_ph,
        final_true_ph=float(env.current_ph),
        final_measured_ph=float(env.current_ph),
        steps=int(env.steps_taken),
        true_success=success,
        measured_success=success,
        overshoots=overshoots,
        acid_added_ml=float(env.acid_volume),
        base_added_ml=float(env.base_volume),
    )


def run_seed_job(job: tuple) -> list[EpisodeResult]:
    (
        seed,
        tasks_per_seed,
        methods,
        imitation_weights,
        rl_weights,
        device,
        bayesian_particles,
        bayesian_source,
        seed_555_csv,
        regenerate_seed_555,
        pid_params,
    ) = job
    module = load_module(Path(bayesian_source), f"major_review_bayesian_{seed}")
    imitation = NeuralVolumePolicy(Path(imitation_weights), device) if "imitation" in methods else None
    rl = NeuralVolumePolicy(Path(rl_weights), device) if "rl" in methods else None
    nominal = StressScenario("nominal")
    if seed == 555 and not regenerate_seed_555 and Path(seed_555_csv).exists():
        tasks = load_tasks_csv(Path(seed_555_csv), seed)[:tasks_per_seed]
    else:
        tasks = generate_tasks(seed, tasks_per_seed, nominal)
    results: list[EpisodeResult] = []
    for index, task in enumerate(tasks, 1):
        common_seed = seed * 1_000_003 + task.task_id
        if "bayesian" in methods:
            results.append(execute_bayesian_task(module, task, bayesian_particles, common_seed))
        if imitation is not None:
            results.append(run_neural_policy(imitation, task, nominal, "imitation", common_seed))
        if rl is not None:
            results.append(run_neural_policy(rl, task, nominal, "rl", common_seed))
        if "expert_rule" in methods:
            results.append(execute_rule_task(module, task, "expert_rule", ExpertRuleController()))
        if "pid" in methods:
            results.append(
                execute_rule_task(
                    module,
                    task,
                    "pid",
                    AdaptivePIDController(**pid_params),
                )
            )
        if index % 250 == 0:
            print(f"seed {seed}: {index}/{len(tasks)} tasks")
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_summaries(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    methods = sorted({row["method"] for row in rows})
    metrics = ["success_rate_percent", "successful_steps_mean", "overshoot_rate_percent", "final_abs_error_mean"]
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        result: dict[str, float | int | str] = {"method": method, "seeds": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if not math.isnan(float(row[metric]))]
            result[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
            result[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def mcnemar_rows(results: list[EpisodeResult]) -> list[dict]:
    rows: list[dict] = []
    methods = sorted({row.method for row in results})
    seeds = sorted({row.seed for row in results})
    for scope_seed in seeds + [None]:
        scope = [row for row in results if scope_seed is None or row.seed == scope_seed]
        by_method = {
            method: {(row.seed, row.task_id): row.true_success for row in scope if row.method == method}
            for method in methods
        }
        scope_rows: list[dict] = []
        for method_a, method_b in itertools.combinations(methods, 2):
            keys = sorted(set(by_method[method_a]) & set(by_method[method_b]))
            stats = exact_mcnemar(
                [by_method[method_a][key] for key in keys],
                [by_method[method_b][key] for key in keys],
            )
            scope_rows.append(
                {
                    "scope": "pooled" if scope_seed is None else f"seed_{scope_seed}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "matched_tasks": len(keys),
                    **stats,
                }
            )
        adjusted = holm_adjust([float(row["p_value_exact_two_sided"]) for row in scope_rows])
        for row, value in zip(scope_rows, adjusted):
            row["p_value_holm"] = value
        rows.extend(scope_rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    review_root = base.parent
    repo = review_root / "repo_reviewcopy"
    bundled_source = base / "inputs" / "bayesian_controller.py"
    bundled_tasks = base / "inputs" / "experiment_summary.csv"
    parser = argparse.ArgumentParser(description="Multi-seed matched benchmark and exact paired tests.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bayesian", "imitation", "rl", "expert_rule", "pid"],
        default=["bayesian", "imitation", "rl", "expert_rule", "pid"],
    )
    parser.add_argument("--imitation-weights", type=Path)
    parser.add_argument("--rl-weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bayesian-particles", type=int, default=1000)
    parser.add_argument(
        "--bayesian-source",
        type=Path,
        default=bundled_source if bundled_source.exists() else repo / "main_code3_modularcopy" / "bayesian_and_data" / "03_evaluate_bayesian_final.py",
    )
    parser.add_argument(
        "--seed-555-csv",
        type=Path,
        default=bundled_tasks if bundled_tasks.exists() else repo / "experiment_summary.csv",
    )
    parser.add_argument("--regenerate-seed-555", action="store_true")
    parser.add_argument("--pid-kp", type=float, default=0.32)
    parser.add_argument("--pid-ki", type=float, default=0.012)
    parser.add_argument("--pid-kd", type=float, default=0.08)
    parser.add_argument("--pid-integral-limit", type=float, default=12.0)
    parser.add_argument("--pid-overshoot-decay", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "multiseed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path(__file__).resolve().parent
    if "imitation" in args.methods and args.imitation_weights is None:
        raise SystemExit("--imitation-weights is required when imitation is selected.")
    if "rl" in args.methods and args.rl_weights is None:
        raise SystemExit("--rl-weights is required when rl is selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nominal = StressScenario("nominal")
    all_results: list[EpisodeResult] = []

    if args.workers > 1:
        if args.device != "cpu":
            raise SystemExit("--workers greater than 1 is supported only with --device cpu.")
        pid_params = {
            "kp": args.pid_kp,
            "ki": args.pid_ki,
            "kd": args.pid_kd,
            "integral_limit": args.pid_integral_limit,
            "overshoot_decay": args.pid_overshoot_decay,
        }
        jobs = [
            (
                seed,
                args.tasks_per_seed,
                tuple(args.methods),
                str(args.imitation_weights.resolve()) if args.imitation_weights else "",
                str(args.rl_weights.resolve()) if args.rl_weights else "",
                args.device,
                args.bayesian_particles,
                str(args.bayesian_source.resolve()),
                str(args.seed_555_csv.resolve()),
                args.regenerate_seed_555,
                pid_params,
            )
            for seed in args.seeds
        ]
        worker_count = min(args.workers, len(jobs))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for seed_results in executor.map(run_seed_job, jobs):
                all_results.extend(seed_results)
    else:
        module = load_module(args.bayesian_source.resolve(), "major_review_bayesian")
        imitation = NeuralVolumePolicy(args.imitation_weights.resolve(), args.device) if "imitation" in args.methods else None
        rl = NeuralVolumePolicy(args.rl_weights.resolve(), args.device) if "rl" in args.methods else None
        for seed in args.seeds:
            if seed == 555 and not args.regenerate_seed_555 and args.seed_555_csv.exists():
                tasks = load_tasks_csv(args.seed_555_csv, seed)[: args.tasks_per_seed]
            else:
                tasks = generate_tasks(seed, args.tasks_per_seed, nominal)
            for index, task in enumerate(tasks, 1):
                common_seed = seed * 1_000_003 + task.task_id
                if "bayesian" in args.methods:
                    all_results.append(execute_bayesian_task(module, task, args.bayesian_particles, common_seed))
                if imitation is not None:
                    all_results.append(run_neural_policy(imitation, task, nominal, "imitation", common_seed))
                if rl is not None:
                    all_results.append(run_neural_policy(rl, task, nominal, "rl", common_seed))
                if "expert_rule" in args.methods:
                    all_results.append(execute_rule_task(module, task, "expert_rule", ExpertRuleController()))
                if "pid" in args.methods:
                    pid = AdaptivePIDController(
                        args.pid_kp,
                        args.pid_ki,
                        args.pid_kd,
                        args.pid_integral_limit,
                        args.pid_overshoot_decay,
                    )
                    all_results.append(execute_rule_task(module, task, "pid", pid))
                if index % 250 == 0:
                    print(f"seed {seed}: {index}/{len(tasks)} tasks")

    task_rows = [row.to_dict() for row in all_results]
    write_csv(args.output_dir / "per_task_results.csv", task_rows)

    seed_summaries: list[dict] = []
    for seed in args.seeds:
        for method in args.methods:
            subset = [row for row in all_results if row.seed == seed and row.method == method]
            if subset:
                seed_summaries.append({"seed": seed, "method": method, **summarize_results(subset)})
    write_csv(args.output_dir / "per_seed_summary.csv", seed_summaries)
    aggregate = aggregate_seed_summaries(seed_summaries)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)

    tests = mcnemar_rows(all_results)
    write_csv(args.output_dir / "mcnemar_tests.csv", tests)
    payload = {
        "settings": portable_settings(vars(args), base),
        "aggregate": aggregate,
        "pooled_mcnemar": [row for row in tests if row["scope"] == "pooled"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
