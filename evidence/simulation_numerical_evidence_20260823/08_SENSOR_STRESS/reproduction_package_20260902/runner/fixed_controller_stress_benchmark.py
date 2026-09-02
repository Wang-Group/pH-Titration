from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
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
from new_pf_controller import RobustPFController
from new_rl_controller import PPOVolumeController
from new_rl_numpy_controller import NumpyPPOVolumeController
from task_distribution import ControlTask, generate_tasks, load_tasks, save_tasks


METHODS = ("new_pf", "new_ppo")
PPO_CHECKPOINT = CONTROLLER_DIR / "models" / "ppo_seed_303.pth"
PPO_NUMPY_CHECKPOINT = CONTROLLER_DIR / "models" / "ppo_seed_303_numpy.npz"
SUCCESS_TOLERANCE = 0.10
STRICT_TOLERANCE = 0.05
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0
_PPO_CONTROLLER_CACHE: dict[tuple[str, str], object] = {}


@dataclass(frozen=True)
class StressRegime:
    name: str
    distribution: str = "nominal"
    observation_noise_sd: float = 0.0
    episode_bias_sd: float = 0.0
    drift_increment_sd: float = 0.0
    response_fraction: float = 1.0
    actuator_log_sd: float = 0.0
    titrant_scale: float = 1.0


REGIMES = {
    item.name: item
    for item in (
        StressRegime("nominal"),
        StressRegime("close_pka", distribution="close_pka"),
        StressRegime("wide_concentration", distribution="wide_concentration"),
        StressRegime("observation_noise_0p01", observation_noise_sd=0.01),
        StressRegime("observation_noise_0p03", observation_noise_sd=0.03),
        StressRegime("observation_noise_0p05", observation_noise_sd=0.05),
        StressRegime("observation_noise_0p10", observation_noise_sd=0.10),
        StressRegime("episode_bias_0p10", episode_bias_sd=0.10),
        StressRegime("random_walk_drift_0p01", drift_increment_sd=0.01),
        StressRegime("response_fraction_0p60", response_fraction=0.60),
        StressRegime("response_fraction_0p70", response_fraction=0.70),
        StressRegime("actuator_log_sd_0p10", actuator_log_sd=0.10),
        StressRegime("titrant_scale_0p90", titrant_scale=0.90),
        StressRegime("titrant_scale_1p10", titrant_scale=1.10),
        StressRegime(
            "combined_unseen",
            observation_noise_sd=0.05,
            episode_bias_sd=0.03,
            drift_increment_sd=0.005,
            response_fraction=0.70,
            actuator_log_sd=0.10,
            titrant_scale=0.90,
        ),
    )
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {
        "task_seed",
        "task_id",
        "true_pair_count",
        "true_success",
        "strict_success",
        "severe_failure",
        "measured_success",
        "false_stop",
        "steps",
        "crossings",
        "benchmark_seed",
    }
    text_fields = {
        "method",
        "regime",
        "distribution",
        "direction",
        "difficulty",
        "pka_family",
        "stop_reason",
    }
    for row in rows:
        for key, value in list(row.items()):
            if key in text_fields:
                continue
            row[key] = int(value) if key in integer_fields else float(value)
    return rows


def make_noise_schedule(seed: int, regime: StressRegime) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    return {
        "episode_bias": float(rng.normal(0.0, regime.episode_bias_sd)),
        "observation_noise": rng.normal(0.0, regime.observation_noise_sd, MAX_STEPS + 1),
        "drift_increment": rng.normal(0.0, regime.drift_increment_sd, MAX_STEPS + 1),
        "actuator_factor": rng.lognormal(0.0, regime.actuator_log_sd, MAX_STEPS),
    }


def sensor_reading(
    true_ph: float,
    previous_measured: float,
    step: int,
    regime: StressRegime,
    schedule: dict,
    accumulated_drift: float,
) -> tuple[float, float]:
    if step > 0:
        accumulated_drift += float(schedule["drift_increment"][step])
    equilibrium_reading = true_ph + float(schedule["episode_bias"]) + accumulated_drift
    if step == 0:
        responded = equilibrium_reading
    else:
        responded = previous_measured + regime.response_fraction * (
            equilibrium_reading - previous_measured
        )
    observed = responded + float(schedule["observation_noise"][step])
    return float(np.round(np.clip(observed, 0.0, 14.0), 2)), accumulated_drift


def build_controller(method: str, seed: int, device: str, ppo_backend: str):
    if method == "new_pf":
        controller = RobustPFController(
            particles=1000,
            seed=seed,
            max_steps=MAX_STEPS,
            max_total_dose_ml=MAX_TOTAL_DOSE_ML,
            titrant_concentration_m=0.1,
        )
        return controller
    cache_key = (ppo_backend, device)
    controller = _PPO_CONTROLLER_CACHE.get(cache_key)
    if controller is None:
        if ppo_backend == "numpy":
            controller = NumpyPPOVolumeController(
                PPO_NUMPY_CHECKPOINT,
                max_steps=MAX_STEPS,
                max_total_dose_ml=MAX_TOTAL_DOSE_ML,
                titrant_concentration_m=0.1,
            )
        else:
            controller = PPOVolumeController(
                PPO_CHECKPOINT,
                device=device,
                max_steps=MAX_STEPS,
                max_total_dose_ml=MAX_TOTAL_DOSE_ML,
                titrant_concentration_m=0.1,
            )
        _PPO_CONTROLLER_CACHE[cache_key] = controller
    return controller


def run_episode(
    task: ControlTask,
    method: str,
    regime: StressRegime,
    random_seed: int,
    device: str,
    ppo_backend: str,
) -> dict:
    schedule = make_noise_schedule(random_seed, regime)
    measured_ph, drift = sensor_reading(
        float(task.initial_ph),
        float(task.initial_ph),
        0,
        regime,
        schedule,
        0.0,
    )
    controller = build_controller(method, random_seed + 17, device, ppo_backend)
    if method == "new_pf":
        controller.reset(
            measured_ph,
            task.target_ph,
            task.initial_volume_ml,
            task.initial_base_moles,
            0.0,
        )
    else:
        controller.reset(measured_ph, task.target_ph)

    true_ph = float(task.initial_ph)
    previous_true_ph = true_ph
    base_moles = float(task.initial_base_moles)
    acid_moles = 0.0
    total_actual = 0.0
    crossings = 0
    decision_ms = 0.0
    update_ms = 0.0
    step = 0
    while not controller.status()["done"] and step < MAX_STEPS:
        started = time.perf_counter()
        action = controller.recommend()
        decision_ms += (time.perf_counter() - started) * 1000.0
        if action.stop:
            break
        remaining = MAX_TOTAL_DOSE_ML - total_actual
        if remaining <= 0.0:
            break
        actual = float(action.volume_ml) * float(schedule["actuator_factor"][step])
        actual = float(np.clip(actual, 0.01, remaining))
        actual_titrant_m = 0.1 * regime.titrant_scale
        if action.reagent == "base":
            base_moles += actual_titrant_m * actual / 1000.0
        else:
            acid_moles += actual_titrant_m * actual / 1000.0
        total_actual += actual
        previous_true_ph = true_ph
        true_ph = solve_ph_scalar(
            task.analyte_conc_m,
            task.pka_values,
            task.initial_volume_ml,
            SolutionState(task.initial_volume_ml + total_actual, base_moles, acid_moles),
        )
        step += 1
        measured_ph, drift = sensor_reading(
            true_ph,
            measured_ph,
            step,
            regime,
            schedule,
            drift,
        )
        crossings += int(
            (previous_true_ph - task.target_ph) * (true_ph - task.target_ph) < 0.0
        )
        started = time.perf_counter()
        controller.observe(measured_ph, actual_volume_ml=actual, reagent=action.reagent)
        update_ms += (time.perf_counter() - started) * 1000.0

    true_error = abs(true_ph - task.target_ph)
    measured_error = abs(measured_ph - task.target_ph)
    return {
        "method": method,
        "regime": regime.name,
        "distribution": regime.distribution,
        "task_seed": task.seed,
        "task_id": task.task_id,
        "direction": task.direction,
        "difficulty": task.difficulty,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "true_success": int(true_error <= SUCCESS_TOLERANCE),
        "strict_success": int(true_error <= STRICT_TOLERANCE),
        "severe_failure": int(true_error > 0.50),
        "measured_success": int(measured_error <= SUCCESS_TOLERANCE),
        "false_stop": int(measured_error <= SUCCESS_TOLERANCE and true_error > SUCCESS_TOLERANCE),
        "steps": step,
        "crossings": crossings,
        "total_volume_ml": total_actual,
        "final_true_ph": true_ph,
        "final_measured_ph": measured_ph,
        "final_abs_error": true_error,
        "decision_time_ms_total": decision_ms,
        "update_time_ms_total": update_ms,
        "controller_time_ms_per_step": (decision_ms + update_ms) / max(step, 1),
        "stop_reason": controller.status()["stop_reason"],
    }


def run_task(payload) -> list[dict]:
    task, regime, benchmark_seed, device, ppo_backend = payload
    common_seed = benchmark_seed * 10_000_019 + task.task_id * 1009
    rows = []
    for method in METHODS:
        row = run_episode(task, method, regime, common_seed, device, ppo_backend)
        row["benchmark_seed"] = benchmark_seed
        rows.append(row)
    return rows


def validate_resume_config(existing: dict, requested: dict) -> None:
    ignored = {"workers"}
    differences = []
    for key in sorted(set(existing) | set(requested)):
        if key in ignored:
            continue
        if existing.get(key) != requested.get(key):
            differences.append(f"{key}: existing={existing.get(key)!r}, requested={requested.get(key)!r}")
    if differences:
        raise RuntimeError(
            "Cannot resume with a different scientific configuration:\n  "
            + "\n  ".join(differences)
        )


def summarize(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["true_success"]]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * np.mean([row["true_success"] for row in rows]),
        "strict_success_rate_percent": 100.0 * np.mean([row["strict_success"] for row in rows]),
        "severe_failure_rate_percent": 100.0 * np.mean([row["severe_failure"] for row in rows]),
        "false_stop_rate_percent": 100.0 * np.mean([row["false_stop"] for row in rows]),
        "steps_mean": float(np.mean([row["steps"] for row in rows])),
        "successful_steps_mean": float(np.mean([row["steps"] for row in successes])) if successes else math.nan,
        "crossings_mean": float(np.mean([row["crossings"] for row in rows])),
        "total_volume_mean_ml": float(np.mean([row["total_volume_ml"] for row in rows])),
        "final_abs_error_mean": float(np.mean([row["final_abs_error"] for row in rows])),
        "controller_time_ms_per_step_mean": float(
            np.mean([row["controller_time_ms_per_step"] for row in rows])
        ),
    }


def exact_mcnemar(reference: list[int], comparison: list[int]) -> tuple[int, int, float]:
    ref_only = sum(a == 1 and b == 0 for a, b in zip(reference, comparison))
    cmp_only = sum(a == 0 and b == 1 for a, b in zip(reference, comparison))
    discordant = ref_only + cmp_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(ref_only, cmp_only), discordant, 0.5).pvalue)
    return ref_only, cmp_only, p_value


def holm_adjust(rows: list[dict], key: str = "raw_p") -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index][key])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * float(rows[index][key]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description="Published new-PF versus selected-PPO paired stress benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--tasks-per-seed", type=int, default=1000)
    parser.add_argument("--regimes", nargs="+", choices=sorted(REGIMES), default=list(REGIMES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ppo-backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "completed_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or min(16, max(1, (os.cpu_count() or 2) - 1))
    effective_device = args.device
    if args.ppo_backend == "numpy":
        effective_device = "cpu"
    elif workers > 1 and args.device != "cpu":
        raise ValueError("Multi-process PyTorch evaluation requires --device cpu; use NumPy for equivalent fast inference")
    config = {
        "seeds": args.seeds,
        "tasks_per_seed": args.tasks_per_seed,
        "regimes": args.regimes,
        "device": effective_device,
        "ppo_backend": args.ppo_backend,
        "workers": workers,
        "methods": list(METHODS),
        "ppo_checkpoint": str(PPO_CHECKPOINT),
        "ppo_numpy_checkpoint": str(PPO_NUMPY_CHECKPOINT),
        "regime_definitions": {name: asdict(REGIMES[name]) for name in args.regimes},
    }
    config_path = output / "RUN_CONFIG.json"
    if args.resume and config_path.is_file():
        validate_resume_config(json.loads(config_path.read_text(encoding="utf-8")), config)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    per_seed: list[dict] = []
    task_cache: dict[tuple[int, str], list[ControlTask]] = {}
    executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        if workers > 1
        else None
    )
    try:
        for regime_name in args.regimes:
            regime = REGIMES[regime_name]
            for benchmark_seed in args.seeds:
                shard_path = shard_dir / f"{regime_name}_seed_{benchmark_seed}.csv"
                if args.resume and shard_path.is_file():
                    rows = read_csv(shard_path)
                    if len(rows) != args.tasks_per_seed * len(METHODS):
                        raise RuntimeError(f"Incomplete resume shard: {shard_path}")
                    print(f"{regime_name} seed {benchmark_seed}: resumed {len(rows)} rows", flush=True)
                else:
                    task_path = output / f"{regime_name}_seed_{benchmark_seed}_tasks.jsonl"
                    cache_key = (benchmark_seed, regime.distribution)
                    if cache_key not in task_cache:
                        # Oracle task generation is expensive. All nominal
                        # stress regimes intentionally share the same task
                        # bank; close-pKa and wide-concentration each get
                        # one independent bank per benchmark seed.
                        candidates = [
                            output / f"{candidate}_seed_{benchmark_seed}_tasks.jsonl"
                            for candidate in args.regimes
                            if REGIMES[candidate].distribution == regime.distribution
                        ]
                        existing_bank = next((path for path in candidates if path.is_file()), None)
                        if existing_bank is not None:
                            task_cache[cache_key] = load_tasks(existing_bank)
                        else:
                            task_cache[cache_key] = generate_tasks(
                                3_000_000 + benchmark_seed,
                                args.tasks_per_seed,
                                f"published_controller_task_bank_{regime.distribution}_{benchmark_seed}",
                                distribution=regime.distribution,
                            )
                    tasks = [
                        replace(task, split=f"published_controller_{regime_name}_{benchmark_seed}")
                        for task in task_cache[cache_key]
                    ]
                    save_tasks(task_path, tasks)
                    payloads = [
                        (task, regime, benchmark_seed, effective_device, args.ppo_backend)
                        for task in tasks
                    ]
                    results = map(run_task, payloads) if executor is None else executor.map(run_task, payloads, chunksize=2)
                    rows = []
                    for index, task_rows in enumerate(results, 1):
                        rows.extend(task_rows)
                        if index % 100 == 0 or index == len(tasks):
                            print(f"{regime_name} seed {benchmark_seed}: {index}/{len(tasks)}", flush=True)
                    write_csv(shard_path, rows)
                all_rows.extend(rows)
                for method in METHODS:
                    subset = [row for row in rows if row["method"] == method]
                    per_seed.append(
                        {"regime": regime_name, "benchmark_seed": benchmark_seed, "method": method, **summarize(subset)}
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    aggregate: list[dict] = []
    for regime_name in args.regimes:
        for method in METHODS:
            subset = [row for row in per_seed if row["regime"] == regime_name and row["method"] == method]
            result = {"regime": regime_name, "method": method, "seed_runs": len(subset)}
            for metric in subset[0]:
                if metric in {"regime", "benchmark_seed", "method"}:
                    continue
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                finite = values[np.isfinite(values)]
                result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
                result[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
            aggregate.append(result)

    tests: list[dict] = []
    continuous: list[dict] = []
    for regime_name in args.regimes:
        subset = [row for row in all_rows if row["regime"] == regime_name]
        lookup = {
            method: {(row["benchmark_seed"], row["task_id"]): row for row in subset if row["method"] == method}
            for method in METHODS
        }
        keys = sorted(set(lookup["new_pf"]) & set(lookup["new_ppo"]))
        pf_success = [lookup["new_pf"][key]["true_success"] for key in keys]
        ppo_success = [lookup["new_ppo"][key]["true_success"] for key in keys]
        pf_only, ppo_only, raw_p = exact_mcnemar(pf_success, ppo_success)
        tests.append(
            {
                "regime": regime_name,
                "comparison": "new_ppo_minus_new_pf",
                "paired_tasks": len(keys),
                "pf_only_success": pf_only,
                "ppo_only_success": ppo_only,
                "success_difference_pp": 100.0 * (np.mean(ppo_success) - np.mean(pf_success)),
                "raw_p": raw_p,
            }
        )
        for metric in ("steps", "crossings", "total_volume_ml", "final_abs_error", "controller_time_ms_per_step"):
            pf_values = np.asarray([lookup["new_pf"][key][metric] for key in keys], dtype=float)
            ppo_values = np.asarray([lookup["new_ppo"][key][metric] for key in keys], dtype=float)
            try:
                paired_p = float(wilcoxon(ppo_values, pf_values, zero_method="zsplit").pvalue)
            except ValueError:
                paired_p = 1.0
            continuous.append(
                {
                    "regime": regime_name,
                    "metric": metric,
                    "ppo_minus_pf_mean": float(np.mean(ppo_values - pf_values)),
                    "raw_p": paired_p,
                }
            )
    holm_adjust(tests)
    holm_adjust(continuous)
    write_csv(output / "all_task_results.csv", all_rows)
    write_csv(output / "per_seed_summary.csv", per_seed)
    write_csv(output / "aggregate_summary.csv", aggregate)
    write_csv(output / "paired_success_tests.csv", tests)
    write_csv(output / "paired_continuous_tests.csv", continuous)
    (output / "BENCHMARK_COMPLETE.json").write_text(
        json.dumps({"config": config, "aggregate_rows": len(aggregate)}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
