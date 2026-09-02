from __future__ import annotations

"""Authoritative 2-representation x 3-control x 3-domain PF factorial.

The sequential-K weighted-parameter cell deliberately calls the recovered
formal JointInferenceController implementation without changing its dosing
logic.  All other cells alter only the chemical representation and/or the
posterior-to-required-volume calculation.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


SEEDS = (101, 202, 303, 404, 555)
DOMAINS = ("sequential_k123", "fixed_two_independent", "independent_j123")
REPRESENTATIONS = ("sequential_k", "independent_j")
STRATEGIES = ("weighted_parameters", "map_order_predictive", "full_predictive")
PARTICLES = 1000
SUCCESS_TOLERANCE = 0.10
STRICT_TOLERANCE = 0.05
SENSOR_RESOLUTION = 0.01
CURVE_PROBE_ML = np.linspace(-10.0, 10.0, 41)

FORMAL_SOURCE: Path | None = None
DATASETS_ROOT: Path | None = None


def configure(formal_source: Path, staging_root: Path) -> None:
    global FORMAL_SOURCE
    FORMAL_SOURCE = formal_source.resolve()
    formal_text = str(FORMAL_SOURCE)
    stage_text = str(staging_root.resolve())
    if formal_text not in sys.path:
        sys.path.insert(0, formal_text)
    # Import the recovered modules first so the independent-J extension reuses
    # their exact numerical constants and chemistry implementation.
    import chemistry_model  # noqa: F401
    import particle_inference  # noqa: F401
    import particle_controllers  # noqa: F401
    if stage_text not in sys.path:
        sys.path.insert(1, stage_text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_object(task: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**task)


def truth_ph(task: dict[str, Any], domain: str, state) -> float:
    from chemistry_model import solve_ph_scalar
    from independent_mixture_pf import solve_independent_ph_scalar

    if domain == "sequential_k123":
        return float(
            solve_ph_scalar(
                float(task["analyte_conc_m"]),
                task["pka_values"],
                float(task["initial_volume_ml"]),
                state,
            )
        )
    return float(
        solve_independent_ph_scalar(
            np.asarray(task["component_concentrations_m"], dtype=float),
            np.asarray(task["pka_values"], dtype=float),
            float(task["initial_volume_ml"]),
            state,
        )
    )


def truth_curve(task: dict[str, Any], domain: str, state) -> np.ndarray:
    from chemistry_model import solve_ph_grid
    from independent_mixture_pf import solve_independent_ph_grid

    positive = np.maximum(CURVE_PROBE_ML, 0.0)
    negative = np.maximum(-CURVE_PROBE_ML, 0.0)
    volumes = float(state.total_volume_ml) + np.abs(CURVE_PROBE_ML)
    bases = float(state.base_moles) + 0.1 * positive / 1000.0
    acids = float(state.acid_moles) + 0.1 * negative / 1000.0
    if domain == "sequential_k123":
        return np.asarray(
            solve_ph_grid(
                float(task["analyte_conc_m"]),
                task["pka_values"],
                float(task["initial_volume_ml"]),
                volumes,
                bases,
                acids,
            ),
            dtype=float,
        )
    return np.asarray(
        solve_independent_ph_grid(
            np.asarray(task["component_concentrations_m"], dtype=float),
            np.asarray(task["pka_values"], dtype=float),
            float(task["initial_volume_ml"]),
            volumes,
            bases,
            acids,
        ),
        dtype=float,
    )


def make_state(controller, signed_volume_ml: float):
    from chemistry_model import SolutionState

    volume = abs(float(signed_volume_ml))
    if signed_volume_ml >= 0.0:
        return SolutionState(
            total_volume_ml=float(controller.total_volume) + volume,
            base_moles=float(controller.base_added_moles) + 0.1 * volume / 1000.0,
            acid_moles=float(controller.acid_added_moles),
        )
    return SolutionState(
        total_volume_ml=float(controller.total_volume) + volume,
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles) + 0.1 * volume / 1000.0,
    )


def plugin_prediction(controller, state) -> float:
    estimate = controller.posterior_estimate()
    if controller.chemical_representation == "sequential_k":
        from chemistry_model import solve_ph_scalar

        return float(
            solve_ph_scalar(
                float(estimate.concentration_m),
                estimate.pka_values,
                float(controller.initial_volume_ml),
                state,
            )
        )
    from independent_mixture_pf import solve_independent_ph_scalar

    return float(
        solve_independent_ph_scalar(
            estimate.component_concentrations_m,
            estimate.pka_values,
            float(controller.initial_volume_ml),
            state,
        )
    )


def predictive_prediction(controller, state, full: bool) -> float:
    inference = controller.inference
    probabilities = np.asarray(inference.model_probabilities, dtype=float)
    if full:
        return float(
            sum(
                probabilities[index - 1]
                * np.sum(
                    inference.banks[index].weights
                    * inference.banks[index].predict(controller.initial_volume_ml, state)
                )
                for index in (1, 2, 3)
            )
        )
    map_order = int(np.argmax(probabilities) + 1)
    bank = inference.banks[map_order]
    return float(
        np.sum(bank.weights * bank.predict(controller.initial_volume_ml, state))
    )


def strategy_prediction(controller, state) -> float:
    if controller.posterior_to_control_strategy == "weighted_parameters":
        return plugin_prediction(controller, state)
    if controller.posterior_to_control_strategy == "map_order_predictive":
        return predictive_prediction(controller, state, full=False)
    if controller.posterior_to_control_strategy == "full_predictive":
        return predictive_prediction(controller, state, full=True)
    raise KeyError(controller.posterior_to_control_strategy)


def response_state(controller, volume_ml: float):
    from chemistry_model import SolutionState

    volume = float(volume_ml)
    if controller.current_ph < controller.target_ph:
        reagent = "Dilute base 2" if controller.use_secondary_reagents else "Dilute base 1"
        concentration = controller.reagents[reagent]
        return SolutionState(
            float(controller.total_volume) + volume,
            float(controller.base_added_moles) + concentration * volume / 1000.0,
            float(controller.acid_added_moles),
        )
    reagent = "Dilute acid 2" if controller.use_secondary_reagents else "Dilute acid 1"
    concentration = controller.reagents[reagent]
    return SolutionState(
        float(controller.total_volume) + volume,
        float(controller.base_added_moles),
        float(controller.acid_added_moles) + concentration * volume / 1000.0,
    )


def make_controller(representation: str, strategy: str, filter_seed: int):
    from particle_controllers import (
        CONTROL_VOLUME_BISECTION_ITERATIONS,
        JointInferenceController,
    )
    from particle_inference import PF_VARIANTS
    from reference import original_bayesian_controller as original
    from independent_mixture_pf import IndependentVariableJParticleFilter

    class StudyController(JointInferenceController):
        chemical_representation = representation
        posterior_to_control_strategy = strategy

        def initialize_task(self, task, max_steps=original.MAX_STEPS):
            super().initialize_task(task, max_steps=max_steps)
            if self.chemical_representation == "independent_j":
                self.inference = IndependentVariableJParticleFilter(
                    self.num_particles,
                    np.random.default_rng(self.filter_seed),
                )
                self._refresh_summary()

        def simulate_observed_ph(self):
            if self._truth_domain == "sequential_k123":
                return super().simulate_observed_ph()
            from chemistry_model import SolutionState

            state = SolutionState(
                total_volume_ml=float(self.total_volume),
                base_moles=float(self.base_added_moles),
                acid_moles=float(self.acid_added_moles),
            )
            return round(truth_ph(self._truth_task, self._truth_domain, state), 2)

        def compute_required_volume(self):
            if (
                self.posterior_to_control_strategy == "weighted_parameters"
                and self.chemical_representation == "sequential_k"
            ):
                # Exact recovered formal baseline path.
                return super().compute_required_volume()

            def objective(volume_ml: float) -> float:
                return strategy_prediction(self, response_state(self, volume_ml)) - self.target_ph

            return original.solve_volume_root(
                objective,
                0.0,
                10.0,
                iterations=CONTROL_VOLUME_BISECTION_ITERATIONS,
            )

    if "pf_pka_conc_variable_k" not in PF_VARIANTS:
        raise RuntimeError("Recovered variable-K PF source is unavailable")
    return StudyController(
        "pf_pka_conc_variable_k",
        num_particles=PARTICLES,
        filter_seed=int(filter_seed),
    )


def initialize_controller(task: dict[str, Any], domain: str, representation: str, strategy: str, common_seed: int):
    common_seed = int(common_seed) % (2**32 - 1)
    np.random.seed(common_seed)
    controller = make_controller(
        representation,
        strategy,
        filter_seed=(common_seed + 17) % (2**32 - 1),
    )
    controller._truth_domain = domain
    controller._truth_task = task
    controller.initialize_task(task_object(task))
    controller.base_added_moles = float(task["initial_base_moles"])
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = float(task["initial_volume_ml"])
    controller.previous_total_volume = float(task["initial_volume_ml"])
    observed_initial_ph = float(
        np.round(float(task["initial_ph"]) / SENSOR_RESOLUTION) * SENSOR_RESOLUTION
    )
    controller.current_ph = observed_initial_ph
    controller.previous_ph = observed_initial_ph
    controller.last_measured_ph = observed_initial_ph
    controller.prev_measured_ph = observed_initial_ph
    controller.last_action_volume = 0.0
    controller.done = False
    return controller


def run_task(task: dict[str, Any], domain: str, representation: str, strategy: str, benchmark_seed: int) -> dict[str, Any]:
    from chemistry_model import SolutionState

    common_seed = benchmark_seed * 10_000_019 + int(task["task_id"]) * 1009
    controller = initialize_controller(task, domain, representation, strategy, common_seed)
    overshoots = 0
    selection_seconds = 0.0
    update_seconds = 0.0
    while not controller.done:
        controller.use_secondary_reagents = False
        started = time.perf_counter()
        action, _ = controller.select_best_action()
        selection_seconds += time.perf_counter() - started
        current_ph, _, done, info = controller.step(action, mode="Simulate")
        overshoots += int(bool(info.get("crossed_target", False)))
        started = time.perf_counter()
        controller.update_posteriors(action, current_ph)
        update_seconds += time.perf_counter() - started
        if done:
            break

    final_state = SolutionState(
        total_volume_ml=float(controller.total_volume),
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles),
    )
    final_true_ph = truth_ph(task, domain, final_state)
    error = abs(final_true_ph - float(controller.target_ph))
    measured_error = abs(float(controller.current_ph) - float(controller.target_ph))
    estimate = controller.posterior_estimate()
    probabilities = np.asarray(estimate.pair_probabilities, dtype=float)
    actual_curve = truth_curve(task, domain, final_state)
    predicted_curve = np.asarray(
        [strategy_prediction(controller, make_state(controller, float(volume))) for volume in CURVE_PROBE_ML],
        dtype=float,
    )
    curve_error = predicted_curve - actual_curve
    true_count = int(task.get("component_count", len(task["pka_values"])))
    steps = int(controller.steps_taken)
    return {
        "domain": domain,
        "representation": representation,
        "strategy": strategy,
        "benchmark_seed": benchmark_seed,
        "task_seed": int(task["seed"]),
        "task_id": int(task["task_id"]),
        "true_component_count": true_count,
        "map_component_count": int(estimate.pair_count),
        "component_count_match": int(int(estimate.pair_count) == true_count),
        "model_probability_1": float(probabilities[0]),
        "model_probability_2": float(probabilities[1]),
        "model_probability_3": float(probabilities[2]),
        "initial_ph": float(task["initial_ph"]),
        "target_ph": float(task["target_ph"]),
        "true_success": int(error <= SUCCESS_TOLERANCE),
        "strict_success": int(error <= STRICT_TOLERANCE),
        "severe_failure": int(error > 0.50),
        "measured_success": int(measured_error <= SUCCESS_TOLERANCE),
        "false_stop": int(measured_error <= SUCCESS_TOLERANCE and error > SUCCESS_TOLERANCE),
        "steps": steps,
        "overshoots": overshoots,
        "total_added_volume_ml": float(controller.acid_volume + controller.base_volume),
        "final_true_ph": final_true_ph,
        "final_measured_ph": float(controller.current_ph),
        "final_abs_error": error,
        "decision_curve_rmse_ph": float(np.sqrt(np.mean(curve_error**2))),
        "decision_curve_mae_ph": float(np.mean(np.abs(curve_error))),
        "decision_curve_max_abs_error_ph": float(np.max(np.abs(curve_error))),
        "final_effective_sample_size": float(estimate.effective_sample_size),
        "selection_time_ms_total": 1000.0 * selection_seconds,
        "posterior_update_time_ms_total": 1000.0 * update_seconds,
        "controller_time_ms_per_step": 1000.0 * (selection_seconds + update_seconds) / max(1, steps),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_shard(job: dict[str, Any]) -> dict[str, Any]:
    configure(Path(job["formal_source"]), Path(job["staging_root"]))
    manifest = Path(job["manifest"])
    tasks = load_tasks(manifest)[int(job["start"]):int(job["stop"])]
    rows = [
        run_task(
            task,
            str(job["domain"]),
            str(job["representation"]),
            str(job["strategy"]),
            int(job["benchmark_seed"]),
        )
        for task in tasks
    ]
    output = Path(job["output"])
    write_csv(output, rows)
    return {
        "domain": job["domain"],
        "representation": job["representation"],
        "strategy": job["strategy"],
        "benchmark_seed": job["benchmark_seed"],
        "start": job["start"],
        "stop": job["stop"],
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean([row["true_success"] for row in rows])),
    }


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    successful = [row for row in rows if int(row["true_success"])]

    def mean(field: str, subset=rows) -> float:
        return float(np.mean([float(row[field]) for row in subset]))

    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * mean("true_success"),
        "strict_success_rate_percent": 100.0 * mean("strict_success"),
        "severe_failure_rate_percent": 100.0 * mean("severe_failure"),
        "successful_additions_mean": mean("steps", successful),
        "overshoots_per_task_mean": mean("overshoots"),
        "final_abs_error_mean": mean("final_abs_error"),
        "component_count_accuracy_percent": 100.0 * mean("component_count_match"),
        "decision_curve_rmse_mean_ph": mean("decision_curve_rmse_ph"),
        "decision_curve_mae_mean_ph": mean("decision_curve_mae_ph"),
        "controller_time_ms_per_step_median": float(
            np.median([float(row["controller_time_ms_per_step"]) for row in rows])
        ),
    }


def aggregate(output: Path, jobs: list[dict[str, Any]]) -> None:
    seed_summaries: list[dict[str, Any]] = []
    cells = sorted({(j["domain"], j["representation"], j["strategy"], j["benchmark_seed"]) for j in jobs})
    for domain, representation, strategy, seed in cells:
        shard_dir = output / "shards" / domain / representation / strategy / f"seed_{seed}"
        rows: list[dict[str, str]] = []
        for path in sorted(shard_dir.glob("*.csv")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        seed_summaries.append(
            {
                "domain": domain,
                "representation": representation,
                "strategy": strategy,
                "benchmark_seed": seed,
                **summarize_rows(rows),
            }
        )
    write_csv(output / "per_seed_summary.csv", seed_summaries)
    aggregate_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for representation in REPRESENTATIONS:
            for strategy in STRATEGIES:
                selected = [
                    row for row in seed_summaries
                    if row["domain"] == domain
                    and row["representation"] == representation
                    and row["strategy"] == strategy
                ]
                if not selected:
                    continue
                entry: dict[str, Any] = {
                    "domain": domain,
                    "representation": representation,
                    "strategy": strategy,
                }
                for field in (
                    "success_rate_percent",
                    "successful_additions_mean",
                    "overshoots_per_task_mean",
                    "final_abs_error_mean",
                    "component_count_accuracy_percent",
                    "decision_curve_rmse_mean_ph",
                    "decision_curve_mae_mean_ph",
                    "controller_time_ms_per_step_median",
                ):
                    values = np.asarray([float(row[field]) for row in selected], dtype=float)
                    entry[field + "_mean"] = float(np.mean(values))
                    entry[field + "_sample_sd"] = float(np.std(values, ddof=1))
                aggregate_rows.append(entry)
    write_csv(output / "aggregate_summary.csv", aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=160)
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--chunks-per-seed", type=int, default=20)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = parser.parse_args()

    configure(args.formal_source, args.staging_root)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_map = {
        "sequential_k123": args.datasets_root / "sequential_k123" / "tasks",
        "fixed_two_independent": args.datasets_root / "fixed_two_independent" / "tasks",
        "independent_j123": args.datasets_root / "independent_j123" / "tasks",
    }
    manifests: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    chunk_size = int(math.ceil(args.tasks_per_seed / args.chunks_per_seed))
    for domain in args.domains:
        for seed in SEEDS:
            manifest = dataset_map[domain] / f"seed_{seed}_tasks.jsonl"
            task_count = sum(1 for line in manifest.open("r", encoding="utf-8") if line.strip())
            if task_count < args.tasks_per_seed:
                raise RuntimeError(f"{manifest} has {task_count} tasks")
            manifests.append({"domain": domain, "seed": seed, "tasks": task_count, "sha256": sha256(manifest)})
            for representation in args.representations:
                for strategy in args.strategies:
                    for start in range(0, args.tasks_per_seed, chunk_size):
                        stop = min(args.tasks_per_seed, start + chunk_size)
                        shard = args.output / "shards" / domain / representation / strategy / f"seed_{seed}" / f"tasks_{start:04d}_{stop:04d}.csv"
                        jobs.append(
                            {
                                "formal_source": str(args.formal_source.resolve()),
                                "staging_root": str(args.staging_root.resolve()),
                                "manifest": str(manifest.resolve()),
                                "domain": domain,
                                "representation": representation,
                                "strategy": strategy,
                                "benchmark_seed": seed,
                                "start": start,
                                "stop": stop,
                                "output": str(shard.resolve()),
                            }
                        )
    config = {
        "study": "authoritative PF representation x posterior-to-control x domain factorial",
        "formal_source": str(args.formal_source.resolve()),
        "formal_source_files": {
            name: sha256(args.formal_source / name)
            for name in ("particle_controllers.py", "particle_inference.py", "chemistry_model.py")
        },
        "manifests": manifests,
        "domains": args.domains,
        "representations": args.representations,
        "strategies": args.strategies,
        "seeds": list(SEEDS),
        "tasks_per_seed": args.tasks_per_seed,
        "particles": PARTICLES,
        "workers": args.workers,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    config_path = args.output / "RUN_CONFIG.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError("Existing output configuration differs")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    pending = [job for job in jobs if not Path(job["output"]).exists()]
    print(f"PF factorial: {len(pending)}/{len(jobs)} shards pending; workers={args.workers}", flush=True)
    completed = len(jobs) - len(pending)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_shard, job): job for job in pending}
            for future in concurrent.futures.as_completed(futures):
                report = future.result()
                completed += 1
                print(
                    f"{completed}/{len(jobs)} {report['domain']} {report['representation']} "
                    f"{report['strategy']} seed={report['benchmark_seed']} "
                    f"tasks={report['start']}:{report['stop']} success={report['success_rate_percent']:.2f}%",
                    flush=True,
                )
    aggregate(args.output, jobs)
    (args.output / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_controller_outcomes": len(args.domains) * len(args.representations) * len(args.strategies) * len(SEEDS) * args.tasks_per_seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("PF factorial complete", flush=True)


if __name__ == "__main__":
    main()
