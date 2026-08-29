from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


DEFAULT_SOURCE = Path(
    r"C:\Users\ZSY\Desktop\FDTD\joint_parameter_bayesian_processed_20260811"
    r"\results\full\triprotic_stepwise_curve_evolution"
)
DEFAULT_CODE = Path(r"C:\Users\ZSY\Desktop\FDTD\joint_parameter_bayesian_processed_20260811")
DEFAULT_TASK_SUMMARY = Path(
    r"C:\Users\ZSY\Desktop\FDTD\submission_numerical_evidence_20260819"
    r"\02_SI\pymc_comparison\pymc_pf_summary.csv"
)


def solve_volume_root(func, lo: float = 0.0, hi: float = 10.0) -> float:
    f_lo = float(func(lo))
    f_hi = float(func(hi))
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return 0.0
    left, right = float(lo), float(hi)
    left_value = f_lo
    for _ in range(32):
        mid = 0.5 * (left + right)
        mid_value = float(func(mid))
        if abs(mid_value) < 1e-10:
            return mid
        if left_value * mid_value < 0.0:
            right = mid
        else:
            left = mid
            left_value = mid_value
    return 0.5 * (left + right)


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "repetitions": int(len(array)),
        "mean_ms": float(np.mean(array)),
        "sd_ms": float(np.std(array, ddof=1)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--task-summary", type=Path, default=DEFAULT_TASK_SUMMARY)
    parser.add_argument("--dose-repetitions", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    source = args.source.resolve()
    code = args.code.resolve()
    sys.path.insert(0, str(code))
    from chemistry_model import SolutionState, solve_ph_scalar

    trajectory = json.loads(
        (source / "triprotic_stepwise_trajectory.json").read_text(encoding="utf-8")
    )
    task = trajectory["task"]
    transitions = trajectory["transitions"]
    raw_rows = []
    stage_rows = []

    for stage in range(1, len(transitions) + 1):
        metadata = []
        posterior = []
        for pair_count in (1, 2, 3):
            stem = f"stage_{stage:02d}_k{pair_count}"
            item = json.loads(
                (source / "pymc_shards" / f"{stem}.json").read_text(encoding="utf-8")
            )
            with np.load(source / "pymc_shards" / f"{stem}.npz") as data:
                posterior.append(
                    (
                        np.asarray(data["concentration_draws_m"], dtype=float),
                        np.asarray(data["pka_draws"], dtype=float),
                    )
                )
            metadata.append(item)
        evidences = np.asarray([item["log_evidence"] for item in metadata], dtype=float)
        probabilities = np.exp(evidences - math.log(3.0) - logsumexp(evidences - math.log(3.0)))
        map_index = int(np.argmax(probabilities))
        concentration_draws, pka_draws = posterior[map_index]
        concentration = float(np.mean(concentration_draws))
        pka = np.mean(pka_draws, axis=0)

        transition = transitions[stage - 1]
        after = transition["after_state"]
        state = SolutionState(
            total_volume_ml=float(after["total_volume_ml"]),
            base_moles=float(after["base_moles"]),
            acid_moles=float(after["acid_moles"]),
        )
        current_ph = float(transition["observed_after_ph"])
        target_ph = float(task["target_ph"])
        direction = "base" if current_ph < target_ph else "acid"

        def decide() -> float:
            def objective(volume_ml: float) -> float:
                if direction == "base":
                    probe = SolutionState(
                        total_volume_ml=state.total_volume_ml + volume_ml,
                        base_moles=state.base_moles + 0.1 * volume_ml / 1000.0,
                        acid_moles=state.acid_moles,
                    )
                else:
                    probe = SolutionState(
                        total_volume_ml=state.total_volume_ml + volume_ml,
                        base_moles=state.base_moles,
                        acid_moles=state.acid_moles + 0.1 * volume_ml / 1000.0,
                    )
                return solve_ph_scalar(
                    concentration,
                    pka,
                    float(task["initial_volume_ml"]),
                    probe,
                ) - target_ph

            continuous = solve_volume_root(objective)
            return float(np.clip(np.round(continuous / 0.01) * 0.01, 0.01, 10.0))

        for _ in range(10):
            decide()
        dose_times_ms = []
        output_volume = math.nan
        for repetition in range(args.dose_repetitions):
            start = time.perf_counter_ns()
            output_volume = decide()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            dose_times_ms.append(elapsed_ms)
            raw_rows.append(
                {
                    "stage": stage,
                    "repetition": repetition,
                    "map_k": map_index + 1,
                    "current_ph": current_ph,
                    "target_ph": target_ph,
                    "direction": direction,
                    "recommended_volume_ml": output_volume,
                    "posterior_to_dose_ms": elapsed_ms,
                }
            )

        sampling_seconds = float(sum(item["runtime_seconds"] for item in metadata))
        dose_summary = summarize_ms(dose_times_ms)
        stage_rows.append(
            {
                "stage": stage,
                "observations_used": stage,
                "map_k": map_index + 1,
                "probability_k1": float(probabilities[0]),
                "probability_k2": float(probabilities[1]),
                "probability_k3": float(probabilities[2]),
                "smc_sampling_seconds_k1_to_k3": sampling_seconds,
                "posterior_to_dose_median_ms": dose_summary["median_ms"],
                "posterior_to_dose_p95_ms": dose_summary["p95_ms"],
                "lower_bound_total_update_to_dose_seconds": sampling_seconds
                + dose_summary["median_ms"] / 1000.0,
                "recommended_volume_ml": output_volume,
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pymc_posterior_to_dose_raw.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with (output_dir / "pymc_stepwise_timing.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    sampling = np.asarray(
        [row["smc_sampling_seconds_k1_to_k3"] for row in stage_rows], dtype=float
    )
    lower_bound = np.asarray(
        [row["lower_bound_total_update_to_dose_seconds"] for row in stage_rows],
        dtype=float,
    )
    task_median = math.nan
    with args.task_summary.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "pymc_pka_conc_variable_k":
                task_median = float(row["runtime_median_seconds"])
                break
    summary = {
        "method": "PyMC variable-K SMC plus posterior-equilibrium dose solve",
        "representative_task": {
            "seed": task["seed"],
            "task_id": task["task_id"],
            "acid_type": task["acid_type"],
            "post_dose_observations": len(transitions),
        },
        "historical_complete_trajectory_inference_median_seconds_15_tasks": task_median,
        "stepwise_smc_sampling_seconds": {
            "minimum": float(np.min(sampling)),
            "median": float(np.median(sampling)),
            "maximum": float(np.max(sampling)),
            "values": sampling.tolist(),
        },
        "lower_bound_update_to_dose_seconds": {
            "minimum": float(np.min(lower_bound)),
            "median": float(np.median(lower_bound)),
            "maximum": float(np.max(lower_bound)),
            "values": lower_bound.tolist(),
        },
        "important_limit": (
            "The SMC shard timer covers pm.sample_smc for K=1,2,3. It excludes "
            "PyMC model construction and file I/O. Adding the measured posterior-to-dose "
            "solve therefore gives a lower bound, not a full end-to-end online latency."
        ),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
    }
    (output_dir / "pymc_stepwise_timing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
