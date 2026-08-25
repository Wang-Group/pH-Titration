from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(
    r"C:\Users\ZSY\Desktop\FDTD\joint_parameter_bayesian_processed_20260811"
    r"\results\full\triprotic_stepwise_curve_evolution"
)
DEFAULT_CODE = Path(r"C:\Users\ZSY\Desktop\FDTD\joint_parameter_bayesian_processed_20260811")


def solve_volume_root(func, lo: float = 0.0, hi: float = 10.0) -> float:
    f_lo = float(func(lo))
    f_hi = float(func(hi))
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return 0.0
    left, right = lo, hi
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    source = args.source.resolve()
    code = args.code.resolve()
    sys.path.insert(0, str(code))
    from benchmark_core import Task
    from chemistry_model import SolutionState, solve_ph_scalar
    from experiment_utils import Transition
    from pymc_inference import fit_pymc_variant

    payload = json.loads(
        (source / "triprotic_stepwise_trajectory.json").read_text(encoding="utf-8")
    )
    task_data = payload["task"]
    task = Task(
        seed=int(task_data["seed"]),
        task_id=int(task_data["task_id"]),
        acid_type=str(task_data["acid_type"]),
        pka_values=tuple(float(value) for value in task_data["pka_values"]),
        initial_ph=float(task_data["initial_ph"]),
        target_ph=float(task_data["target_ph"]),
        initial_volume_ml=float(task_data["initial_volume_ml"]),
        analyte_conc_m=float(task_data["analyte_conc_m"]),
    )
    transition_data = payload["transitions"][0]
    transition = Transition(
        step=int(transition_data["step"]),
        reagent=str(transition_data["reagent"]),
        requested_volume_ml=float(transition_data["requested_volume_ml"]),
        before_state=SolutionState(**transition_data["before_state"]),
        after_state=SolutionState(**transition_data["after_state"]),
        observed_before_ph=float(transition_data["observed_before_ph"]),
        observed_after_ph=float(transition_data["observed_after_ph"]),
    )

    # PyMC is imported above. The outer timer starts immediately before the
    # three fresh K fits and includes model construction, SMC, evidence
    # comparison, and posterior extraction, but not Python/package import.
    fit_start = time.perf_counter()
    fit = fit_pymc_variant(
        task,
        [transition],
        "pymc_pka_conc_variable_k",
        draws=args.draws,
        chains=args.chains,
        seed=101301332,
    )
    full_fit_seconds = time.perf_counter() - fit_start

    estimate = fit.estimate
    state = transition.after_state
    current_ph = transition.observed_after_ph
    direction = "base" if current_ph < task.target_ph else "acid"

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
            estimate.concentration_m,
            estimate.pka_values,
            task.initial_volume_ml,
            probe,
        ) - task.target_ph

    dose_start = time.perf_counter()
    continuous_volume = solve_volume_root(objective)
    volume_ml = float(np.clip(np.round(continuous_volume / 0.01) * 0.01, 0.01, 10.0))
    dose_seconds = time.perf_counter() - dose_start
    result = {
        "status": "COMPLETE",
        "definition": (
            "One fresh variable-K update after the first post-dose pH observation, "
            "followed by a posterior-equilibrium next-dose calculation."
        ),
        "task": task_data,
        "observation_step": 1,
        "draws_per_k": args.draws,
        "chains": args.chains,
        "K_values": [1, 2, 3],
        "pymc_sample_smc_sum_seconds": fit.runtime_seconds,
        "fresh_fit_outer_wall_seconds": full_fit_seconds,
        "posterior_to_dose_seconds": dose_seconds,
        "fresh_update_to_dose_wall_seconds": full_fit_seconds + dose_seconds,
        "selected_k": estimate.pair_count,
        "pair_probabilities": estimate.pair_probabilities.tolist(),
        "posterior_concentration_m": estimate.concentration_m,
        "posterior_pka": estimate.pka_values.tolist(),
        "direction": direction,
        "recommended_volume_ml": volume_ml,
        "timing_scope": {
            "included": [
                "construction of three fresh PyMC models",
                "SMC sampling for K=1,2,3",
                "marginal-likelihood comparison",
                "posterior extraction",
                "posterior-equilibrium dose root solve and 0.01 mL quantization",
            ],
            "excluded": ["Python startup", "module import", "file I/O"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pymc_one_step_end_to_end.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
