from __future__ import annotations

"""Controlled observation-to-action timing on the same 100 task cases.

Run one method per fresh process. The launcher is responsible for setting the
same thread-related environment variables for every process. This worker pins
itself to one logical CPU, applies the same process priority, performs the same
warm-up protocol, and measures the same new-observation-to-next-action path.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path


for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
PRIMARY = EVIDENCE / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
CHECKPOINTS = EVIDENCE / "02_TEACHER_AND_IMITATION" / "checkpoints"
PYMC_SOURCE = EVIDENCE / "13_SOURCE_ARCHIVES" / "joint_parameter_bayesian_code_current"
SEEDS = (101, 202, 303, 404, 555)
SELECTED_TASK_IDS = (
    1,
    159,
    317,
    475,
    632,
    790,
    948,
    1106,
    1264,
    1422,
    1579,
    1737,
    1895,
    2053,
    2211,
    2369,
    2526,
    2684,
    2842,
    3000,
)
TITRANT_M = 0.1
PREVIOUS_DOSE_ML = 0.01
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


@dataclass(frozen=True)
class Transition:
    step: int
    reagent: str
    requested_volume_ml: float
    before_state: object
    after_state: object
    observed_before_ph: float
    observed_after_ph: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def apply_windows_execution_controls(cpu_index: int) -> dict:
    controls = {
        "requested_cpu_index": cpu_index,
        "affinity_applied": False,
        "priority_applied": False,
    }
    if sys.platform != "win32":
        controls["note"] = "Windows affinity and priority controls were not applicable."
        return controls

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    set_affinity = kernel32.SetProcessAffinityMask
    set_affinity.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    set_affinity.restype = ctypes.c_int
    set_priority = kernel32.SetPriorityClass
    set_priority.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    set_priority.restype = ctypes.c_int

    process = get_current_process()
    mask = ctypes.c_size_t(1 << cpu_index)
    if not set_affinity(process, mask):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    controls["affinity_applied"] = True
    controls["affinity_mask_hex"] = hex(mask.value)
    if not set_priority(process, ABOVE_NORMAL_PRIORITY_CLASS):
        raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
    controls["priority_applied"] = True
    controls["priority_class"] = "ABOVE_NORMAL_PRIORITY_CLASS"
    return controls


def load_task_payloads() -> tuple[list[dict], list[dict]]:
    measured: list[dict] = []
    warmup: list[dict] = []
    selected = set(SELECTED_TASK_IDS)
    for benchmark_seed in SEEDS:
        path = PRIMARY / "tasks" / f"seed_{benchmark_seed}_tasks.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle if line.strip()]
        by_id = {int(row["task_id"]): row for row in payloads}
        for task_id in SELECTED_TASK_IDS:
            row = dict(by_id[task_id])
            row["benchmark_seed"] = benchmark_seed
            measured.append(row)
        # One warm-up case per benchmark seed is enough to exercise imports,
        # allocator/cache paths, and controller initialization without making
        # the expensive PyMC worker dominate the protocol.
        extras = [row for row in payloads if int(row["task_id"]) not in selected][:1]
        for payload in extras:
            row = dict(payload)
            row["benchmark_seed"] = benchmark_seed
            warmup.append(row)
    if len(measured) != 100 or len(warmup) != 5:
        raise RuntimeError(
            f"Expected 100 measured and 20 warm-up cases, found {len(measured)} and {len(warmup)}"
        )
    return measured, warmup


def build_case(payload: dict, solve_ph_scalar, SolutionState) -> dict:
    before_ph = float(np.round(float(payload["initial_ph"]), 2))
    target_ph = float(payload["target_ph"])
    reagent = "base" if before_ph < target_ph else "acid"
    base_moles = float(payload["initial_base_moles"])
    acid_moles = 0.0
    if reagent == "base":
        base_moles += TITRANT_M * PREVIOUS_DOSE_ML / 1000.0
    else:
        acid_moles += TITRANT_M * PREVIOUS_DOSE_ML / 1000.0
    after_ph = solve_ph_scalar(
        float(payload["analyte_conc_m"]),
        tuple(payload["pka_values"]),
        float(payload["initial_volume_ml"]),
        SolutionState(
            total_volume_ml=float(payload["initial_volume_ml"]) + PREVIOUS_DOSE_ML,
            base_moles=base_moles,
            acid_moles=acid_moles,
        ),
    )
    return {
        "benchmark_seed": int(payload["benchmark_seed"]),
        "task_seed": int(payload["seed"]),
        "task_id": int(payload["task_id"]),
        "acid_type": str(payload["acid_type"]),
        "pka_values": tuple(float(value) for value in payload["pka_values"]),
        "analyte_conc_m": float(payload["analyte_conc_m"]),
        "before_ph": before_ph,
        "observed_ph": float(np.round(np.clip(after_ph, 0.0, 14.0), 2)),
        "target_ph": target_ph,
        "reagent": reagent,
        "previous_volume_ml": PREVIOUS_DOSE_ML,
        "initial_volume_ml": float(payload["initial_volume_ml"]),
        "initial_base_moles": float(payload["initial_base_moles"]),
    }


def timed_neural(controller, case: dict, ControllerAction) -> tuple[int, int]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    controller.current_ph = case["before_ph"]
    controller.previous_ph = case["before_ph"]
    controller.target_ph = case["target_ph"]
    controller.last_requested_volume_ml = 0.0
    controller.total_actual_volume_ml = 0.0
    controller.base_added_ml = 0.0
    controller.acid_added_ml = 0.0
    controller.steps = 0
    controller.done = False
    controller.stop_reason = "running"
    controller.initialized = True
    controller.pending_action = ControllerAction(
        stop=False,
        reagent=case["reagent"],
        volume_ml=case["previous_volume_ml"],
    )
    controller.observe(
        measured_ph=case["observed_ph"],
        actual_volume_ml=case["previous_volume_ml"],
        reagent=case["reagent"],
    )
    action = controller.recommend()
    cpu_elapsed = time.process_time_ns() - cpu_started
    wall_elapsed = time.perf_counter_ns() - wall_started
    if action.stop:
        raise RuntimeError("Unexpected stop in neural observation-to-action timing")
    return wall_elapsed, cpu_elapsed


def timed_pf(controller, case: dict, ControllerAction) -> tuple[int, int]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    controller.current_ph = case["before_ph"]
    controller.previous_ph = case["before_ph"]
    controller.target_ph = case["target_ph"]
    controller.total_volume_ml = case["initial_volume_ml"]
    controller.base_moles = case["initial_base_moles"]
    controller.acid_moles = 0.0
    controller.base_added_ml = 0.0
    controller.acid_added_ml = 0.0
    controller.last_action_volume_ml = 0.0
    controller.steps = 0
    controller.done = False
    controller.stop_reason = "running"
    controller.pending_action = ControllerAction(
        stop=False,
        reagent=case["reagent"],
        volume_ml=case["previous_volume_ml"],
    )
    controller.observe(
        measured_ph=case["observed_ph"],
        actual_volume_ml=case["previous_volume_ml"],
        reagent=case["reagent"],
    )
    action = controller.recommend()
    cpu_elapsed = time.process_time_ns() - cpu_started
    wall_elapsed = time.perf_counter_ns() - wall_started
    if action.stop:
        raise RuntimeError("Unexpected stop in PF observation-to-action timing")
    estimate = controller.posterior_estimate()
    return wall_elapsed, cpu_elapsed, {
        "selected_k": int(estimate.pair_count),
        "posterior_concentration_m": float(estimate.concentration_m),
        "posterior_pka_values": json.dumps(np.asarray(estimate.pka_values).tolist()),
        "posterior_pka_sd": json.dumps(np.asarray(estimate.pka_sd).tolist()),
        "pair_probabilities": json.dumps(np.asarray(estimate.pair_probabilities).tolist()),
    }


def solve_volume_root(func, lo: float = 0.0, hi: float = 10.0) -> float:
    f_lo = float(func(lo))
    f_hi = float(func(hi))
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0 or f_lo * f_hi > 0.0:
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


def make_pymc_objects(case: dict, Task, SolutionState):
    task = Task(
        seed=case["task_seed"],
        task_id=case["task_id"],
        acid_type=case["acid_type"],
        pka_values=case["pka_values"],
        initial_ph=case["before_ph"],
        target_ph=case["target_ph"],
        initial_volume_ml=case["initial_volume_ml"],
        analyte_conc_m=case["analyte_conc_m"],
    )
    before_state = SolutionState(
        total_volume_ml=case["initial_volume_ml"],
        base_moles=case["initial_base_moles"],
        acid_moles=0.0,
    )
    after_state = SolutionState(
        total_volume_ml=case["initial_volume_ml"] + PREVIOUS_DOSE_ML,
        base_moles=(
            case["initial_base_moles"]
            + (TITRANT_M * PREVIOUS_DOSE_ML / 1000.0 if case["reagent"] == "base" else 0.0)
        ),
        acid_moles=(
            TITRANT_M * PREVIOUS_DOSE_ML / 1000.0 if case["reagent"] == "acid" else 0.0
        ),
    )
    transition = Transition(
        step=1,
        reagent=case["reagent"],
        requested_volume_ml=PREVIOUS_DOSE_ML,
        before_state=before_state,
        after_state=after_state,
        observed_before_ph=case["before_ph"],
        observed_after_ph=case["observed_ph"],
    )
    return task, transition


def posterior_next_dose(fit, task, transition, solve_ph_scalar, SolutionState) -> float:
    estimate = fit.estimate
    state = transition.after_state
    direction = "base" if transition.observed_after_ph < task.target_ph else "acid"

    def objective(volume_ml: float) -> float:
        if direction == "base":
            probe = SolutionState(
                total_volume_ml=state.total_volume_ml + volume_ml,
                base_moles=state.base_moles + TITRANT_M * volume_ml / 1000.0,
                acid_moles=state.acid_moles,
            )
        else:
            probe = SolutionState(
                total_volume_ml=state.total_volume_ml + volume_ml,
                base_moles=state.base_moles,
                acid_moles=state.acid_moles + TITRANT_M * volume_ml / 1000.0,
            )
        return solve_ph_scalar(
            estimate.concentration_m,
            estimate.pka_values,
            task.initial_volume_ml,
            probe,
        ) - task.target_ph

    continuous = solve_volume_root(objective)
    return float(np.clip(np.round(continuous / 0.01) * 0.01, 0.01, 10.0))


def timed_pymc(
    case: dict,
    repeat: int,
    fit_pymc_variant,
    Task,
    SolutionState,
    solve_ph_scalar,
    draws: int,
    chains: int,
    seed: int,
) -> tuple[int, int, dict]:
    task, transition = make_pymc_objects(case, Task, SolutionState)
    # Keep warm-up and measured seeds non-negative for NumPy/PyMC while
    # retaining a deterministic, case-specific stream.
    repeat_key = abs(int(repeat)) + 1
    inference_seed = (
        seed
        + repeat_key * 10_000_019
        + case["benchmark_seed"] * 1009
        + case["task_id"]
    ) % (2**32 - 1)
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    fit = fit_pymc_variant(
        task,
        [transition],
        "pymc_pka_conc_variable_k",
        draws=draws,
        chains=chains,
        seed=inference_seed,
    )
    next_dose = posterior_next_dose(fit, task, transition, solve_ph_scalar, SolutionState)
    cpu_elapsed = time.process_time_ns() - cpu_started
    wall_elapsed = time.perf_counter_ns() - wall_started
    return wall_elapsed, cpu_elapsed, {
        "selected_k": fit.estimate.pair_count,
        "posterior_concentration_m": fit.estimate.concentration_m,
        "posterior_pka_values": json.dumps(np.asarray(fit.estimate.pka_values).tolist()),
        "posterior_pka_sd": json.dumps(np.asarray(fit.estimate.pka_sd).tolist()),
        "pair_probabilities": json.dumps(np.asarray(fit.estimate.pair_probabilities).tolist()),
        "recommended_volume_ml": next_dose,
        "inference_seed": inference_seed,
    }


def summarize(method: str, rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    wall_values = np.asarray([row["wall_ns"] for row in rows], dtype=np.float64) / 1e6
    cpu_values = np.asarray([row["process_cpu_ns"] for row in rows], dtype=np.float64) / 1e6
    per_task_rows: list[dict] = []
    task_keys = sorted({(row["benchmark_seed"], row["task_id"]) for row in rows})
    for benchmark_seed, task_id in task_keys:
        subset = [
            row
            for row in rows
            if row["benchmark_seed"] == benchmark_seed and row["task_id"] == task_id
        ]
        per_task_rows.append(
            {
                "method": method,
                "benchmark_seed": benchmark_seed,
                "task_id": task_id,
                "repeats": len(subset),
                "median_wall_ms": float(np.median([row["wall_ns"] for row in subset]) / 1e6),
                "median_process_cpu_ms": float(
                    np.median([row["process_cpu_ns"] for row in subset]) / 1e6
                ),
            }
        )
    per_task_wall = np.asarray([row["median_wall_ms"] for row in per_task_rows], dtype=float)
    per_task_cpu = np.asarray([row["median_process_cpu_ms"] for row in per_task_rows], dtype=float)
    repeat_rows: list[dict] = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        subset = [row for row in rows if int(row["repeat"]) == repeat]
        repeat_wall = np.asarray([row["wall_ns"] for row in subset], dtype=np.float64) / 1e6
        repeat_rows.append(
            {
                "method": method,
                "repeat": repeat,
                "tasks": len(subset),
                "median_wall_ms": float(np.median(repeat_wall)),
                "p05_wall_ms": float(np.percentile(repeat_wall, 5)),
                "p95_wall_ms": float(np.percentile(repeat_wall, 95)),
            }
        )
    summary = {
        "method": method,
        "unique_tasks": len(task_keys),
        "repeats_per_task": int(len(rows) / len(task_keys)),
        "measurements": len(rows),
        "primary_median_of_task_median_wall_ms": float(np.median(per_task_wall)),
        "median_of_task_median_process_cpu_ms": float(np.median(per_task_cpu)),
        "all_measurement_median_wall_ms": float(np.median(wall_values)),
        "all_measurement_p05_wall_ms": float(np.percentile(wall_values, 5)),
        "all_measurement_p95_wall_ms": float(np.percentile(wall_values, 95)),
        "all_measurement_median_process_cpu_ms": float(np.median(cpu_values)),
    }
    return summary, per_task_rows, repeat_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("imitation", "ppo", "pf_1000", "pf_10000", "pf_100000", "pymc"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-index", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Development-only limit; formal runs leave this unset.",
    )
    parser.add_argument(
        "--warmup-limit",
        type=int,
        default=None,
        help="Development-only limit; formal runs leave this unset.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.cpu_index < 0 or args.cpu_index >= (os.cpu_count() or 1):
        raise ValueError("cpu-index is outside the available logical CPU range")

    controls = apply_windows_execution_controls(args.cpu_index)
    measured_payloads, warmup_payloads = load_task_payloads()
    if args.case_limit is not None:
        if args.case_limit < 1 or args.case_limit > len(measured_payloads):
            raise ValueError("case-limit is outside the available measured cases")
        measured_payloads = measured_payloads[: args.case_limit]
    if args.warmup_limit is not None:
        if args.warmup_limit < 0 or args.warmup_limit > len(warmup_payloads):
            raise ValueError("warmup-limit is outside the available warm-up cases")
        warmup_payloads = warmup_payloads[: args.warmup_limit]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(PYMC_SOURCE))
    from chemistry_model import SolutionState, solve_ph_scalar

    measured_cases = [build_case(row, solve_ph_scalar, SolutionState) for row in measured_payloads]
    warmup_cases = [build_case(row, solve_ph_scalar, SolutionState) for row in warmup_payloads]
    common_metadata = {
        "benchmark_seeds": list(SEEDS),
        "selected_task_ids_per_seed": list(SELECTED_TASK_IDS),
        "unique_tasks": len(measured_cases),
        "warmup_tasks": len(warmup_cases),
        "repeats": args.repeats,
        "formal_protocol": args.case_limit is None and args.warmup_limit is None,
        "timing_scope": "new pH observation to next action",
        "timed_operations": "state bookkeeping, observation/posterior update, and next-action selection",
        "excluded": [
            "Python startup",
            "module import",
            "checkpoint loading",
            "controller construction/reset",
            "task loading",
            "chemical transition calculation",
            "liquid delivery",
            "mixing",
            "sensor acquisition",
            "file I/O",
        ],
        "case_construction": "one 0.01 mL pre-dose and rounded post-dose pH for each locked task",
        "thread_environment": {
            variable: os.environ.get(variable)
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "execution_controls": controls,
        "clock": {
            "wall": vars(time.get_clock_info("perf_counter")),
            "process_cpu": vars(time.get_clock_info("process_time")),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
    }

    raw_rows: list[dict] = []
    if args.method == "pymc":
        from benchmark_core import Task
        from pymc_inference import fit_pymc_variant

        import pymc

        common_metadata["environment"]["pymc"] = pymc.__version__
        common_metadata["pymc"] = {
            "draws_per_k": args.draws,
            "chains": args.chains,
            "k_values": [1, 2, 3],
        }
        for index, case in enumerate(warmup_cases, 1):
            timed_pymc(
                case,
                -index,
                fit_pymc_variant,
                Task,
                SolutionState,
                solve_ph_scalar,
                args.draws,
                args.chains,
                args.seed,
            )
            print(f"{args.method}: warm-up {index}/{len(warmup_cases)}", flush=True)
        for repeat in range(args.repeats):
            order = np.random.default_rng(args.seed + repeat).permutation(len(measured_cases))
            for position, case_index in enumerate(order, 1):
                case = measured_cases[int(case_index)]
                wall_ns, cpu_ns, diagnostics = timed_pymc(
                    case,
                    repeat,
                    fit_pymc_variant,
                    Task,
                    SolutionState,
                    solve_ph_scalar,
                    args.draws,
                    args.chains,
                    args.seed,
                )
                raw_rows.append(
                    {
                        "method": args.method,
                        "repeat": repeat + 1,
                        "order_position": position,
                        **{key: case[key] for key in (
                            "benchmark_seed",
                            "task_seed",
                            "task_id",
                            "pka_values",
                            "analyte_conc_m",
                            "initial_volume_ml",
                            "initial_base_moles",
                            "before_ph",
                            "observed_ph",
                            "target_ph",
                            "reagent",
                            "previous_volume_ml",
                        )},
                        "wall_ns": wall_ns,
                        "process_cpu_ns": cpu_ns,
                        **diagnostics,
                    }
                )
                print(
                    f"{args.method}: repeat {repeat + 1}/{args.repeats}, "
                    f"case {position}/{len(measured_cases)}, {wall_ns / 1e9:.3f} s",
                    flush=True,
                )
    else:
        import torch

        from controllers.controller_api import ControllerAction
        from controllers.new_pf_controller import RobustPFController
        from controllers.new_rl_controller import PPOVolumeController

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        common_metadata["environment"]["torch"] = torch.__version__
        common_metadata["environment"]["torch_threads"] = torch.get_num_threads()

        controller = None
        particle_count = None
        checkpoint = None
        if args.method in ("imitation", "ppo"):
            checkpoint = CHECKPOINTS / (
                "imitation_best.pth" if args.method == "imitation" else "principal_ppo_seed_303.pth"
            )
            controller = PPOVolumeController(
                checkpoint,
                device="cpu",
                verify_selected_checkpoint=args.method == "ppo",
            )
            common_metadata["checkpoint_sha256"] = sha256(checkpoint)
        else:
            particle_count = int(args.method.split("_")[1])
            common_metadata["particle_count"] = particle_count

        def execute(case: dict, repeat: int) -> tuple[int, int]:
            if particle_count is None:
                controller.reset(case["before_ph"], case["target_ph"])
                return timed_neural(controller, case, ControllerAction)
            pf_seed = (
                case["task_seed"] * 1_000_003
                + case["task_id"]
                + repeat * 10_000_019
            ) % (2**32 - 1)
            pf_controller = RobustPFController(
                particles=particle_count,
                seed=pf_seed,
                max_steps=50,
                max_total_dose_ml=50.0,
            )
            pf_controller.reset(
                case["before_ph"],
                case["target_ph"],
                case["initial_volume_ml"],
                case["initial_base_moles"],
                0.0,
            )
            return timed_pf(pf_controller, case, ControllerAction)

        for index, case in enumerate(warmup_cases, 1):
            execute(case, -index)
        for repeat in range(args.repeats):
            order = np.random.default_rng(args.seed + repeat).permutation(len(measured_cases))
            for position, case_index in enumerate(order, 1):
                case = measured_cases[int(case_index)]
                timed_result = execute(case, repeat)
                if len(timed_result) == 2:
                    wall_ns, cpu_ns = timed_result
                    diagnostics = {}
                else:
                    wall_ns, cpu_ns, diagnostics = timed_result
                raw_rows.append(
                    {
                        "method": args.method,
                        "repeat": repeat + 1,
                        "order_position": position,
                        **{key: case[key] for key in (
                            "benchmark_seed",
                            "task_seed",
                            "task_id",
                            "pka_values",
                            "analyte_conc_m",
                            "initial_volume_ml",
                            "initial_base_moles",
                            "before_ph",
                            "observed_ph",
                            "target_ph",
                            "reagent",
                            "previous_volume_ml",
                        )},
                        "wall_ns": wall_ns,
                        "process_cpu_ns": cpu_ns,
                        **diagnostics,
                    }
                )
            print(f"{args.method}: completed repeat {repeat + 1}/{args.repeats}", flush=True)

    summary, per_task_rows, repeat_rows = summarize(args.method, raw_rows)
    common_metadata["method"] = args.method
    common_metadata["summary"] = summary
    write_csv(output / "raw.csv", raw_rows)
    write_csv(output / "per_task_summary.csv", per_task_rows)
    write_csv(output / "repeat_summary.csv", repeat_rows)
    write_csv(output / "summary.csv", [summary])
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(common_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
