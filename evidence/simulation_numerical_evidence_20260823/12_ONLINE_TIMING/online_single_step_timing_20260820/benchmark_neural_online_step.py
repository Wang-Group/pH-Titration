from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_RELEASE = Path(r"C:\Users\ZSY\Desktop\FDTD\new_pf_ppo_controllers_20260812_release")
DEFAULT_IMITATION = Path(
    r"C:\Users\ZSY\Documents\xwechat_files\wxid_mkx0ewygqoen22_124f\msg\file\2026-08"
    r"\bayesian_external_rule_ablation_20260811\bayesian_external_rule_ablation_20260811"
    r"\results\complete_study_standard_v1\02_imitation\imitation_best.pth"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_ns(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64) / 1e6
    return {
        "repetitions": int(len(array)),
        "mean_ms": float(np.mean(array)),
        "sd_ms": float(np.std(array, ddof=1)),
        "median_ms": float(np.median(array)),
        "p05_ms": float(np.percentile(array, 5)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
    }


def build_states(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < count:
        current = float(rng.uniform(1.0, 13.0))
        target = float(rng.uniform(2.0, 12.0))
        if abs(current - target) <= 0.15:
            continue
        recent_change = float(rng.uniform(-2.0, 2.0))
        previous_dose = float(rng.integers(1, 1001)) * 0.01
        states.append(
            [current, target, recent_change, current - target, previous_dose]
        )
    return np.asarray(states, dtype=np.float32)


def neural_decision(actor, normalizer, state: np.ndarray) -> tuple[int, float, str]:
    """Hot single-step path used by both neural policies.

    This includes NumPy normalization, tensor construction, one actor forward
    pass, argmax, volume decoding, and the disclosed pH-direction rule.
    """
    normalized = normalizer.transform_numpy(state)
    tensor = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits = actor(tensor)
        action_class = int(torch.argmax(logits, dim=1).item())
    volume_ml = (action_class + 1) * 0.01
    reagent = "base" if float(state[0]) < float(state[1]) else "acid"
    return action_class, volume_ml, reagent


def full_hot_recommend(controller, state: np.ndarray):
    """Execute the released controller's complete hot recommend() path."""
    controller.current_ph = float(state[0])
    controller.target_ph = float(state[1])
    controller.previous_ph = float(state[0] - state[2])
    controller.last_requested_volume_ml = float(state[4])
    controller.total_actual_volume_ml = min(controller.total_actual_volume_ml, 1.0)
    controller.steps = 1
    controller.done = False
    controller.stop_reason = "running"
    controller.pending_action = None
    return controller.recommend()


def observation_to_next_action(controller, state: np.ndarray, controller_action_type):
    """Include observation bookkeeping before the complete next recommendation."""
    observed_ph = float(state[0])
    target_ph = float(state[1])
    before_ph = observed_ph - float(state[2])
    previous_volume = max(float(state[4]), 0.01)
    previous_reagent = "base" if before_ph < target_ph else "acid"
    controller.current_ph = before_ph
    controller.previous_ph = before_ph
    controller.target_ph = target_ph
    controller.last_requested_volume_ml = 0.0
    controller.total_actual_volume_ml = 0.0
    controller.base_added_ml = 0.0
    controller.acid_added_ml = 0.0
    controller.steps = 0
    controller.done = False
    controller.stop_reason = "running"
    controller.initialized = True
    controller.pending_action = controller_action_type(
        stop=False,
        reagent=previous_reagent,
        volume_ml=previous_volume,
    )
    controller.observe(
        measured_ph=observed_ph,
        actual_volume_ml=previous_volume,
        reagent=previous_reagent,
    )
    return controller.recommend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--imitation", type=Path, default=DEFAULT_IMITATION)
    parser.add_argument("--repetitions", type=int, default=30000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    if args.repetitions < args.blocks or args.repetitions % args.blocks:
        raise ValueError("repetitions must be divisible by blocks")
    release = args.release.resolve()
    imitation_path = args.imitation.resolve()
    ppo_path = (release / "models" / "ppo_seed_303.pth").resolve()
    for path in (release, imitation_path, ppo_path):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(release))
    from models import checkpoint_sha256, load_actor_checkpoint
    from new_rl_controller import PPOVolumeController
    from controller_api import ControllerAction

    # Single-request deployment latency is measured with one CPU thread. This
    # avoids counting thread-pool scheduling overhead as neural computation.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoints = {
        "PF-distilled imitation": imitation_path,
        "PPO from PF imitation": ppo_path,
    }
    loaded = {}
    controllers = {}
    load_rows = []
    for method, checkpoint in checkpoints.items():
        start = time.perf_counter_ns()
        actor, normalizer, metadata = load_actor_checkpoint(checkpoint, torch.device("cpu"))
        actor.eval()
        load_ns = time.perf_counter_ns() - start
        loaded[method] = (actor, normalizer, metadata)
        controller = PPOVolumeController(
            checkpoint,
            device="cpu",
            verify_selected_checkpoint=(method == "PPO from PF imitation"),
        )
        controller.reset(2.0, 8.0)
        controllers[method] = controller
        load_rows.append(
            {
                "method": method,
                "checkpoint": str(checkpoint),
                "checkpoint_file_sha256": sha256_file(checkpoint),
                "actor_tensor_sha256": checkpoint_sha256(actor.state_dict()),
                "load_time_ms": load_ns / 1e6,
                "metadata": metadata,
            }
        )

    states = build_states(args.repetitions, args.seed)
    warm_states = build_states(args.warmup, args.seed + 1)
    for actor, normalizer, _ in loaded.values():
        for state in warm_states:
            neural_decision(actor, normalizer, state)
    for method, controller in controllers.items():
        for state in warm_states:
            action = full_hot_recommend(controller, state)
            if action.stop:
                raise RuntimeError(f"Unexpected stop during warmup for {method}")
            action = observation_to_next_action(controller, state, ControllerAction)
            if action.stop:
                raise RuntimeError(f"Unexpected observation-cycle stop for {method}")

    timer_overhead = []
    for _ in range(20000):
        start = time.perf_counter_ns()
        timer_overhead.append(time.perf_counter_ns() - start)

    raw_rows = []
    latencies: dict[tuple[str, str], list[int]] = {
        (method, scope): []
        for method in loaded
        for scope in (
            "core neural decision",
            "full hot recommend wrapper",
            "new pH observation to next action",
        )
    }
    block_size = args.repetitions // args.blocks
    methods = list(loaded)
    for block in range(args.blocks):
        order = methods if block % 2 == 0 else list(reversed(methods))
        block_start = block * block_size
        block_stop = block_start + block_size
        for method in order:
            actor, normalizer, _ = loaded[method]
            for index in range(block_start, block_stop):
                state = states[index]
                start = time.perf_counter_ns()
                action_class, volume_ml, reagent = neural_decision(
                    actor, normalizer, state
                )
                elapsed = time.perf_counter_ns() - start
                latencies[(method, "core neural decision")].append(elapsed)
                raw_rows.append(
                    {
                        "method": method,
                        "timing_scope": "core neural decision",
                        "block": block + 1,
                        "state_index": index,
                        "current_ph": float(state[0]),
                        "target_ph": float(state[1]),
                        "recent_ph_change": float(state[2]),
                        "current_error": float(state[3]),
                        "previous_dose_ml": float(state[4]),
                        "action_class": action_class,
                        "volume_ml": volume_ml,
                        "reagent": reagent,
                        "latency_ns": elapsed,
                    }
                )
                controller = controllers[method]
                start = time.perf_counter_ns()
                action = full_hot_recommend(controller, state)
                wrapper_elapsed = time.perf_counter_ns() - start
                if action.stop:
                    raise RuntimeError(f"Unexpected stop during timing for {method}")
                latencies[(method, "full hot recommend wrapper")].append(wrapper_elapsed)
                raw_rows.append(
                    {
                        "method": method,
                        "timing_scope": "full hot recommend wrapper",
                        "block": block + 1,
                        "state_index": index,
                        "current_ph": float(state[0]),
                        "target_ph": float(state[1]),
                        "recent_ph_change": float(state[2]),
                        "current_error": float(state[3]),
                        "previous_dose_ml": float(state[4]),
                        "action_class": action.diagnostics["action_class"],
                        "volume_ml": action.volume_ml,
                        "reagent": action.reagent,
                        "latency_ns": wrapper_elapsed,
                    }
                )
                start = time.perf_counter_ns()
                next_action = observation_to_next_action(
                    controller, state, ControllerAction
                )
                cycle_elapsed = time.perf_counter_ns() - start
                if next_action.stop:
                    raise RuntimeError(f"Unexpected observation-cycle stop for {method}")
                latencies[(method, "new pH observation to next action")].append(
                    cycle_elapsed
                )
                raw_rows.append(
                    {
                        "method": method,
                        "timing_scope": "new pH observation to next action",
                        "block": block + 1,
                        "state_index": index,
                        "current_ph": float(state[0]),
                        "target_ph": float(state[1]),
                        "recent_ph_change": float(state[2]),
                        "current_error": float(state[3]),
                        "previous_dose_ml": float(state[4]),
                        "action_class": next_action.diagnostics["action_class"],
                        "volume_ml": next_action.volume_ml,
                        "reagent": next_action.reagent,
                        "latency_ns": cycle_elapsed,
                    }
                )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "neural_single_step_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary = []
    for method in methods:
        for scope in (
            "core neural decision",
            "full hot recommend wrapper",
            "new pH observation to next action",
        ):
            row = {"method": method, "timing_scope": scope}
            row.update(summarize_ns(latencies[(method, scope)]))
            load_row = next(item for item in load_rows if item["method"] == method)
            row["one_time_checkpoint_load_ms"] = load_row["load_time_ms"]
            row["checkpoint_file_sha256"] = load_row["checkpoint_file_sha256"]
            row["actor_tensor_sha256"] = load_row["actor_tensor_sha256"]
            summary.append(row)

    summary_path = output_dir / "neural_single_step_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "timer": "time.perf_counter_ns",
        "timer_overhead_ns": {
            "median": statistics.median(timer_overhead),
            "p95": float(np.percentile(timer_overhead, 95)),
        },
        "settings": vars(args) | {
            "release": str(release),
            "imitation": str(imitation_path),
            "output_dir": str(output_dir),
        },
        "checkpoint_loads": load_rows,
    }
    (output_dir / "neural_environment.json").write_text(
        json.dumps(environment, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
