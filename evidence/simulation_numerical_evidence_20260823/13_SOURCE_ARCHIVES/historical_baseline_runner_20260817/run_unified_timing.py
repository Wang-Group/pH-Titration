from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from baseline_controllers import (
    PRESPECIFIED_PID,
    FixedGainPIDController,
    PIDConfig,
    SimpleRuleController,
)
from chemistry_model import SolutionState, solve_ph_scalar
from models import StateNormalizer, VolumeActor
from new_pf_controller import RobustPFController
from task_distribution import load_tasks


EXPECTED_CHECKPOINT_SHA256 = {
    "imitation": "71ae0176252d55c2a30b5d69afc6033ee1b0f4ac045e05b689ae7031c488adcc",
    "ppo": "496797be6be22dacd9f7360e7980a983dca816b7bded19597c4be6beb58abd23",
}
MAX_STEPS = 50
MAX_TOTAL_DOSE_ML = 50.0
SUCCESS_TOLERANCE_PH = 0.10


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


def load_actor(path: Path, device: torch.device) -> tuple[VolumeActor, StateNormalizer]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    actor = VolumeActor().to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    normalizer = StateNormalizer(
        np.asarray(payload["state_mean"], dtype=np.float32),
        np.asarray(payload["state_std"], dtype=np.float32),
    )
    return actor, normalizer


class NetworkRuntime:
    def __init__(self, actor: VolumeActor, normalizer: StateNormalizer, device: torch.device) -> None:
        self.actor = actor
        self.normalizer = normalizer
        self.device = device

    def reset(self, current_ph: float, target_ph: float) -> None:
        self.current_ph = float(current_ph)
        self.previous_ph = float(current_ph)
        self.target_ph = float(target_ph)
        self.last_requested_volume_ml = 0.0
        self.cap_ml: float | None = None

    def recommend(self) -> tuple[str, float]:
        state = np.asarray(
            [
                self.current_ph,
                self.target_ph,
                self.current_ph - self.previous_ph,
                self.current_ph - self.target_ph,
                self.last_requested_volume_ml,
            ],
            dtype=np.float32,
        )
        normalized = self.normalizer.transform_numpy(state)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(tensor)
            action_class = int(torch.argmax(logits, dim=1).item())
        requested = round((action_class + 1) * 0.01, 2)
        if self.cap_ml is None:
            executed = requested
        else:
            class_cap = max(0.01, math.floor((self.cap_ml + 1e-12) * 100.0) / 100.0)
            executed = round(min(requested, class_cap), 2)
        reagent = "base" if self.current_ph < self.target_ph else "acid"
        return reagent, executed

    def observe(self, measured_ph: float, actual_volume_ml: float) -> None:
        before = self.current_ph
        after = float(measured_ph)
        crossed = (before - self.target_ph) * (after - self.target_ph) < 0.0
        error_increased = abs(after - self.target_ph) > abs(before - self.target_ph)
        self.previous_ph = before
        self.current_ph = after
        self.last_requested_volume_ml = float(actual_volume_ml)
        if crossed or error_increased:
            new_cap = max(float(actual_volume_ml) / 2.0, 0.01)
            self.cap_ml = new_cap if self.cap_ml is None else min(self.cap_ml, new_cap)


def chemistry_step(task, total_volume_ml: float, base_moles: float, acid_moles: float, reagent: str, volume_ml: float):
    total_volume_ml += volume_ml
    if reagent == "base":
        base_moles += 0.1 * volume_ml / 1000.0
    else:
        acid_moles += 0.1 * volume_ml / 1000.0
    true_ph = solve_ph_scalar(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        SolutionState(total_volume_ml, base_moles, acid_moles),
    )
    return total_volume_ml, base_moles, acid_moles, float(true_ph)


def build_runtime(method: str, task, package: Path, device: torch.device, actors: dict, selected_pid: dict | None):
    if method == "pf_1000":
        seed = int(task.seed * 1_000_003 + task.task_id) % (2**32 - 1)
        controller = RobustPFController(
            particles=1000,
            seed=seed,
            max_steps=MAX_STEPS,
            max_total_dose_ml=MAX_TOTAL_DOSE_ML,
        )
        controller.reset(
            task.initial_ph,
            task.target_ph,
            task.initial_volume_ml,
            task.initial_base_moles,
            0.0,
        )
        return controller
    if method in {"imitation", "ppo"}:
        actor, normalizer = actors[method]
        runtime = NetworkRuntime(actor, normalizer, device)
        runtime.reset(float(np.round(task.initial_ph, 2)), task.target_ph)
        return runtime
    if method == "prespecified_pid":
        runtime = FixedGainPIDController(PRESPECIFIED_PID)
    elif method == "selected_pid":
        if selected_pid is None:
            raise ValueError("selected_pid parameters are required")
        runtime = FixedGainPIDController(PIDConfig.from_mapping(selected_pid))
    elif method == "simple_rule":
        runtime = SimpleRuleController()
    else:
        raise ValueError(method)
    runtime.reset(float(np.round(task.initial_ph, 2)), task.target_ph)
    return runtime


def run_episode(method: str, task, package: Path, device: torch.device, actors: dict, selected_pid: dict | None) -> dict:
    runtime = build_runtime(method, task, package, device, actors, selected_pid)
    true_ph = float(task.initial_ph)
    measured_ph = float(np.round(true_ph, 2))
    total_volume_ml = float(task.initial_volume_ml)
    base_moles = float(task.initial_base_moles)
    acid_moles = 0.0
    total_added_ml = 0.0
    steps = 0
    decision_ns = 0
    update_ns = 0

    while (
        abs(measured_ph - task.target_ph) > SUCCESS_TOLERANCE_PH
        and steps < MAX_STEPS
        and total_added_ml < MAX_TOTAL_DOSE_ML - 1e-12
    ):
        started = time.perf_counter_ns()
        if method == "pf_1000":
            action = runtime.recommend()
            if action.stop:
                decision_ns += time.perf_counter_ns() - started
                break
            reagent = str(action.reagent)
            requested = float(action.volume_ml)
        elif method in {"imitation", "ppo"}:
            reagent, requested = runtime.recommend()
        else:
            reagent = "base" if measured_ph < task.target_ph else "acid"
            requested = float(runtime.recommend(measured_ph))
        decision_ns += time.perf_counter_ns() - started

        remaining = MAX_TOTAL_DOSE_ML - total_added_ml
        actual = float(np.clip(requested, 0.01, remaining))
        before_true_ph = true_ph
        total_volume_ml, base_moles, acid_moles, true_ph = chemistry_step(
            task, total_volume_ml, base_moles, acid_moles, reagent, actual
        )
        measured_ph = float(np.round(np.clip(true_ph, 0.0, 14.0), 2))
        crossed = (before_true_ph - task.target_ph) * (true_ph - task.target_ph) < 0.0

        started = time.perf_counter_ns()
        if method == "pf_1000":
            runtime.observe(measured_ph, actual, reagent)
        elif method in {"imitation", "ppo"}:
            runtime.observe(measured_ph, actual)
        else:
            runtime.observe(measured_ph, reagent == "base", actual, crossed)
        update_ns += time.perf_counter_ns() - started
        total_added_ml += actual
        steps += 1

    controller_ms = (decision_ns + update_ns) / 1_000_000.0
    return {
        "method": method,
        "task_seed": task.seed,
        "task_id": task.task_id,
        "steps": steps,
        "success": int(abs(true_ph - task.target_ph) <= SUCCESS_TOLERANCE_PH),
        "decision_ms_total": decision_ns / 1_000_000.0,
        "update_ms_total": update_ns / 1_000_000.0,
        "controller_ms_total": controller_ms,
        "controller_ms_per_step": controller_ms / max(1, steps),
    }


def summarize(rows: list[dict], method: str) -> dict:
    values = np.asarray(
        [row["controller_ms_per_step"] for row in rows if row["method"] == method],
        dtype=float,
    )
    decisions = np.asarray(
        [row["decision_ms_total"] / max(1, row["steps"]) for row in rows if row["method"] == method],
        dtype=float,
    )
    updates = np.asarray(
        [row["update_ms_total"] / max(1, row["steps"]) for row in rows if row["method"] == method],
        dtype=float,
    )
    return {
        "method": method,
        "episodes": len(values),
        "controller_ms_per_step_median": float(np.median(values)),
        "controller_ms_per_step_mean": float(np.mean(values)),
        "controller_ms_per_step_p95": float(np.percentile(values, 95)),
        "decision_ms_per_step_mean": float(np.mean(decisions)),
        "update_ms_per_step_mean": float(np.mean(updates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified software-only controller timing")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--warmup-tasks", type=int, default=10)
    parser.add_argument("--task-seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--selected-pid-json", type=Path, default=None)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    output = (args.output_dir or package / "results_unified_timing").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.tasks < 1:
        raise ValueError("--tasks must be positive")

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    checkpoint_paths = {
        "imitation": package / "models" / "imitation_best.pth",
        "ppo": package / "models" / "ppo_seed_303.pth",
    }
    for method, path in checkpoint_paths.items():
        actual = sha256(path)
        if actual != EXPECTED_CHECKPOINT_SHA256[method]:
            raise RuntimeError(f"Unexpected {method} checkpoint hash: {actual}")
    actors = {method: load_actor(path, device) for method, path in checkpoint_paths.items()}

    selected_pid = None
    methods = ["prespecified_pid", "simple_rule", "imitation", "ppo", "pf_1000"]
    if args.selected_pid_json is not None:
        payload = json.loads(args.selected_pid_json.read_text(encoding="utf-8"))
        selected_pid = payload.get("parameters", payload)
        methods.insert(1, "selected_pid")

    all_tasks = load_tasks(package / "tasks" / f"seed_{args.task_seed}_tasks.jsonl")
    tasks = all_tasks[: args.tasks]
    warmup = all_tasks[args.tasks : args.tasks + args.warmup_tasks]
    if len(warmup) < args.warmup_tasks:
        warmup = all_tasks[: args.warmup_tasks]
    for method in methods:
        for task in warmup:
            run_episode(method, task, package, device, actors, selected_pid)

    rows = []
    for method in methods:
        for index, task in enumerate(tasks, 1):
            rows.append(run_episode(method, task, package, device, actors, selected_pid))
            if index % 20 == 0 or index == len(tasks):
                print(f"{method}: {index}/{len(tasks)}", flush=True)
    summary = [summarize(rows, method) for method in methods]
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
    }
    protocol = {
        "timed": "controller decision plus post-observation controller update",
        "excluded": [
            "checkpoint loading",
            "PF initialization/reset",
            "task loading",
            "chemical-system simulation",
            "liquid delivery",
            "mixing",
            "sensor stabilization and acquisition",
            "file input/output",
        ],
        "neural_timing_includes": "state construction, normalization, actor inference, argmax, reagent direction, and persistent cap update",
        "pf_timing_includes": "1000-particle next-action calculation and posterior update",
        "summary_unit": "episode-level controller milliseconds per dosing step",
    }
    write_csv(output / "task_timing.csv", rows)
    write_csv(output / "summary.csv", summary)
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "task_seed": args.task_seed,
                "tasks": args.tasks,
                "warmup_tasks": args.warmup_tasks,
                "methods": methods,
                "hardware": hardware,
                "protocol": protocol,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
