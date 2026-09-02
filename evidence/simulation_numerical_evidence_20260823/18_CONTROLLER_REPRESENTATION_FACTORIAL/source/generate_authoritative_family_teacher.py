from __future__ import annotations

"""Generate F2-F6 teacher data from the recovered formal PF source."""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


FAMILIES = {
    "F2": ("sequential_k123", "sequential_k", "map_order_predictive"),
    "F3": ("sequential_k123", "sequential_k", "full_predictive"),
    "F4": ("independent_j123", "sequential_k", "weighted_parameters"),
    "F5": ("sequential_k123", "independent_j", "weighted_parameters"),
    "F6": ("independent_j123", "independent_j", "weighted_parameters"),
}

FORMAL = None
RUNNER = None
TASK_DOMAIN = ""
REPRESENTATION = ""
STRATEGY = ""
INDEPENDENT_GENERATOR = None


def row_from_task(task) -> dict:
    if isinstance(task, dict):
        return dict(task)
    if is_dataclass(task):
        return asdict(task)
    return dict(vars(task))


def object_from_row(row: dict):
    return SimpleNamespace(**row)


def load_independent_generator(path: Path):
    spec = importlib.util.spec_from_file_location("independent_j123_generator_authoritative", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load independent-J generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def family_generate_tasks(seed: int, count: int, split: str, task_id_offset: int = 0):
    if TASK_DOMAIN == "sequential_k123":
        tasks = FORMAL._ORIGINAL_GENERATE_TASKS(seed, count, split, task_id_offset=task_id_offset)
        return tasks
    generated = INDEPENDENT_GENERATOR.generate_seed(seed, count)
    tasks = []
    for index, task in enumerate(generated, 1):
        row = row_from_task(task)
        row["task_id"] = int(task_id_offset) + index
        row["split"] = split
        tasks.append(object_from_row(row))
    return tasks


def save_tasks(path: Path, tasks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(row_from_task(task), separators=(",", ":")) + "\n")


def load_tasks(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [object_from_row(json.loads(line)) for line in handle if line.strip()]


def run_teacher_episode(task, particles, seed, perturb_probability, perturb_log_sd):
    row = row_from_task(task)
    rng = np.random.default_rng(seed + 91)
    controller = RUNNER.initialize_controller(
        row,
        TASK_DOMAIN,
        REPRESENTATION,
        STRATEGY,
        int(seed),
    )
    pending = []
    perturbed_steps = 0
    overshoots = 0
    while not controller.done:
        controller.use_secondary_reagents = False
        state = controller.get_state().astype(np.float32)
        action, _ = controller.select_best_action()
        teacher_volume = float(action[1])
        teacher_class = int(np.clip(round(teacher_volume / 0.01) - 1, 0, 999))
        direction = "base" if controller.current_ph < controller.target_ph else "acid"
        reagent = "Dilute base 1" if direction == "base" else "Dilute acid 1"
        executed_volume = teacher_volume
        perturbed = rng.random() < perturb_probability
        if perturbed:
            executed_volume *= float(rng.lognormal(0.0, perturb_log_sd))
            executed_volume += float(rng.normal(0.0, 0.03))
            executed_volume = float(np.clip(executed_volume, 0.01, 10.0))
            perturbed_steps += 1
        previous_error = abs(controller.current_ph - controller.target_ph)
        current_ph, _, done, info = controller.step((reagent, executed_volume), mode="Simulate")
        overshoots += int(bool(info.get("crossed_target", False)))
        controller.update_posteriors((reagent, executed_volume), current_ph)
        pending.append(
            {
                "state": state,
                "label": teacher_class,
                "teacher_volume_ml": teacher_volume,
                "executed_volume_ml": executed_volume,
                "direction": direction,
                "error_bin": FORMAL._error_bin(previous_error),
                "volume_bin": FORMAL._volume_bin(teacher_volume),
                "perturbed": int(perturbed),
                "task_id": int(row["task_id"]),
                "step": int(controller.steps_taken),
                "acid_type": row["acid_type"],
                "difficulty": row["difficulty"],
            }
        )
        dose_overhead_now = max(
            0.0,
            controller.acid_volume + controller.base_volume - float(row["oracle_required_volume_ml"]),
        )
        if (
            not done
            and (
                controller.steps_taken >= FORMAL.MAX_QUALITY_STEPS
                or overshoots > FORMAL.MAX_QUALITY_OVERSHOOTS
                or dose_overhead_now > FORMAL.MAX_DOSE_OVERHEAD_ML
            )
        ):
            break
        if done:
            break

    measured_final_error = abs(controller.current_ph - controller.target_ph)
    from chemistry_model import SolutionState

    final_state = SolutionState(
        total_volume_ml=float(controller.total_volume),
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles),
    )
    final_true_ph = RUNNER.truth_ph(row, TASK_DOMAIN, final_state)
    final_error = abs(final_true_ph - controller.target_ph)
    total_volume_ml = controller.acid_volume + controller.base_volume
    dose_overhead_ml = max(0.0, total_volume_ml - float(row["oracle_required_volume_ml"]))
    endpoint_score = float(np.clip(1.0 - final_error / 0.10, 0.0, 1.0))
    step_score = float(np.clip(1.0 - max(0, controller.steps_taken - 1) / 29.0, 0.0, 1.0))
    overshoot_score = 1.0 / (1.0 + overshoots)
    dose_score = float(np.exp(-dose_overhead_ml / 10.0))
    quality_score = (
        0.40 * endpoint_score
        + 0.25 * step_score
        + 0.15 * overshoot_score
        + 0.20 * dose_score
    )
    finite_records = all(
        np.all(np.isfinite(item["state"]))
        and np.isfinite(item["teacher_volume_ml"])
        and np.isfinite(item["executed_volume_ml"])
        for item in pending
    )
    quality_pass = bool(
        final_error <= 0.10
        and finite_records
        and 1 <= controller.steps_taken <= FORMAL.MAX_QUALITY_STEPS
        and overshoots <= FORMAL.MAX_QUALITY_OVERSHOOTS
        and dose_overhead_ml <= FORMAL.MAX_DOSE_OVERHEAD_ML
        and quality_score >= FORMAL.QUALITY_SCORE_THRESHOLD
    )
    true_count = int(row.get("component_count", len(row["pka_values"])))
    for item in pending:
        item["quality_score"] = quality_score
        item["pka_family"] = row["pka_family"]
        item["true_pair_count"] = true_count
        item["current_ph_bin"] = FORMAL._ph_bin(float(item["state"][0]))
        item["target_ph_bin"] = FORMAL._ph_bin(float(item["state"][1]))
        item["concentration_bin"] = FORMAL._concentration_bin(float(row["analyte_conc_m"]))
        item["initial_volume_bin"] = FORMAL._initial_volume_bin(float(row["initial_volume_ml"]))
    metrics = {
        "task_id": int(row["task_id"]),
        "direction": row["direction"],
        "initial_ph": float(row["initial_ph"]),
        "target_ph": float(row["target_ph"]),
        "success": int(final_error <= 0.10),
        "strict_success": int(final_error <= 0.05),
        "steps": int(controller.steps_taken),
        "overshoots": overshoots,
        "final_error": final_error,
        "final_true_ph": final_true_ph,
        "final_measured_ph": float(controller.current_ph),
        "measured_final_error": measured_final_error,
        "total_volume_ml": total_volume_ml,
        "oracle_required_volume_ml": float(row["oracle_required_volume_ml"]),
        "dose_overhead_ml": dose_overhead_ml,
        "quality_score": quality_score,
        "quality_pass": int(quality_pass),
        "perturbed_steps": perturbed_steps,
        "states_retained": len(pending) if quality_pass else 0,
    }
    return pending if quality_pass else [], metrics


def configure(args) -> None:
    global FORMAL, RUNNER, TASK_DOMAIN, REPRESENTATION, STRATEGY, INDEPENDENT_GENERATOR
    TASK_DOMAIN, REPRESENTATION, STRATEGY = FAMILIES[args.family]
    formal_text = str(args.formal_source.resolve())
    staging_text = str(args.staging_root.resolve())
    if formal_text not in sys.path:
        sys.path.insert(0, formal_text)
    if staging_text not in sys.path:
        sys.path.insert(1, staging_text)
    import run_full_pf_factorial as runner
    runner.configure(args.formal_source, args.staging_root)
    RUNNER = runner
    import generate_teacher_dataset as formal
    import task_distribution

    FORMAL = formal
    FORMAL._ORIGINAL_GENERATE_TASKS = task_distribution.generate_tasks
    FORMAL.generate_tasks = family_generate_tasks
    FORMAL.save_tasks = save_tasks
    FORMAL.load_tasks = load_tasks
    FORMAL.run_teacher_episode = run_teacher_episode
    if TASK_DOMAIN == "independent_j123":
        # The formal quality-control screen is retained, but its acid-type
        # labels must describe the independent-component generator rather than
        # the sequential mono/di/triprotic generator.
        FORMAL.EXPECTED_LEVELS = dict(FORMAL.EXPECTED_LEVELS)
        FORMAL.EXPECTED_LEVELS["acid_type"] = (
            "1_independent_monoprotic_component",
            "2_independent_monoprotic_components",
            "3_independent_monoprotic_components",
        )
    INDEPENDENT_GENERATOR = load_independent_generator(args.independent_generator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--independent-generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=34)
    parser.add_argument("--train-tasks", type=int, default=5000)
    parser.add_argument("--validation-tasks", type=int, default=500)
    parser.add_argument("--min-train-states", type=int, default=60000)
    parser.add_argument("--min-validation-states", type=int, default=12000)
    parser.add_argument("--test-tasks", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure(args)
    protocol = {
        "family": args.family,
        "task_domain": TASK_DOMAIN,
        "representation": REPRESENTATION,
        "posterior_to_control": STRATEGY,
        "formal_source": str(args.formal_source.resolve()),
        "train_tasks_initial": args.train_tasks,
        "validation_tasks_initial": args.validation_tasks,
        "minimum_train_states": args.min_train_states,
        "minimum_validation_states": args.min_validation_states,
        "workers": args.workers,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "FAMILY_PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Existing family protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    old_argv = sys.argv
    sys.argv = [
        "generate_teacher_dataset.py",
        "--output-dir", str(args.output),
        "--train-tasks", str(args.train_tasks),
        "--validation-tasks", str(args.validation_tasks),
        "--test-tasks", str(args.test_tasks),
        "--min-train-states", str(args.min_train_states),
        "--min-validation-states", str(args.min_validation_states),
        "--particles", "1000",
        "--workers", str(args.workers),
        *( ["--resume"] if args.resume else [] ),
    ]
    try:
        FORMAL.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
