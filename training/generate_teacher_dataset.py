from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .chemistry_model import SolutionState, solve_ph_scalar
    from .particle_controllers import build_controller
    from .task_distribution import ControlTask, generate_tasks, load_tasks, save_tasks
except ImportError:  # pragma: no cover - direct script compatibility
    from chemistry_model import SolutionState, solve_ph_scalar
    from particle_controllers import build_controller
    from task_distribution import ControlTask, generate_tasks, load_tasks, save_tasks


TEACHER_VARIANT = "pf_pka_conc_variable_k"
DATASET_VERSION = 6
QUALITY_CONTROL_VERSION = 6
QUALITY_SCORE_THRESHOLD = 0.30
MAX_QUALITY_STEPS = 30
MAX_QUALITY_OVERSHOOTS = 8
MAX_DOSE_OVERHEAD_ML = 20.0

EXPECTED_LEVELS = {
    "direction": ("acid", "base"),
    "error_bin": ("0.10-0.30", "0.30-1.00", "1.00-3.00", ">=3.00"),
    "volume_bin": ("0.01-0.10", "0.11-0.50", "0.51-2.00", "2.01-5.00", "5.01-10.00"),
    "acid_type": ("monoprotic", "diprotic", "triprotic"),
    "difficulty": ("near", "medium", "far"),
    "pka_family": ("single", "separated", "overlapping"),
    "true_pair_count": (1, 2, 3),
    "current_ph_bin": ("1.5-4", "4-7", "7-10", "10-12"),
    "target_ph_bin": ("1.5-4", "4-7", "7-10", "10-12"),
    "concentration_bin": ("0.03-0.06", "0.06-0.12", "0.12-0.18"),
    "initial_volume_bin": ("8-10.67", "10.67-13.33", "13.33-16"),
}

JOINT_COVERAGE = (
    ("direction_x_error", ("direction", "error_bin"), 0.005),
    ("direction_x_type_x_difficulty", ("direction", "acid_type", "difficulty"), 0.002),
    ("direction_x_pka_family", ("direction", "pka_family"), 0.005),
)


def _initialize_teacher(task: ControlTask, particles: int, seed: int):
    np.random.seed(seed)
    controller = build_controller(TEACHER_VARIANT, particles, seed + 17)
    controller.initialize_task(task)
    controller.base_added_moles = float(task.initial_base_moles)
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = float(task.initial_volume_ml)
    controller.previous_total_volume = float(task.initial_volume_ml)
    observed_initial_ph = float(np.round(task.initial_ph, 2))
    controller.current_ph = observed_initial_ph
    controller.previous_ph = observed_initial_ph
    controller.last_measured_ph = observed_initial_ph
    controller.prev_measured_ph = observed_initial_ph
    controller.last_action_volume = 0.0
    controller.done = False
    return controller


def _error_bin(error: float) -> str:
    if error < 0.30:
        return "0.10-0.30"
    if error < 1.00:
        return "0.30-1.00"
    if error < 3.00:
        return "1.00-3.00"
    return ">=3.00"


def _volume_bin(volume: float) -> str:
    if volume <= 0.10:
        return "0.01-0.10"
    if volume <= 0.50:
        return "0.11-0.50"
    if volume <= 2.00:
        return "0.51-2.00"
    if volume <= 5.00:
        return "2.01-5.00"
    return "5.01-10.00"


def _ph_bin(value: float) -> str:
    if value < 4.0:
        return "1.5-4"
    if value < 7.0:
        return "4-7"
    if value < 10.0:
        return "7-10"
    return "10-12"


def _concentration_bin(value: float) -> str:
    if value < 0.06:
        return "0.03-0.06"
    if value < 0.12:
        return "0.06-0.12"
    return "0.12-0.18"


def _initial_volume_bin(value: float) -> str:
    if value < 10.67:
        return "8-10.67"
    if value < 13.33:
        return "10.67-13.33"
    return "13.33-16"


def run_teacher_episode(
    task: ControlTask,
    particles: int,
    seed: int,
    perturb_probability: float,
    perturb_log_sd: float,
):
    rng = np.random.default_rng(seed + 91)
    controller = _initialize_teacher(task, particles, seed)
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
                "error_bin": _error_bin(previous_error),
                "volume_bin": _volume_bin(teacher_volume),
                "perturbed": int(perturbed),
                "task_id": task.task_id,
                "step": controller.steps_taken,
                "acid_type": task.acid_type,
                "difficulty": task.difficulty,
            }
        )
        dose_overhead_now = max(
            0.0,
            controller.acid_volume + controller.base_volume - task.oracle_required_volume_ml,
        )
        if (
            not done
            and (
                controller.steps_taken >= MAX_QUALITY_STEPS
                or overshoots > MAX_QUALITY_OVERSHOOTS
                or dose_overhead_now > MAX_DOSE_OVERHEAD_ML
            )
        ):
            break
        if done:
            break
    measured_final_error = abs(controller.current_ph - controller.target_ph)
    final_state = SolutionState(
        total_volume_ml=float(controller.total_volume),
        base_moles=float(controller.base_added_moles),
        acid_moles=float(controller.acid_added_moles),
    )
    final_true_ph = solve_ph_scalar(
        task.analyte_conc_m,
        task.pka_values,
        task.initial_volume_ml,
        final_state,
    )
    final_error = abs(final_true_ph - controller.target_ph)
    total_volume_ml = controller.acid_volume + controller.base_volume
    dose_overhead_ml = max(0.0, total_volume_ml - task.oracle_required_volume_ml)
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
        np.all(np.isfinite(row["state"]))
        and np.isfinite(row["teacher_volume_ml"])
        and np.isfinite(row["executed_volume_ml"])
        for row in pending
    )
    quality_pass = bool(
        final_error < 0.10
        and finite_records
        and 1 <= controller.steps_taken <= MAX_QUALITY_STEPS
        and overshoots <= MAX_QUALITY_OVERSHOOTS
        and dose_overhead_ml <= MAX_DOSE_OVERHEAD_ML
        and quality_score >= QUALITY_SCORE_THRESHOLD
    )
    for row in pending:
        row["quality_score"] = quality_score
        row["pka_family"] = task.pka_family
        row["true_pair_count"] = len(task.pka_values)
        row["current_ph_bin"] = _ph_bin(float(row["state"][0]))
        row["target_ph_bin"] = _ph_bin(float(row["state"][1]))
        row["concentration_bin"] = _concentration_bin(task.analyte_conc_m)
        row["initial_volume_bin"] = _initial_volume_bin(task.initial_volume_ml)
    metrics = {
        "task_id": task.task_id,
        "direction": task.direction,
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "success": int(final_error < 0.10),
        "strict_success": int(final_error < 0.05),
        "steps": controller.steps_taken,
        "overshoots": overshoots,
        "final_error": final_error,
        "final_true_ph": final_true_ph,
        "final_measured_ph": controller.current_ph,
        "measured_final_error": measured_final_error,
        "total_volume_ml": total_volume_ml,
        "oracle_required_volume_ml": task.oracle_required_volume_ml,
        "dose_overhead_ml": dose_overhead_ml,
        "quality_score": quality_score,
        "quality_pass": int(quality_pass),
        "perturbed_steps": perturbed_steps,
        "states_retained": len(pending) if quality_pass else 0,
    }
    return pending if quality_pass else [], metrics


def _record_key(row):
    rounded_state = tuple(np.round(np.asarray(row["state"], dtype=float), 3).tolist())
    return rounded_state + (int(row["label"]),)


def _coverage_deficits(records, minimum_states):
    if minimum_states <= 0:
        return []
    counts = {
        field: Counter(row[field] for row in records)
        for field in EXPECTED_LEVELS
    }
    requirements = {
        "direction": (EXPECTED_LEVELS["direction"], int(math.ceil(minimum_states / 2))),
        "error_bin": (EXPECTED_LEVELS["error_bin"], max(5, int(0.01 * minimum_states))),
        "volume_bin": (EXPECTED_LEVELS["volume_bin"], max(5, int(0.005 * minimum_states))),
        "acid_type": (EXPECTED_LEVELS["acid_type"], max(5, int(0.05 * minimum_states))),
        "difficulty": (EXPECTED_LEVELS["difficulty"], max(5, int(0.05 * minimum_states))),
        "pka_family": (EXPECTED_LEVELS["pka_family"], max(5, int(0.03 * minimum_states))),
        "true_pair_count": (EXPECTED_LEVELS["true_pair_count"], max(5, int(0.05 * minimum_states))),
        "current_ph_bin": (EXPECTED_LEVELS["current_ph_bin"], max(5, int(0.02 * minimum_states))),
        "target_ph_bin": (EXPECTED_LEVELS["target_ph_bin"], max(5, int(0.02 * minimum_states))),
        "concentration_bin": (EXPECTED_LEVELS["concentration_bin"], max(5, int(0.05 * minimum_states))),
        "initial_volume_bin": (EXPECTED_LEVELS["initial_volume_bin"], max(5, int(0.05 * minimum_states))),
    }
    deficits = []
    if len(records) < minimum_states:
        deficits.append(f"total:{len(records)}/{minimum_states}")
    for field, (values, required) in requirements.items():
        for value in sorted(values):
            observed = counts[field][value]
            if observed < required:
                deficits.append(f"{field}={value}:{observed}/{required}")
    for name, fields, fraction in JOINT_COVERAGE:
        required = max(2, int(fraction * minimum_states))
        joint_counts = Counter(tuple(row[field] for field in fields) for row in records)
        expected_combinations = itertools.product(*(EXPECTED_LEVELS[field] for field in fields))
        for combination in expected_combinations:
            observed = joint_counts[combination]
            if observed < required:
                label = "|".join(str(value) for value in combination)
                deficits.append(f"{name}={label}:{observed}/{required}")
    return deficits


def _diversity_audit_rows(split, records):
    total = max(1, len(records))
    rows = []
    for field, levels in EXPECTED_LEVELS.items():
        counts = Counter(row[field] for row in records)
        for level in levels:
            rows.append(
                {
                    "split": split,
                    "dimension": field,
                    "level": level,
                    "count": counts[level],
                    "percent": 100.0 * counts[level] / total,
                }
            )
    for name, fields, _ in JOINT_COVERAGE:
        counts = Counter(tuple(row[field] for field in fields) for row in records)
        for combination in itertools.product(*(EXPECTED_LEVELS[field] for field in fields)):
            count = counts[combination]
            rows.append(
                {
                    "split": split,
                    "dimension": name,
                    "level": "|".join(str(value) for value in combination),
                    "count": count,
                    "percent": 100.0 * count / total,
                }
            )
    return rows


def _weights(records):
    cells = Counter(
        (row["direction"], row["error_bin"], row["volume_bin"], row["acid_type"])
        for row in records
    )
    raw = np.asarray(
        [1.0 / math.sqrt(cells[(row["direction"], row["error_bin"], row["volume_bin"], row["acid_type"])]) for row in records],
        dtype=np.float32,
    )
    quality = np.asarray([row["quality_score"] for row in records], dtype=np.float32)
    raw *= 0.5 + quality
    raw /= float(np.mean(raw))
    return np.clip(raw, 0.25, 4.0)


def save_dataset(path: Path, records) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.stack([row["state"] for row in records]).astype(np.float32)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    weights = _weights(records)
    task_ids = np.asarray([row["task_id"] for row in records], dtype=np.int32)
    np.savez_compressed(path, states=states, labels=labels, weights=weights, task_ids=task_ids)
    metadata_path = path.with_name(path.stem + "_state_metadata.csv")
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task_id", "step", "acid_type", "difficulty", "direction", "error_bin",
            "volume_bin", "pka_family", "true_pair_count", "teacher_volume_ml",
            "current_ph_bin", "target_ph_bin", "concentration_bin", "initial_volume_bin",
            "executed_volume_ml", "perturbed", "quality_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})
    return {
        "states": len(records),
        "tasks_with_states": int(len(set(task_ids.tolist()))),
        "direction_counts": Counter(row["direction"] for row in records),
        "error_bin_counts": Counter(row["error_bin"] for row in records),
        "volume_bin_counts": Counter(row["volume_bin"] for row in records),
        "perturbed_state_percent": 100.0 * float(np.mean([row["perturbed"] for row in records])),
        "quality_score_mean": float(np.mean([row["quality_score"] for row in records])),
        "quality_score_min": float(np.min([row["quality_score"] for row in records])),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }


def _run_teacher_payload(payload):
    return run_teacher_episode(*payload)


def generate_split(tasks, particles, seed_offset, perturb_probability, perturb_log_sd, workers):
    records = []
    metrics = []
    payloads = [
        (
            task,
            particles,
            seed_offset + task.task_id * 1009,
            perturb_probability,
            perturb_log_sd,
        )
        for task in tasks
    ]
    if workers == 1:
        results = map(_run_teacher_payload, payloads)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_run_teacher_payload, payloads, chunksize=4)
    try:
        for index, (task_records, task_metrics) in enumerate(results, 1):
            records.extend(task_records)
            metrics.append(task_metrics)
            progress_interval = min(100, max(10, len(tasks)))
            if index % progress_interval == 0 or index == len(tasks):
                print(f"teacher {tasks[index - 1].split}: {index}/{len(tasks)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)
    return records, metrics


def generate_quality_controlled_split(
    split,
    base_seed,
    initial_tasks,
    minimum_states,
    particles,
    perturb_probability,
    perturb_log_sd,
    workers,
):
    all_tasks = []
    all_metrics = []
    unique_records = {}
    round_index = 0
    top_up_batch = max(
        20,
        min(2000, max(2, initial_tasks // 2, minimum_states // 20)),
    )
    if top_up_batch % 2:
        top_up_batch += 1
    maximum_tasks = max(
        initial_tasks * 4,
        initial_tasks + 2000,
        int(math.ceil(minimum_states / 2.0)),
    )
    audit = []
    while True:
        batch_size = initial_tasks if round_index == 0 else min(
            top_up_batch,
            maximum_tasks - len(all_tasks),
        )
        if batch_size <= 0:
            deficits = _coverage_deficits(list(unique_records.values()), minimum_states)
            raise RuntimeError(
                f"Could not meet {split} data-quality targets within {maximum_tasks} candidate tasks: "
                + "; ".join(deficits)
            )
        if batch_size % 2 and len(all_tasks) + batch_size < maximum_tasks:
            batch_size += 1
        task_seed = base_seed + round_index * 100_003
        batch_tasks = generate_tasks(
            task_seed,
            batch_size,
            split,
            task_id_offset=len(all_tasks),
        )
        batch_records, batch_metrics = generate_split(
            batch_tasks,
            particles,
            base_seed * 13,
            perturb_probability,
            perturb_log_sd,
            workers,
        )
        all_tasks.extend(batch_tasks)
        all_metrics.extend(batch_metrics)
        for record in batch_records:
            key = _record_key(record)
            previous = unique_records.get(key)
            if previous is None or record["quality_score"] > previous["quality_score"]:
                unique_records[key] = record
        records = list(unique_records.values())
        deficits = _coverage_deficits(records, minimum_states)
        audit.append(
            {
                "round": round_index + 1,
                "candidate_tasks_total": len(all_tasks),
                "quality_pass_tasks": sum(row["quality_pass"] for row in all_metrics),
                "unique_retained_states": len(records),
                "acid_states": sum(row["direction"] == "acid" for row in records),
                "base_states": sum(row["direction"] == "base" for row in records),
                "remaining_deficits": " | ".join(deficits),
            }
        )
        print(
            f"{split} quality audit round {round_index + 1}: "
            f"tasks={len(all_tasks)}, unique_states={len(records)}, deficits={len(deficits)}",
            flush=True,
        )
        if not deficits:
            return all_tasks, records, all_metrics, audit
        round_index += 1


def write_rows(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate imitation data from the robust PF teacher")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tasks", type=int, default=5000)
    parser.add_argument("--validation-tasks", type=int, default=500)
    parser.add_argument("--test-tasks", type=int, default=1000)
    parser.add_argument("--min-train-states", type=int, default=0)
    parser.add_argument("--min-validation-states", type=int, default=0)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--perturb-probability", type=float, default=0.25)
    parser.add_argument("--perturb-log-sd", type=float, default=0.25)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Teacher worker processes; 0 selects up to eight processes automatically.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.particles < 60:
        raise ValueError("The variable-K teacher requires at least 60 total particles.")
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    if workers < 1:
        raise ValueError("--workers must be zero or a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = args.output_dir / "TEACHER_DATA_COMPLETE.json"
    if args.resume and complete.exists():
        existing = json.loads(complete.read_text(encoding="utf-8"))
        expected = {
            "train_tasks": args.train_tasks,
            "validation_tasks": args.validation_tasks,
            "test_tasks": args.test_tasks,
            "min_train_states": args.min_train_states,
            "min_validation_states": args.min_validation_states,
            "particles": args.particles,
            "perturb_probability": args.perturb_probability,
            "perturb_log_sd": args.perturb_log_sd,
        }
        if existing.get("dataset_version") == DATASET_VERSION and existing.get("requested_config") == expected:
            print(f"Teacher data already complete: {complete}")
            return
        print("Existing teacher data use an older or different configuration; regenerating.", flush=True)

    split_specs = {
        "train": (310_001, args.train_tasks, args.min_train_states),
        "validation": (410_001, args.validation_tasks, args.min_validation_states),
    }
    tasks_by_split = {}
    summaries = {}
    for split, (base_seed, initial_tasks, minimum_states) in split_specs.items():
        split_summary_path = args.output_dir / f"{split}_teacher_summary.json"
        dataset_path = args.output_dir / f"{split}_teacher_dataset.npz"
        metadata_path = args.output_dir / f"{split}_teacher_dataset_state_metadata.csv"
        task_results_path = args.output_dir / f"{split}_teacher_task_results.csv"
        tasks_path = args.output_dir / f"{split}_tasks.jsonl"
        quality_audit_path = args.output_dir / f"{split}_quality_generation_audit.csv"
        diversity_audit_path = args.output_dir / f"{split}_diversity_audit.csv"
        if (
            args.resume
            and split_summary_path.exists()
            and dataset_path.exists()
            and metadata_path.exists()
            and task_results_path.exists()
            and tasks_path.exists()
            and quality_audit_path.exists()
            and diversity_audit_path.exists()
        ):
            existing_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
            if (
                existing_summary.get("quality_control_version") == QUALITY_CONTROL_VERSION
                and existing_summary.get("minimum_unique_states") == minimum_states
                and existing_summary.get("particles") == args.particles
                and existing_summary.get("initial_candidate_tasks") == initial_tasks
                and existing_summary.get("perturb_probability") == (
                    args.perturb_probability if split == "train" else 0.0
                )
            ):
                summaries[split] = existing_summary
                tasks_by_split[split] = load_tasks(tasks_path)
                print(f"Teacher {split} split already complete: {split_summary_path}", flush=True)
                continue
        probability = args.perturb_probability if split == "train" else 0.0
        tasks, records, metrics, audit = generate_quality_controlled_split(
            split,
            base_seed,
            initial_tasks,
            minimum_states,
            args.particles,
            probability,
            args.perturb_log_sd,
            workers,
        )
        if not records:
            raise RuntimeError(f"No quality-screened teacher states for {split}")
        tasks_by_split[split] = tasks
        save_tasks(tasks_path, tasks)
        summaries[split] = save_dataset(dataset_path, records)
        summaries[split]["teacher_task_success_percent"] = 100.0 * float(np.mean([row["success"] for row in metrics]))
        summaries[split]["quality_pass_task_percent"] = 100.0 * float(np.mean([row["quality_pass"] for row in metrics]))
        summaries[split]["candidate_tasks"] = len(tasks)
        summaries[split]["minimum_unique_states"] = minimum_states
        summaries[split]["initial_candidate_tasks"] = initial_tasks
        summaries[split]["perturb_probability"] = probability
        summaries[split]["particles"] = args.particles
        summaries[split]["quality_control_version"] = QUALITY_CONTROL_VERSION
        write_rows(task_results_path, metrics)
        write_rows(quality_audit_path, audit)
        write_rows(diversity_audit_path, _diversity_audit_rows(split, records))
        split_summary_path.write_text(
            json.dumps(summaries[split], indent=2, default=dict),
            encoding="utf-8",
        )

    test_seed = 510_001
    tasks_by_split["test"] = generate_tasks(test_seed, args.test_tasks, "test")
    save_tasks(args.output_dir / "test_tasks.jsonl", tasks_by_split["test"])

    payload = {
        "dataset_version": DATASET_VERSION,
        "requested_config": {
            "train_tasks": args.train_tasks,
            "validation_tasks": args.validation_tasks,
            "test_tasks": args.test_tasks,
            "min_train_states": args.min_train_states,
            "min_validation_states": args.min_validation_states,
            "particles": args.particles,
            "perturb_probability": args.perturb_probability,
            "perturb_log_sd": args.perturb_log_sd,
        },
        "teacher_variant": TEACHER_VARIANT,
        "particles": args.particles,
        "task_distribution": "variable concentration, volume, K, ordered pKa, initial neutralization, balanced direction and difficulty, reachable within 30 mL",
        "split_seeds": {"train": 310_001, "validation": 410_001, "test": test_seed},
        "initial_candidate_task_counts": {
            "train": args.train_tasks,
            "validation": args.validation_tasks,
            "test": args.test_tasks,
        },
        "minimum_unique_state_counts": {
            "train": args.min_train_states,
            "validation": args.min_validation_states,
        },
        "final_candidate_task_counts": {
            split: len(tasks) for split, tasks in tasks_by_split.items()
        },
        "accepted_task_direction_counts": {
            split: dict(Counter(task.direction for task in tasks))
            for split, tasks in tasks_by_split.items()
        },
        "perturb_probability_train": args.perturb_probability,
        "perturb_log_sd": args.perturb_log_sd,
        "teacher_workers": workers,
        "quality_screen": {
            "teacher": TEACHER_VARIANT,
            "endpoint_abs_error_max": 0.10,
            "quality_score_min": QUALITY_SCORE_THRESHOLD,
            "steps_max": MAX_QUALITY_STEPS,
            "overshoots_max": MAX_QUALITY_OVERSHOOTS,
            "dose_overhead_max_ml": MAX_DOSE_OVERHEAD_ML,
            "nonfinite_records_removed": True,
            "near_duplicate_key": "state rounded to 0.001 plus action class; keep higher-quality record",
            "coverage": "marginal physicochemical/state coverage plus predefined joint-stratum minima",
        },
        "summaries": summaries,
    }
    direction_audit = []
    for split, tasks in tasks_by_split.items():
        state_counts = summaries.get(split, {}).get("direction_counts", {})
        for direction in ("acid", "base"):
            direction_audit.append(
                {
                    "split": split,
                    "direction": direction,
                    "accepted_tasks": sum(task.direction == direction for task in tasks),
                    "retained_teacher_states": state_counts.get(direction, ""),
                }
            )
    write_rows(args.output_dir / "direction_balance_audit.csv", direction_audit)
    complete.write_text(json.dumps(payload, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
