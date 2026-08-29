from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from experiment_utils import run_control_episode
from io_utils import write_json
from particle_inference import PRIOR_PKA_HIGH, PRIOR_PKA_LOW, build_filter
from task_generation import generate_comparison_tasks


def load_original_module(path: Path):
    spec = importlib.util.spec_from_file_location("particle_inference_original_snapshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bank_state(inference):
    rows = []
    for pair_count, bank in sorted(inference.banks.items()):
        values = np.asarray(bank.pka_particles, dtype=float)
        outside = (values < PRIOR_PKA_LOW) | (values > PRIOR_PKA_HIGH)
        rows.append(
            {
                "pair_count": int(pair_count),
                "minimum_pka": float(np.min(values)),
                "maximum_pka": float(np.max(values)),
                "outside_particle_value_fraction": float(np.mean(outside)),
                "outside_particle_row_fraction": float(np.mean(np.any(outside, axis=1))),
            }
        )
    return rows


def summarize(records):
    final = [item for record in records for item in record["final_banks"]]
    return {
        "tasks": len(records),
        "model_banks": len(final),
        "banks_outside_at_final": sum(item["outside_particle_value_fraction"] > 0.0 for item in final),
        "banks_outside_at_any_update": sum(any(value > 0.0 for value in record["maximum_outside_fraction_by_k"][str(k)]) for record in records for k in (1, 2, 3)),
        "minimum_pka_observed": min(item["minimum_pka"] for item in final),
        "maximum_pka_observed": max(item["maximum_pka"] for item in final),
        "maximum_outside_particle_value_fraction": max(item["outside_particle_value_fraction"] for item in final),
        "maximum_outside_particle_row_fraction": max(item["outside_particle_row_fraction"] for item in final),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproducible before/after PF pKa support audit")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("diagnostics_corrected") / "prior_support_before_after.json")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    original = load_original_module(base / "original_source_snapshot" / "particle_inference.py")
    tasks = generate_comparison_tasks(
        args.seed,
        args.tasks,
        "variable_concentration",
        minimum_initial_error_ph=1.0,
    )
    original_records = []
    corrected_records = []
    started = time.perf_counter()
    for task in tasks:
        task_seed = task.seed * 1_000_003 + task.task_id
        _, transitions, _ = run_control_episode(
            task,
            "pf_pka_only_k3",
            args.particles,
            task_seed,
            keep_trajectory=True,
        )
        filters = {
            "original": original.build_filter("pf_pka_conc_variable_k", args.particles, task_seed + 30_000),
            "corrected": build_filter("pf_pka_conc_variable_k", args.particles, task_seed + 30_000),
        }
        task_records = {}
        for name, inference in filters.items():
            maxima = {str(k): [] for k in (1, 2, 3)}
            for transition in transitions:
                inference.update(
                    task.initial_volume_ml,
                    transition.before_state,
                    transition.after_state,
                    transition.observed_before_ph,
                    transition.observed_after_ph,
                )
                for item in bank_state(inference):
                    maxima[str(item["pair_count"])].append(item["outside_particle_value_fraction"])
            task_records[name] = {
                "seed": int(task.seed),
                "task_id": int(task.task_id),
                "trajectory_steps": len(transitions),
                "maximum_outside_fraction_by_k": maxima,
                "final_banks": bank_state(inference),
            }
        original_records.append(task_records["original"])
        corrected_records.append(task_records["corrected"])

    payload = {
        "settings": {
            "seed": args.seed,
            "tasks": args.tasks,
            "particles": args.particles,
            "pka_support": [PRIOR_PKA_LOW, PRIOR_PKA_HIGH],
            "trajectory": "identical corrected pKa-only K=3 closed-loop trajectory replayed by both variable-K filters",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "original_snapshot": summarize(original_records),
        "corrected": summarize(corrected_records),
        "validation": {
            "original_escape_detected": summarize(original_records)["banks_outside_at_any_update"] > 0,
            "corrected_all_updates_within_support": summarize(corrected_records)["banks_outside_at_any_update"] == 0,
        },
        "per_task_original": original_records,
        "per_task_corrected": corrected_records,
    }
    write_json(args.output, payload)
    print(payload["validation"])
    print(payload["original_snapshot"])
    print(payload["corrected"])


if __name__ == "__main__":
    main()
