from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
TRAINING_BLOCK = EVIDENCE / "02_TEACHER_AND_IMITATION"
DEFAULT_OUTPUT = EVIDENCE / "15_PPO_STEP_COST_TUNING"
CANDIDATES = (0.0, 0.0025, 0.005, 0.01)


def candidate_name(value: float) -> str:
    return f"step_cost_{value:g}".replace(".", "p")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_candidate(output_dir: Path, value: float, seed: int) -> dict:
    path = output_dir / candidate_name(value) / f"seed_{seed}" / "COMPLETE.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed candidate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload["best_validation"]
    return {
        "step_cost": value,
        "candidate": candidate_name(value),
        "training_seed": seed,
        "train_interactions": payload["train_interactions"],
        "best_checkpoint_source": payload["best_checkpoint_source"],
        "best_environment_steps": payload["best_environment_steps"],
        "validation_tasks": validation["tasks"],
        "validation_success_rate_percent": validation["success_rate_percent"],
        "validation_severe_failure_rate_percent": validation["severe_failure_rate_percent"],
        "validation_false_stop_rate_percent": validation["false_stop_rate_percent"],
        "validation_successful_steps_mean": validation["successful_steps_mean"],
        "validation_steps_mean": validation["steps_mean"],
        "validation_overshoots_mean": validation["overshoots_mean"],
        "validation_total_volume_mean_ml": validation["total_volume_mean_ml"],
        "validation_final_abs_error_mean": validation["final_abs_error_mean"],
        "actor_sha256": payload["actor_sha256"],
    }


def select_best(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            float(row["validation_success_rate_percent"]),
            -float(row["validation_final_abs_error_mean"]),
            -float(row["validation_steps_mean"]),
        ),
    )


def aggregate(output_dir: Path, seed: int) -> None:
    rows = [load_candidate(output_dir, value, seed) for value in CANDIDATES]
    best = select_best(rows)
    baseline = next(row for row in rows if math.isclose(float(row["step_cost"]), 0.005))
    for row in rows:
        row["success_difference_vs_0p005_pp"] = (
            float(row["validation_success_rate_percent"])
            - float(baseline["validation_success_rate_percent"])
        )
    write_csv(output_dir / "candidate_validation_summary.csv", rows)
    report = [
        "# PPO step-cost coefficient tuning",
        "",
        f"Candidates were retrained with training seed {seed} under the standard PPO protocol and selected using the 500-task validation set only.",
        "The locked benchmark test sets were not used for coefficient selection.",
        "",
        "| Step cost | Validation success (%) | Validation final error | Validation steps | Best checkpoint | Best step |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        report.append(
            f"| {float(row['step_cost']):.4f} | {float(row['validation_success_rate_percent']):.2f} | "
            f"{float(row['validation_final_abs_error_mean']):.4f} | "
            f"{float(row['validation_steps_mean']):.2f} | {row['best_checkpoint_source']} | "
            f"{row['best_environment_steps']} |"
        )
    report.extend([
        "",
        f"Selected coefficient by the prespecified lexicographic criterion (success rate, then final error, then steps): `{float(best['step_cost']):g}`.",
        "",
        "This is a one-training-seed coefficient-tuning study; the selected value should be confirmed in a multi-seed sensitivity analysis before being treated as a fully independent hyperparameter estimate.",
    ])
    (output_dir / "RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    config = {
        "study_id": "ppo_step_cost_tuning",
        "candidate_step_costs": list(CANDIDATES),
        "training_seed": seed,
        "selection_data": "500-task validation set",
        "locked_test_used_for_selection": False,
        "training_protocol": {
            "train_interactions_target": 100000,
            "training_pool_tasks": 5000,
            "validation_tasks": 500,
            "evaluation_interval": 10000,
            "ppo_batch_steps": 2048,
            "ppo_epochs": 4,
            "minibatch_size": 256,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.20,
            "entropy_coefficient": 0.005,
            "reward_variant": "full",
        },
        "selected_step_cost": float(best["step_cost"]),
        "imitation_checkpoint": (
            TRAINING_BLOCK / "checkpoints" / "imitation_best.pth"
        ).relative_to(ROOT).as_posix(),
    }
    (output_dir / "RUN_CONFIG.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune the PPO step-cost coefficient on validation tasks")
    parser.add_argument("--phase", choices=("aggregate",), default="aggregate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-seed", type=int, default=303)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(args.output_dir.resolve(), args.training_seed)


if __name__ == "__main__":
    main()
