from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES = {
    # Minimal execution check only. Results from this profile are not used in
    # the manuscript or the archived publication evidence.
    "smoke": {
        "train_tasks": 80,
        "validation_tasks": 30,
        "test_tasks": 30,
        "min_train_states": 0,
        "min_validation_states": 0,
        "teacher_particles": 60,
        "imitation_seeds": [11],
        "imitation_epochs": 2,
        "imitation_validation_tasks": 20,
        "ppo_seeds": [303],
        "ppo_interactions": 1_000,
        "ppo_pool_size": 100,
        "ppo_validation_tasks": 30,
        "ppo_eval_interval": 500,
        "ppo_batch_steps": 256,
    },
    "standard": {
        "train_tasks": 5_000,
        "validation_tasks": 500,
        "test_tasks": 1_000,
        "min_train_states": 60_000,
        "min_validation_states": 12_000,
        "teacher_particles": 1_000,
        "imitation_seeds": [11, 22, 33],
        "imitation_epochs": 30,
        "imitation_validation_tasks": 500,
        "ppo_seeds": [101, 202, 303, 404, 555],
        "ppo_interactions": 100_000,
        "ppo_pool_size": 5_000,
        "ppo_validation_tasks": 500,
        "ppo_eval_interval": 10_000,
        "ppo_batch_steps": 2_048,
    },
    "full": {
        "train_tasks": 10_000,
        "validation_tasks": 1_000,
        "test_tasks": 2_000,
        "min_train_states": 120_000,
        "min_validation_states": 24_000,
        "teacher_particles": 1_000,
        "imitation_seeds": [11, 22, 33],
        "imitation_epochs": 50,
        "imitation_validation_tasks": 500,
        "ppo_seeds": [101, 202, 303, 404, 555],
        "ppo_interactions": 200_000,
        "ppo_pool_size": 10_000,
        "ppo_validation_tasks": 500,
        "ppo_eval_interval": 10_000,
        "ppo_batch_steps": 2_048,
    },
}


def run(command: list[str], cwd: Path) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PF-teacher, imitation, and PPO training only"
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    training = root / "training"
    profile = PROFILES[args.profile]
    output = args.output_dir.resolve()
    teacher = output / "teacher_data"
    imitation = output / "imitation"
    ppo = output / "ppo"
    output.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "generate_teacher_dataset.py",
            "--output-dir", str(teacher),
            "--train-tasks", str(profile["train_tasks"]),
            "--validation-tasks", str(profile["validation_tasks"]),
            "--test-tasks", str(profile["test_tasks"]),
            "--min-train-states", str(profile["min_train_states"]),
            "--min-validation-states", str(profile["min_validation_states"]),
            "--particles", str(profile["teacher_particles"]),
            "--workers", str(args.workers),
        ],
        training,
    )
    run(
        [
            sys.executable,
            "train_imitation.py",
            "--data-dir", str(teacher),
            "--output-dir", str(imitation),
            "--seeds", *map(str, profile["imitation_seeds"]),
            "--epochs", str(profile["imitation_epochs"]),
            "--closed-loop-validation-tasks", str(profile["imitation_validation_tasks"]),
            "--device", args.device,
        ],
        training,
    )
    run(
        [
            sys.executable,
            "train_ppo.py",
            "--imitation-checkpoint", str(imitation / "imitation_best.pth"),
            "--data-dir", str(teacher),
            "--output-dir", str(ppo),
            "--seeds", *map(str, profile["ppo_seeds"]),
            "--train-interactions", str(profile["ppo_interactions"]),
            "--training-pool-size", str(profile["ppo_pool_size"]),
            "--validation-tasks", str(profile["ppo_validation_tasks"]),
            "--eval-interval", str(profile["ppo_eval_interval"]),
            "--ppo-batch-steps", str(profile["ppo_batch_steps"]),
            "--device", args.device,
        ],
        training,
    )
    print(f"Training-only pipeline complete: {output}", flush=True)


if __name__ == "__main__":
    main()
