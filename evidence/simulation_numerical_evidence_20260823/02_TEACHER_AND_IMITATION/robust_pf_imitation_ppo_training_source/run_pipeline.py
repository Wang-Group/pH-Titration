from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROFILES = {
    "quick": {
        "teacher_train_tasks": 50,
        "teacher_validation_tasks": 20,
        "locked_test_tasks": 50,
        "minimum_train_states": 200,
        "minimum_validation_states": 80,
        "teacher_particles": 100,
        "imitation_seeds": [11],
        "imitation_epochs": 3,
        "imitation_batch_size": 128,
        "imitation_validation_tasks": 20,
        "imitation_patience": 3,
        "ppo_seeds": [101],
        "ppo_interactions": 1000,
        "ppo_training_pool": 100,
        "ppo_validation_tasks": 50,
        "ppo_eval_interval": 250,
        "ppo_batch_steps": 256,
        "ppo_epochs": 2,
        "ppo_minibatch": 64,
        "stress_tasks": 20,
    },
    "standard": {
        "teacher_train_tasks": 5000,
        "teacher_validation_tasks": 500,
        "locked_test_tasks": 1000,
        "minimum_train_states": 60000,
        "minimum_validation_states": 12000,
        "teacher_particles": 1000,
        "imitation_seeds": [11, 22, 33],
        "imitation_epochs": 30,
        "imitation_batch_size": 512,
        "imitation_validation_tasks": 500,
        "imitation_patience": 8,
        "ppo_seeds": [101, 202, 303, 404, 555],
        "ppo_interactions": 100000,
        "ppo_training_pool": 5000,
        "ppo_validation_tasks": 500,
        "ppo_eval_interval": 10000,
        "ppo_batch_steps": 2048,
        "ppo_epochs": 4,
        "ppo_minibatch": 256,
        "stress_tasks": 300,
    },
    "full": {
        "teacher_train_tasks": 15000,
        "teacher_validation_tasks": 1500,
        "locked_test_tasks": 3000,
        "minimum_train_states": 120000,
        "minimum_validation_states": 24000,
        "teacher_particles": 1000,
        "imitation_seeds": [11, 22, 33, 44, 55],
        "imitation_epochs": 50,
        "imitation_batch_size": 512,
        "imitation_validation_tasks": 1000,
        "imitation_patience": 10,
        "ppo_seeds": [101, 202, 303, 404, 555],
        "ppo_interactions": 200000,
        "ppo_training_pool": 15000,
        "ppo_validation_tasks": 1000,
        "ppo_eval_interval": 10000,
        "ppo_batch_steps": 4096,
        "ppo_epochs": 4,
        "ppo_minibatch": 512,
        "stress_tasks": 1000,
    },
}


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access CUDA.")
    return requested


def run_stage(name: str, command: list[str], marker: Path, resume: bool, log_path: Path) -> None:
    if resume and marker.exists():
        message = f"[{name}] already complete: {marker}"
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")
        return
    display = subprocess.list2cmdline(command)
    print(f"\n[{name}] {display}\n", flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTHONHASHSEED", "0")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{name}] {display}\n")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Stage {name!r} failed with exit code {code}. See {log_path}.")
    if not marker.exists():
        raise RuntimeError(f"Stage {name!r} exited successfully but did not create {marker}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click robust PF teacher -> imitation -> PPO pipeline")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument("--workers", type=int, default=0, help="Teacher processes; 0 selects automatically")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    package_dir = Path(__file__).resolve().parent
    output_dir = (args.output_dir or package_dir / "results" / f"{args.profile}_run").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --resume or choose another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    config = dict(PROFILES[args.profile])
    device = resolve_device(args.device)
    config.update(
        {
            "profile": args.profile,
            "device": device,
            "teacher_workers": args.workers,
            "teacher_variant": "pf_pka_conc_variable_k",
            "rl_algorithm": "PPO",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started_unix_time": time.time(),
        }
    )
    (output_dir / "RUN_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    data_dir = output_dir / "01_teacher_data"
    imitation_dir = output_dir / "02_imitation"
    ppo_dir = output_dir / "03_ppo"
    report_dir = output_dir / "04_evaluation"
    resume_flag = ["--resume"] if args.resume else []

    run_stage(
        "teacher data",
        [
            sys.executable,
            "generate_teacher_dataset.py",
            "--output-dir", str(data_dir),
            "--train-tasks", str(config["teacher_train_tasks"]),
            "--validation-tasks", str(config["teacher_validation_tasks"]),
            "--test-tasks", str(config["locked_test_tasks"]),
            "--min-train-states", str(config["minimum_train_states"]),
            "--min-validation-states", str(config["minimum_validation_states"]),
            "--particles", str(config["teacher_particles"]),
            "--workers", str(args.workers),
            *resume_flag,
        ],
        data_dir / "TEACHER_DATA_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "imitation learning",
        [
            sys.executable,
            "train_imitation.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(imitation_dir),
            "--seeds", *[str(seed) for seed in config["imitation_seeds"]],
            "--epochs", str(config["imitation_epochs"]),
            "--batch-size", str(config["imitation_batch_size"]),
            "--closed-loop-validation-tasks", str(config["imitation_validation_tasks"]),
            "--early-stopping-patience", str(config["imitation_patience"]),
            "--device", device,
            *resume_flag,
        ],
        imitation_dir / "IMITATION_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "PPO refinement",
        [
            sys.executable,
            "train_ppo.py",
            "--imitation-checkpoint", str(imitation_dir / "imitation_best.pth"),
            "--data-dir", str(data_dir),
            "--output-dir", str(ppo_dir),
            "--seeds", *[str(seed) for seed in config["ppo_seeds"]],
            "--train-interactions", str(config["ppo_interactions"]),
            "--training-pool-size", str(config["ppo_training_pool"]),
            "--validation-tasks", str(config["ppo_validation_tasks"]),
            "--eval-interval", str(config["ppo_eval_interval"]),
            "--ppo-batch-steps", str(config["ppo_batch_steps"]),
            "--ppo-epochs", str(config["ppo_epochs"]),
            "--minibatch-size", str(config["ppo_minibatch"]),
            "--device", device,
            *resume_flag,
        ],
        ppo_dir / "PPO_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "locked evaluation",
        [
            sys.executable,
            "evaluate_and_report.py",
            "--data-dir", str(data_dir),
            "--imitation-checkpoint", str(imitation_dir / "imitation_best.pth"),
            "--ppo-dir", str(ppo_dir),
            "--output-dir", str(report_dir),
            "--ppo-seeds", *[str(seed) for seed in config["ppo_seeds"]],
            "--stress-tasks", str(config["stress_tasks"]),
            "--teacher-particles", str(config["teacher_particles"]),
            "--device", device,
        ],
        report_dir / "EVALUATION_COMPLETE.json",
        args.resume,
        log_path,
    )
    completed = {
        "profile": args.profile,
        "device": device,
        "output_dir": str(output_dir),
        "result_summary": str(report_dir / "RESULT_SUMMARY.md"),
        "completed_unix_time": time.time(),
    }
    (output_dir / "PIPELINE_COMPLETE.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"\nPipeline complete. Results: {report_dir}\n", flush=True)


if __name__ == "__main__":
    main()
