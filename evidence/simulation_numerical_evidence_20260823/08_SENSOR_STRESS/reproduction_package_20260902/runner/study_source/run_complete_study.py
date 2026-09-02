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


PIPELINE_PROTOCOL_VERSION = 8


PROFILES = {
    "quick": {
        "ablation_seeds": [101],
        "ablation_tasks_per_seed": 50,
        "ablation_particles": 100,
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
        "posterior_tasks_per_seed": 10,
        "posterior_particles": 100,
        "posterior_checkpoints": [0, 1, 2, 3],
        "rl_audit_evaluation_seeds": [701],
        "rl_audit_tasks_per_seed": 20,
    },
    "standard": {
        "ablation_seeds": [101, 202, 303, 404, 555],
        "ablation_tasks_per_seed": 3000,
        "ablation_particles": 1000,
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
        "posterior_tasks_per_seed": 300,
        "posterior_particles": 1000,
        "posterior_checkpoints": [0, 1, 2, 3, 5, 8, 12],
        "rl_audit_evaluation_seeds": [701, 702, 703, 704, 705],
        "rl_audit_tasks_per_seed": 500,
    },
    "full": {
        "ablation_seeds": [101, 202, 303, 404, 555],
        "ablation_tasks_per_seed": 3000,
        "ablation_particles": 1000,
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
        "posterior_tasks_per_seed": 1000,
        "posterior_particles": 1000,
        "posterior_checkpoints": [0, 1, 2, 3, 5, 8, 12],
        "rl_audit_evaluation_seeds": [701, 702, 703, 704, 705],
        "rl_audit_tasks_per_seed": 1000,
    },
}


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access CUDA.")
    return requested


def validate_resume_config(existing: dict, requested: dict) -> None:
    ignored = {
        "started_unix_time",
        "last_resumed_unix_time",
        "resume_count",
        "pipeline_protocol_version",
        "pf_fit_distribution_protocol_version",
    }
    differences = []
    for key in sorted(set(existing) | set(requested)):
        if key in ignored:
            continue
        if existing.get(key) != requested.get(key):
            differences.append(
                f"{key}: existing={existing.get(key)!r}, requested={requested.get(key)!r}"
            )
    if differences:
        raise RuntimeError(
            "Cannot resume with a different run configuration. Choose a new "
            "--output-dir or restore the original arguments:\n  "
            + "\n  ".join(differences)
        )


def validate_teacher_selection(ablation_dir: Path, output_dir: Path) -> dict:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    with (ablation_dir / "aggregate_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    success = {row["policy"]: float(row["success_rate_percent_mean"]) for row in rows}
    if set(success) != {"hybrid_full", "hybrid_no_overshoot_cap", "posterior_direct"}:
        raise RuntimeError("Ablation output does not contain the three required policies")
    selected = max(success, key=success.get)
    if selected != "hybrid_full":
        raise RuntimeError(
            f"Ablation selected {selected!r}, but the neural teacher implementation is hybrid_full. "
            "Update the teacher implementation before training."
        )
    payload = {
        "selected_teacher_policy": selected,
        "selection_metric": "success_rate_percent_mean",
        "policy_success_rates_percent": success,
        "ablation_dir": str(ablation_dir),
    }
    (output_dir / "TEACHER_SELECTION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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
    parser.add_argument("--ablation-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument("--workers", type=int, default=0, help="Teacher processes; 0 selects automatically")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    package_dir = Path(__file__).resolve().parent
    output_dir = (args.output_dir or package_dir / "results" / f"{args.profile}_run").resolve()
    ablation_dir = (
        args.ablation_dir
        or package_dir / "results" / f"bayesian_rule_ablation_{args.profile}_v2"
    ).resolve()
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
            "teacher_controller_policy": "hybrid_full",
            "ablation_protocol_version": 2,
            "teacher_dataset_version": 6,
            "teacher_quality_control_version": 6,
            "action_count": 1000,
            "action_volume_range_ml": [0.01, 10.0],
            "sensor_resolution_ph": 0.01,
            "evaluation_protocol_version": 2,
            "posterior_diagnostic_protocol_version": 1,
            "pf_fit_distribution_protocol_version": 1,
            "rl_effectiveness_protocol_version": 1,
            "ph_solver_iterations": 60,
            "oracle_volume_bisection_iterations": 32,
            "control_volume_bisection_iterations": 32,
            "rl_algorithm": "PPO",
            "pipeline_protocol_version": PIPELINE_PROTOCOL_VERSION,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started_unix_time": time.time(),
        }
    )
    config_path = output_dir / "RUN_CONFIG.json"
    if args.resume and config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_config(existing_config, config)
        config["started_unix_time"] = existing_config.get(
            "started_unix_time", config["started_unix_time"]
        )
        config["last_resumed_unix_time"] = time.time()
        config["resume_count"] = int(existing_config.get("resume_count", 0)) + 1
    (config_path).write_text(json.dumps(config, indent=2), encoding="utf-8")

    data_dir = output_dir / "01_teacher_data"
    imitation_dir = output_dir / "02_imitation"
    ppo_dir = output_dir / "03_ppo"
    report_dir = output_dir / "04_evaluation"
    posterior_dir = output_dir / "05_posterior_diagnostics"
    two_network_dir = output_dir / "06_two_network_evaluation"
    rl_effectiveness_dir = output_dir / "07_rl_effectiveness"
    pf_fit_distribution_dir = output_dir / "08_pf_fit_distributions"
    resume_flag = ["--resume"] if args.resume else []

    run_stage(
        "PF external-rule ablation",
        [
            sys.executable,
            "bayesian_external_rule_ablation.py",
            "--output-dir", str(ablation_dir),
            "--seeds", *[str(seed) for seed in config["ablation_seeds"]],
            "--tasks-per-seed", str(config["ablation_tasks_per_seed"]),
            "--particles", str(config["ablation_particles"]),
            "--workers", str(args.workers),
            *resume_flag,
        ],
        ablation_dir / "ABLATION_COMPLETE.json",
        args.resume,
        log_path,
    )
    validate_teacher_selection(ablation_dir, output_dir)

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
    run_stage(
        "posterior curve and parameter diagnostics",
        [
            sys.executable,
            "posterior_diagnostics.py",
            "--output-dir", str(posterior_dir),
            "--seeds", *[str(seed) for seed in config["ppo_seeds"]],
            "--tasks-per-seed", str(config["posterior_tasks_per_seed"]),
            "--particles", str(config["posterior_particles"]),
            "--checkpoints", *[str(value) for value in config["posterior_checkpoints"]],
            "--workers", str(args.workers),
            *resume_flag,
        ],
        posterior_dir / "POSTERIOR_DIAGNOSTICS_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "selected imitation and PPO network evaluation",
        [
            sys.executable,
            "evaluate_two_networks.py",
            "--pipeline-dir", str(output_dir),
            "--output-dir", str(two_network_dir),
        ],
        two_network_dir / "TWO_NETWORK_EVALUATION_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "RL effectiveness audit under unseen interventions",
        [
            sys.executable,
            "evaluate_rl_effectiveness.py",
            "--pipeline-dir", str(output_dir),
            "--output-dir", str(rl_effectiveness_dir),
            "--device", device,
            "--evaluation-seeds",
            *[str(seed) for seed in config["rl_audit_evaluation_seeds"]],
            "--tasks-per-seed", str(config["rl_audit_tasks_per_seed"]),
        ],
        rl_effectiveness_dir / "RL_EFFECTIVENESS_COMPLETE.json",
        args.resume,
        log_path,
    )
    run_stage(
        "PF fit-error distribution analysis",
        [
            sys.executable,
            "analyze_pf_fit_distributions.py",
            "--posterior-dir", str(posterior_dir),
            "--output-dir", str(pf_fit_distribution_dir),
        ],
        pf_fit_distribution_dir / "PF_FIT_DISTRIBUTION_COMPLETE.json",
        args.resume,
        log_path,
    )
    completed = {
        "profile": args.profile,
        "device": device,
        "output_dir": str(output_dir),
        "result_summary": str(report_dir / "RESULT_SUMMARY.md"),
        "posterior_summary": str(posterior_dir / "POSTERIOR_DIAGNOSTIC_SUMMARY.md"),
        "two_network_summary": str(two_network_dir / "TWO_NETWORK_EVALUATION.md"),
        "rl_effectiveness_summary": str(
            rl_effectiveness_dir / "RL_EFFECTIVENESS_AUDIT.md"
        ),
        "pf_fit_distribution_summary": str(
            pf_fit_distribution_dir / "PF_FIT_DISTRIBUTION_ANALYSIS_CN.md"
        ),
        "completed_unix_time": time.time(),
    }
    (output_dir / "PIPELINE_COMPLETE.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"\nPipeline complete. Results: {report_dir}\n", flush=True)


if __name__ == "__main__":
    main()
