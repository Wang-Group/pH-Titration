from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import py_compile
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the pH-control source release.")
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    python_files = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "evidence" not in path.parts
    )
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    importlib.import_module("controllers")
    importlib.import_module("training.task_distribution")

    evidence_root = root / "evidence" / "simulation_numerical_evidence_20260823"
    required_files = [
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements-lock.txt",
        root / "controllers" / "__init__.py",
        root / "training" / "__init__.py",
        evidence_root / "README_CN.md",
        evidence_root / "00_INDEX_AND_PROTOCOLS" / "SIMULATION_STUDY_INDEX.csv",
        evidence_root / "00_INDEX_AND_PROTOCOLS" / "CURRENT_SIMULATION_CLAIMS.csv",
        evidence_root / "13_SOURCE_ARCHIVES" / "README_CN.md",
        evidence_root / "13_SOURCE_ARCHIVES" / "major_reviewer_evidence_source" / "multiseed_benchmark.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "major_reviewer_evidence_source" / "pid_tuning.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "historical_baseline_runner_20260817" / "baseline_controllers.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "historical_baseline_runner_20260817" / "run_missing_baselines.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "historical_baseline_runner_20260817" / "tune_pid.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "primary_locked_benchmark_source" / "task_distribution.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "primary_locked_benchmark_source" / "run_matched_evaluation.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "joint_parameter_bayesian_code_current" / "run_pf_multiseed_control.py",
        evidence_root / "13_SOURCE_ARCHIVES" / "historical_fixed3000_runner_source" / "benchmark_core.py",
        evidence_root / "02_TEACHER_AND_IMITATION" / "CHECKPOINT_PROVENANCE.json",
        evidence_root / "02_TEACHER_AND_IMITATION" / "datasets" / "train_teacher_dataset.npz",
        evidence_root / "02_TEACHER_AND_IMITATION" / "datasets" / "validation_teacher_dataset.npz",
        evidence_root / "02_TEACHER_AND_IMITATION" / "checkpoints" / "imitation_best.pth",
        evidence_root / "02_TEACHER_AND_IMITATION" / "robust_pf_imitation_ppo_training_source" / "train_imitation.py",
        evidence_root / "02_TEACHER_AND_IMITATION" / "robust_pf_imitation_ppo_training_source" / "train_ppo.py",
        evidence_root / "02_TEACHER_AND_IMITATION" / "robust_pf_imitation_ppo_training_source" / "task_distribution.py",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "CHECKPOINT_PROVENANCE.json",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "checkpoints" / "ppo_seed_303.pth",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "RUN_CONFIG.json",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "SOURCE_RUN_CONFIG.json",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "PPO_COMPLETE.json",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "common_locked_test_tasks.jsonl",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "evaluation" / "all_task_results.csv",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "evaluation" / "aggregate_summary.csv",
        evidence_root / "03_PPO_TRAINING_STABILITY" / "PPO_STABILITY_AUDIT.json",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "PRIMARY_BENCHMARK_SUMMARY.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "REGENERATED_PRIMARY_SUMMARY.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "PRIMARY_REPRODUCTION_AUDIT.json",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "PRIMARY_MANIFEST_REGENERATION_AUDIT.json",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "publication_tables" / "primary_controller_comparison.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "publication_tables" / "primary_controller_comparison.md",
        evidence_root / "05_RL_ALGORITHM_SCREEN" / "direction_assisted_release" / "results_full" / "aggregate_summary.csv",
        evidence_root / "06_POSTERIOR_RECOVERY" / "current_complete_study" / "aggregate_posterior_summary.csv",
        evidence_root / "08_SENSOR_STRESS" / "current_pf_noise_stress" / "aggregate_summary.csv",
        evidence_root / "10_PARTICLE_SCALING" / "particle_scaling_summary" / "summary.csv",
        evidence_root / "11_PYMC_COMPARISON" / "pymc_pf_comparison" / "pymc_pf_summary.csv",
        evidence_root / "12_ONLINE_TIMING" / "online_single_step_timing_20260820" / "FINAL_SINGLE_STEP_TIMING_SUMMARY.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "rule_baseline_replay" / "aggregate_summary.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "rule_baseline_replay" / "REPLAY_PROVENANCE.json",
        root / "scripts" / "replay_primary_rule_baselines.py",
        root / "scripts" / "audit_primary_benchmark.py",
        root / "scripts" / "verify_primary_manifest_regeneration.py",
        root / "scripts" / "generate_publication_tables.py",
        root / "scripts" / "sanitize_checkpoint_metadata.py",
        root / "scripts" / "audit_ppo_stability.py",
        root / "tests" / "test_release_contracts.py",
    ]
    formal_root = evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
    primary_task_root = formal_root / "tasks"
    primary_seeds = (101, 202, 303, 404, 555)
    for seed in primary_seeds:
        required_files.append(primary_task_root / f"seed_{seed}_tasks.jsonl")
        required_files.append(formal_root / f"seed_{seed}_task_results.csv")
        required_files.append(formal_root / "pf_reference" / f"seed_{seed}_task_results.csv")
    stability_root = evidence_root / "03_PPO_TRAINING_STABILITY"
    for seed in primary_seeds:
        required_files.append(stability_root / "checkpoints" / f"ppo_seed_{seed}.pth")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "COMPLETE.json")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "training_tasks.jsonl")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "validation_tasks.jsonl")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "locked_test_results.csv")
    missing = [str(path.relative_to(root)) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit("Missing evidence files:\n" + "\n".join(missing))

    index_path = evidence_root / "00_INDEX_AND_PROTOCOLS" / "SIMULATION_STUDY_INDEX.csv"
    if index_path.stat().st_size < 100:
        raise SystemExit("Simulation study index is unexpectedly empty.")

    # The locked primary benchmark is task-level data: 5 x 3,000 unique rows,
    # with one result row per method for each matched task.
    task_counts = {
        seed: count_jsonl(primary_task_root / f"seed_{seed}_tasks.jsonl")
        for seed in primary_seeds
    }
    if task_counts != {seed: 3000 for seed in primary_seeds}:
        raise SystemExit(f"Primary task-manifest count mismatch: {task_counts}")
    all_results = formal_root / "all_task_results.csv"
    if count_data_rows(all_results) != 45000:
        raise SystemExit("Primary all_task_results.csv must contain 45,000 data rows.")
    with all_results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(int(row["benchmark_seed"]), int(row["task_id"])) for row in rows}
    methods = {row["method"] for row in rows}
    if len(keys) != 15000 or methods != {"pf_teacher", "imitation", "ppo"}:
        raise SystemExit(
            f"Primary matching-result structure mismatch: {len(keys)} tasks, methods={sorted(methods)}"
        )
    for seed in primary_seeds:
        if count_data_rows(formal_root / f"seed_{seed}_task_results.csv") != 9000:
            raise SystemExit(f"Primary result count mismatch for seed {seed}")
        if count_data_rows(formal_root / "pf_reference" / f"seed_{seed}_task_results.csv") != 9000:
            raise SystemExit(f"Primary PF-reference count mismatch for seed {seed}")

    replay_root = formal_root / "rule_baseline_replay"
    replay_results = replay_root / "all_rule_baseline_results.csv"
    if not replay_results.is_file() or count_data_rows(replay_results) != 45000:
        raise SystemExit("Rule/PID replay must contain 45,000 task-level rows.")
    replay_methods = set()
    with replay_results.open("r", encoding="utf-8-sig", newline="") as handle:
        replay_methods = {row["method"] for row in csv.DictReader(handle)}
    if replay_methods != {"simple_rule", "prespecified_pid", "tuned_pid"}:
        raise SystemExit(f"Rule/PID replay methods mismatch: {sorted(replay_methods)}")
    baseline_summary = replay_root / "aggregate_summary.csv"
    with baseline_summary.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = {row["method"]: row for row in csv.DictReader(handle)}
    expected_baselines = {
        "simple_rule": (77.28, 16.111345970976874, 0.11061467247185386),
        "prespecified_pid": (84.59333333333333, 17.14398254509369, 0.22142239877687014),
        "tuned_pid": (92.43999999999998, 14.747293644978054, 0.15041957401413186),
    }
    for method, (success, steps, error) in expected_baselines.items():
        row = baseline_rows.get(method)
        if row is None:
            raise SystemExit(f"Missing recovered baseline summary: {method}")
        observed = (
            float(row["success_rate_percent_mean"]),
            float(row["successful_steps_mean_mean"]),
            float(row["final_abs_error_mean_mean"]),
        )
        if any(abs(a - b) > 1e-9 for a, b in zip(observed, (success, steps, error))):
            raise SystemExit(f"Recovered baseline summary mismatch for {method}: {observed}")
    audit_path = formal_root / "PRIMARY_REPRODUCTION_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("unique_tasks") != 15000:
        raise SystemExit("Primary benchmark audit report is invalid")
    manifest_audit = json.loads(
        (formal_root / "PRIMARY_MANIFEST_REGENERATION_AUDIT.json").read_text(encoding="utf-8")
    )
    if manifest_audit.get("status") != "PASS" or not all(
        item.get("match") for item in manifest_audit.get("checks", [])
    ):
        raise SystemExit("Primary manifest regeneration audit is invalid")

    provenance_path = evidence_root / "02_TEACHER_AND_IMITATION" / "CHECKPOINT_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    teacher_root = provenance_path.parent
    for item in provenance["teacher_datasets"]:
        path = teacher_root / item["file"]
        if sha256(path) != item["sha256"].lower():
            raise SystemExit(f"Teacher dataset hash mismatch: {path.name}")
    imitation = provenance["imitation_checkpoint"]
    imitation_path = teacher_root / imitation["file"]
    if sha256(imitation_path) != imitation["sha256"].lower():
        raise SystemExit("Imitation checkpoint hash mismatch")
    for item in provenance["principal_ppo_checkpoints"]:
        path = teacher_root / item["file"]
        if sha256(path) != item["sha256"].lower():
            raise SystemExit(f"Principal PPO checkpoint hash mismatch: seed {item['seed']}")

    checkpoint = root / "controllers" / "models" / "ppo_seed_303.pth"
    checkpoint_hash = sha256(checkpoint)
    expected_hash = "4004d7a09768fc5ac3f448523f53cb22210ed919ca7e713f13d9aa693cc19de5"
    if checkpoint_hash != expected_hash:
        raise SystemExit(f"Checkpoint hash mismatch: {checkpoint_hash}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validation = payload.get("metadata", {}).get("validation", {})
    if "strict_success_rate_percent" in validation:
        raise SystemExit("Released PPO metadata contains the internal strict-success field")
    protocol = payload.get("metadata", {}).get("protocol", {})
    if protocol.get("protocol_version") != "2026.08":
        raise SystemExit("Released PPO checkpoint is missing protocol version 2026.08")
    if protocol.get("protocol_profile") != "training_environment_strict":
        raise SystemExit("Released PPO checkpoint has the wrong training protocol profile")
    numpy_weights = root / "controllers" / "models" / "ppo_seed_303_numpy.npz"
    with np.load(numpy_weights, allow_pickle=False) as archive:
        numpy_metadata = json.loads(str(archive["metadata_json"].item()))
    if "strict_success_rate_percent" in numpy_metadata.get("validation", {}):
        raise SystemExit("Released NumPy PPO metadata contains the internal strict-success field")
    if numpy_metadata.get("protocol") != protocol:
        raise SystemExit("PyTorch and NumPy PPO protocol metadata differ")

    stability_audit = importlib.import_module("scripts.audit_ppo_stability")
    stability_audit.main()

    if not args.skip_self_test:
        module = importlib.import_module("controllers.controller_package_self_test")
        module.main()

    report = {
        "status": "pass",
        "compiled_python_files": len(python_files),
        "evidence_required_files": len(required_files),
        "checkpoint_sha256": checkpoint_hash,
        "evidence_root": str(evidence_root),
        "primary_task_counts": task_counts,
        "primary_unique_tasks": len(keys),
        "primary_result_rows": len(rows),
        "primary_methods": sorted(methods),
        "teacher_and_principal_checkpoints_verified": True,
        "ppo_stability_audit": "PASS",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
