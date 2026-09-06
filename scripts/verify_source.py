from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PRIVATE_LOCAL_DIRECTORIES = {"ph4github_analysiscopy"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".toml",
    ".yaml", ".yml", ".xml", ".svg", ".sh", ".cmd", ".bat", ".ini", ".cfg",
}


def hash_byte_candidates(path: Path) -> tuple[bytes, ...]:
    """Allow only LF/CRLF conversion in UTF-8 text, in either direction.

    Supplied manifests may hash CRLF files while GitHub ZIPs contain LF blobs.
    Conversely, Windows may check out CRLF for an LF manifest. Binary files,
    whitespace, BOMs, final-newline presence and bare CR characters are not
    normalized away.
    """
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES or b"\x00" in data:
        return (data,)
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return (data,)
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return tuple(dict.fromkeys((data, lf, crlf)))


def validate_public_paths(paths) -> None:
    for path in paths:
        top = str(path).replace("\\", "/").split("/", 1)[0].casefold()
        if top in PRIVATE_LOCAL_DIRECTORIES:
            raise ValueError(f"Internal working-copy material must not be published: {path}")


def verify_public_layout(root: Path) -> None:
    if (root / ".git").exists():
        # Ignored local drafts may remain: only Git's index is public input.
        result = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                                check=True, capture_output=True)
        paths = result.stdout.decode("utf-8").split("\0")
    else:
        # Source ZIPs have no index: each extracted top-level item is public.
        paths = [path.name for path in root.iterdir()]
    validate_public_paths(paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_manifest_entry(path: Path, row: dict[str, str]) -> bool:
    """Match digest and byte count in either permitted text newline style."""
    expected_size = int(row["bytes"])
    expected_hash = row["sha256"].lower()
    return any(
        len(candidate) == expected_size
        and hashlib.sha256(candidate).hexdigest() == expected_hash
        for candidate in hash_byte_candidates(path)
    )


def matches_sha256_allowing_crlf(path: Path, expected_hash: str) -> bool:
    """Match original bytes, allowing only LF/CRLF text conversion both ways."""
    return any(hashlib.sha256(candidate).hexdigest() == expected_hash.lower()
               for candidate in hash_byte_candidates(path))


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
    verify_public_layout(root)
    from scripts.fetch_lfs_assets import find_lfs_pointers
    missing_lfs = find_lfs_pointers(root)
    if missing_lfs:
        raise SystemExit(
            f"Git LFS assets are still pointers ({len(missing_lfs)} files). "
            "For a source ZIP, run python scripts/fetch_lfs_assets.py "
            "(--ref COMMIT for an older archive); for a Git clone, run git lfs pull. "
            "Then rerun this verifier. No result hashes were bypassed."
        )

    python_files = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "evidence" not in path.parts
        and path.relative_to(root).parts[0].casefold() not in PRIVATE_LOCAL_DIRECTORIES
    )
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    importlib.import_module("controllers")
    importlib.import_module("training.task_distribution")

    evidence_root = root / "evidence" / "simulation_numerical_evidence_20260823"
    sensor_repro_root = (
        evidence_root
        / "08_SENSOR_STRESS"
        / "reproduction_package_20260902"
    )
    sensor_repro_results = sensor_repro_root / "results" / "fixed_pf_ppo_stress"
    matched_timing_root = evidence_root / "16_MATCHED_TIMING_RECOVERY_100TASKS"
    matched_timing_results = matched_timing_root / "results"
    pf_closed_loop_root = evidence_root / "17_PF_CLOSED_LOOP_TIMING_100TASKS"
    pf_closed_loop_results = pf_closed_loop_root / "results"
    required_files = [
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements-lock.txt",
        root / "controllers" / "__init__.py",
        root / "training" / "__init__.py",
        evidence_root / "README_CN.md",
        evidence_root / "FILE_MANIFEST_SHA256.csv",
        evidence_root / "PACKAGE_VALIDATION.json",
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
        sensor_repro_root / "README.md",
        sensor_repro_root / "source_archive_SHA256SUMS.txt",
        sensor_repro_root / "runner" / "fixed_controller_stress_benchmark.py",
        sensor_repro_root / "runner" / "RUN_FIXED_CONTROLLER_STRESS.cmd",
        sensor_repro_root / "runner" / "study_source" / "task_distribution.py",
        sensor_repro_root / "runner" / "controllers_release" / "new_pf_controller.py",
        sensor_repro_root / "runner" / "controllers_release" / "new_rl_numpy_controller.py",
        sensor_repro_root / "runner" / "controllers_release" / "models" / "ppo_seed_303.pth",
        sensor_repro_root / "runner" / "controllers_release" / "models" / "ppo_seed_303_numpy.npz",
        sensor_repro_results / "all_task_results.csv",
        sensor_repro_results / "aggregate_summary.csv",
        sensor_repro_results / "per_seed_summary.csv",
        sensor_repro_results / "paired_success_tests.csv",
        sensor_repro_results / "paired_continuous_tests.csv",
        sensor_repro_results / "RUN_CONFIG.json",
        sensor_repro_results / "BENCHMARK_COMPLETE.json",
        evidence_root / "10_PARTICLE_SCALING" / "particle_scaling_summary" / "summary.csv",
        evidence_root / "11_PYMC_COMPARISON" / "pymc_pf_comparison" / "pymc_pf_summary.csv",
        evidence_root / "12_ONLINE_TIMING" / "online_single_step_timing_20260820" / "FINAL_SINGLE_STEP_TIMING_SUMMARY.csv",
        matched_timing_root / "PROTOCOL_AND_RESULTS.md",
        matched_timing_results / "CONTROLLED_RESULT_SUMMARY.csv",
        matched_timing_results / "CONTROLLED_RUN_CONFIG.json",
        matched_timing_results / "MATCHED_RUN_CONFIG.json",
        matched_timing_results / "POSTERIOR_RECOVERY_SUMMARY.csv",
        matched_timing_results / "POSTERIOR_RECOVERY_TASK_RESULTS.csv",
        matched_timing_results / "RESULTS_MATCHED.md",
        pf_closed_loop_root / "PROTOCOL_AND_RESULTS.md",
        pf_closed_loop_results / "PF_CLOSED_LOOP_OUTCOME_SUMMARY.csv",
        pf_closed_loop_results / "PF_CLOSED_LOOP_TIMING_SUMMARY.csv",
        pf_closed_loop_results / "PUBLICATION_TIMING_SCOPE_SUMMARY.csv",
        pf_closed_loop_results / "RELEASE_VALIDATION.json",
        evidence_root / "15_PPO_STEP_COST_TUNING" / "README.md",
        evidence_root / "15_PPO_STEP_COST_TUNING" / "RUN_CONFIG.json",
        evidence_root / "15_PPO_STEP_COST_TUNING" / "candidate_validation_summary.csv",
        evidence_root / "15_PPO_STEP_COST_TUNING" / "evaluation_full_5x3000" / "RESULT_SUMMARY.md",
        evidence_root / "15_PPO_STEP_COST_TUNING" / "evaluation_full_5x3000" / "benchmark_seed_mean_sd_summary.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "README.md",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "MANIFEST_SHA256.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "SOURCE_SHA256.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "CHECKPOINT_SHA256.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "table_s14_posterior_to_control.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "table_s15_pf_representation.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "table_s16_policy_families.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "family_method_summary.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "per_policy_summary.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "per_evaluation_cell_summary.csv",
        evidence_root / "18_CONTROLLER_REPRESENTATION_FACTORIAL" / "results" / "RUN_COMPLETE.json",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "rule_baseline_replay" / "aggregate_summary.csv",
        evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "rule_baseline_replay" / "REPLAY_PROVENANCE.json",
        root / "scripts" / "replay_primary_rule_baselines.py",
        root / "scripts" / "audit_primary_benchmark.py",
        root / "scripts" / "verify_primary_manifest_regeneration.py",
        root / "scripts" / "generate_publication_tables.py",
        root / "scripts" / "sanitize_checkpoint_metadata.py",
        root / "scripts" / "audit_ppo_stability.py",
        root / "scripts" / "audit_controller_representation_factorial.py",
        root / "scripts" / "run_ppo_step_cost_tuning.py",
        root / "scripts" / "evaluate_ppo_step_cost_tuning_full_benchmark.py",
        root / "scripts" / "benchmark_controlled_observation_to_action_100tasks.py",
        root / "scripts" / "run_controlled_timing_100tasks.py",
        root / "scripts" / "finalize_matched_timing_recovery_100tasks.py",
        root / "scripts" / "benchmark_pf_first_n_step_full_stats.py",
        root / "scripts" / "finalize_pf_closed_loop_timing_100tasks.py",
        root / "tests" / "test_release_contracts.py",
    ]
    formal_root = evidence_root / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
    primary_task_root = formal_root / "tasks"
    primary_seeds = (101, 202, 303, 404, 555)
    for seed in primary_seeds:
        required_files.append(primary_task_root / f"seed_{seed}_tasks.jsonl")
        required_files.append(formal_root / f"seed_{seed}_task_results.csv")
        required_files.append(formal_root / "pf_reference" / f"seed_{seed}_task_results.csv")
    sensor_regimes = (
        "nominal",
        "close_pka",
        "wide_concentration",
        "observation_noise_0p01",
        "observation_noise_0p03",
        "observation_noise_0p05",
        "observation_noise_0p10",
        "episode_bias_0p10",
        "random_walk_drift_0p01",
        "response_fraction_0p60",
        "response_fraction_0p70",
        "actuator_log_sd_0p10",
        "titrant_scale_0p90",
        "titrant_scale_1p10",
        "combined_unseen",
    )
    for regime in sensor_regimes:
        for seed in primary_seeds:
            required_files.append(sensor_repro_results / f"{regime}_seed_{seed}_tasks.jsonl")
            required_files.append(
                sensor_repro_results / "completed_shards" / f"{regime}_seed_{seed}.csv"
            )
    stability_root = evidence_root / "03_PPO_TRAINING_STABILITY"
    for seed in primary_seeds:
        required_files.append(stability_root / "checkpoints" / f"ppo_seed_{seed}.pth")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "COMPLETE.json")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "training_tasks.jsonl")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "validation_tasks.jsonl")
        required_files.append(stability_root / "runs" / f"seed_{seed}" / "locked_test_results.csv")
    step_cost_root = evidence_root / "15_PPO_STEP_COST_TUNING"
    step_cost_labels = ("step_cost_0", "step_cost_0p0025", "step_cost_0p005", "step_cost_0p01")
    for label in step_cost_labels:
        required_files.append(step_cost_root / label / "seed_303" / "best_ppo.pth")
        required_files.append(step_cost_root / label / "seed_303" / "COMPLETE.json")
        required_files.append(step_cost_root / label / "seed_303" / "training_tasks.jsonl")
        required_files.append(step_cost_root / label / "seed_303" / "validation_tasks.jsonl")
    for label in ("original_full", *step_cost_labels):
        for seed in primary_seeds:
            required_files.append(
                step_cost_root
                / "evaluation_full_5x3000"
                / "tasks"
                / label
                / f"benchmark_seed_{seed}_task_results.csv"
            )
    for method in ("imitation", "ppo", "pf_1000", "pf_10000", "pf_100000", "pymc"):
        required_files.append(matched_timing_results / method / "raw.csv")
        required_files.append(matched_timing_results / method / "summary.csv")
        required_files.append(matched_timing_results / method / "RUN_CONFIG.json")
    for method in ("pf_1000", "pf_10000", "pf_100000"):
        for name in (
            "task_results.csv",
            "per_step_timing.csv",
            "trajectories.jsonl",
            "closed_loop_summary.csv",
            "timing_first_n_summary.csv",
        ):
            required_files.append(pf_closed_loop_results / method / name)
    missing = [str(path.relative_to(root)) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit("Missing evidence files:\n" + "\n".join(missing))

    sensor_task_files = sorted(sensor_repro_results.glob("*_tasks.jsonl"))
    sensor_shard_files = sorted((sensor_repro_results / "completed_shards").glob("*.csv"))
    if len(sensor_task_files) != 75 or len(sensor_shard_files) != 75:
        raise SystemExit("Sensor-stress reproduction package has an incomplete task/shard set")
    if any(count_jsonl(path) != 1000 for path in sensor_task_files):
        raise SystemExit("Sensor-stress task manifest row count mismatch")
    if any(count_data_rows(path) != 2000 for path in sensor_shard_files):
        raise SystemExit("Sensor-stress result shard row count mismatch")
    if count_data_rows(sensor_repro_results / "all_task_results.csv") != 150000:
        raise SystemExit("Sensor-stress combined task-result row count mismatch")

    sensor_manifest = {}
    with (sensor_repro_root / "source_archive_SHA256SUMS.txt").open(
        "r", encoding="utf-8-sig"
    ) as handle:
        for line in handle:
            if line.strip():
                digest, archive_path = line.rstrip("\r\n").split("  ", 1)
                sensor_manifest[archive_path] = digest.lower()

    def sensor_manifest_key(path: Path) -> str | None:
        relative = path.relative_to(sensor_repro_root).as_posix()
        if relative.startswith("controller_source/"):
            return relative.removeprefix("controller_source/")
        if relative.startswith("runner/controllers_release/"):
            return relative.removeprefix("runner/")
        if relative.startswith("runner/study_source/"):
            return relative.removeprefix("runner/")
        if relative.startswith("runner/"):
            return relative.removeprefix("runner/")
        if relative.startswith("results/fixed_pf_ppo_stress/"):
            return "formal_results/" + relative.removeprefix("results/")
        return None

    sensor_hashed_files = 0
    for path in sensor_repro_root.rglob("*"):
        if not path.is_file() or path.name in {"README.md", "source_archive_SHA256SUMS.txt"}:
            continue
        manifest_key = sensor_manifest_key(path)
        expected_hash = sensor_manifest.get(manifest_key or "")
        if expected_hash is None:
            raise SystemExit(f"Sensor-stress file is absent from the source manifest: {path}")
        if not matches_sha256_allowing_crlf(path, expected_hash):
            raise SystemExit(f"Sensor-stress source/result hash mismatch: {path}")
        sensor_hashed_files += 1
    if sensor_hashed_files != 246:
        raise SystemExit(
            f"Sensor-stress source/result manifest coverage mismatch: {sensor_hashed_files}"
        )

    index_path = evidence_root / "00_INDEX_AND_PROTOCOLS" / "SIMULATION_STUDY_INDEX.csv"
    if index_path.stat().st_size < 100:
        raise SystemExit("Simulation study index is unexpectedly empty.")

    # Block 16 retains the matched single-step timing calls and the paired
    # one-observation PF/PyMC recovery analysis. Current full-trajectory PF
    # timing is validated separately below from block 17.
    expected_timing_ms = {
        "imitation": 0.15495,
        "ppo": 0.15390,
        "pf_1000": 22.99615,
        "pf_10000": 101.45055,
        "pf_100000": 900.93545,
        "pymc": 14407.37565,
    }
    with (matched_timing_results / "CONTROLLED_RESULT_SUMMARY.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        timing_rows = {row["method"]: row for row in csv.DictReader(handle)}
    if set(timing_rows) != set(expected_timing_ms):
        raise SystemExit("Matched timing summary contains an unexpected method set")
    for method, expected in expected_timing_ms.items():
        observed = float(timing_rows[method]["primary_median_of_task_median_wall_ms"])
        if abs(observed - expected) > 1e-9:
            raise SystemExit(f"Matched timing median mismatch for {method}: {observed}")
        if count_data_rows(matched_timing_results / method / "raw.csv") != 100:
            raise SystemExit(f"Matched timing raw row count mismatch for {method}")

    matched_config = json.loads(
        (matched_timing_results / "MATCHED_RUN_CONFIG.json").read_text(encoding="utf-8")
    )
    if (
        matched_config.get("status") != "completed_and_audited"
        or matched_config.get("same_task_and_input_audit") is not True
        or matched_config.get("tasks") != 100
    ):
        raise SystemExit("Matched timing task/input audit is invalid")
    pymc_config = matched_config.get("worker_configs", {}).get("pymc", {}).get("pymc", {})
    if pymc_config != {"draws_per_k": 300, "chains": 1, "k_values": [1, 2, 3]}:
        raise SystemExit(f"Matched timing PyMC configuration mismatch: {pymc_config}")

    package_validation = json.loads(
        (evidence_root / "PACKAGE_VALIDATION.json").read_text(encoding="utf-8")
    )
    expected_validation = {
        "status": "PASS",
        "same_task_and_input_audit": True,
        "task_cases_per_method": 100,
        "methods": 6,
        "pymc_draws_per_k": 300,
        "pymc_chains": 1,
    }
    if package_validation.get("matched_timing_recovery_100tasks") != expected_validation:
        raise SystemExit("Package validation does not describe the matched timing block")

    expected_pf_closed_loop_validation = {
        "status": "PASS",
        "task_cases_per_particle_count": 100,
        "step_measurements": {
            "pf_1000": 592,
            "pf_10000": 589,
            "pf_100000": 577,
        },
        "success_rate_percent": {
            "pf_1000": 97.0,
            "pf_10000": 97.0,
            "pf_100000": 97.0,
        },
    }

    expected_controller_representation_validation = {
        "status": "PASS",
        "locked_manifests": 15,
        "tasks_per_manifest": 3000,
        "publication_table_rows": {"S14": 3, "S15": 3, "S16": 15},
        "policy_evaluation_cells": 450,
        "policy_task_outcomes": 1_350_000,
    }
    if (
        package_validation.get("controller_representation_factorial")
        != expected_controller_representation_validation
    ):
        raise SystemExit(
            "Package validation does not describe the controller/representation factorial"
        )
    released_pf_validation = json.loads(
        (pf_closed_loop_results / "RELEASE_VALIDATION.json").read_text(encoding="utf-8")
    )
    for key, expected in expected_pf_closed_loop_validation.items():
        if released_pf_validation.get(key) != expected:
            raise SystemExit(f"PF closed-loop validation mismatch for {key}")
    if released_pf_validation.get("publication_timing_scope_summary_regenerated") is not True:
        raise SystemExit("PF closed-loop publication timing audit is missing")
    if package_validation.get("pf_closed_loop_timing_100tasks") != {
        "status": "PASS",
        "task_cases_per_particle_count": 100,
        "step_measurements": {"pf_1000": 592, "pf_10000": 589, "pf_100000": 577},
        "success_rate_percent": {"pf_1000": 97.0, "pf_10000": 97.0, "pf_100000": 97.0},
    }:
        raise SystemExit("Package validation does not describe the PF closed-loop timing block")

    expected_pf_closed_loop = {
        "pf_1000": (592, 40.1311, 97.0, 0.06000232336082756),
        "pf_10000": (589, 93.0455, 97.0, 0.0640039490755104),
        "pf_100000": (577, 594.127, 97.0, 0.05953077002847626),
    }
    with (pf_closed_loop_results / "PF_CLOSED_LOOP_TIMING_SUMMARY.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        pf_timing_rows = {row["method"]: row for row in csv.DictReader(handle)}
    with (pf_closed_loop_results / "PF_CLOSED_LOOP_OUTCOME_SUMMARY.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        pf_outcome_rows = {row["method"]: row for row in csv.DictReader(handle)}
    if set(pf_timing_rows) != set(expected_pf_closed_loop) or set(pf_outcome_rows) != set(
        expected_pf_closed_loop
    ):
        raise SystemExit("PF closed-loop summary method set mismatch")
    for method, (n_steps, median_ms, success, mean_error) in expected_pf_closed_loop.items():
        timing = pf_timing_rows[method]
        outcome = pf_outcome_rows[method]
        observed = (
            int(timing["n_step_measurements"]),
            float(timing["total_decision_median_ms"]),
            float(outcome["success_rate_percent"]),
            float(outcome["final_abs_error_ph_mean"]),
        )
        expected = (n_steps, median_ms, success, mean_error)
        if observed[0] != expected[0] or any(
            abs(value - target) > 1e-12 for value, target in zip(observed[1:], expected[1:])
        ):
            raise SystemExit(f"PF closed-loop summary mismatch for {method}: {observed}")

    with (evidence_root / "FILE_MANIFEST_SHA256.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        manifest_rows = {row["path"]: row for row in csv.DictReader(handle)}
    if package_validation.get("manifest_entries") != len(manifest_rows):
        raise SystemExit("Evidence manifest count does not match package validation")
    for protocol_root in (matched_timing_root, pf_closed_loop_root):
        for path in sorted(protocol_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(evidence_root).as_posix()
            row = manifest_rows.get(relative)
            if row is None:
                raise SystemExit(f"Timing file is absent from evidence manifest: {relative}")
            if not matches_manifest_entry(path, row):
                raise SystemExit(f"Timing manifest mismatch: {relative}")

    expected_recovery = {
        "PF": (33.0, 45.197764468185, 0.6738121046053155, 3.179720456744687),
        "PyMC": (29.0, 47.48075227477676, 0.6342257084598905, 3.075972471302178),
    }
    with (matched_timing_results / "POSTERIOR_RECOVERY_SUMMARY.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        recovery_rows = {row["method"]: row for row in csv.DictReader(handle)}
    if set(recovery_rows) != set(expected_recovery):
        raise SystemExit("Matched posterior-recovery summary contains an unexpected method")
    for method, expected in expected_recovery.items():
        row = recovery_rows[method]
        observed = (
            float(row["model_order_accuracy_percent"]),
            float(row["concentration_relative_error_median_percent"]),
            float(row["pka_matched_mae_median"]),
            float(row["full_curve_rmse_0_33ml_median_ph"]),
        )
        if any(abs(a - b) > 1e-9 for a, b in zip(observed, expected)):
            raise SystemExit(f"Matched posterior-recovery mismatch for {method}: {observed}")
    if count_data_rows(matched_timing_results / "POSTERIOR_RECOVERY_TASK_RESULTS.csv") != 200:
        raise SystemExit("Matched posterior-recovery task table must contain 200 rows")

    # The local PPO step-cost screen reports 0 through 0.01. The independently
    # retrained 0.005 checkpoint remains distinct from the original selected
    # 0.005 checkpoint so stochastic training variation is visible.
    step_cost_config = json.loads((step_cost_root / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    if step_cost_config.get("candidate_step_costs") != [0.0, 0.0025, 0.005, 0.01]:
        raise SystemExit("PPO step-cost screen must report exactly 0 through 0.01")
    expected_step_cost_success = {
        "original_full": (93.94666666666667, 0.6291970191354129),
        "step_cost_0": (89.17333333333333, 0.5106640556939307),
        "step_cost_0p0025": (88.66, 0.43614472623456374),
        "step_cost_0p005": (91.86, 0.5614465444031349),
        "step_cost_0p01": (89.17333333333333, 0.5106640556939307),
    }
    summary_path = step_cost_root / "evaluation_full_5x3000" / "benchmark_seed_mean_sd_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        step_cost_summary = {row["label"]: row for row in csv.DictReader(handle)}
    if set(step_cost_summary) != set(expected_step_cost_success):
        raise SystemExit("PPO step-cost summary contains an unexpected network")
    for label, expected in expected_step_cost_success.items():
        row = step_cost_summary[label]
        observed = (
            float(row["success_rate_percent_mean_across_benchmark_seeds"]),
            float(row["success_rate_percent_sd_across_benchmark_seeds"]),
        )
        if any(abs(a - b) > 1e-9 for a, b in zip(observed, expected)):
            raise SystemExit(f"PPO step-cost summary mismatch for {label}: {observed}")
    for label in step_cost_labels:
        if count_jsonl(step_cost_root / label / "seed_303" / "training_tasks.jsonl") != 5000:
            raise SystemExit(f"PPO step-cost training manifest mismatch: {label}")
        if count_jsonl(step_cost_root / label / "seed_303" / "validation_tasks.jsonl") != 500:
            raise SystemExit(f"PPO step-cost validation manifest mismatch: {label}")
    for label in expected_step_cost_success:
        for seed in primary_seeds:
            path = step_cost_root / "evaluation_full_5x3000" / "tasks" / label / f"benchmark_seed_{seed}_task_results.csv"
            if count_data_rows(path) != 3000:
                raise SystemExit(f"PPO step-cost locked result count mismatch: {label}, seed {seed}")

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

    controller_representation_audit = importlib.import_module(
        "scripts.audit_controller_representation_factorial"
    )
    controller_representation_audit.main()

    internal_rule_audit = importlib.import_module('scripts.audit_pf_internal_rule_ablation')
    internal_rule_report = internal_rule_audit.audit()

    primary_ppo_audit = importlib.import_module('scripts.audit_primary_ppo_five_seeds')
    primary_ppo_report = primary_ppo_audit.audit()

    local_curve_audit = importlib.import_module('scripts.audit_pf_local_curves')
    local_curve_report = local_curve_audit.audit()

    if not args.skip_self_test:
        module = importlib.import_module("controllers.controller_package_self_test")
        module.main()

    report = {
        "status": "pass",
        "compiled_python_files": len(python_files),
        "public_layout_audit": "PASS",
        "evidence_required_files": len(required_files),
        "checkpoint_sha256": checkpoint_hash,
        "evidence_root": str(evidence_root),
        "primary_task_counts": task_counts,
        "primary_unique_tasks": len(keys),
        "primary_result_rows": len(rows),
        "primary_methods": sorted(methods),
        "teacher_and_principal_checkpoints_verified": True,
        "ppo_stability_audit": "PASS",
        "controller_representation_factorial_audit": "PASS",
        "pf_internal_rule_ablation_audit": internal_rule_report['status'],
        "pf_internal_rule_ablation_task_results": internal_rule_report['task_results'],
        "primary_ppo_five_seed_reevaluation": primary_ppo_report,
        "pf_local_response": local_curve_report,
        "sensor_stress_reproduction": {
            "task_manifests": len(sensor_task_files),
            "result_shards": len(sensor_shard_files),
            "task_results": 150000,
            "manifest_files_verified": sensor_hashed_files,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
