from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent / "rl_direction_assisted_processed_20260807_release"
ZIP_PATH = ROOT.parent / "rl_direction_assisted_processed_20260807.zip"
ZIP_VALIDATION = ROOT.parent / "rl_direction_assisted_processed_20260807_ZIP_VALIDATION.json"
ZIP_HASH = ROOT.parent / "rl_direction_assisted_processed_20260807.zip.sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative: str, target_root: Path, source_root: Path = ROOT) -> None:
    source = source_root / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def remove_existing(path: Path) -> None:
    resolved = path.resolve()
    expected_parent = ROOT.parent.resolve()
    if resolved.parent != expected_parent:
        raise RuntimeError(f"Refusing to remove path outside the delivery parent: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def main() -> None:
    for path in (RELEASE, ZIP_PATH, ZIP_VALIDATION, ZIP_HASH):
        remove_existing(path)
    RELEASE.mkdir(parents=True)

    processed_files = [
        "benchmark_core.py",
        "check_environment.py",
        "direction_assisted_rl_comparison.py",
        "merge_shards.py",
        "validate_results.py",
        "analyze_results.py",
        "build_release.py",
        "run_parallel.ps1",
        "run_experiment.ps1",
        "RUN_FULL.cmd",
        "RUN_FULL_CUDA.cmd",
        "RUN_QUICK_TEST.cmd",
        "README_CN.md",
        "EXPERIMENT_PROTOCOL.md",
        "requirements.txt",
        "models/imitation.pth",
    ]
    current_hash_lines = [f"{sha256(ROOT / relative)}  {relative}" for relative in processed_files]
    (ROOT / "SOURCE_SHA256_CURRENT.txt").write_text("\n".join(current_hash_lines) + "\n", encoding="utf-8")
    processed_files.append("SOURCE_SHA256_CURRENT.txt")
    processed_files.append("SOURCE_SHA256_ORIGINAL.txt")
    for relative in processed_files:
        copy_file(relative, RELEASE / "processed_source")

    shutil.copytree(
        ROOT / "original_source_snapshot",
        RELEASE / "original_source_snapshot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    top_level = [
        "README_PROCESSED_CN.md",
        "ANALYSIS_CN_20260807.md",
        "REVIEWER_RESPONSE_EN_20260807.md",
        "BUGFIX_AND_RUN_LOG_20260807.md",
        "PACKAGE_CONTENTS.txt",
        "EXPERIMENT_PROTOCOL.md",
        "SOURCE_SHA256_ORIGINAL.txt",
        "SOURCE_SHA256_CURRENT.txt",
    ]
    for relative in top_level:
        copy_file(relative, RELEASE)

    shutil.copytree(ROOT / "results_full", RELEASE / "results_full")
    shutil.copytree(ROOT / "results_quick_corrected", RELEASE / "results_quick_corrected")
    shutil.copytree(ROOT / "results_quick_supplied_from_author", RELEASE / "supplied_quick_reference")

    logs = RELEASE / "logs"
    logs.mkdir(parents=True)
    for pattern in ("*.log", "*.err", "launch_manifest*.json"):
        for source in sorted((ROOT / "results_shards").glob(pattern)):
            shutil.copy2(source, logs / source.name)
    if (ROOT / "logs").exists():
        for source in sorted((ROOT / "logs").glob("*")):
            if source.is_file():
                shutil.copy2(source, logs / source.name)

    strict = json.loads((ROOT / "results_full" / "RESULT_VALIDATION_STRICT.json").read_text(encoding="utf-8"))
    internal = json.loads((ROOT / "results_full" / "RESULT_VALIDATION.json").read_text(encoding="utf-8"))
    quick = json.loads((ROOT / "results_quick_corrected" / "RESULT_VALIDATION_STRICT.json").read_text(encoding="utf-8"))
    original_hash_lines = (ROOT / "SOURCE_SHA256_ORIGINAL.txt").read_text(encoding="utf-8").splitlines()
    original_hashes = {
        parts[1]: parts[0]
        for line in original_hash_lines
        if len(parts := line.split(maxsplit=1)) == 2 and len(parts[0]) == 64
    }
    original_snapshot_errors = []
    for relative, expected_hash in original_hashes.items():
        path = ROOT / "original_source_snapshot" / Path(relative)
        if not path.exists() or sha256(path).upper() != expected_hash.upper():
            original_snapshot_errors.append(relative)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_source_directory": r"C:\Users\ZSY\Documents\xwechat_files\wxid_mkx0ewygqoen22_124f\msg\file\2026-08\rl_direction_assisted_comparison_20260807\rl_direction_assisted_comparison_20260807",
        "working_copy": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "protocol_id": strict.get("protocol_id"),
        "control_allocation": "Neural actor selects volume; external rule selects acid/base direction.",
        "source_snapshot_status": "PASS" if not original_snapshot_errors else "FAIL",
        "source_snapshot_errors": original_snapshot_errors,
        "full_validation_status": strict.get("status"),
        "quick_validation_status": quick.get("status"),
        "condition_count": strict.get("condition_count"),
        "task_row_count": strict.get("task_row_count"),
        "learning_curve_row_count": strict.get("learning_curve_row_count"),
        "model_count": strict.get("model_count"),
        "internal_validation_status": internal.get("status"),
        "code_changes": [
            "empty trajectory and batch guards",
            "protocol-complete resume fingerprint",
            "correct random actor seed metadata",
            "fixed PyTorch thread controls",
            "bounded plot uncertainty band",
            "parallel runner, deterministic merger, endpoint analysis, and strict validator",
        ],
    }
    (RELEASE / "PROVENANCE_20260807.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    required = [
        RELEASE / "ANALYSIS_CN_20260807.md",
        RELEASE / "REVIEWER_RESPONSE_EN_20260807.md",
        RELEASE / "processed_source" / "direction_assisted_rl_comparison.py",
        RELEASE / "original_source_snapshot" / "direction_assisted_rl_comparison.py",
        RELEASE / "results_full" / "task_results.csv",
        RELEASE / "results_full" / "aggregate_summary.csv",
        RELEASE / "results_full" / "paired_initialization_tests.csv",
        RELEASE / "results_full" / "paired_algorithm_tests.csv",
        RELEASE / "results_full" / "learning_endpoint_summary.csv",
        RELEASE / "results_full" / "PERFORMANCE_HIGHLIGHTS.md",
        RELEASE / "results_full" / "RESULT_VALIDATION_STRICT.json",
    ]
    error_logs = [path.name for path in logs.glob("*.err") if path.stat().st_size > 0]
    release_errors = [str(path.relative_to(RELEASE)) for path in required if not path.exists()]
    if provenance["source_snapshot_status"] != "PASS":
        release_errors.append("original source snapshot hash mismatch")
    if strict.get("status") != "PASS" or internal.get("status") != "PASS" or quick.get("status") != "PASS":
        release_errors.append("one or more result validations failed")
    if strict.get("condition_count") != 30 or strict.get("task_row_count") != 30_000 or strict.get("model_count") != 30:
        release_errors.append("full result counts do not match the protocol")
    if error_logs:
        release_errors.append(f"nonempty shard error logs: {error_logs}")
    release_validation = {
        "status": "PASS" if not release_errors else "FAIL",
        "errors": release_errors,
        "condition_count": strict.get("condition_count"),
        "task_row_count": strict.get("task_row_count"),
        "learning_curve_row_count": strict.get("learning_curve_row_count"),
        "model_count": strict.get("model_count"),
        "log_file_count": len(list(logs.glob("*"))),
        "nonempty_error_logs": error_logs,
    }
    (RELEASE / "RELEASE_VALIDATION.json").write_text(json.dumps(release_validation, indent=2), encoding="utf-8")
    if release_errors:
        raise RuntimeError(f"Release validation failed: {release_errors}")

    manifest_lines = []
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            manifest_lines.append(f"{sha256(path)}  {path.relative_to(RELEASE).as_posix()}")
    (RELEASE / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(RELEASE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RELEASE).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.testzip()
        zip_members = sorted(name for name in archive.namelist() if not name.endswith("/"))
    release_members = sorted(path.relative_to(RELEASE).as_posix() for path in RELEASE.rglob("*") if path.is_file())
    zip_validation = {
        "status": "PASS" if zip_members == release_members else "FAIL",
        "zip": str(ZIP_PATH),
        "sha256": sha256(ZIP_PATH),
        "member_count": len(zip_members),
        "size_bytes": ZIP_PATH.stat().st_size,
        "missing_members": sorted(set(release_members) - set(zip_members)),
        "unexpected_members": sorted(set(zip_members) - set(release_members)),
    }
    ZIP_VALIDATION.write_text(json.dumps(zip_validation, indent=2), encoding="utf-8")
    ZIP_HASH.write_text(f"{zip_validation['sha256']}  {ZIP_PATH.name}\n", encoding="utf-8")
    if zip_validation["status"] != "PASS":
        raise RuntimeError("ZIP member validation failed")
    print(json.dumps(zip_validation, indent=2))


if __name__ == "__main__":
    main()
