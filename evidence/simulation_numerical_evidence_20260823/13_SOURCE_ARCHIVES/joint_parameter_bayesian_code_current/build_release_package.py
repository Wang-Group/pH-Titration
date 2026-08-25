from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_PATTERNS = (
    "*.py",
    "*.md",
    "*.txt",
    "*.yml",
    "*.csv",
    "*.cmd",
    "*.ipynb",
)


def copy_directory(source: Path, destination: Path):
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"),
    )


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the curated reviewer evidence ZIP")
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    release_dir = args.release_dir.resolve()
    zip_path = args.zip_path.resolve()
    allowed_parent = work_dir.parent.resolve()
    if release_dir.parent != allowed_parent or zip_path.parent != allowed_parent:
        raise RuntimeError("Release outputs must be direct children of the workspace root")
    if release_dir.exists():
        raise FileExistsError(f"Release directory already exists: {release_dir}")

    validation_path = work_dir / "results" / "full" / "VALIDATION_REPORT.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("Full result validation did not pass")

    release_dir.mkdir(parents=True)
    for name in (
        "00_READ_ME_FIRST_CN.md",
        "01_FULL_ANALYSIS_CN.md",
        "02_REVIEWER_RESPONSE_DRAFT_EN.md",
        "03_BUGFIX_AND_RUN_RECORD_CN.md",
        "04_LIMITATIONS_AND_CLAIM_BOUNDARIES_CN.md",
    ):
        shutil.copy2(work_dir / "reports" / name, release_dir / name)

    source_dir = release_dir / "code_current"
    source_dir.mkdir()
    copied = set()
    for pattern in SOURCE_PATTERNS:
        for path in work_dir.glob(pattern):
            if path.name in copied or path.name == "SHA256SUMS.csv":
                continue
            shutil.copy2(path, source_dir / path.name)
            copied.add(path.name)

    copy_directory(work_dir / "original_source_snapshot", release_dir / "code_original_snapshot")
    copy_directory(work_dir / "reference", release_dir / "reference")
    copy_directory(work_dir / "results" / "full", release_dir / "results_full")
    copy_directory(work_dir / "results" / "quick", release_dir / "diagnostics" / "quick_smoke")
    copy_directory(work_dir / "diagnostics_original", release_dir / "diagnostics" / "prior_bound_before_fix")
    copy_directory(work_dir / "diagnostics_corrected", release_dir / "diagnostics" / "corrected_checks")
    copy_directory(work_dir / "logs", release_dir / "logs")
    shutil.copy2(work_dir / "SHA256SUMS.csv", release_dir / "ORIGINAL_SOURCE_SHA256SUMS.csv")

    inventory = {
        "package_created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_profile": "full",
        "formal_result_validation": validation,
        "original_source_manifest": "ORIGINAL_SOURCE_SHA256SUMS.csv",
        "notes": [
            "results_full contains the formal evidence used in the analysis.",
            "diagnostics/quick_smoke is installation and interface evidence only.",
            "The Python virtual environment is intentionally excluded.",
        ],
    }
    (release_dir / "PACKAGE_INVENTORY.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    manifest_path = release_dir / "SHA256_MANIFEST.csv"
    files = sorted(path for path in release_dir.rglob("*") if path.is_file() and path != manifest_path)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "size_bytes", "sha256"))
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "relative_path": path.relative_to(release_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    archive_base = zip_path.with_suffix("")
    created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=release_dir.parent, base_dir=release_dir.name))
    if created != zip_path:
        raise RuntimeError(f"Unexpected archive path: {created}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC test failed: {bad_member}")
        zip_entries = len(archive.infolist())

    summary = {
        "release_directory": str(release_dir),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "zip_entries": zip_entries,
        "crc_test": "PASS",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
