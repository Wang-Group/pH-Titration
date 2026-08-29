from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "physical_experiments"
OUTPUT = ROOT / "release_archives" / "physical_experiments_20260825.zip"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def included(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and path.name != "job_plot_reproduced_reported_method.png"
    )


def build() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"Missing physical-data directory: {SOURCE}")

    files = [path for path in sorted(SOURCE.rglob("*")) if included(path)]
    if not files:
        raise SystemExit("No physical-data files found")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path(SOURCE.name) / path.relative_to(SOURCE))

    archive_names = zipfile.ZipFile(OUTPUT).namelist()
    forbidden = [name for name in archive_names if "__pycache__" in name or name.endswith(".pyc")]
    if forbidden:
        raise SystemExit(f"Forbidden cache files entered the archive: {forbidden[:5]}")

    validation = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file": OUTPUT.name,
        "sha256": digest(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "archive_entries": len(archive_names),
        "includes_physical_control_notebook": False,
        "directories": ["casein", "cu_ssa", "mixed_acid", "wastewater"],
    }
    OUTPUT.with_suffix(OUTPUT.suffix + ".validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    build()
