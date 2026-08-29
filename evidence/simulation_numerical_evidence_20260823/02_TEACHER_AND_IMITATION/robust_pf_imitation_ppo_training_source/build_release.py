from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {".venv", "__pycache__", "results"}
EXCLUDED_SUFFIXES = {".pyc", ".zip"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.name in {"SHA256SUMS.txt"} or path.name.endswith(".sha256.txt"):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def main() -> None:
    root = Path(__file__).resolve().parent
    required = {
        "run_pipeline.py",
        "RUN_QUICK.cmd",
        "RUN_STANDARD.cmd",
        "RUN_STANDARD_CUDA.cmd",
        "RUN_FULL.cmd",
        "INSTALL_ENV.cmd",
        "INSTALL_ENV_CUDA.cmd",
        "ONE_CLICK.ipynb",
        "README_CN.md",
        "EXPERIMENT_PROTOCOL.md",
        "DATASET_DESIGN.md",
        "generate_teacher_dataset.py",
        "train_imitation.py",
        "train_ppo.py",
        "evaluate_and_report.py",
        "particle_controllers.py",
        "particle_inference.py",
        "chemistry_model.py",
        "reference/original_bayesian_controller.py",
    }
    missing = sorted(name for name in required if not (root / name).exists())
    if missing:
        raise FileNotFoundError(f"Required release files are missing: {missing}")

    files = release_files(root)
    manifest_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    manifest_path = root / "SHA256SUMS.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    files = release_files(root) + [manifest_path]

    zip_path = root.parent / f"{root.name}.zip"
    archive_root = root.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            archive.write(path, f"{archive_root}/{path.relative_to(root).as_posix()}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP CRC validation failed for {bad_file}")
        for line in manifest_lines:
            expected, relative = line.split("  ", 1)
            archived = archive.read(f"{archive_root}/{relative}")
            actual = sha256_bytes(archived)
            if actual != expected:
                raise RuntimeError(f"ZIP content hash mismatch for {relative}")

    zip_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256.txt")
    checksum_path.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="ascii")
    print(f"Release files: {len(files)}")
    print(f"ZIP: {zip_path}")
    print(f"SHA-256: {zip_hash}")


if __name__ == "__main__":
    main()
