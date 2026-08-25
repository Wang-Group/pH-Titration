"""Remove internal-only validation fields from released policy metadata.

The actor tensors are preserved byte-for-byte logically; only the serialized
metadata is rewritten.  Provenance files are updated with the new file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
REMOVE_KEYS = {"strict_success_rate_percent"}
PROTOCOL_METADATA = {
    "protocol_family": "pH-control",
    "protocol_version": "2026.08",
    "protocol_profile": "training_environment_strict",
    "training_profile": "training_environment_strict",
    "deployment_profile": "deployment_api_strict",
    "formal_evaluation_profile": "formal_evaluation",
    "observed_ph_resolution": 0.01,
    "training_stop_operator": "<",
    "training_success_operator": "<",
    "training_persistent_overshoot_cap": "disabled",
    "deployment_stop_operator": "<",
    "success_tolerance_ph": 0.10,
    "max_steps": 50,
    "max_total_dose_ml": 50.0,
    "deployment_persistent_overshoot_cap": "enabled",
    "formal_evaluation_stop_operator": "<=",
    "formal_evaluation_success_operator": "<=",
    "formal_evaluation_persistent_overshoot_cap": "enabled_for_neural_policy",
}
LEGACY_PROTOCOL_KEYS = {
    "protocol_family",
    "protocol_version",
    "training_protocol_profile",
    "deployment_protocol_profile",
    "observed_ph_resolution",
    "training_stop_operator",
    "training_success_operator",
    "success_tolerance_ph",
    "max_steps",
    "max_total_dose_ml",
    "deployment_persistent_overshoot_cap",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_metadata(metadata: object) -> tuple[object, bool]:
    if not isinstance(metadata, dict):
        return metadata, False
    changed = False
    cleaned = dict(metadata)
    validation = cleaned.get("validation")
    if isinstance(validation, dict):
        validation_clean = {
            key: value for key, value in validation.items() if key not in REMOVE_KEYS
        }
        changed = validation_clean != validation
        cleaned["validation"] = validation_clean
    for key in LEGACY_PROTOCOL_KEYS:
        if key in cleaned:
            del cleaned[key]
            changed = True
    if cleaned.get("protocol") != PROTOCOL_METADATA:
        cleaned["protocol"] = dict(PROTOCOL_METADATA)
        changed = True
    return cleaned, changed


def atomic_torch_save(path: Path, payload: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".pth", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sanitize_torch(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata, changed = sanitize_metadata(payload.get("metadata"))
    if changed:
        payload["metadata"] = metadata
        atomic_torch_save(path, payload)
    return changed


def atomic_npz_save(path: Path, payload: dict[str, np.ndarray]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        np.savez(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sanitize_npz(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    cleaned, changed = sanitize_metadata(metadata)
    if changed:
        payload["metadata_json"] = np.asarray(json.dumps(cleaned, sort_keys=True))
        atomic_npz_save(path, payload)
    return changed


def main() -> None:
    torch_paths = [
        ROOT / "controllers" / "models" / "ppo_seed_303.pth",
        *sorted(
            (EVIDENCE / "02_TEACHER_AND_IMITATION" / "checkpoints").glob(
                "principal_ppo_seed_*.pth"
            )
        ),
        *sorted(
            (EVIDENCE / "03_PPO_TRAINING_STABILITY" / "checkpoints").glob(
                "ppo_seed_*.pth"
            )
        ),
    ]
    changed = [str(path.relative_to(ROOT)) for path in torch_paths if sanitize_torch(path)]
    numpy_path = ROOT / "controllers" / "models" / "ppo_seed_303_numpy.npz"
    if sanitize_npz(numpy_path):
        changed.append(str(numpy_path.relative_to(ROOT)))

    provenance_paths = [
        EVIDENCE / "02_TEACHER_AND_IMITATION" / "CHECKPOINT_PROVENANCE.json",
        EVIDENCE / "03_PPO_TRAINING_STABILITY" / "CHECKPOINT_PROVENANCE.json",
    ]
    for provenance_path in provenance_paths:
        payload = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        entries = payload if isinstance(payload, list) else payload.get(
            "principal_ppo_checkpoints", payload.get("checkpoints", [])
        )
        if isinstance(entries, list):
            for entry in entries:
                relative = entry.get("checkpoint") or entry.get("file")
                if not relative:
                    continue
                candidate = ROOT / relative if str(relative).startswith("evidence/") else provenance_path.parent / relative
                if candidate.is_file():
                    key = "released_file_sha256" if "released_file_sha256" in entry else "sha256"
                    entry[key] = sha256(candidate).upper()
            provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    run_config = EVIDENCE / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation" / "RUN_CONFIG.json"
    config = json.loads(run_config.read_text(encoding="utf-8"))
    config["ppo_checkpoint_sha256"] = sha256(ROOT / "controllers" / "models" / "ppo_seed_303.pth")
    run_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"changed": changed, "deployment_sha256": sha256(torch_paths[0]), "numpy_sha256": sha256(numpy_path)}, indent=2))


if __name__ == "__main__":
    main()
