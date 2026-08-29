from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from .protocol import training_checkpoint_protocol_metadata
except ImportError:  # pragma: no cover - direct script compatibility
    from protocol import training_checkpoint_protocol_metadata


ACTION_COUNT = 1000
ACTION_VOLUMES_ML = np.arange(1, ACTION_COUNT + 1, dtype=np.float32) * 0.01


class VolumeActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_COUNT),
        )

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.net(normalized_state)


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = VolumeActor()
        self.critic = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


@dataclass(frozen=True)
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform_numpy(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, dtype=np.float32) - self.mean) / self.std

    def transform_tensor(self, state: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=state.dtype, device=state.device)
        std = torch.as_tensor(self.std, dtype=state.dtype, device=state.device)
        return (state - mean) / std


def checkpoint_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a checkpoint file without deserializing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_actor_checkpoint(path, actor, normalizer, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("protocol", training_checkpoint_protocol_metadata())
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "state_mean": np.asarray(normalizer.mean, dtype=np.float32),
            "state_std": np.asarray(normalizer.std, dtype=np.float32),
            "metadata": metadata,
        },
        path,
    )


def load_actor_checkpoint(path, device, expected_file_sha256: str | None = None):
    path = Path(path)
    if expected_file_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_file_sha256:
            raise RuntimeError(
                f"Checkpoint hash mismatch: expected {expected_file_sha256}, got {actual}"
            )
    payload = torch.load(path, map_location=device, weights_only=False)
    actor = VolumeActor().to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    normalizer = StateNormalizer(
        np.asarray(payload["state_mean"], dtype=np.float32),
        np.asarray(payload["state_std"], dtype=np.float32),
    )
    return actor, normalizer, payload.get("metadata", {})
