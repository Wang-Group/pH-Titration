from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controller_api import (
    PRIMARY_TITRANT_CONCENTRATION_M,
    ControllerAction,
    normalize_reagent,
    quantize_ph,
    stop_action,
)
from models import checkpoint_sha256, load_actor_checkpoint


SELECTED_PPO_TRAINING_SEED = 303
SELECTED_PPO_FILE_SHA256 = "bafd85f896945245f4a2275764ee74cfb458aae78cbe91f5c17396c24fd22f1c"
SELECTED_PPO_ACTOR_SHA256 = "8c2ccdbc7879d1d54b151eeccb76ed3d354e58ace23e8052f6619d57369214fb"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PPOVolumeController:
    """Deterministic deployment wrapper for the validation-selected PPO actor."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        success_tolerance_ph: float = 0.10,
        max_steps: int = 50,
        max_total_dose_ml: float = 50.0,
        titrant_concentration_m: float = PRIMARY_TITRANT_CONCENTRATION_M,
        verify_selected_checkpoint: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.actor, self.normalizer, self.metadata = load_actor_checkpoint(
            self.checkpoint, self.device
        )
        self.actor.eval()
        self.checkpoint_file_sha256 = file_sha256(self.checkpoint)
        self.actor_sha256 = checkpoint_sha256(self.actor.state_dict())
        if verify_selected_checkpoint:
            if self.checkpoint_file_sha256 != SELECTED_PPO_FILE_SHA256:
                raise RuntimeError(
                    "Checkpoint file hash does not match the validation-selected PPO seed 303"
                )
            if self.actor_sha256 != SELECTED_PPO_ACTOR_SHA256:
                raise RuntimeError(
                    "Actor tensor hash does not match the validation-selected PPO seed 303"
                )
        self.success_tolerance_ph = float(success_tolerance_ph)
        self.max_steps = int(max_steps)
        self.max_total_dose_ml = float(max_total_dose_ml)
        self.titrant_concentration_m = float(titrant_concentration_m)
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self.current_ph = math.nan
        self.previous_ph = math.nan
        self.target_ph = math.nan
        self.last_requested_volume_ml = 0.0
        self.total_actual_volume_ml = 0.0
        self.base_added_ml = 0.0
        self.acid_added_ml = 0.0
        self.steps = 0
        self.done = False
        self.stop_reason = "not_initialized"
        self.initialized = False
        self.pending_action: ControllerAction | None = None
        self.last_observation: dict[str, Any] | None = None

    def reset(self, initial_measured_ph: float, target_ph: float) -> dict[str, Any]:
        self._reset_runtime()
        self.current_ph = quantize_ph(initial_measured_ph)
        self.previous_ph = self.current_ph
        self.target_ph = float(target_ph)
        if not np.isfinite(self.target_ph):
            raise ValueError("target_ph must be finite")
        self.initialized = True
        if abs(self.current_ph - self.target_ph) <= self.success_tolerance_ph:
            self.done = True
            self.stop_reason = "initial_success"
        else:
            self.stop_reason = "running"
        return self.status()

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("Call reset() before requesting an action")

    def state_vector(self) -> np.ndarray:
        self._require_initialized()
        return np.asarray(
            [
                self.current_ph,
                self.target_ph,
                self.current_ph - self.previous_ph,
                self.current_ph - self.target_ph,
                self.last_requested_volume_ml,
            ],
            dtype=np.float32,
        )

    def _stop_if_needed(self) -> None:
        if abs(self.current_ph - self.target_ph) <= self.success_tolerance_ph:
            self.done = True
            self.stop_reason = "measured_success"
        elif self.steps >= self.max_steps:
            self.done = True
            self.stop_reason = "max_steps"
        elif self.total_actual_volume_ml >= self.max_total_dose_ml - 1e-12:
            self.done = True
            self.stop_reason = "dose_limit"

    def recommend(self) -> ControllerAction:
        self._require_initialized()
        self._stop_if_needed()
        if self.done:
            return stop_action(self.stop_reason, self.status())
        if self.pending_action is not None:
            raise RuntimeError("The previous action has not been acknowledged by observe()")
        state = self.state_vector()
        normalized = self.normalizer.transform_numpy(state)
        tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(tensor)
            action_class = int(torch.argmax(logits, dim=1).item())
        volume_ml = (action_class + 1) * 0.01
        reagent = "base" if self.current_ph < self.target_ph else "acid"
        action = ControllerAction(
            stop=False,
            reagent=reagent,
            volume_ml=round(volume_ml, 2),
            titrant_concentration_m=self.titrant_concentration_m,
            diagnostics={
                "state": state.tolist(),
                "normalized_state": normalized.tolist(),
                "action_class": action_class,
                "checkpoint_training_seed": SELECTED_PPO_TRAINING_SEED,
                "checkpoint_environment_steps": self.metadata.get("environment_steps"),
            },
        )
        self.pending_action = action
        return action

    def observe(
        self,
        measured_ph: float,
        actual_volume_ml: float | None = None,
        reagent: str | None = None,
    ) -> dict[str, Any]:
        self._require_initialized()
        if self.pending_action is None:
            raise RuntimeError("Call recommend() before observe()")
        expected_reagent = str(self.pending_action.reagent)
        executed_reagent = (
            expected_reagent if reagent is None else normalize_reagent(reagent)
        )
        if executed_reagent != expected_reagent:
            raise ValueError(
                f"Executed reagent {executed_reagent!r} does not match requested "
                f"reagent {expected_reagent!r}"
            )
        actual = (
            float(self.pending_action.volume_ml)
            if actual_volume_ml is None
            else float(actual_volume_ml)
        )
        if not np.isfinite(actual) or actual <= 0.0:
            raise ValueError("actual_volume_ml must be finite and positive")

        observed_before = self.current_ph
        observed_after = quantize_ph(measured_ph)
        self.previous_ph = observed_before
        self.current_ph = observed_after
        self.last_requested_volume_ml = float(self.pending_action.volume_ml)
        self.total_actual_volume_ml += actual
        if executed_reagent == "base":
            self.base_added_ml += actual
        else:
            self.acid_added_ml += actual
        self.steps += 1
        self.last_observation = {
            "step": self.steps,
            "observed_before_ph": observed_before,
            "observed_after_ph": observed_after,
            "requested_volume_ml": self.pending_action.volume_ml,
            "actual_volume_ml": actual,
            "reagent": executed_reagent,
        }
        self.pending_action = None
        self._stop_if_needed()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "controller": "ppo_volume_seed_303",
            "initialized": self.initialized,
            "done": self.done,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "current_measured_ph": self.current_ph,
            "target_ph": self.target_ph,
            "total_actual_volume_ml": self.total_actual_volume_ml,
            "base_added_ml": self.base_added_ml,
            "acid_added_ml": self.acid_added_ml,
            "checkpoint": str(self.checkpoint),
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "actor_sha256": self.actor_sha256,
            "metadata": self.metadata,
            "last_observation": self.last_observation,
        }
