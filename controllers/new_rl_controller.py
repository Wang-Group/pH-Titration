from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:  # Supports both package imports and legacy direct execution.
    from .controller_api import (
        DEFAULT_MAX_TOTAL_DOSE_ML,
        MAX_ACTION_VOLUME_ML,
        MIN_ACTION_VOLUME_ML,
        PRIMARY_TITRANT_CONCENTRATION_M,
        ControllerAction,
        PersistentOvershootCap,
        normalize_reagent,
        quantize_ph,
        remaining_dose_ml,
        stop_action,
        validate_controller_config,
    )
    from .models import ACTION_COUNT, checkpoint_sha256, load_actor_checkpoint
    from .protocol import controller_protocol_metadata
except ImportError:  # pragma: no cover - exercised only by direct script use
    from controller_api import (
        DEFAULT_MAX_TOTAL_DOSE_ML,
        MAX_ACTION_VOLUME_ML,
        MIN_ACTION_VOLUME_ML,
        PRIMARY_TITRANT_CONCENTRATION_M,
        ControllerAction,
        PersistentOvershootCap,
        normalize_reagent,
        quantize_ph,
        remaining_dose_ml,
        stop_action,
        validate_controller_config,
    )
    from models import ACTION_COUNT, checkpoint_sha256, load_actor_checkpoint
    from protocol import controller_protocol_metadata


SELECTED_PPO_TRAINING_SEED = 303
SELECTED_PPO_FILE_SHA256 = "4004d7a09768fc5ac3f448523f53cb22210ed919ca7e713f13d9aa693cc19de5"
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
        max_total_dose_ml: float | None = DEFAULT_MAX_TOTAL_DOSE_ML,
        titrant_concentration_m: float = PRIMARY_TITRANT_CONCENTRATION_M,
        verify_selected_checkpoint: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_file_sha256 = file_sha256(self.checkpoint)
        if verify_selected_checkpoint and self.checkpoint_file_sha256 != SELECTED_PPO_FILE_SHA256:
            raise RuntimeError(
                "Checkpoint file hash does not match the validation-selected PPO seed 303"
            )
        tolerance, steps, dose_limit, concentration = validate_controller_config(
            success_tolerance_ph=success_tolerance_ph,
            max_steps=max_steps,
            max_total_dose_ml=max_total_dose_ml,
            titrant_concentration_m=titrant_concentration_m,
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.actor, self.normalizer, self.metadata = load_actor_checkpoint(
            self.checkpoint,
            self.device,
            expected_file_sha256=(
                SELECTED_PPO_FILE_SHA256 if verify_selected_checkpoint else None
            ),
        )
        self.actor.eval()
        self.actor_sha256 = checkpoint_sha256(self.actor.state_dict())
        if verify_selected_checkpoint and self.actor_sha256 != SELECTED_PPO_ACTOR_SHA256:
            raise RuntimeError(
                "Actor tensor hash does not match the validation-selected PPO seed 303"
            )
        self.success_tolerance_ph = tolerance
        self.max_steps = steps
        self.max_total_dose_ml = dose_limit
        self.titrant_concentration_m = concentration
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
        self.overshoot_cap = PersistentOvershootCap()

    def reset(self, initial_measured_ph: float, target_ph: float) -> dict[str, Any]:
        self._reset_runtime()
        self.current_ph = quantize_ph(initial_measured_ph)
        self.previous_ph = self.current_ph
        self.target_ph = float(target_ph)
        if not np.isfinite(self.target_ph):
            raise ValueError("target_ph must be finite")
        self.initialized = True
        if abs(self.current_ph - self.target_ph) < self.success_tolerance_ph:
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
        if abs(self.current_ph - self.target_ph) < self.success_tolerance_ph:
            self.done = True
            self.stop_reason = "measured_success"
        elif self.steps >= self.max_steps:
            self.done = True
            self.stop_reason = "max_steps"
        elif (
            self.max_total_dose_ml is not None
            and self.total_actual_volume_ml >= self.max_total_dose_ml - 1e-12
        ):
            self.done = True
            self.stop_reason = "dose_limit"

    def recommend(self) -> ControllerAction:
        self._require_initialized()
        self._stop_if_needed()
        if self.done:
            return stop_action(self.stop_reason, self.status())
        if self.pending_action is not None:
            raise RuntimeError("The previous action has not been acknowledged by observe()")
        remaining = remaining_dose_ml(
            self.total_actual_volume_ml, self.max_total_dose_ml
        )
        if remaining < MIN_ACTION_VOLUME_ML - 1e-12:
            self.done = True
            self.stop_reason = "dose_limit"
            return stop_action(self.stop_reason, self.status())
        state = self.state_vector()
        normalized = self.normalizer.transform_numpy(state)
        tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(tensor)
            max_action_count = min(
                ACTION_COUNT,
                int(math.floor((min(MAX_ACTION_VOLUME_ML, remaining) + 1e-9) / MIN_ACTION_VOLUME_ML)),
            )
            if max_action_count < 1:
                self.done = True
                self.stop_reason = "dose_limit"
                return stop_action(self.stop_reason, self.status())
            masked_logits = logits.clone()
            masked_logits[:, max_action_count:] = -torch.inf
            action_class = int(torch.argmax(masked_logits, dim=1).item())
        uncapped_volume_ml = (action_class + 1) * 0.01
        volume_ml, cap_applied = self.overshoot_cap.apply(uncapped_volume_ml)
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
                "uncapped_volume_ml": round(uncapped_volume_ml, 2),
                "overshoot_cap_ml": self.overshoot_cap.cap_ml,
                "overshoot_cap_applied": cap_applied,
                "remaining_dose_ml": remaining,
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
        remaining = remaining_dose_ml(
            self.total_actual_volume_ml, self.max_total_dose_ml
        )
        if actual > remaining + 1e-9:
            raise ValueError(
                f"actual_volume_ml={actual:.6g} exceeds remaining dose limit "
                f"({remaining:.6g} mL)"
            )

        observed_before = self.current_ph
        observed_after = quantize_ph(measured_ph)
        action_diagnostics = dict(self.pending_action.diagnostics)
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
            "overshoot_cap_applied": bool(action_diagnostics.get("overshoot_cap_applied", False)),
        }
        self.pending_action = None
        cap_triggered = self.overshoot_cap.update(
            observed_before,
            observed_after,
            self.target_ph,
            actual,
        )
        self.last_observation["overshoot_cap_event_triggered"] = cap_triggered
        self.last_observation["overshoot_cap_after_step_ml"] = self.overshoot_cap.cap_ml
        self._stop_if_needed()
        return self.status()

    def status(self) -> dict[str, Any]:
        status = {
            "controller": "ppo_volume_seed_303",
            "protocol": controller_protocol_metadata(
                "deployment_api_strict",
                stop_tolerance_ph=self.success_tolerance_ph,
                max_steps=self.max_steps,
                max_total_dose_ml=self.max_total_dose_ml,
                persistent_overshoot_cap="enabled_neural_persistent_half-dose_cap",
            ),
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
        status.update(self.overshoot_cap.status())
        return status
