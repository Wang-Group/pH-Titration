from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from controller_api import (
    PRIMARY_TITRANT_CONCENTRATION_M,
    ControllerAction,
    normalize_reagent,
    quantize_ph,
    stop_action,
)


SELECTED_PPO_NUMPY_SHA256 = "39f0aca8f98d4c4fdfc222c476e5421ef8a7776fb5803c9099f4666c17b06efe"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NumpyPPOVolumeController:
    """Pure-NumPy inference wrapper for the validation-selected PPO actor.

    The weights are exported from the archived PyTorch checkpoint. Keeping
    deployment inference NumPy-only prevents importing PyTorch/Intel OpenMP
    into the physical-device Jupyter kernel.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        success_tolerance_ph: float = 0.10,
        max_steps: int = 50,
        max_total_dose_ml: float = 50.0,
        titrant_concentration_m: float = PRIMARY_TITRANT_CONCENTRATION_M,
    ) -> None:
        del device  # Compatibility with the former PyTorch deployment API.
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        payload = np.load(self.weights_path, allow_pickle=False)
        self.weights = [
            np.asarray(payload[f"w{i}"], dtype=np.float32)
            for i in range(3)
        ]
        self.biases = [
            np.asarray(payload[f"b{i}"], dtype=np.float32)
            for i in range(3)
        ]
        self.state_mean = np.asarray(payload["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(payload["state_std"], dtype=np.float32)
        self.metadata = json.loads(str(payload["metadata_json"].item()))
        self.weights_sha256 = file_sha256(self.weights_path)
        if self.weights_sha256 != SELECTED_PPO_NUMPY_SHA256:
            raise RuntimeError(
                "NumPy PPO weights do not match the validation-selected deployment file"
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
        self._stop_if_needed()
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

    def _forward(self, state: np.ndarray) -> np.ndarray:
        value = (state - self.state_mean) / self.state_std
        value = np.maximum(value @ self.weights[0].T + self.biases[0], 0.0)
        value = np.maximum(value @ self.weights[1].T + self.biases[1], 0.0)
        return value @ self.weights[2].T + self.biases[2]

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
            raise RuntimeError("The previous action has not been acknowledged")
        state = self.state_vector()
        logits = self._forward(state[None, :])[0]
        action_class = int(np.argmax(logits))
        volume_ml = (action_class + 1) * 0.01
        reagent = "base" if self.current_ph < self.target_ph else "acid"
        action = ControllerAction(
            stop=False,
            reagent=reagent,
            volume_ml=round(volume_ml, 2),
            titrant_concentration_m=self.titrant_concentration_m,
            diagnostics={
                "state": state.tolist(),
                "action_class": action_class,
                "checkpoint_training_seed": self.metadata.get("seed", 303),
                "checkpoint_environment_steps": self.metadata.get("environment_steps"),
                "inference_backend": "numpy",
                "numpy_checkpoint_sha256": self.weights_sha256,
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
        executed_reagent = expected_reagent if reagent is None else normalize_reagent(reagent)
        if executed_reagent != expected_reagent:
            raise ValueError("Executed reagent does not match requested reagent")
        actual = float(self.pending_action.volume_ml) if actual_volume_ml is None else float(actual_volume_ml)
        if not np.isfinite(actual) or actual <= 0.0:
            raise ValueError("actual_volume_ml must be finite and positive")
        before = self.current_ph
        after = quantize_ph(measured_ph)
        self.previous_ph = before
        self.current_ph = after
        self.last_requested_volume_ml = float(self.pending_action.volume_ml)
        self.total_actual_volume_ml += actual
        if executed_reagent == "base":
            self.base_added_ml += actual
        else:
            self.acid_added_ml += actual
        self.steps += 1
        self.last_observation = {
            "step": self.steps,
            "observed_before_ph": before,
            "observed_after_ph": after,
            "requested_volume_ml": self.pending_action.volume_ml,
            "actual_volume_ml": actual,
            "reagent": executed_reagent,
        }
        self.pending_action = None
        self._stop_if_needed()
        return self.status()

    def observe_physical(
        self,
        measured_ph: float,
        actual_volume_ml: float,
        actual_titrant_concentration_m: float,
        reagent: str,
    ) -> dict[str, Any]:
        actual = float(actual_volume_ml)
        concentration = float(actual_titrant_concentration_m)
        if not np.isfinite(actual) or actual <= 0.0:
            raise ValueError("actual_volume_ml must be finite and positive")
        if not np.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("actual_titrant_concentration_m must be finite and positive")
        equivalent_volume_ml = actual * concentration / self.titrant_concentration_m
        status = self.observe(measured_ph, actual_volume_ml=equivalent_volume_ml, reagent=reagent)
        if self.last_observation is not None:
            self.last_observation.update(
                {
                    "actual_pump_volume_ml": actual,
                    "actual_titrant_concentration_m": concentration,
                    "actual_equivalent_volume_ml": equivalent_volume_ml,
                }
            )
        return status

    def status(self) -> dict[str, Any]:
        return {
            "controller": "ppo_volume_seed_303_numpy",
            "initialized": self.initialized,
            "done": self.done,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "current_measured_ph": self.current_ph,
            "target_ph": self.target_ph,
            "total_actual_volume_ml": self.total_actual_volume_ml,
            "base_added_ml": self.base_added_ml,
            "acid_added_ml": self.acid_added_ml,
            "weights_path": str(self.weights_path),
            "numpy_checkpoint_sha256": self.weights_sha256,
            "metadata": self.metadata,
            "last_observation": self.last_observation,
        }
