from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


SENSOR_RESOLUTION_PH = 0.01
PRIMARY_TITRANT_CONCENTRATION_M = 0.1
MIN_ACTION_VOLUME_ML = 0.01
MAX_ACTION_VOLUME_ML = 10.0
DEFAULT_MAX_TOTAL_DOSE_ML = 50.0


def validate_controller_config(
    *,
    success_tolerance_ph: float,
    max_steps: int,
    max_total_dose_ml: float | None,
    titrant_concentration_m: float,
) -> tuple[float, int, float | None, float]:
    """Validate shared safety limits before loading or running a controller."""
    try:
        tolerance = float(success_tolerance_ph)
    except (TypeError, ValueError) as exc:
        raise ValueError("success_tolerance_ph must be finite and positive") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("success_tolerance_ph must be finite and positive")
    try:
        steps = int(max_steps)
        steps_is_integral = float(max_steps) == steps
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_steps must be a positive integer") from exc
    if isinstance(max_steps, bool) or not steps_is_integral or steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_total_dose_ml is None:
        dose_limit = None
    else:
        try:
            dose_limit = float(max_total_dose_ml)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_total_dose_ml must be finite and nonnegative") from exc
        if not np.isfinite(dose_limit) or dose_limit < 0.0:
            raise ValueError("max_total_dose_ml must be finite and nonnegative")
    try:
        concentration = float(titrant_concentration_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("titrant_concentration_m must be finite and positive") from exc
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("titrant_concentration_m must be finite and positive")
    return tolerance, steps, dose_limit, concentration


def remaining_dose_ml(total_added_ml: float, max_total_dose_ml: float | None) -> float:
    if max_total_dose_ml is None:
        return float(MAX_ACTION_VOLUME_ML)
    return max(0.0, float(max_total_dose_ml) - float(total_added_ml))


def quantize_ph(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("Measured pH must be finite")
    return float(np.round(float(value) / SENSOR_RESOLUTION_PH) * SENSOR_RESOLUTION_PH)


def normalize_reagent(value: str) -> str:
    normalized = str(value).strip().lower()
    if "base" in normalized:
        return "base"
    if "acid" in normalized:
        return "acid"
    raise ValueError(f"Reagent must identify acid or base, got {value!r}")


class PersistentOvershootCap:
    """Shared post-overshoot dose cap used by deployment and evaluation.

    The cap is updated after an observed target crossing or an increase in
    absolute observed-pH error. Once activated, it can only decrease and is
    applied at the same 0.01 mL resolution as controller actions.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.cap_ml: float | None = None
        self.events = 0
        self.applied_steps = 0

    def apply(self, volume_ml: float) -> tuple[float, bool]:
        """Return the executable volume and whether the persistent cap applied."""
        requested = round(float(np.clip(volume_ml, MIN_ACTION_VOLUME_ML, MAX_ACTION_VOLUME_ML)), 2)
        if not self.enabled or self.cap_ml is None:
            return requested, False
        class_cap = max(
            MIN_ACTION_VOLUME_ML,
            math.floor((float(self.cap_ml) + 1e-12) * 100.0) / 100.0,
        )
        capped = round(min(requested, class_cap), 2)
        applied = capped < requested - 1e-12
        if applied:
            self.applied_steps += 1
        return capped, applied

    def update(
        self,
        before_ph: float,
        after_ph: float,
        target_ph: float,
        actual_volume_ml: float,
    ) -> bool:
        """Update the cap after one observed dose and return whether it fired."""
        if not self.enabled:
            return False
        actual = float(actual_volume_ml)
        if not np.isfinite(actual) or actual <= 0.0:
            raise ValueError("actual_volume_ml must be finite and positive")
        crossed = (float(before_ph) - float(target_ph)) * (
            float(after_ph) - float(target_ph)
        ) < 0.0
        error_increased = abs(float(after_ph) - float(target_ph)) > abs(
            float(before_ph) - float(target_ph)
        )
        if not (crossed or error_increased):
            return False
        new_cap = max(actual / 2.0, MIN_ACTION_VOLUME_ML)
        self.cap_ml = new_cap if self.cap_ml is None else min(self.cap_ml, new_cap)
        self.events += 1
        return True

    def status(self) -> dict[str, float | int | bool | None]:
        return {
            "overshoot_cap_enabled": self.enabled,
            "overshoot_cap_ml": self.cap_ml,
            "overshoot_cap_events": self.events,
            "overshoot_cap_applied_steps": self.applied_steps,
        }


@dataclass(frozen=True)
class ControllerAction:
    stop: bool
    reagent: str | None
    volume_ml: float
    titrant_concentration_m: float = PRIMARY_TITRANT_CONCENTRATION_M
    reason: str = "running"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stop_action(reason: str, diagnostics: dict[str, Any] | None = None) -> ControllerAction:
    return ControllerAction(
        stop=True,
        reagent=None,
        volume_ml=0.0,
        reason=reason,
        diagnostics=diagnostics or {},
    )
