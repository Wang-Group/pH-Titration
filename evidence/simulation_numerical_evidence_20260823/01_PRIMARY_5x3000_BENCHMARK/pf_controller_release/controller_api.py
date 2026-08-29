from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


SENSOR_RESOLUTION_PH = 0.01
PRIMARY_TITRANT_CONCENTRATION_M = 0.1
MIN_ACTION_VOLUME_ML = 0.01
MAX_ACTION_VOLUME_ML = 10.0


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
