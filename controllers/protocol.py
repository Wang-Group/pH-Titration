from __future__ import annotations

"""Protocol metadata shared by the released controller APIs."""

PROTOCOL_FAMILY = "pH-control"
PROTOCOL_VERSION = "2026.08"
_UNSET = object()

FORMAL_EVALUATION_PROFILE = {
    "protocol_family": PROTOCOL_FAMILY,
    "protocol_version": PROTOCOL_VERSION,
    "protocol_profile": "formal_evaluation",
    "observed_ph_resolution": 0.01,
    "controller_stop_operator": "<=",
    "controller_stop_tolerance_ph": 0.10,
    "reported_success_source": "final_unrounded_equilibrium_ph",
    "reported_success_operator": "<=",
    "reported_success_tolerance_ph": 0.10,
    "max_steps": 50,
    "max_total_dose_ml": 50.0,
    "persistent_overshoot_cap": "enabled_for_formal_neural_policy_evaluation",
}

TRAINING_ENVIRONMENT_PROFILE = {
    "protocol_family": PROTOCOL_FAMILY,
    "protocol_version": PROTOCOL_VERSION,
    "protocol_profile": "training_environment_strict",
    "observed_ph_resolution": 0.01,
    "controller_stop_operator": "<",
    "controller_stop_tolerance_ph": 0.10,
    "reported_success_source": "final_unrounded_equilibrium_ph",
    "reported_success_operator": "<",
    "reported_success_tolerance_ph": 0.10,
    "max_steps": 50,
    "max_total_dose_ml": 50.0,
    "persistent_overshoot_cap": "disabled_in_training_environment",
}

DEPLOYMENT_API_PROFILE = {
    "protocol_family": PROTOCOL_FAMILY,
    "protocol_version": PROTOCOL_VERSION,
    "protocol_profile": "deployment_api_strict",
    "observed_ph_resolution": 0.01,
    "controller_stop_operator": "<",
    "controller_stop_tolerance_ph": 0.10,
    "reported_success_source": None,
    "reported_success_operator": None,
    "reported_success_tolerance_ph": None,
    "max_steps": 50,
    "max_total_dose_ml": 50.0,
    "persistent_overshoot_cap": "enabled",
}


def controller_protocol_metadata(
    profile: str,
    *,
    stop_tolerance_ph: float | None = None,
    max_steps: int | None = None,
    max_total_dose_ml: float | None | object = _UNSET,
    persistent_overshoot_cap: str | None = None,
) -> dict[str, object]:
    profiles = {
        "formal_evaluation": FORMAL_EVALUATION_PROFILE,
        "training_environment_strict": TRAINING_ENVIRONMENT_PROFILE,
        "deployment_api_strict": DEPLOYMENT_API_PROFILE,
    }
    try:
        metadata = dict(profiles[profile])
    except KeyError as exc:
        raise ValueError(f"Unknown protocol profile: {profile}") from exc
    if stop_tolerance_ph is not None:
        metadata["controller_stop_tolerance_ph"] = float(stop_tolerance_ph)
    if max_steps is not None:
        metadata["max_steps"] = int(max_steps)
    if max_total_dose_ml is not _UNSET:
        metadata["max_total_dose_ml"] = (
            None if max_total_dose_ml is None else float(max_total_dose_ml)
        )
    if persistent_overshoot_cap is not None:
        metadata["persistent_overshoot_cap"] = persistent_overshoot_cap
    return metadata
