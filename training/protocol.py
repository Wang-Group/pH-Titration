from __future__ import annotations

"""Protocol metadata embedded in newly generated training checkpoints."""


def training_checkpoint_protocol_metadata() -> dict[str, object]:
    return {
        "protocol_family": "pH-control",
        "protocol_version": "2026.08",
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
