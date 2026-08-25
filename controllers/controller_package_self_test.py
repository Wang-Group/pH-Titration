from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .chemistry_model import SolutionState, solve_ph_scalar
    from .new_pf_controller import RobustPFController
    from .new_rl_controller import PPOVolumeController
    from .new_rl_numpy_controller import NumpyPPOVolumeController
except ImportError:  # pragma: no cover - direct script compatibility
    from chemistry_model import SolutionState, solve_ph_scalar
    from new_pf_controller import RobustPFController
    from new_rl_controller import PPOVolumeController
    from new_rl_numpy_controller import NumpyPPOVolumeController


SAMPLE = {
    "pka_values": (4.565982225120272,),
    "analyte_conc_m": 0.1791423294444583,
    "initial_volume_ml": 8.79662341754318,
    "initial_base_moles": 0.000606739365843235,
    "initial_ph": 4.363,
    "target_ph": 7.721,
}


def advance(state: SolutionState, reagent: str, volume_ml: float) -> SolutionState:
    added = 0.1 * volume_ml / 1000.0
    if reagent == "base":
        return SolutionState(
            state.total_volume_ml + volume_ml,
            state.base_moles + added,
            state.acid_moles,
        )
    return SolutionState(
        state.total_volume_ml + volume_ml,
        state.base_moles,
        state.acid_moles + added,
    )


def run_controller(controller) -> dict:
    state = SolutionState(
        SAMPLE["initial_volume_ml"],
        SAMPLE["initial_base_moles"],
        0.0,
    )
    actions = []
    while True:
        action = controller.recommend()
        if action.stop:
            break
        if not 0.01 <= action.volume_ml <= 10.0:
            raise AssertionError("Action volume is outside 0.01-10.00 mL")
        state = advance(state, str(action.reagent), action.volume_ml)
        true_ph = solve_ph_scalar(
            SAMPLE["analyte_conc_m"],
            SAMPLE["pka_values"],
            SAMPLE["initial_volume_ml"],
            state,
        )
        measured_ph = float(np.round(true_ph, 2))
        controller.observe(measured_ph, action.volume_ml, action.reagent)
        actions.append(
            {
                "step": len(actions) + 1,
                "reagent": action.reagent,
                "volume_ml": action.volume_ml,
                "true_ph": true_ph,
                "measured_ph": measured_ph,
            }
        )
    final_error = abs(actions[-1]["true_ph"] - SAMPLE["target_ph"])
    return {
        "steps": len(actions),
        "final_true_ph": actions[-1]["true_ph"],
        "final_abs_error": final_error,
        "success": final_error < 0.10,
        "total_volume_ml": float(sum(row["volume_ml"] for row in actions)),
        "stop_reason": controller.status()["stop_reason"],
        "actions": actions,
    }


def run_safety_checks(root: Path) -> dict:
    """Exercise the action mask, hard dose boundary, and constructor checks."""
    results = {}
    pf = RobustPFController(particles=1000, seed=7, max_total_dose_ml=0.05)
    pf.reset(4.0, 8.0, 10.0)
    action = pf.recommend()
    results["pf_action_respects_cap"] = action.volume_ml <= 0.05
    try:
        pf.observe(4.1, actual_volume_ml=0.06, reagent=action.reagent)
    except ValueError:
        results["pf_rejects_overcap_delivery"] = True
    else:
        results["pf_rejects_overcap_delivery"] = False

    checkpoint = root / "models" / "ppo_seed_303.pth"
    ppo = PPOVolumeController(checkpoint, device="cpu", max_total_dose_ml=0.05)
    ppo.reset(4.0, 8.0)
    results["ppo_action_respects_cap"] = ppo.recommend().volume_ml <= 0.05

    numpy_controller = NumpyPPOVolumeController(
        root / "models" / "ppo_seed_303_numpy.npz", max_total_dose_ml=0.05
    )
    numpy_controller.reset(4.0, 8.0)
    results["numpy_action_respects_cap"] = (
        numpy_controller.recommend().volume_ml <= 0.05
    )

    invalid_parameters = (
        {"success_tolerance_ph": -0.1},
        {"max_steps": 0},
        {"max_total_dose_ml": -1.0},
        {"titrant_concentration_m": 0.0},
    )
    results["invalid_parameters_rejected"] = True
    for parameters in invalid_parameters:
        try:
            RobustPFController(**parameters)
        except ValueError:
            continue
        results["invalid_parameters_rejected"] = False
    if not all(results.values()):
        raise AssertionError(f"Safety checks failed: {results}")
    return results


def main() -> None:
    root = Path(__file__).resolve().parent
    pf = RobustPFController(particles=1000, seed=101)
    pf.reset(
        SAMPLE["initial_ph"],
        SAMPLE["target_ph"],
        SAMPLE["initial_volume_ml"],
        SAMPLE["initial_base_moles"],
    )
    ppo = PPOVolumeController(root / "models" / "ppo_seed_303.pth", device="cpu")
    ppo.reset(SAMPLE["initial_ph"], SAMPLE["target_ph"])
    report = {
        "status": "pass",
        "sample": SAMPLE,
        "pf": run_controller(pf),
        "ppo": run_controller(ppo),
        "ppo_checkpoint_file_sha256": ppo.checkpoint_file_sha256,
        "ppo_actor_sha256": ppo.actor_sha256,
        "safety": run_safety_checks(root),
    }
    if not report["pf"]["success"] or not report["ppo"]["success"]:
        raise AssertionError("A controller failed the packaged closed-loop sample")
    output = root / "SELF_TEST_REPORT.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "pf": {key: report["pf"][key] for key in ("steps", "final_abs_error", "total_volume_ml")},
        "ppo": {key: report["ppo"][key] for key in ("steps", "final_abs_error", "total_volume_ml")},
        "report": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
