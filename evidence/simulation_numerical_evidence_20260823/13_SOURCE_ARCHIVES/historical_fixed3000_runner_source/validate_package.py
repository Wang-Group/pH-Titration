from __future__ import annotations

import argparse
import json
import platform
import struct
import sys
from pathlib import Path

import matplotlib
import numpy
import scipy
import torch

from benchmark_core import NeuralVolumePolicy
from challenge_common import CONFIRM_STRESS_SCENARIOS, SCENARIOS, feature_dim, load_bayesian_module
from train_candidates import CANDIDATES


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=base / "package_validation.json")
    args = parser.parse_args()
    required = [
        base / "benchmark_core.py",
        base / "challenge_common.py",
        base / "train_candidates.py",
        base / "evaluate_candidates.py",
        base / "build_report.py",
        base / "inputs" / "bayesian_controller.py",
        base / "models" / "imitation.pth",
        base / "models" / "reinforcement.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    NeuralVolumePolicy(base / "models" / "imitation.pth", "cpu")
    NeuralVolumePolicy(base / "models" / "reinforcement.pth", "cpu")
    ppo_paths = sorted((base / "models" / "ppo_reference").glob("ppo_full_seed*.pth"))
    for path in ppo_paths:
        NeuralVolumePolicy(path, "cpu")
    load_bayesian_module(base / "inputs" / "bayesian_controller.py")
    payload = {
        "status": "ok",
        "python": sys.version,
        "python_64_bit": struct.calcsize("P") * 8 == 64,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "candidate_count": len(CANDIDATES),
        "candidate_names": sorted(CANDIDATES),
        "scenario_count": len(SCENARIOS),
        "confirmatory_scenario_count": len(CONFIRM_STRESS_SCENARIOS),
        "ppo_reference_models": len(ppo_paths),
        "feature_dimensions": {
            "basic": feature_dim("basic"),
            "history": feature_dim("history"),
            "filtered": feature_dim("filtered"),
            "history_residual": feature_dim("history", True),
        },
    }
    if not payload["python_64_bit"] or sys.version_info < (3, 10):
        raise SystemExit("64-bit Python 3.10 or newer is required.")
    if len(ppo_paths) != 5:
        raise SystemExit(f"Expected five reference PPO models, found {len(ppo_paths)}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
