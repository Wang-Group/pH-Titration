from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

from benchmark_core import NeuralVolumePolicy, load_tasks_csv, portable_path


def import_source(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("portable_bayesian_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import bundled Bayesian source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ["PHAdjustmentEnv", "solve_pH_batch", "MAX_STEPS", "SUCCESS_THRESHOLD"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Bundled Bayesian source is missing: {', '.join(missing)}")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate the portable reviewer-analysis package.")
    parser.add_argument("--imitation", type=Path, required=True)
    parser.add_argument("--reinforcement", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=base / "package_validation.json")
    args = parser.parse_args()

    source = base / "inputs" / "bayesian_controller.py"
    tasks_csv = base / "inputs" / "experiment_summary.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    if not tasks_csv.exists():
        raise FileNotFoundError(tasks_csv)
    import_source(source)
    tasks = load_tasks_csv(tasks_csv, 555)
    if len(tasks) < 3000:
        raise RuntimeError(f"Expected at least 3000 bundled tasks, found {len(tasks)}")

    imitation = NeuralVolumePolicy(args.imitation.resolve(), args.device)
    reinforcement = NeuralVolumePolicy(args.reinforcement.resolve(), args.device)
    state = np.array([3.0, 7.0, 0.0, -4.0, 0.0], dtype=np.float32)
    imitation_volume = imitation.select_volume(state)
    reinforcement_volume = reinforcement.select_volume(state)

    payload = {
        "status": "ok",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "bundled_tasks": len(tasks),
        "imitation_weights": portable_path(args.imitation, base),
        "reinforcement_weights": portable_path(args.reinforcement, base),
        "imitation_test_volume_ml": imitation_volume,
        "reinforcement_test_volume_ml": reinforcement_volume,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
