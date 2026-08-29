from __future__ import annotations

import hashlib
import platform
import struct
from pathlib import Path

import matplotlib
import numpy
import numpy as np
import scipy
import torch

from benchmark_core import (
    DiscreteVolumeRegressor,
    PolicyEnvironment,
    StressScenario,
    Task,
    _unwrap_state_dict,
)


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "imitation.pth"
EXPECTED_SHA256 = "5a58cf32af0392ac5a8aeaa5db296cd0a2fafed9d03c984ef20e841ebb8772b0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("A 64-bit Python installation is required.")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weight file: {WEIGHTS}")
    actual_hash = sha256(WEIGHTS)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected imitation weight SHA-256: {actual_hash}")
    model = DiscreteVolumeRegressor()
    payload = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(_unwrap_state_dict(payload), strict=True)
    output = model(torch.zeros(1, 5))
    if tuple(output.shape) != (1, 1000):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

    base_task = Task(1, 1, "monoprotic", (4.76,), 0.0, 10.0)
    base_env = PolicyEnvironment(base_task, StressScenario("nominal"), np.random.default_rng(1))
    base_env.step(10.0)
    if not (base_env.base_moles > 0 and base_env.acid_moles == 0):
        raise RuntimeError("The external direction rule did not choose base below the target")
    if base_env.maximum_requested_volume() != 10.0:
        raise RuntimeError("The action range was reduced after a step")

    acid_task = Task(1, 2, "monoprotic", (4.76,), 0.0, 2.0)
    acid_env = PolicyEnvironment(acid_task, StressScenario("nominal"), np.random.default_rng(2))
    acid_env.step(1.0)
    if not (acid_env.acid_moles > 0 and acid_env.base_moles == 0):
        raise RuntimeError("The external direction rule did not choose acid above the target")
    if hasattr(acid_env, "use_secondary") or hasattr(acid_env, "overshoot_threshold_ml"):
        raise RuntimeError("Legacy titrant-switching or overshoot-masking state is still present")
    print(f"Python: {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
    print(f"PyTorch: {torch.__version__}; NumPy: {numpy.__version__}; SciPy: {scipy.__version__}; Matplotlib: {matplotlib.__version__}")
    print(f"Imitation weights: OK ({actual_hash})")
    print("Control allocation: volume-only actor with external acid/base direction rule")
    print("Legacy overshoot masking and dilute-titrant switching: disabled")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
