from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


TRAIN_SEEDS = [101, 202, 303, 404, 555]
CANDIDATES = [
    "ppo_nominal",
    "ppo_robust",
    "a2c_robust",
    "ppo_history_robust",
    "sac_history_robust",
    "ppo_residual_robust",
    "ppo_filtered_robust",
    "ppo_conservative_robust",
    "td3_filtered_robust",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    problems: list[str] = []
    if sys.version_info < (3, 10):
        problems.append(f"Python 3.10+ is required; found {platform.python_version()}")

    versions = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            problems.append(f"Cannot import {package}: {exc}")

    required = [
        base / "models" / "imitation.pth",
        base / "models" / "reinforcement.pth",
        base / "inputs" / "bayesian_controller.py",
        base / "run_fixed3000.py",
        base / "analyze_fixed3000.py",
    ]
    required.extend(
        base / "models" / "ppo_reference" / f"ppo_full_seed{seed}.pth"
        for seed in TRAIN_SEEDS
    )
    required.extend(
        base / "candidate_models" / f"{candidate}_seed{seed}.pth"
        for candidate in CANDIDATES
        for seed in TRAIN_SEEDS
    )
    missing = [str(path.relative_to(base)) for path in required if not path.exists()]
    if missing:
        problems.append("Missing files: " + ", ".join(missing))

    result = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "required_files": len(required),
        "missing_files": missing,
        "status": "FAIL" if problems else "PASS",
        "problems": problems,
    }
    (base / "PACKAGE_VALIDATION_FIXED3000.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
