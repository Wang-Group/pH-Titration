from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz
import matplotlib
import numpy as np
import pymc
import scipy

from benchmark_core import solve_ph
from chemistry_model import SolutionState, solve_ph_particles, solve_ph_scalar
from io_utils import write_json
from particle_inference import (
    PRIOR_PKA_HIGH,
    PRIOR_PKA_LOW,
    FixedKParticleFilter,
)


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "original_source_snapshot"
MANIFEST = SNAPSHOT / "SHA256SUMS.csv"
EXPECTED_VERSIONS = {
    "numpy": "2.2.6",
    "scipy": "1.17.1",
    "matplotlib": "3.11.1",
    "pymc": "5.28.5",
    "arviz": "0.23.4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_snapshot() -> int:
    errors = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = SNAPSHOT / Path(row["Path"])
        if not path.exists() or sha256(path) != row["SHA256"].upper():
            errors.append(row["Path"])
    if errors:
        raise RuntimeError(f"Original source snapshot mismatch: {errors}")
    return len(rows)


def validate_chemistry() -> None:
    rng = np.random.default_rng(20260811)
    for _ in range(50):
        pair_count = int(rng.integers(1, 4))
        pkas = tuple(sorted(rng.uniform(2.0, 8.0, pair_count)))
        concentration = float(rng.uniform(0.03, 0.25))
        base_moles = float(rng.uniform(0.0, 0.003))
        acid_moles = float(rng.uniform(0.0, 0.003))
        added_volume = float(rng.uniform(0.0, 20.0))
        expected = solve_ph(pkas, 11.0, concentration, base_moles, acid_moles, added_volume)
        actual = solve_ph_scalar(
            concentration,
            pkas,
            11.0,
            SolutionState(11.0 + added_volume, base_moles, acid_moles),
        )
        if abs(expected - actual) > 1e-7:
            raise RuntimeError("Scalar chemistry model does not match benchmark_core")

    concentrations = rng.uniform(0.03, 0.25, 40)
    pka_matrix = np.sort(rng.uniform(2.0, 8.0, size=(40, 3)), axis=1)
    state = SolutionState(17.0, 0.0012, 0.0004)
    vectorized = solve_ph_particles(concentrations, pka_matrix, 11.0, state)
    scalar = np.asarray([
        solve_ph_scalar(concentration, pkas, 11.0, state)
        for concentration, pkas in zip(concentrations, pka_matrix)
    ])
    if float(np.max(np.abs(vectorized - scalar))) > 1e-10:
        raise RuntimeError("Particle chemistry solver does not match scalar solver")


def validate_prior_support() -> None:
    inference = FixedKParticleFilter(300, 3, True, np.random.default_rng(91))
    inference.weights.fill(0.0)
    inference.weights[0] = 1.0
    inference.update(
        11.0,
        SolutionState(11.0, 0.0, 0.0),
        SolutionState(12.0, 0.0001, 0.0),
        3.0,
        3.2,
    )
    if float(np.min(inference.pka_particles)) < PRIOR_PKA_LOW:
        raise RuntimeError("PF regularization escaped below the frozen pKa prior")
    if float(np.max(inference.pka_particles)) > PRIOR_PKA_HIGH:
        raise RuntimeError("PF regularization escaped above the frozen pKa prior")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen source, environment, and numerical model")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("A 64-bit Python installation is required")
    actual_versions = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pymc": pymc.__version__,
        "arviz": arviz.__version__,
    }
    if actual_versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"Dependency version mismatch: {actual_versions}")
    manifest_entries = validate_snapshot()
    validate_chemistry()
    validate_prior_support()
    payload = {
        "status": "PASS",
        "python": platform.python_version(),
        "python_bits": struct.calcsize("P") * 8,
        "versions": actual_versions,
        "original_manifest_entries": manifest_entries,
        "chemistry_consistency": "PASS",
        "pf_pka_support": [PRIOR_PKA_LOW, PRIOR_PKA_HIGH],
        "gpu_policy": "CPU multiprocessing; current NumPy/PyMC black-box implementation is not CUDA-enabled",
    }
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
