from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SEEDS = (101, 202, 303, 404, 555)
COMPONENT_PROBABILITIES = (0.40, 0.35, 0.25)
TITRANT_M = 0.100
DIFFICULTY_RANGES = {
    "near": (0.20, 0.70),
    "medium": (0.70, 2.00),
    "far": (2.00, 4.00),
}


@dataclass(frozen=True)
class IndependentMixtureTask:
    seed: int
    task_id: int
    split: str
    acid_type: str
    component_count: int
    pka_values: tuple[float, ...]
    analyte_conc_m: float
    component_concentrations_m: tuple[float, ...]
    component_mole_fractions: tuple[float, ...]
    initial_volume_ml: float
    initial_base_equivalents: float
    initial_base_moles: float
    initial_ph: float
    target_ph: float
    direction: str
    difficulty: str
    pka_family: str
    oracle_required_volume_ml: float


def mixture_ph(
    component_concentrations_m,
    pka_values,
    initial_volume_ml: float,
    total_volume_ml: float,
    base_moles: float,
    acid_moles: float,
    iterations: int = 60,
) -> float:
    total_volume_l = total_volume_ml / 1000.0
    dilution = initial_volume_ml / total_volume_ml
    concentrations = np.asarray(component_concentrations_m, dtype=float) * dilution
    kas = np.power(10.0, -np.asarray(pka_values, dtype=float))
    c_na = base_moles / total_volume_l
    c_hcl = acid_moles / total_volume_l

    def balance(ph: float) -> float:
        hydrogen = 10.0 ** (-ph)
        hydroxide = 1.0e-14 / hydrogen
        charge = float(np.sum(concentrations * kas / (kas + hydrogen)))
        return hydrogen + c_na - hydroxide - c_hcl - charge

    lo, hi = 0.0, 14.0
    f_lo = balance(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def response_after_dose(
    task: IndependentMixtureTask, direction: str, dose_ml: float
) -> float:
    base_moles = task.initial_base_moles
    acid_moles = 0.0
    if direction == "base":
        base_moles += TITRANT_M * dose_ml / 1000.0
    else:
        acid_moles += TITRANT_M * dose_ml / 1000.0
    return mixture_ph(
        task.component_concentrations_m,
        task.pka_values,
        task.initial_volume_ml,
        task.initial_volume_ml + dose_ml,
        base_moles,
        acid_moles,
    )


def required_volume(
    task: IndependentMixtureTask, maximum_ml: float = 30.0
) -> float | None:
    direction = "base" if task.target_ph > task.initial_ph else "acid"
    end_ph = response_after_dose(task, direction, maximum_ml)
    if direction == "base" and end_ph < task.target_ph:
        return None
    if direction == "acid" and end_ph > task.target_ph:
        return None
    lo, hi = 0.0, maximum_ml
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        value = response_after_dose(task, direction, mid)
        if (direction == "base" and value < task.target_ph) or (
            direction == "acid" and value > task.target_ph
        ):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def component_counts(count: int, rng: np.random.Generator) -> np.ndarray:
    first = int(round(COMPONENT_PROBABILITIES[0] * count))
    second = int(round(COMPONENT_PROBABILITIES[1] * count))
    third = count - first - second
    values = np.asarray([1] * first + [2] * second + [3] * third, dtype=int)
    rng.shuffle(values)
    return values


def sample_pkas(
    rng: np.random.Generator, component_count: int, family: str
) -> tuple[float, ...]:
    if component_count == 1:
        return (float(rng.uniform(1.8, 9.2)),)
    gap_range = (0.20, 0.70) if family == "overlapping" else (0.90, 2.00)
    gaps = rng.uniform(gap_range[0], gap_range[1], size=component_count - 1)
    total_gap = float(np.sum(gaps))
    first = float(rng.uniform(1.8, 9.5 - total_gap))
    return tuple(np.concatenate([[first], first + np.cumsum(gaps)]).tolist())


def sample_fractions(
    rng: np.random.Generator, component_count: int
) -> tuple[float, ...]:
    if component_count == 1:
        return (1.0,)
    if component_count == 2:
        first = float(rng.uniform(0.20, 0.80))
        return (first, 1.0 - first)
    for _ in range(1000):
        values = rng.dirichlet(np.full(component_count, 2.0))
        if float(np.min(values)) >= 0.10:
            return tuple(values.tolist())
    raise RuntimeError("Could not sample non-degenerate three-component fractions")


def generate_seed(seed: int, count: int) -> list[IndependentMixtureTask]:
    rng = np.random.default_rng(2_000_000 + seed)
    counts = component_counts(count, rng)
    directions = np.asarray(
        ["acid"] * (count // 2) + ["base"] * (count - count // 2),
        dtype=object,
    )
    rng.shuffle(directions)
    families = np.asarray(
        ["overlapping"] * (count // 2) + ["separated"] * (count - count // 2),
        dtype=object,
    )
    rng.shuffle(families)
    tasks: list[IndependentMixtureTask] = []
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("Could not generate enough reachable mixture tasks")
        index = len(tasks)
        component_count = int(counts[index])
        family = "single" if component_count == 1 else str(families[index])
        pkas = sample_pkas(rng, component_count, family)
        fractions = sample_fractions(rng, component_count)
        total_concentration = float(
            np.exp(rng.uniform(math.log(0.03), math.log(0.18)))
        )
        component_concentrations = tuple(
            total_concentration * value for value in fractions
        )
        initial_volume_ml = float(rng.uniform(8.0, 16.0))
        if rng.random() < 0.18:
            initial_equivalents = float(
                rng.uniform(0.02, 0.18)
                if rng.random() < 0.5
                else rng.uniform(0.82, 0.98)
            )
        else:
            initial_equivalents = float(rng.uniform(0.08, 0.92))
        total_acid_moles = total_concentration * initial_volume_ml / 1000.0
        initial_base_moles = initial_equivalents * total_acid_moles
        initial_ph_exact = mixture_ph(
            component_concentrations,
            pkas,
            initial_volume_ml,
            initial_volume_ml,
            initial_base_moles,
            0.0,
        )
        if not 1.5 <= initial_ph_exact <= 12.0:
            continue
        direction = str(directions[index])
        difficulty = str(
            rng.choice(["near", "medium", "far"], p=[0.30, 0.45, 0.25])
        )
        low, high = DIFFICULTY_RANGES[difficulty]
        delta = float(rng.uniform(low, high))
        target_ph_exact = (
            initial_ph_exact + delta if direction == "base" else initial_ph_exact - delta
        )
        if not 1.5 <= target_ph_exact <= 12.0:
            continue
        provisional = IndependentMixtureTask(
            seed=2_000_000 + seed,
            task_id=index + 1,
            split=f"independent_j123_seed_{seed}",
            acid_type=f"{component_count}_independent_monoprotic_component"
            + ("s" if component_count != 1 else ""),
            component_count=component_count,
            pka_values=pkas,
            analyte_conc_m=total_concentration,
            component_concentrations_m=component_concentrations,
            component_mole_fractions=fractions,
            initial_volume_ml=initial_volume_ml,
            initial_base_equivalents=initial_equivalents,
            initial_base_moles=initial_base_moles,
            initial_ph=float(np.round(initial_ph_exact, 3)),
            target_ph=float(np.round(target_ph_exact, 3)),
            direction=direction,
            difficulty=difficulty,
            pka_family=family,
            oracle_required_volume_ml=0.0,
        )
        oracle = required_volume(provisional)
        if oracle is None or oracle < 0.01:
            continue
        tasks.append(
            IndependentMixtureTask(
                **{
                    **asdict(provisional),
                    "oracle_required_volume_ml": float(oracle),
                }
            )
        )
    return tasks


def write_manifest(path: Path, tasks: list[IndependentMixtureTask]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(
                json.dumps(asdict(task), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-seed", type=int, default=3000)
    args = parser.parse_args()
    root = args.output.resolve()
    records = []
    for seed in SEEDS:
        tasks = generate_seed(seed, args.count_per_seed)
        path = root / "tasks" / f"seed_{seed}_tasks.jsonl"
        digest = write_manifest(path, tasks)
        count_by_j = {
            str(j): sum(task.component_count == j for task in tasks)
            for j in (1, 2, 3)
        }
        records.append(
            {
                "benchmark_seed": seed,
                "tasks": len(tasks),
                "sha256": digest,
                "acid_direction_tasks": sum(task.direction == "acid" for task in tasks),
                "base_direction_tasks": sum(task.direction == "base" for task in tasks),
                "component_count_tasks": count_by_j,
            }
        )
    protocol = {
        "name": "independent_monoprotic_J123_benchmark",
        "benchmark_seeds": list(SEEDS),
        "tasks_per_seed": args.count_per_seed,
        "truth_model": "J=1,2,3 independent monoprotic components",
        "component_count_probabilities": {
            "1": COMPONENT_PROBABILITIES[0],
            "2": COMPONENT_PROBABILITIES[1],
            "3": COMPONENT_PROBABILITIES[2],
        },
        "total_concentration_m": "log-uniform[0.03,0.18]",
        "component_fractions": (
            "J=1 fixed; J=2 uniform[0.20,0.80]; "
            "J=3 Dirichlet(2,2,2) restricted to min fraction >=0.10"
        ),
        "pka_families": (
            "single for J=1; otherwise overlapping consecutive gaps 0.20-0.70 "
            "or separated gaps 0.90-2.00"
        ),
        "initial_volume_ml": "uniform[8,16]",
        "direction": "exactly 50% acid and 50% base per seed",
        "difficulty": {"near": 0.30, "medium": 0.45, "far": 0.25},
        "titrant_m": TITRANT_M,
        "manifest_records": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "BENCHMARK_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()

