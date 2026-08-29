from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chemistry_model import SolutionState, solve_ph_scalar


@dataclass(frozen=True)
class ControlTask:
    seed: int
    task_id: int
    split: str
    acid_type: str
    pka_values: tuple[float, ...]
    analyte_conc_m: float
    initial_volume_ml: float
    initial_base_equivalents: float
    initial_base_moles: float
    initial_ph: float
    target_ph: float
    direction: str
    difficulty: str
    pka_family: str
    oracle_required_volume_ml: float


DIFFICULTY_RANGES = {
    "near": (0.20, 0.70),
    "medium": (0.70, 2.00),
    "far": (2.00, 4.00),
}


def _sample_pkas(rng: np.random.Generator, force_family: str | None = None):
    pair_count = int(rng.choice([1, 2, 3], p=[0.40, 0.35, 0.25]))
    close = force_family == "close" or (force_family is None and rng.random() < 0.15)
    if pair_count == 1:
        values = [float(rng.uniform(2.0, 8.5))]
        family = "single"
    else:
        first = float(rng.uniform(1.8, 5.0 if pair_count == 2 else 3.8))
        values = [first]
        for _ in range(pair_count - 1):
            gap = float(rng.uniform(0.20, 0.70) if close else rng.uniform(0.90, 2.50))
            values.append(values[-1] + gap)
        if values[-1] > 9.5:
            shift = values[-1] - 9.5
            values = [value - shift for value in values]
        family = "overlapping" if close else "separated"
    label = {1: "monoprotic", 2: "diprotic", 3: "triprotic"}[pair_count]
    return label, tuple(values), family


def _ph_for_equivalents(
    concentration_m: float,
    pka_values: tuple[float, ...],
    volume_ml: float,
    equivalents: float,
) -> tuple[float, float]:
    analyte_moles = concentration_m * volume_ml / 1000.0
    base_moles = equivalents * analyte_moles
    state = SolutionState(volume_ml, base_moles, 0.0)
    return solve_ph_scalar(concentration_m, pka_values, volume_ml, state), base_moles


def _response_after_dose(
    concentration_m: float,
    pka_values: tuple[float, ...],
    initial_volume_ml: float,
    initial_base_moles: float,
    direction: str,
    dose_ml: float,
    titrant_m: float = 0.1,
) -> float:
    if direction == "base":
        state = SolutionState(
            initial_volume_ml + dose_ml,
            initial_base_moles + titrant_m * dose_ml / 1000.0,
            0.0,
        )
    else:
        state = SolutionState(
            initial_volume_ml + dose_ml,
            initial_base_moles,
            titrant_m * dose_ml / 1000.0,
        )
    return solve_ph_scalar(concentration_m, pka_values, initial_volume_ml, state)


def _required_volume(
    concentration_m: float,
    pka_values: tuple[float, ...],
    initial_volume_ml: float,
    initial_base_moles: float,
    initial_ph: float,
    target_ph: float,
    maximum_ml: float = 30.0,
) -> float | None:
    direction = "base" if target_ph > initial_ph else "acid"
    end_ph = _response_after_dose(
        concentration_m,
        pka_values,
        initial_volume_ml,
        initial_base_moles,
        direction,
        maximum_ml,
    )
    if direction == "base" and end_ph < target_ph:
        return None
    if direction == "acid" and end_ph > target_ph:
        return None
    lo, hi = 0.0, maximum_ml
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        value = _response_after_dose(
            concentration_m,
            pka_values,
            initial_volume_ml,
            initial_base_moles,
            direction,
            mid,
        )
        if (direction == "base" and value < target_ph) or (
            direction == "acid" and value > target_ph
        ):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_tasks(
    seed: int,
    count: int,
    split: str,
    distribution: str = "nominal",
    task_id_offset: int = 0,
) -> list[ControlTask]:
    rng = np.random.default_rng(seed)
    tasks: list[ControlTask] = []
    requested_directions = np.asarray(
        ["acid"] * (count // 2) + ["base"] * (count - count // 2),
        dtype=object,
    )
    rng.shuffle(requested_directions)
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("Could not generate enough reachable tasks")
        force_family = "close" if distribution == "close_pka" else None
        acid_type, pka_values, pka_family = _sample_pkas(rng, force_family)
        if distribution == "wide_concentration":
            concentration_m = float(np.exp(rng.uniform(np.log(0.015), np.log(0.25))))
        else:
            concentration_m = float(np.exp(rng.uniform(np.log(0.03), np.log(0.18))))
        initial_volume_ml = float(rng.uniform(8.0, 16.0))
        pair_count = len(pka_values)
        if rng.random() < 0.18:
            if rng.random() < 0.5:
                initial_equivalents = float(rng.uniform(0.02, min(0.18, pair_count - 0.02)))
            else:
                initial_equivalents = float(rng.uniform(max(0.02, pair_count - 0.18), pair_count - 0.02))
        else:
            initial_equivalents = float(rng.uniform(0.08, max(0.081, pair_count - 0.08)))
        initial_ph, initial_base_moles = _ph_for_equivalents(
            concentration_m,
            pka_values,
            initial_volume_ml,
            initial_equivalents,
        )
        if not 1.5 <= initial_ph <= 12.0:
            continue

        # Keep the accepted task set exactly direction-balanced. If a sampled
        # chemistry is not reachable, resample that chemistry for the same slot.
        direction = str(requested_directions[len(tasks)])
        difficulty = str(rng.choice(["near", "medium", "far"], p=[0.30, 0.45, 0.25]))
        low, high = DIFFICULTY_RANGES[difficulty]
        delta = float(rng.uniform(low, high))
        target_ph = initial_ph + delta if direction == "base" else initial_ph - delta
        if not 1.5 <= target_ph <= 12.0:
            continue
        required = _required_volume(
            concentration_m,
            pka_values,
            initial_volume_ml,
            initial_base_moles,
            initial_ph,
            target_ph,
        )
        if required is None or required < 0.01:
            continue
        task_id = task_id_offset + len(tasks) + 1
        tasks.append(
            ControlTask(
                seed=seed,
                task_id=task_id,
                split=split,
                acid_type=acid_type,
                pka_values=tuple(float(value) for value in pka_values),
                analyte_conc_m=concentration_m,
                initial_volume_ml=initial_volume_ml,
                initial_base_equivalents=initial_equivalents,
                initial_base_moles=initial_base_moles,
                initial_ph=float(np.round(initial_ph, 3)),
                target_ph=float(np.round(target_ph, 3)),
                direction=direction,
                difficulty=difficulty,
                pka_family=pka_family,
                oracle_required_volume_ml=float(required),
            )
        )
    return tasks


def save_tasks(path: Path, tasks: list[ControlTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> list[ControlTask]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["pka_values"] = tuple(payload["pka_values"])
            tasks.append(ControlTask(**payload))
    return tasks
