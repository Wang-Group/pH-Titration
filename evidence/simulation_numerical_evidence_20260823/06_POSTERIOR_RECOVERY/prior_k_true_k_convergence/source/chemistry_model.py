from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


WATER_KW = 1e-14
PH_SOLVER_ITERATIONS = 60


@dataclass(frozen=True)
class SolutionState:
    total_volume_ml: float
    base_moles: float
    acid_moles: float


def acid_charge_factor(ph: np.ndarray | float, pka_values: np.ndarray | Sequence[float]):
    """Mean negative charge per analytical acid molecule."""
    ph_array = np.asarray(ph, dtype=float)
    pka = np.asarray(pka_values, dtype=float)
    hydrogen = np.power(10.0, -ph_array)
    kas = np.power(10.0, -np.clip(pka, -100.0, 100.0))

    if pka.ndim == 1:
        denominator = np.ones_like(hydrogen, dtype=float)
        cumulative = np.ones_like(hydrogen, dtype=float)
        numerator = np.zeros_like(hydrogen, dtype=float)
        for index, ka in enumerate(kas):
            cumulative = cumulative * ka / hydrogen
            denominator = denominator + cumulative
            numerator = numerator + (index + 1) * cumulative
        return numerator / denominator

    if pka.ndim != 2:
        raise ValueError("pKa input must be one- or two-dimensional")
    hydrogen = np.broadcast_to(hydrogen, (pka.shape[0],))
    denominator = np.ones(pka.shape[0], dtype=float)
    cumulative = np.ones(pka.shape[0], dtype=float)
    numerator = np.zeros(pka.shape[0], dtype=float)
    for index in range(pka.shape[1]):
        cumulative = cumulative * kas[:, index] / hydrogen
        denominator = denominator + cumulative
        numerator = numerator + (index + 1) * cumulative
    return numerator / denominator


def solve_ph_scalar(
    concentration_m: float,
    pka_values: Sequence[float],
    initial_volume_ml: float,
    state: SolutionState,
    iterations: int = PH_SOLVER_ITERATIONS,
) -> float:
    total_volume_l = state.total_volume_ml / 1000.0
    analyte_moles = initial_volume_ml / 1000.0 * concentration_m
    c_analyte = analyte_moles / total_volume_l
    c_na = state.base_moles / total_volume_l
    c_hcl = state.acid_moles / total_volume_l

    def balance(ph: float) -> float:
        hydrogen = 10.0 ** (-ph)
        hydroxide = WATER_KW / hydrogen
        charge = c_analyte * float(acid_charge_factor(ph, pka_values))
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


def solve_ph_particles(
    concentrations_m: np.ndarray,
    pka_matrix: np.ndarray,
    initial_volume_ml: float,
    state: SolutionState,
    iterations: int = PH_SOLVER_ITERATIONS,
) -> np.ndarray:
    concentrations = np.asarray(concentrations_m, dtype=float)
    pkas = np.asarray(pka_matrix, dtype=float)
    if pkas.ndim != 2 or concentrations.shape != (pkas.shape[0],):
        raise ValueError("Particle concentrations and pKa matrix have incompatible shapes")

    total_volume_l = state.total_volume_ml / 1000.0
    c_analyte = initial_volume_ml / 1000.0 * concentrations / total_volume_l
    c_na = state.base_moles / total_volume_l
    c_hcl = state.acid_moles / total_volume_l

    lo = np.zeros(pkas.shape[0], dtype=float)
    hi = np.full(pkas.shape[0], 14.0, dtype=float)

    def balance(ph: np.ndarray) -> np.ndarray:
        hydrogen = np.power(10.0, -ph)
        hydroxide = WATER_KW / hydrogen
        charge = c_analyte * acid_charge_factor(ph, pkas)
        return hydrogen + c_na - hydroxide - c_hcl - charge

    f_lo = balance(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        left = f_lo * f_mid < 0.0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi)


def solve_ph_grid(
    concentration_m: float,
    pka_values: Sequence[float],
    initial_volume_ml: float,
    total_volume_ml: np.ndarray,
    base_moles: np.ndarray,
    acid_moles: np.ndarray,
    iterations: int = PH_SOLVER_ITERATIONS,
) -> np.ndarray:
    volumes = np.asarray(total_volume_ml, dtype=float)
    base = np.asarray(base_moles, dtype=float)
    acid = np.asarray(acid_moles, dtype=float)
    volumes, base, acid = np.broadcast_arrays(volumes, base, acid)
    total_volume_l = volumes / 1000.0
    c_analyte = initial_volume_ml / 1000.0 * float(concentration_m) / total_volume_l
    c_na = base / total_volume_l
    c_hcl = acid / total_volume_l
    pka = np.asarray(pka_values, dtype=float)

    def balance(ph: np.ndarray) -> np.ndarray:
        hydrogen = np.power(10.0, -ph)
        hydroxide = WATER_KW / hydrogen
        charge = c_analyte * acid_charge_factor(ph, pka)
        return hydrogen + c_na - hydroxide - c_hcl - charge

    lo = np.zeros_like(volumes, dtype=float)
    hi = np.full_like(volumes, 14.0, dtype=float)
    f_lo = balance(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        left = f_lo * f_mid < 0.0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi)


def response_curve(
    concentration_m: float,
    pka_values: Sequence[float],
    initial_volume_ml: float,
    state: SolutionState,
    signed_probe_ml: np.ndarray,
    titrant_concentration_m: float = 0.1,
) -> np.ndarray:
    signed = np.asarray(signed_probe_ml, dtype=float)
    base_ml = np.maximum(0.0, signed)
    acid_ml = np.maximum(0.0, -signed)
    return solve_ph_grid(
        concentration_m,
        pka_values,
        initial_volume_ml,
        state.total_volume_ml + base_ml + acid_ml,
        state.base_moles + titrant_concentration_m * base_ml / 1000.0,
        state.acid_moles + titrant_concentration_m * acid_ml / 1000.0,
    )


def full_base_curve(
    concentration_m: float,
    pka_values: Sequence[float],
    initial_volume_ml: float,
    initial_ph: float,
    base_grid_ml: np.ndarray,
    titrant_concentration_m: float = 0.1,
) -> np.ndarray:
    volumes = np.asarray(base_grid_ml, dtype=float)
    raw_array = solve_ph_grid(
        concentration_m,
        pka_values,
        initial_volume_ml,
        initial_volume_ml + volumes,
        titrant_concentration_m * volumes / 1000.0,
        np.zeros_like(volumes),
    )
    return float(initial_ph) + raw_array - raw_array[0]
