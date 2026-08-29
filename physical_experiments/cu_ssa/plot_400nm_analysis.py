"""Reproduce the publication Cu-SSA UV-Vis and 400 nm analysis figure.

The upper panel plots the measured spectra. The lower panel fits the raw
400 nm absorbance using sample-specific post-adjustment concentrations and a
1:1 Cu-SSA mass-balance model. Panel labels are deliberately separated from
descriptive titles so the figure follows conventional journal styling.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
JOB_DATA = BASE_DIR / "job_experiment_data.csv"
UVVIS_DATA = BASE_DIR / "uvvis_spectra.csv"
OUTPUT_PNG = BASE_DIR / "cu_ssa_400nm_analysis.png"
OUTPUT_PDF = BASE_DIR / "cu_ssa_400nm_analysis.pdf"

STOCK_CONCENTRATION_MOL_L = 0.0147
INITIAL_MIXTURE_VOLUME_ML = 11.02


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def complex_concentration(
    k_app: float,
    total_cu: np.ndarray,
    total_ssa: np.ndarray,
) -> np.ndarray:
    """Return the physical root of K[Cu][SSA] = [CuSSA]."""

    coefficient = k_app * (total_cu + total_ssa) + 1.0
    discriminant = coefficient**2 - 4.0 * k_app**2 * total_cu * total_ssa
    return (coefficient - np.sqrt(discriminant)) / (2.0 * k_app)


def linear_fit_at_log_k(
    log_k: float,
    total_cu: np.ndarray,
    total_ssa: np.ndarray,
    absorbance: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit background and response coefficients for a fixed log(Kapp)."""

    k_app = math.exp(log_k)
    complex_value = complex_concentration(k_app, total_cu, total_ssa)
    design = np.column_stack(
        (
            np.ones(absorbance.size),
            total_cu - complex_value,
            total_ssa - complex_value,
            complex_value,
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, absorbance, rcond=None)
    prediction = design @ coefficients
    residual_sum_squares = float(np.sum((absorbance - prediction) ** 2))
    return residual_sum_squares, coefficients, prediction


def fit_mass_balance(
    total_cu: np.ndarray,
    total_ssa: np.ndarray,
    absorbance: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimize log(Kapp), solving the linear optical terms analytically."""

    lower = math.log(1.0)
    upper = math.log(1.0e6)
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left_probe = upper - golden_ratio * (upper - lower)
    right_probe = lower + golden_ratio * (upper - lower)
    left_value = linear_fit_at_log_k(left_probe, total_cu, total_ssa, absorbance)[0]
    right_value = linear_fit_at_log_k(right_probe, total_cu, total_ssa, absorbance)[0]

    for _ in range(200):
        if left_value < right_value:
            upper = right_probe
            right_probe = left_probe
            right_value = left_value
            left_probe = upper - golden_ratio * (upper - lower)
            left_value = linear_fit_at_log_k(
                left_probe, total_cu, total_ssa, absorbance
            )[0]
        else:
            lower = left_probe
            left_probe = right_probe
            left_value = right_value
            right_probe = lower + golden_ratio * (upper - lower)
            right_value = linear_fit_at_log_k(
                right_probe, total_cu, total_ssa, absorbance
            )[0]

    log_k = (lower + upper) / 2.0
    _, coefficients, prediction = linear_fit_at_log_k(
        log_k, total_cu, total_ssa, absorbance
    )
    return math.exp(log_k), coefficients, prediction


def main() -> None:
    job_rows = read_csv(JOB_DATA)
    uvvis_rows = read_csv(UVVIS_DATA)

    x_ssa = np.array([float(row["x_ssa"]) for row in job_rows])
    final_volume_ml = np.array(
        [float(row["final_measurement_volume_ml"]) for row in job_rows]
    )
    raw_a400 = np.array(
        [float(row["raw_absorbance_at_400_nm"]) for row in job_rows]
    )

    total_cu = (
        STOCK_CONCENTRATION_MOL_L
        * INITIAL_MIXTURE_VOLUME_ML
        * (1.0 - x_ssa)
        / final_volume_ml
    )
    total_ssa = (
        STOCK_CONCENTRATION_MOL_L
        * INITIAL_MIXTURE_VOLUME_ML
        * x_ssa
        / final_volume_ml
    )
    k_app, optical_coefficients, fitted_a400 = fit_mass_balance(
        total_cu, total_ssa, raw_a400
    )

    residual_sum_squares = float(np.sum((raw_a400 - fitted_a400) ** 2))
    total_sum_squares = float(np.sum((raw_a400 - raw_a400.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    rmse = math.sqrt(residual_sum_squares / raw_a400.size)

    # Guard the published numerical result while allowing platform-level
    # floating-point variation in the nonlinear search.
    if not math.isclose(k_app, 911.84, rel_tol=2.0e-4):
        raise ValueError(f"Unexpected Kapp: {k_app}")
    if not math.isclose(r_squared, 0.991593, abs_tol=2.0e-6):
        raise ValueError(f"Unexpected R2: {r_squared}")
    if not math.isclose(rmse, 0.00497498, abs_tol=2.0e-7):
        raise ValueError(f"Unexpected RMSE: {rmse}")

    spectra: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in uvvis_rows:
        spectra[int(row["spectrum_number"])].append(
            (float(row["wavelength_nm"]), float(row["absorbance"]))
        )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.2, 6.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.022),
        height_ratios=(1.28, 1.0),
        left=0.10,
        right=0.91,
        bottom=0.09,
        top=0.965,
        hspace=0.30,
        wspace=0.05,
    )
    spectra_axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])
    fit_axis = figure.add_subplot(grid[1, 0])

    color_map = mpl.colormaps["viridis"]
    normalization = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for spectrum_number, composition in enumerate(x_ssa, start=1):
        spectrum = sorted(spectra[spectrum_number])
        wavelength = np.array([point[0] for point in spectrum])
        absorbance = np.array([point[1] for point in spectrum])
        spectra_axis.plot(
            wavelength,
            absorbance,
            color=color_map(normalization(composition)),
            linewidth=1.0,
        )

    spectra_axis.axvline(400.0, color="#666666", linestyle="--", linewidth=0.9)
    spectra_axis.text(407.0, 0.405, "400 nm", color="#555555", va="top")
    spectra_axis.set_xlim(350.0, 800.0)
    spectra_axis.set_ylim(0.0, 0.43)
    spectra_axis.set_xlabel("Wavelength (nm)")
    spectra_axis.set_ylabel("Absorbance")

    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=normalization, cmap=color_map),
        cax=colorbar_axis,
    )
    colorbar.set_label("SSA mole fraction, χ$_{\\mathrm{SSA}}$")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    fit_axis.scatter(
        x_ssa,
        raw_a400,
        c=x_ssa,
        cmap=color_map,
        norm=normalization,
        s=30,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label=r"Observed $A_{400}$",
    )
    fit_axis.plot(
        x_ssa,
        fitted_a400,
        color="#202020",
        marker="o",
        markersize=2.7,
        linewidth=1.35,
        label="1:1 mass-balance fit",
        zorder=2,
    )
    fit_axis.axvline(0.5, color="#777777", linestyle=":", linewidth=0.9)
    fit_axis.set_xlim(0.02, 0.98)
    fit_axis.set_ylim(0.045, 0.228)
    fit_axis.set_xlabel("SSA mole fraction, χ$_{\\mathrm{SSA}}$")
    fit_axis.set_ylabel("Raw absorbance at 400 nm")
    fit_axis.legend(frameon=False, loc="upper right", handlelength=2.4)

    for panel_label, axis in (("(a)", spectra_axis), ("(b)", fit_axis)):
        axis.text(
            -0.06,
            1.01,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        axis.grid(color="#d7d7d7", linewidth=0.45, alpha=0.45)
        axis.set_axisbelow(True)

    figure.savefig(OUTPUT_PNG, dpi=600, facecolor="white")
    figure.savefig(OUTPUT_PDF, facecolor="white")
    plt.close(figure)

    print(f"Kapp = {k_app:.6f} L mol^-1")
    print(f"log10(Kapp) = {math.log10(k_app):.6f}")
    print(f"R2 = {r_squared:.6f}")
    print(f"RMSE = {rmse:.8f}")
    print(
        "Optical coefficients [background, Cu, SSA, CuSSA] = "
        + np.array2string(optical_coefficients, precision=7)
    )
    print(f"Wrote {OUTPUT_PNG.name}")
    print(f"Wrote {OUTPUT_PDF.name}")


if __name__ == "__main__":
    main()
