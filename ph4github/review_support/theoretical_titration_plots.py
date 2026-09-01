from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KW = 1e-14
TITRANT_CONC = 0.1
TARGET_PH = 6.0
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "reviewer_response" / "theoretical_titration"
PHBA_EFFECTIVE_CONC = 0.0362
TPA_EFFECTIVE_CONC = 0.000102


@dataclass(frozen=True)
class AcidComponent:
    name: str
    initial_moles: float
    pKas: tuple[float, ...]


@dataclass(frozen=True)
class MixtureDefinition:
    key: str
    title: str
    description: str
    initial_volume_ml: float
    components: tuple[AcidComponent, ...]


def acid_charge_factor(pH: float, pKas: tuple[float, ...]) -> float:
    H = 10 ** (-pH)
    Kas = [10 ** (-pKa) for pKa in pKas]
    order = len(Kas)

    coeffs = [1.0]
    running = 1.0
    for Ka in Kas:
        running *= Ka
        coeffs.append(running)

    terms = [coeffs[i] * (H ** (order - i)) for i in range(order + 1)]
    denominator = sum(terms)
    return sum(i * terms[i] for i in range(order + 1)) / denominator


def charge_balance(pH: float, c_na: float, c_hcl: float, components: tuple[AcidComponent, ...], total_volume_l: float) -> float:
    H = 10 ** (-pH)
    OH = KW / H
    acid_charge = 0.0
    for component in components:
        formal_conc = component.initial_moles / total_volume_l
        acid_charge += formal_conc * acid_charge_factor(pH, component.pKas)
    return H + c_na - OH - c_hcl - acid_charge


def solve_mixture_ph(base_ml: float, acid_ml: float, mixture: MixtureDefinition) -> float:
    total_volume_l = (mixture.initial_volume_ml + base_ml + acid_ml) / 1000.0
    c_na = (base_ml * TITRANT_CONC / 1000.0) / total_volume_l
    c_hcl = (acid_ml * TITRANT_CONC / 1000.0) / total_volume_l

    lo, hi = 0.0, 14.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if charge_balance(mid, c_na, c_hcl, mixture.components, total_volume_l) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def build_curve(mixture: MixtureDefinition, net_titrant_grid_ml: np.ndarray) -> np.ndarray:
    ph_values = []
    for net_ml in net_titrant_grid_ml:
        if net_ml >= 0:
            ph_values.append(solve_mixture_ph(base_ml=float(net_ml), acid_ml=0.0, mixture=mixture))
        else:
            ph_values.append(solve_mixture_ph(base_ml=0.0, acid_ml=float(-net_ml), mixture=mixture))
    return np.array(ph_values)


def interpolate_crossing(x: np.ndarray, y: np.ndarray, target: float) -> tuple[float | None, float | None]:
    for idx in range(len(x) - 1):
        y0 = y[idx] - target
        y1 = y[idx + 1] - target
        if y0 == 0:
            return float(x[idx]), float(idx)
        if y0 * y1 <= 0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            return float(x[idx] + frac * (x[idx + 1] - x[idx])), float(idx + frac)
    return None, None


def local_slope(x: np.ndarray, y: np.ndarray, x0: float | None) -> float | None:
    if x0 is None:
        return None
    idx = int(np.searchsorted(x, x0))
    idx = min(max(idx, 1), len(x) - 2)
    dx = x[idx + 1] - x[idx - 1]
    dy = y[idx + 1] - y[idx - 1]
    return float(dy / dx)


def write_curve_csv(output_csv: Path, grid: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mixture", "net_titrant_ml", "pH"])
        for key, values in curves.items():
            for net_ml, ph in zip(grid, values):
                writer.writerow([key, f"{net_ml:.4f}", f"{ph:.6f}"])


def write_summary_json(output_json: Path, summary: dict[str, dict[str, float | str | None]]) -> None:
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_curves(
    mixtures: tuple[MixtureDefinition, ...],
    grid: np.ndarray,
    curves: dict[str, np.ndarray],
    summary: dict[str, dict[str, float | str | None]],
    output_svg: Path,
    output_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    colors = ["#1b4965", "#5fa8d3", "#ca6702", "#6a994e"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mixture, color in zip(axes, mixtures, colors):
        y = curves[mixture.key]
        ax.plot(grid, y, color=color, linewidth=2.2)
        ax.axhspan(TARGET_PH - 0.1, TARGET_PH + 0.1, color="#d8e2dc", alpha=0.7)
        ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle="--")
        ax.scatter([0.0], [y[np.searchsorted(grid, 0.0)]], color=color, s=28, zorder=4)

        crossing = summary[mixture.key]["target_crossing_ml"]
        if crossing is not None:
            ax.axvline(float(crossing), color=color, linewidth=1.0, linestyle=":")
            ax.scatter([float(crossing)], [TARGET_PH], color=color, edgecolor="white", s=42, zorder=5)

        ax.set_title(f"{mixture.title}\n{mixture.description}")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0.5, 12.5)
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("pH")
    axes[2].set_ylabel("pH")
    axes[2].set_xlabel("Net titrant volume (mL)\nnegative = HCl, positive = NaOH")
    axes[3].set_xlabel("Net titrant volume (mL)\nnegative = HCl, positive = NaOH")
    fig.suptitle(
        "Representative theoretical titration curves for Figure 2 mixed-acid systems\n"
        "using literature/database pKa values and solubility-limited effective concentrations",
        y=0.98,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg")
    fig.savefig(output_png, format="png", dpi=220)
    plt.close(fig)


def main() -> None:
    # Representative literature/database values used for qualitative visualization.
    # pHBA: pKa1 = 4.54, phenolic pKa2 = 9.32
    # TPA: pKa1 = 3.54, pKa2 = 4.46
    # Acetic acid: pKa ~ 4.76
    # SSA: sulfonic group treated as strongly acidic (pKa ~ -2.8), carboxyl ~ 2.49, phenol ~ 11.5
    #
    # Solubility handling:
    # The SI states that pHBA and TPA stock solutions were prepared as nominal 0.05 M mixtures,
    # but the clear supernatant was used because of limited solubility. To reflect the liquid
    # phase actually transferred to the flask, the theoretical curves below use approximate
    # solubility-limited effective concentrations for pHBA and TPA rather than the nominal
    # weighed concentrations.
    mixtures = (
        MixtureDefinition(
            key="mixture_1",
            title="Mixture 1",
            description=(
                "pHBA saturated-solution clear supernatant (6 mL)\n"
                "+ TPA saturated-solution clear supernatant (6 mL)"
            ),
            initial_volume_ml=12.0,
            components=(
                AcidComponent("pHBA", PHBA_EFFECTIVE_CONC * 0.006, (4.54, 9.32)),
                AcidComponent("TPA", TPA_EFFECTIVE_CONC * 0.006, (3.54, 4.46)),
            ),
        ),
        MixtureDefinition(
            key="mixture_2",
            title="Mixture 2",
            description=(
                "pHBA saturated-solution clear supernatant (6 mL)\n"
                "+ 0.10 M acetic acid (6 mL)"
            ),
            initial_volume_ml=12.0,
            components=(
                AcidComponent("pHBA", PHBA_EFFECTIVE_CONC * 0.006, (4.54, 9.32)),
                AcidComponent("Acetic acid", 0.10 * 0.006, (4.76,)),
            ),
        ),
        MixtureDefinition(
            key="mixture_3",
            title="Mixture 3",
            description=(
                "pHBA saturated-solution clear supernatant (6 mL)\n"
                "+ 14.7 mM SSA (6 mL)"
            ),
            initial_volume_ml=12.0,
            components=(
                AcidComponent("pHBA", PHBA_EFFECTIVE_CONC * 0.006, (4.54, 9.32)),
                AcidComponent("SSA", 0.0147 * 0.006, (-2.80, 2.49, 11.50)),
            ),
        ),
        MixtureDefinition(
            key="mixture_4",
            title="Mixture 4",
            description=(
                "TPA saturated-solution clear supernatant (6 mL)\n"
                "+ 0.10 M acetic acid (6 mL)"
            ),
            initial_volume_ml=12.0,
            components=(
                AcidComponent("TPA", TPA_EFFECTIVE_CONC * 0.006, (3.54, 4.46)),
                AcidComponent("Acetic acid", 0.10 * 0.006, (4.76,)),
            ),
        ),
    )

    grid = np.linspace(-2.0, 16.0, 721)
    curves = {mixture.key: build_curve(mixture, grid) for mixture in mixtures}

    summary: dict[str, dict[str, float | str | None]] = {}
    for mixture in mixtures:
        values = curves[mixture.key]
        crossing_ml, _ = interpolate_crossing(grid, values, TARGET_PH)
        summary[mixture.key] = {
            "title": mixture.title,
            "description": mixture.description,
            "model_basis": "solubility-limited effective dissolved concentrations for pHBA and TPA; nominal concentrations for acetic acid and SSA",
            "initial_pH": float(values[np.searchsorted(grid, 0.0)]),
            "target_crossing_ml": crossing_ml,
            "slope_at_target_dpH_per_mL": local_slope(grid, values, crossing_ml),
            "max_pH_on_grid": float(values.max()),
            "min_pH_on_grid": float(values.min()),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_curve_csv(OUTPUT_DIR / "theoretical_titration_curves.csv", grid, curves)
    write_summary_json(OUTPUT_DIR / "theoretical_titration_summary.json", summary)
    plot_curves(
        mixtures,
        grid,
        curves,
        summary,
        OUTPUT_DIR / "theoretical_titration_curves.svg",
        OUTPUT_DIR / "theoretical_titration_curves.png",
    )

    print(f"Wrote curve data to: {OUTPUT_DIR / 'theoretical_titration_curves.csv'}")
    print(f"Wrote summary to: {OUTPUT_DIR / 'theoretical_titration_summary.json'}")
    print(f"Wrote figure to: {OUTPUT_DIR / 'theoretical_titration_curves.svg'}")
    print(f"Wrote figure to: {OUTPUT_DIR / 'theoretical_titration_curves.png'}")


if __name__ == "__main__":
    main()
