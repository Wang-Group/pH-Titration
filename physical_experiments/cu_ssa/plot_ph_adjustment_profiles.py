"""Reproduce the stepwise Cu-SSA pH-adjustment profiles (Figure 6/S6)."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "ph_adjustment_data.csv"
OUTPUT_PNG = BASE_DIR / "cu_ssa_ph_adjustment_profiles.png"
OUTPUT_PDF = BASE_DIR / "cu_ssa_ph_adjustment_profiles.pdf"

ACID_COLOR = "#cf3f55"
BASE_COLOR = "#2a7fb8"
PH_COLOR = "#30343b"
TARGET_COLOR = "#4e936c"


def read_runs() -> OrderedDict[str, list[dict[str, str]]]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    runs: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        runs.setdefault(row["run_label"], []).append(row)
    for run_rows in runs.values():
        run_rows.sort(key=lambda row: int(row["point_index"]))

    if len(rows) != 92 or len(runs) != 11:
        raise ValueError(f"Expected 92 observations across 11 runs, got {len(rows)} and {len(runs)}")
    final_ph = [float(run_rows[-1]["measured_ph"]) for run_rows in runs.values()]
    if min(final_ph) < 4.17 or max(final_ph) > 4.22:
        raise ValueError(f"Unexpected final-pH range: {min(final_ph):.3f}-{max(final_ph):.3f}")
    return runs


def blend_with_white(hex_color: str, strength: float) -> tuple[float, float, float]:
    base = mpl.colors.to_rgb(hex_color)
    return tuple(1.0 - strength * (1.0 - channel) for channel in base)


def dose_color(reagent: str, dose_ml: float, maximum_dose_ml: float) -> tuple[float, float, float]:
    base = ACID_COLOR if reagent == "acid" else BASE_COLOR
    relative = math.sqrt(max(dose_ml, 0.0) / maximum_dose_ml)
    return blend_with_white(base, 0.38 + 0.62 * relative)


def main() -> None:
    runs = read_runs()
    all_post_doses = [
        float(row["dose_volume_ml"])
        for run_rows in runs.values()
        for row in run_rows
        if row["phase"] == "post_dose"
    ]
    maximum_dose_ml = max(all_post_doses)
    if not math.isclose(min(all_post_doses), 0.01) or not math.isclose(maximum_dose_ml, 6.25):
        raise ValueError("Unexpected observed dose range")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(4, 3, figsize=(6.701, 7.915), squeeze=False)
    figure.subplots_adjust(
        left=0.10,
        right=0.92,
        bottom=0.12,
        top=0.855,
        hspace=0.49,
        wspace=0.20,
    )

    panel_letters = "abcdefghijk"
    for panel_index, ((run_label, run_rows), axis) in enumerate(
        zip(runs.items(), axes.flat)
    ):
        steps = [int(row["point_index"]) for row in run_rows]
        doses = [float(row["dose_volume_ml"]) for row in run_rows]
        ph_values = [float(row["measured_ph"]) for row in run_rows]
        reagents = [row["reagent"] for row in run_rows]
        target_ph = float(run_rows[0]["target_ph"])
        x_ssa = float(run_rows[0]["system"].rsplit("=", 1)[1])

        colors = [
            dose_color(reagent, dose, maximum_dose_ml) if reagent else "#ffffff"
            for reagent, dose in zip(reagents, doses)
        ]
        axis.bar(
            steps,
            doses,
            width=0.72,
            color=colors,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        axis.set_ylim(0.0, 7.0)
        axis.set_yticks((0.0, 2.0, 4.0, 6.0))
        axis.set_xlim(-0.45, max(steps) + 0.45)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.grid(axis="y", color="#d6dce1", linewidth=0.45, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        ph_axis = axis.twinx()
        ph_axis.plot(
            steps,
            ph_values,
            color=PH_COLOR,
            linewidth=1.15,
            marker="o",
            markersize=3.1,
            markerfacecolor="white",
            markeredgecolor=PH_COLOR,
            markeredgewidth=0.75,
            zorder=4,
        )
        ph_axis.axhline(
            target_ph,
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=0.9,
            zorder=3,
        )
        # Leave clear headroom above the largest measured pH values so the
        # trace and open-circle marker never intrude into the panel title.
        ph_axis.set_ylim(1.0, 12.0)
        ph_axis.set_yticks((2.5, 5.0, 7.5, 10.0))
        ph_axis.spines["top"].set_visible(False)

        row_index, column_index = divmod(panel_index, 3)
        if column_index == 0:
            axis.set_ylabel("Added volume (mL)")
        else:
            axis.tick_params(labelleft=False)
        # The final row has only two populated panels, so panel (k) is the
        # rightmost axis there and must carry that row's visible pH scale.
        if column_index == 2 or panel_index == len(runs) - 1:
            ph_axis.set_ylabel("pH")
        else:
            ph_axis.tick_params(right=False, labelright=False)
            ph_axis.spines["right"].set_visible(False)

        axis.text(
            0.0,
            1.035,
            f"({panel_letters[panel_index]})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
        axis.text(
            0.12,
            1.035,
            f"χ$_{{\\mathrm{{SSA}}}}$ = {x_ssa:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
        )

    axes[3, 2].axis("off")
    figure.supxlabel("Sequential dosing step", y=0.078, fontsize=10)

    legend_handles = [
        Patch(facecolor=blend_with_white(ACID_COLOR, 0.88), label="Acid added"),
        Patch(facecolor=blend_with_white(BASE_COLOR, 0.88), label="Base added"),
        Line2D(
            [0],
            [0],
            color=PH_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=PH_COLOR,
            linewidth=1.15,
            markersize=4.0,
            label="Measured pH after each step",
        ),
        Line2D(
            [0],
            [0],
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=0.9,
            label="Target pH",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=2.2,
        columnspacing=1.5,
    )

    figure.savefig(OUTPUT_PNG, dpi=600, facecolor="white")
    figure.savefig(OUTPUT_PDF, facecolor="white")
    plt.close(figure)

    print(f"Runs: {len(runs)}")
    print(f"Observed dose range: {min(all_post_doses):.2f}-{maximum_dose_ml:.2f} mL")
    print(f"Wrote {OUTPUT_PNG.name}")
    print(f"Wrote {OUTPUT_PDF.name}")


if __name__ == "__main__":
    main()
