from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
JOB_DATA = BASE_DIR / "job_experiment_data.csv"
PH_DATA = BASE_DIR / "ph_adjustment_data.csv"
UVVIS_DATA = BASE_DIR / "uvvis_spectra.csv"
OUTPUT_CSV = BASE_DIR / "job_analysis_reproduced.csv"
OUTPUT_SUMMARY = BASE_DIR / "job_analysis_reproduced_summary.txt"
OUTPUT_PLOT = BASE_DIR / "job_plot_reproduced_reported_method.png"
PHYSICAL_DOSE_MIN_ML = 0.01
PHYSICAL_DOSE_MAX_ML = 10.00


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    predictions = [slope * x + intercept for x in xs]
    residual_ss = sum((y - prediction) ** 2 for y, prediction in zip(ys, predictions))
    total_ss = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - residual_ss / total_ss
    return slope, intercept, r_squared


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance


def intersection_analysis(
    left_x: list[float],
    left_y: list[float],
    right_x: list[float],
    right_y: list[float],
    equimolar_response: float,
    concentration: float,
) -> dict[str, float]:
    left_slope, left_intercept, left_r_squared = linear_fit(left_x, left_y)
    right_slope, right_intercept, right_r_squared = linear_fit(right_x, right_y)
    intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
    intersection_response = left_slope * intersection_x + left_intercept
    alpha = (intersection_response - equimolar_response) / intersection_response
    k_app = (1.0 - alpha) / (concentration * alpha**2)
    return {
        "left_slope": left_slope,
        "left_intercept": left_intercept,
        "left_r_squared": left_r_squared,
        "right_slope": right_slope,
        "right_intercept": right_intercept,
        "right_r_squared": right_r_squared,
        "intersection_x": intersection_x,
        "intersection_response": intersection_response,
        "alpha": alpha,
        "k_app": k_app,
        "p_k_ex": math.log10(k_app),
    }


job_rows = read_csv(JOB_DATA)
ph_rows = read_csv(PH_DATA)
uvvis_rows = read_csv(UVVIS_DATA)

post_dose_volumes = [
    float(row["dose_volume_ml"])
    for row in ph_rows
    if row["phase"] == "post_dose"
]
if not post_dose_volumes:
    raise ValueError("No post-dose volumes found in ph_adjustment_data.csv")
if any(
    volume < PHYSICAL_DOSE_MIN_ML or volume > PHYSICAL_DOSE_MAX_ML
    for volume in post_dose_volumes
):
    raise ValueError(
        "A logged Cu-SSA dose is outside the physical pump range "
        f"{PHYSICAL_DOSE_MIN_ML:.2f}-{PHYSICAL_DOSE_MAX_ML:.2f} mL"
    )
largest_logged_dose_ml = max(post_dose_volumes)

uvvis_by_spectrum: dict[int, list[dict[str, str]]] = defaultdict(list)
for row in uvvis_rows:
    uvvis_by_spectrum[int(row["spectrum_number"])].append(row)

ph_rows_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in ph_rows:
    ph_rows_by_run[row["run_label"]].append(row)

x_values: list[float] = []
raw_values: list[float] = []
corrected_values: list[float] = []
output_rows: list[dict[str, str]] = []

for row in job_rows:
    spectrum_number = int(row["spectrum_number"])
    x_ssa = float(row["x_ssa"])
    raw_absorbance = float(row["raw_absorbance_at_400_nm"])
    initial_volume = float(row["initial_mixture_volume_ml"])
    final_volume = float(row["final_measurement_volume_ml"])
    final_ph = float(row["final_ph"])
    corrected_absorbance = raw_absorbance * final_volume / initial_volume

    nearest_spectrum_row = min(
        uvvis_by_spectrum[spectrum_number],
        key=lambda item: abs(float(item["wavelength_nm"]) - 400.0),
    )
    spectrum_absorbance = float(nearest_spectrum_row["absorbance"])
    spectrum_wavelength = float(nearest_spectrum_row["wavelength_nm"])
    if not close(raw_absorbance, spectrum_absorbance, 1e-12):
        raise ValueError(f"Spectrum {spectrum_number}: A400 does not match uvvis_spectra.csv")

    run_label = f"CuSSA-{spectrum_number:02d}"
    run_ph_rows = ph_rows_by_run.get(run_label, [])
    if not run_ph_rows:
        raise ValueError(f"{run_label}: no pH-adjustment records found")
    final_ph_row = max(run_ph_rows, key=lambda item: int(item["point_index"]))
    logged_final_ph = float(final_ph_row["measured_ph"])
    logged_final_volume = initial_volume + sum(
        float(item["dose_volume_ml"])
        for item in run_ph_rows
        if item["phase"] == "post_dose"
    )
    if not close(final_ph, logged_final_ph, 1e-12):
        raise ValueError(f"{run_label}: final pH does not match ph_adjustment_data.csv")
    if not close(final_volume, logged_final_volume, 1e-9):
        raise ValueError(f"{run_label}: final volume does not match the cumulative dose log")

    x_values.append(x_ssa)
    raw_values.append(raw_absorbance)
    corrected_values.append(corrected_absorbance)
    output_rows.append(
        {
            "spectrum_number": str(spectrum_number),
            "x_ssa": f"{x_ssa:.12f}",
            "wavelength_nm": f"{spectrum_wavelength:.7f}",
            "raw_absorbance_at_400_nm": f"{raw_absorbance:.9f}",
            "initial_volume_ml": f"{initial_volume:.2f}",
            "final_volume_ml": f"{final_volume:.2f}",
            "final_ph": f"{final_ph:.2f}",
            "dilution_corrected_absorbance": f"{corrected_absorbance:.12f}",
        }
    )

equimolar_response = corrected_values[5]
stock_concentration = 0.0147
equimolar_component_volume_ml = 5.51
initial_equimolar_volume_ml = 11.02
final_equimolar_volume_ml = float(job_rows[5]["final_measurement_volume_ml"])
initial_component_concentration = (
    stock_concentration * equimolar_component_volume_ml / initial_equimolar_volume_ml
)
final_component_concentration = (
    initial_component_concentration
    * initial_equimolar_volume_ml
    / final_equimolar_volume_ml
)

# The reported regressions include the theoretical zero-response endpoints and
# include the X = 0.500 observation in both branches.
reported_analysis = intersection_analysis(
    [0.0] + x_values[:6],
    [0.0] + corrected_values[:6],
    x_values[5:] + [1.0],
    corrected_values[5:] + [0.0],
    equimolar_response,
    final_component_concentration,
)
left_slope = reported_analysis["left_slope"]
left_intercept = reported_analysis["left_intercept"]
left_r_squared = reported_analysis["left_r_squared"]
right_slope = reported_analysis["right_slope"]
right_intercept = reported_analysis["right_intercept"]
right_r_squared = reported_analysis["right_r_squared"]
intersection_x = reported_analysis["intersection_x"]
intersection_response = reported_analysis["intersection_response"]
alpha = reported_analysis["alpha"]
k_app = reported_analysis["k_app"]
p_k_ex = reported_analysis["p_k_ex"]
literature_p_k_ex = 3.295

# These alternatives demonstrate why the regression convention must be disclosed.
fit_variants = [
    (
        "Synthetic endpoints included; X=0.500 included in both branches (reported)",
        reported_analysis,
    ),
    (
        "Measured points only; X=0.500 included in both branches",
        intersection_analysis(
            x_values[:6],
            corrected_values[:6],
            x_values[5:],
            corrected_values[5:],
            equimolar_response,
            final_component_concentration,
        ),
    ),
    (
        "Measured points only; X=0.500 excluded from both branches",
        intersection_analysis(
            x_values[:5],
            corrected_values[:5],
            x_values[6:],
            corrected_values[6:],
            equimolar_response,
            final_component_concentration,
        ),
    ),
    (
        "Synthetic endpoints included; X=0.500 excluded from both branches",
        intersection_analysis(
            [0.0] + x_values[:5],
            [0.0] + corrected_values[:5],
            x_values[6:] + [1.0],
            corrected_values[6:] + [0.0],
            equimolar_response,
            final_component_concentration,
        ),
    ),
]

expected_values = {
    "intersection_x": (intersection_x, 0.491635895),
    "intersection_response": (intersection_response, 0.449671585),
    "equimolar_response": (equimolar_response, 0.401126017),
    "alpha": (alpha, 0.107957828),
    "final_component_concentration": (final_component_concentration, 0.003890346),
    "k_app": (k_app, 19673.8294),
    "p_k_ex": (p_k_ex, 4.293888901),
}
for name, (actual, expected) in expected_values.items():
    tolerance = max(1e-8, abs(expected) * 1e-7)
    if not close(actual, expected, tolerance):
        raise ValueError(f"Unexpected {name}: {actual} (expected approximately {expected})")

with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)

summary_lines = [
    "Cu(II)-SSA Job analysis reproduction",
    "====================================",
    "",
    "Dilution correction: A_corr = A_raw * V_final / 11.02",
    f"Physical pump range checked: {PHYSICAL_DOSE_MIN_ML:.2f}-{PHYSICAL_DOSE_MAX_ML:.2f} mL per dosing command.",
    f"Largest logged Cu-SSA dose: {largest_logged_dose_ml:.2f} mL.",
    "Reported-fit convention: Cu-rich fit includes (0, 0); SSA-rich fit includes (1, 0).",
    "The X = 0.500 observation is included in both fitted branches.",
    "",
    f"Observed maximum X_SSA: {x_values[corrected_values.index(max(corrected_values))]:.6f}",
    f"Cu-rich slope/intercept/R2: {left_slope:.9f}, {left_intercept:.9f}, {left_r_squared:.9f}",
    f"SSA-rich slope/intercept/R2: {right_slope:.9f}, {right_intercept:.9f}, {right_r_squared:.9f}",
    f"Fitted intersection X_SSA: {intersection_x:.9f}",
    f"Intersection response A_fit: {intersection_response:.9f}",
    f"Equimolar corrected response: {equimolar_response:.9f}",
    f"Dissociated fraction alpha: {alpha:.9f}",
    f"Final equimolar component concentration: {final_component_concentration:.9f} mol L^-1",
    f"K_app: {k_app:.6f} L mol^-1",
    f"pK_ex = log10(K_app): {p_k_ex:.9f}",
    f"Difference from literature pK_ex = {literature_p_k_ex:.3f}: {p_k_ex - literature_p_k_ex:.9f}",
    "",
    "Fit-convention sensitivity:",
]
for label, analysis in fit_variants:
    summary_lines.extend(
        [
            label,
            "  X_intersection={:.9f}; A_fit={:.9f}; R2_left={:.6f}; R2_right={:.6f}; "
            "K_app={:.3f}; pK_ex={:.6f}".format(
                analysis["intersection_x"],
                analysis["intersection_response"],
                analysis["left_r_squared"],
                analysis["right_r_squared"],
                analysis["k_app"],
                analysis["p_k_ex"],
            ),
        ]
    )
summary_lines.extend(
    [
    "",
    "Document audit:",
    "The manuscript, SI, and response values are reproduced by the reported-fit convention above.",
    "The documents should explicitly state that the theoretical zero-response endpoints were included.",
    "The 1:1 composition conclusion is stable, but K_app is sensitive to the branch-fit convention.",
    ]
)
OUTPUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

try:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    figure.suptitle("Cu(II)-SSA continuous-variation reproduction", fontsize=16, weight="bold")

    axes[0].plot(x_values, raw_values, "o-", label="Raw A400")
    axes[0].plot(x_values, corrected_values, "s-", label="Dilution-corrected A400")
    axes[0].set_title("Dilution correction")
    axes[0].set_xlabel("SSA mole fraction, X_SSA")
    axes[0].set_ylabel("Absorbance response at 400 nm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].scatter(x_values, corrected_values, color="#2d7a59", label="Measured data", zorder=3)
    axes[1].scatter([0.0, 1.0], [0.0, 0.0], facecolors="none", edgecolors="#666666",
                    s=70, label="Theoretical endpoints", zorder=4)
    plot_left_x = [0.0, intersection_x]
    plot_right_x = [intersection_x, 1.0]
    axes[1].plot(plot_left_x, [left_slope * x + left_intercept for x in plot_left_x],
                 color="#3d78ad", label="Cu-rich fit")
    axes[1].plot(plot_right_x, [right_slope * x + right_intercept for x in plot_right_x],
                 color="#c85243", label="SSA-rich fit")
    axes[1].scatter([0.5], [equimolar_response], marker="D", s=80, color="#d89216",
                    label="Observed maximum", zorder=5)
    axes[1].scatter([intersection_x], [intersection_response], marker="X", s=110, color="#222222",
                    label="Fitted intersection", zorder=6)
    axes[1].axvline(0.5, color="#d89216", linestyle=":", linewidth=1)
    axes[1].axvline(intersection_x, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_title("Reported branch-fit convention")
    axes[1].set_xlabel("SSA mole fraction, X_SSA")
    axes[1].set_ylabel("Dilution-corrected A400")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.01, 0.48)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].text(
        0.03,
        0.97,
        "Cu-rich R2 = {:.4f}\nSSA-rich R2 = {:.4f}\nX_intersection = {:.4f}\nA_fit = {:.6f}".format(
            left_r_squared,
            right_r_squared,
            intersection_x,
            intersection_response,
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )
    figure.savefig(OUTPUT_PLOT, dpi=300)
except ImportError:
    print("matplotlib is not installed; numerical outputs were created without a plot.")

print("\n".join(summary_lines))
print(f"\nWrote {OUTPUT_CSV.name}")
print(f"Wrote {OUTPUT_SUMMARY.name}")
if OUTPUT_PLOT.exists():
    print(f"Wrote {OUTPUT_PLOT.name}")
