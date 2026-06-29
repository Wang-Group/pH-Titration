from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "wastewater": {
        "files": [
            "data/all_data/wastewater/WasteWater1.csv",
            "data/all_data/wastewater/WasteWater2.csv",
            "data/all_data/wastewater/WasteWater3.csv",
        ],
        "target_ph": 7.0,
        "output_name": "wastewater_pH7.svg",
    }
}


def load_adjustment_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if "actual_volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["actual_volume"], errors="coerce").fillna(0.0)
    elif "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    else:
        raise KeyError(f"Missing volume column in {path}")

    if "reagent" not in frame.columns:
        raise KeyError(f"Missing reagent column in {path}")
    frame["reagent"] = frame["reagent"].astype(str).str.strip().str.lower()

    for required_column in ("pH_before", "pH_after"):
        if required_column not in frame.columns:
            raise KeyError(f"Missing {required_column} column in {path}")

    if "step" not in frame.columns or frame["step"].isnull().any():
        frame.insert(0, "step", np.arange(1, len(frame) + 1, dtype=int))
    frame["step"] = frame["step"].astype(int)

    step_zero = {
        "step": 0,
        "reagent": "none",
        "volume": 0.0,
        "pH_before": frame.loc[0, "pH_before"],
        "pH_after": frame.loc[0, "pH_before"],
        "pH_change": 0.0,
    }
    return pd.concat([pd.DataFrame([step_zero]), frame], ignore_index=True)


def resolve_dataset_files(dataset_name: str) -> tuple[list[Path], float, Path]:
    config = DATASETS[dataset_name]
    files = [ROOT / relative_path for relative_path in config["files"]]
    output_path = ROOT / "output" / "repro" / config["output_name"]
    return files, float(config["target_ph"]), output_path


def plot_ph_files(files: list[Path], target_ph: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [load_adjustment_table(path) for path in files]

    acid_cmap = LinearSegmentedColormap.from_list(
        "repro_acid",
        plt.get_cmap("YlOrRd")(np.linspace(0.25, 1.0, 256)),
    )
    base_cmap = LinearSegmentedColormap.from_list(
        "repro_base",
        plt.get_cmap("GnBu")(np.linspace(0.25, 1.0, 256)),
    )

    acid_values = pd.concat([frame.loc[frame["reagent"] == "acid", "volume"] for frame in frames], ignore_index=True)
    base_values = pd.concat([frame.loc[frame["reagent"] == "base", "volume"] for frame in frames], ignore_index=True)
    acid_norm = mcolors.Normalize(vmin=acid_values.min() if not acid_values.empty else 0.0, vmax=acid_values.max() if not acid_values.empty else 1.0)
    base_norm = mcolors.Normalize(vmin=base_values.min() if not base_values.empty else 0.0, vmax=base_values.max() if not base_values.empty else 1.0)

    if acid_norm.vmin == acid_norm.vmax:
        acid_norm = mcolors.Normalize(vmin=acid_norm.vmin, vmax=acid_norm.vmin + 1e-9)
    if base_norm.vmin == base_norm.vmax:
        base_norm = mcolors.Normalize(vmin=base_norm.vmin, vmax=base_norm.vmin + 1e-9)

    figure, axes = plt.subplots(1, len(frames), figsize=(5.5 * len(frames), 4.5), squeeze=False, sharey=False)
    axes_flat = axes.flatten()

    for axis, frame, path in zip(axes_flat, frames, files):
        steps = frame["step"]
        volumes = frame["volume"]
        for step, volume, reagent in zip(steps, volumes, frame["reagent"]):
            if reagent == "acid":
                color = acid_cmap(acid_norm(volume))
                axis.bar(step, volume, width=0.6, color=color, edgecolor="none", zorder=2)
            elif reagent == "base":
                color = base_cmap(base_norm(volume))
                axis.bar(step, volume, width=0.6, color=color, edgecolor="none", zorder=2)
            else:
                axis.bar(step, volume, width=0.6, color="#cccccc", edgecolor="none", zorder=2)

        twin_axis = axis.twinx()
        twin_axis.plot(steps, frame["pH_after"], marker="x", linestyle="None", color="black", zorder=3)
        twin_axis.axhline(target_ph, color="gray", linestyle="--", linewidth=1.0, zorder=1)

        axis.set_xlabel("Step")
        axis.set_ylabel("Volume (mL)")
        twin_axis.set_ylabel("pH")
        axis.set_title(path.stem)
        axis.set_xticks(list(range(0, int(steps.max()) + 1, 2)) or [0])

    figure.suptitle(f"Bundled wastewater adjustment examples (target pH {target_ph:.1f})", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a representative plot from bundled repository files.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="wastewater",
        help="Named bundled dataset to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to output/repro/<dataset>.svg.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files, target_ph, default_output = resolve_dataset_files(args.dataset)
    output_path = args.output if args.output is not None else default_output
    plot_ph_files(files, target_ph, output_path)
    print(f"Generated plot: {output_path}")


if __name__ == "__main__":
    main()
