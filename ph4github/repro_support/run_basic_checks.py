from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "main_code3.ipynb",
    "plot.ipynb",
    "experiment_summary.csv",
    "data/bayesian.txt",
    "data/m_network.txt",
    "data/PIDexperiment.txt",
    "data/reinforced_network.txt",
    "data/all_data/milk/milk.xlsx",
    "data/all_data/mixed_acid/1-1.xlsx",
    "data/all_data/SSA/SSA1.csv",
    "data/all_data/wastewater/WasteWater1.csv",
]

TABLE_SAMPLES = [
    "experiment_summary.csv",
    "data/all_data/wastewater/WasteWater1.csv",
    "data/all_data/SSA/SSA1.csv",
    "data/all_data/mixed_acid/1-1.xlsx",
    "data/all_data/milk/milk.xlsx",
]

TEXT_LOGS = [
    "data/bayesian.txt",
    "data/m_network.txt",
    "data/PIDexperiment.txt",
    "data/reinforced_network.txt",
]


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def summarize_table(path: Path) -> dict[str, object]:
    frame = load_table(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "preview": frame.head(3).to_dict(orient="records"),
    }


def summarize_text_log(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    experiment_like_lines = sum(1 for line in lines if "Experiment" in line or "experiment" in line)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "lines": len(lines),
        "experiment_like_lines": experiment_like_lines,
        "preview": lines[:5],
    }


def summarize_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown"]
    code_titles = []
    for index, cell in enumerate(code_cells, start=1):
        lines = [line.strip() for line in "".join(cell.get("source", [])).splitlines() if line.strip()]
        if lines:
            code_titles.append({"cell_index": index, "first_line": lines[0][:160]})
    return {
        "path": str(path.relative_to(ROOT)),
        "total_cells": len(notebook.get("cells", [])),
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "first_code_lines": code_titles[:12],
    }


def analyze_plot_notebook_paths(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    absolute_paths: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        absolute_paths.extend(re.findall(r"[A-Za-z]:\\[^\"\n]+", source))

    unique_paths = sorted(set(absolute_paths))
    missing_paths = [candidate for candidate in unique_paths if not Path(candidate).exists()]
    return {
        "absolute_path_occurrences": len(absolute_paths),
        "unique_absolute_paths": len(unique_paths),
        "missing_unique_absolute_paths": len(missing_paths),
        "example_paths": unique_paths[:12],
        "example_missing_paths": missing_paths[:12],
    }


def analyze_main_notebook_portability(path: Path) -> dict[str, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")
    return {
        "uppercase_open_modes": len(re.findall(r"open\([^)]*,\s*['\"](?:R|W)['\"]", source)),
        "capitalized_cpu_device": source.count("torch.device('Cpu')") + source.count('torch.device("Cpu")'),
        "incorrect_main_guard": source.count('__name__ == "Main"') + source.count("__name__ == 'Main'"),
        "hard_coded_ph4github_relative_paths": source.count('Path("ph4github")') + source.count("Path('ph4github')"),
    }


def collect_summary() -> dict[str, object]:
    missing_required = [relative_path for relative_path in REQUIRED_PATHS if not (ROOT / relative_path).exists()]

    summary = {
        "repo_root": str(ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "packages": {
            "numpy": distribution_version("numpy"),
            "pandas": distribution_version("pandas"),
            "scipy": distribution_version("scipy"),
            "matplotlib": distribution_version("matplotlib"),
            "openpyxl": distribution_version("openpyxl"),
            "PyYAML": distribution_version("PyYAML"),
            "torch": distribution_version("torch"),
            "scikit-learn": distribution_version("scikit-learn"),
            "seaborn": distribution_version("seaborn"),
            "shap": distribution_version("shap"),
        },
        "missing_required_paths": missing_required,
        "tables": [summarize_table(ROOT / relative_path) for relative_path in TABLE_SAMPLES],
        "text_logs": [summarize_text_log(ROOT / relative_path) for relative_path in TEXT_LOGS],
        "notebooks": {
            "main_code3": summarize_notebook(ROOT / "main_code3.ipynb"),
            "plot": summarize_notebook(ROOT / "plot.ipynb"),
        },
        "path_audit": analyze_plot_notebook_paths(ROOT / "plot.ipynb"),
        "portability_flags": analyze_main_notebook_portability(ROOT / "main_code3.ipynb"),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight repository checks against bundled files.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "output" / "repro" / "basic_checks.json",
        help="Where to write the JSON summary.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = collect_summary()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote basic checks to: {args.output_json}")
    print(f"Missing required paths: {len(summary['missing_required_paths'])}")
    print(
        "plot.ipynb absolute path occurrences: {} ({} unique, {} missing)".format(
            summary["path_audit"]["absolute_path_occurrences"],
            summary["path_audit"]["unique_absolute_paths"],
            summary["path_audit"]["missing_unique_absolute_paths"],
        )
    )
    print(
        "main_code3.ipynb portability flags: {}".format(
            summary["portability_flags"]
        )
    )


if __name__ == "__main__":
    main()
