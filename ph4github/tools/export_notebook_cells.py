from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def slugify(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[#/\s\"']+", "", cleaned)
    cleaned = cleaned.replace("\\", " ")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()
    return cleaned[:48] or "cell"


def first_nonempty_line(source: list[str]) -> str:
    for line in source:
        if line.strip():
            return line.rstrip("\n")
    return ""


def export_notebook_cells(notebook_path: Path, output_dir: Path) -> list[dict[str, str]]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    code_index = 0
    for raw_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        code_index += 1
        source = cell.get("source", [])
        first_line = first_nonempty_line(source)
        slug = slugify(first_line)
        filename = f"cell_{code_index:02d}_{slug}.py"
        output_path = output_dir / filename

        header = [
            f"# Source notebook: {notebook_path.name}",
            f"# Raw notebook cell index: {raw_index}",
            f"# Code-cell export index: {code_index}",
        ]
        if first_line:
            header.append(f"# First non-empty line: {first_line}")
        header.append("")

        output_path.write_text("\n".join(header) + "".join(source), encoding="utf-8", newline="\n")
        manifest.append(
            {
                "raw_notebook_cell_index": str(raw_index),
                "code_cell_export_index": str(code_index),
                "output_file": filename,
                "first_nonempty_line": first_line,
            }
        )

    manifest_path = output_dir / "cell_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_notebook_cell_index",
                "code_cell_export_index",
                "output_file",
                "first_nonempty_line",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export each code cell of a Jupyter notebook to its own .py file.")
    parser.add_argument("--notebook", type=Path, required=True, help="Path to the source .ipynb file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-cell .py files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_notebook_cells(args.notebook, args.output_dir)
    print(f"Exported {len(manifest)} code cells to: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'cell_manifest.csv'}")


if __name__ == "__main__":
    main()
