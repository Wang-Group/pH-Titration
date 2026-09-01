# main_code3 Notebook Cell Exports

This folder contains a structural export of each code cell from `main_code3.ipynb`.

## Purpose

- Make the large notebook easier to browse on GitHub.
- Let you inspect, diff, and search individual notebook blocks as plain Python files.
- Provide a lightweight stepping stone before any future notebook refactor.

## Notes

- These files are direct cell exports, not a cleaned module refactor.
- Repeated imports and repeated class/function definitions from the notebook are preserved as-is.
- `cell_manifest.csv` records the mapping from raw notebook cell index to exported file name.

## Regeneration

From `ph4github/`, run:

```powershell
py -3.11 tools\export_notebook_cells.py --notebook main_code3.ipynb --output-dir main_code3_cells
```
