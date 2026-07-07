# pH Titration Code and Data Package

This folder is the main code and data package for the pH-titration project.

## Contents

- `main_code3.ipynb`
  Main working notebook containing simulation, imitation-learning, reinforcement-learning, Bayesian-style evaluation, interpretation, and baseline code blocks.
- `plot.ipynb`
  Figure-generation notebook with mixed-acid, wastewater, milk, SSA, UV-Vis, and Job-plot related plotting blocks.
- `experiment_summary.csv`
  Bundled simulation benchmark table used by several analyses and helper scripts.
- `data/`
  Bundled experimental tables and archived raw text outputs.
- `repro_support/`
  Small helper scripts for basic repository checks, a representative bundled plot, and the extracted PID baseline.
- `review_support/`
  Reviewer-response analysis scripts including the expert-rule baseline, timing benchmark, and theoretical titration curves.
- `tools/export_notebook_cells.py`
  Utility that exports each code cell from a notebook to its own `.py` file.
- `main_code3_cells/`
  Generated per-cell Python exports from `main_code3.ipynb`.

## Quick start

Use Python 3.11 if possible:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Minimal helper commands

```powershell
py -3.11 repro_support\run_basic_checks.py
py -3.11 repro_support\run_example_plots.py --dataset wastewater
py -3.11 repro_support\run_pid_baseline.py --limit 100
py -3.11 tools\export_notebook_cells.py --notebook main_code3.ipynb --output-dir main_code3_cells
```

## Reviewer-response scripts

The following scripts were added during the manuscript revision:

- `review_support\expert_rule_baseline.py`
- `review_support\timing_comparison_benchmark.py`
- `review_support\theoretical_titration_plots.py`

These are useful when reproducing the additional analyses used to answer reviewer questions about heuristic baselines, controller timing, and mixed-acid titration-curve interpretation.

## Notes

- The original notebooks are still monolithic and contain repeated code blocks.
- `plot.ipynb` still contains hard-coded absolute paths in several cells.
- `main_code3_cells/` is only a structural export of the notebook cells; it is meant to help code navigation and GitHub browsing, not to replace a full refactor yet.

See `reproducibility_notes.md` for more environment details.
