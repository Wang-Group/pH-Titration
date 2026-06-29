# pH Titration Reproducibility Package

This repository copy was prepared as a reproducibility-oriented working version of the original `ph4github` materials. The goal is not a full scientific refactor. The goal is to make the current package easier for a third party to inspect, validate, and partially rerun.

## What is in this repository

- `main_code3.ipynb`
  Main analysis notebook. It contains multiple independent blocks for simulated data generation, supervised learning, reinforcement learning, Bayesian-style evaluation, model interpretation, and a revised PID baseline.
- `plot.ipynb`
  Figure-generation notebook. It contains several plotting variants for wastewater, mixed-acid, milk, SSA, and UV/Jobs-style plots.
- `experiment_summary.csv`
  Root-level summary table of simulated titration experiments used by several notebook sections and by the standalone PID helper script.
- `data/`
  Bundled experimental tables and archived text outputs.
- `output/`
  Output directory. The helper scripts in `repro_support/` write their results to `output/repro/`.
- `repro_support/`
  Added for this reproducibility pass. It contains small runnable scripts, report files, and manuscript-facing notes.

## Repository layout

```text
ph4github_reprocopy/
|-- data/
|   |-- all_data/
|   |   |-- milk/
|   |   |-- mixed_acid/
|   |   |-- SSA/
|   |   `-- wastewater/
|   |-- bayesian.txt
|   |-- m_network.txt
|   |-- PIDexperiment.txt
|   `-- reinforced_network.txt
|-- output/
|-- experiment_summary.csv
|-- main_code3.ipynb
|-- plot.ipynb
|-- requirements.txt
|-- reproducibility_notes.md
`-- repro_support/
```

## Quick start

The commands below assume a Python 3.11 interpreter. On the audit machine, `python` pointed to Python 3.4.1, so `py -3.11` was used explicitly.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Minimal runnable entry points

Run a basic repository audit and data-loading smoke test:

```powershell
py -3.11 repro_support\run_basic_checks.py
```

Generate a representative wastewater figure using only bundled files and relative paths:

```powershell
py -3.11 repro_support\run_example_plots.py --dataset wastewater
```

Run the extracted revised PID baseline on a subset of `experiment_summary.csv`:

```powershell
py -3.11 repro_support\run_pid_baseline.py --limit 100
```

Outputs are written to `output/repro/`.

## Execution map

The notebooks are monolithic and contain repeated code blocks, so the mapping below is approximate but useful:

- `main_code3.ipynb`
  - Cells 1-4: acid-system simulation utilities and synthetic dataset generation.
  - Cells 5-8: discrete regression model training and evaluation.
  - Cells 9-18: reinforcement learning training and evaluation variants.
  - Cell 20: Bayesian / online-update style evaluation written as a more script-like block.
  - Cell 22: archived experiment log generation.
  - Cells 24-30: SHAP and correlation analysis from saved text/model outputs.
  - Cell 32: ablation experiment block.
  - Cell 34: revised PID baseline. This block was extracted into `repro_support/run_pid_baseline.py`.
- `plot.ipynb`
  - Early cells: generic pH trajectory plotting helpers.
  - Middle cells: wastewater, milk, mixed-acid, and SSA plotting variants.
  - Later cells: UV-Vis, Jobs plot, and biuret-related figures.

## Known limitations

- The notebooks contain no markdown guidance and should be treated as lab-style working notebooks rather than polished pipelines.
- `plot.ipynb` contains many hard-coded absolute Windows paths and references several files that are not bundled in this repository copy.
- Full reruns of the machine-learning sections in `main_code3.ipynb` require additional packages such as PyTorch and were not fully smoke-tested in this pass.
- Exact historical package versions used for the original study were not preserved. `requirements.txt` and `reproducibility_notes.md` therefore document a practical tested setup, not a reconstructed historical environment.

## Where to look next

- `repro_support/reproducibility_report.md` for the audit summary and smoke-test results.
- `repro_support/manuscript_repro_text.md` for manuscript/SI wording suggestions.
- `reproducibility_notes.md` for dependency and environment details.
