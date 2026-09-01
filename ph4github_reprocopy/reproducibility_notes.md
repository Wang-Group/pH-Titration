# Reproducibility Notes

## Scope

These notes describe the practical environment used for the reproducibility audit performed on 2026-06-26. They do not claim to reconstruct the exact historical software stack used during the original study.

## Tested interpreter on the audit machine

- Preferred interpreter: `py -3.11`
- Observed version: Python 3.11.9

The default `python` command on the audit machine resolved to Python 3.4.1 without the scientific packages needed for this repository. For that reason, the helper scripts and example commands use `py -3.11`.

## Packages available during smoke testing

The following packages were confirmed in the working Python 3.11 environment:

- `numpy` 2.4.4
- `pandas` 2.3.3
- `scipy` 1.17.1
- `matplotlib` 3.10.9
- `openpyxl` 3.1.5
- `PyYAML` 6.0.3

These were enough for:

- loading the bundled CSV and XLSX data files
- generating a representative plot from bundled wastewater files
- running the extracted revised PID baseline

## Notebook-only or partially tested imports

The notebooks also import the following packages:

- `torch`
- `scikit-learn`
- `seaborn`
- `shap`

Those imports were observed directly in the notebooks, but they were not available in the audit environment and the corresponding notebook sections were not fully rerun here.

## Important environment caveats

### Jupyter notebooks are not drop-in pipelines

The notebooks preserve the original working structure of the project. They contain repeated code blocks, no markdown instructions, and multiple independent `if __name__ == ...` style execution blocks.

### Hard-coded paths remain in the notebooks

`plot.ipynb` still contains many machine-specific absolute paths, including examples under:

- `C:\Users\ZSY\...`
- `C:\Users\SYLZ\...`
- `E:\GitHub\ph4git\ph4git\ph4github\...`
- `Z:\...`

Some of those paths point to files that are not bundled in this repository copy. The standalone helper scripts in `repro_support/` avoid those machine-specific paths and only use files present inside this repository.

### Full ML reruns still need manual setup

The supervised-learning, reinforcement-learning, SHAP, and some plotting sections in `main_code3.ipynb` and `plot.ipynb` still require manual dependency installation and, in some cases, further path cleanup. This reproducibility pass focuses on:

- documenting the structure honestly
- exposing relative-path entry points for key bundled materials
- extracting one minimal baseline script that can run independently

## Recommended command style

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python repro_support\run_basic_checks.py
python repro_support\run_example_plots.py --dataset wastewater
python repro_support\run_pid_baseline.py --limit 100
```
