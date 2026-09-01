# Reproducibility Report

## Working copies used

All edits in this pass were made only in copied materials.

- Repo copy: `E:\GitHub\ph4git\pH-Titration\ph4github_reprocopy`
- Manuscript copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\manuscript\Hybrid Bayesian Inference and Reinforcement Learning for Autonomous pH Adjustment in Diverse Chemical Systems2.0_reprocopy.docx`
- SI copy: `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\supplementary_information\SI-V19_reprocopy.docx`
- Reviewer PDF source inspected by path only: `C:\Users\Admin\Downloads\SC-EDG-05-2026-003882 Round 1.pdf`

## Audit summary

The copied code repository is compact but under-documented. It contains:

- two notebooks: `main_code3.ipynb` and `plot.ipynb`
- one root-level summary table: `experiment_summary.csv`
- one bundled `data/` tree with four released result logs and four subdatasets
- an empty `output/` directory before this reproducibility pass
- no README and no dependency specification before this pass

The strongest barriers to reproduction were:

- no project-specific README or run instructions
- no environment specification
- notebooks without markdown guidance
- heavy reliance on monolithic notebook cells instead of stable entry points
- extensive hard-coded absolute paths in `plot.ipynb`
- several notebook portability issues in `main_code3.ipynb`, including uppercase file modes, `__name__ == "Main"` guards, and notebook blocks that assume a `ph4github` parent directory name

## Repository contents identified

### Main notebooks / scripts

- `main_code3.ipynb`
  Mixed notebook containing simulation utilities, imitation learning, reinforcement learning, Bayesian-style online updating, SHAP/correlation analysis, ablation experiments, and a revised PID baseline.
- `plot.ipynb`
  Mixed notebook containing many figure-generation variants for wastewater, milk, mixed-acid, SSA, UV-Vis, Jobs plot, and biuret-related plots.

### Data folders

- `data/all_data/milk/`
- `data/all_data/mixed_acid/`
- `data/all_data/SSA/`
- `data/all_data/wastewater/`

### Bundled text outputs

- `data/bayesian.txt`
- `data/m_network.txt`
- `data/PIDexperiment.txt`
- `data/reinforced_network.txt`

### Output folders

- Original state before this pass: `output/` existed but was empty.
- Added in this pass: `output/repro/`

## Execution map

### `main_code3.ipynb`

- Cells 1-4: acid system / pH simulation utilities and synthetic dataset generation.
- Cells 5-8: discrete regression model training and evaluation.
- Cells 9-18: reinforcement-learning training and evaluation variants.
- Cell 20: Bayesian / online-update style evaluation against the experiment summary table.
- Cell 22: archived experiment log generation.
- Cells 24-30: SHAP and correlation analyses using saved outputs and learned models.
- Cell 32: ablation experiments.
- Cell 34: revised PID baseline with a script-like structure. This was extracted into `repro_support/run_pid_baseline.py`.

### `plot.ipynb`

- Early cells: generic pH trajectory plotting functions.
- Middle cells: wastewater, milk, mixed-acid, and SSA figure variants.
- Later cells: UV-Vis, Jobs plot, and biuret-related plotting.

### Likely correspondence to results / figures

This mapping is approximate because the notebooks are not organized as labeled figure scripts.

- Wastewater adjustment figure generation appears in `plot.ipynb` cells using bundled `data/all_data/wastewater/*.csv`.
- Milk adjustment figure generation appears in `plot.ipynb` cells using `data/all_data/milk/milk.xlsx`.
- Mixed-acid figure generation appears in `plot.ipynb` cells using `data/all_data/mixed_acid/*.xlsx`.
- SSA adjustment figure generation appears in `plot.ipynb` cells using `data/all_data/SSA/*.csv`.
- UV-Vis / Jobs / biuret style figures in later `plot.ipynb` cells depend on additional local files not bundled in this repo copy.
- The revised PID baseline corresponds to the text output style already archived as `data/PIDexperiment.txt` and is now exposed as `repro_support/run_pid_baseline.py`.

## Dependency summary

### Observed directly from notebook imports

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `openpyxl`
- `torch`
- `scikit-learn`
- `seaborn`
- `shap`
- `PyYAML`

### Confirmed available in the audit Python 3.11 environment

- `numpy` 2.4.4
- `pandas` 2.3.3
- `scipy` 1.17.1
- `matplotlib` 3.10.9
- `openpyxl` 3.1.5
- `PyYAML` 6.0.3

### Observed but not available during this audit

- `torch`
- `scikit-learn`
- `seaborn`
- `shap`

### Important environment note

On this machine, the default `python` command resolved to Python 3.4.1 without the scientific stack. Reproducibility commands therefore use `py -3.11`.

## Hard-coded path issues

`plot.ipynb` contains extensive machine-specific paths.

Measured during the audit:

- 103 absolute-path occurrences
- 75 unique absolute paths
- 61 unique absolute paths missing on the audit machine

Path roots observed include:

- `C:\Users\ZSY\...`
- `C:\Users\SYLZ\...`
- `C:\Users\Admin\Desktop`
- `E:\GitHub\ph4git\ph4git\ph4github\...`
- `Z:\自动化小组\0-papers in progress\2025-张思远-pH-titration\fig\...`

Additional portability flags found in `main_code3.ipynb`:

- 17 uppercase file-mode uses such as `'W'` and `'R'`
- 10 uses of `torch.device('Cpu')`
- 15 incorrect main guards using `__name__ == "Main"`
- 5 hard-coded assumptions about a parent folder named `ph4github`

These issues are why the reproducibility pass favored wrapper scripts and documentation over attempting to declare the notebooks themselves as clean pipelines.

## Files added in this pass

- `README.md`
- `requirements.txt`
- `reproducibility_notes.md`
- `repro_support/run_basic_checks.py`
- `repro_support/run_example_plots.py`
- `repro_support/run_pid_baseline.py`
- `repro_support/manuscript_repro_text.md`
- `repro_support/reproducibility_report.md`

## Smoke-test results

### 1. Repository / data checks

Command:

```powershell
py -3.11 repro_support\run_basic_checks.py
```

Result:

- Passed
- Output written to `output/repro/basic_checks.json`
- All required bundled repo paths checked by the script were present
- Confirmed `experiment_summary.csv` contains 3000 rows
- Confirmed bundled wastewater, SSA, mixed-acid, and milk tables can be read with the tested environment

### 2. Representative bundled plot

Command:

```powershell
py -3.11 repro_support\run_example_plots.py --dataset wastewater
```

Result:

- Passed
- Output written to `output/repro/wastewater_pH7.svg`
- File size after generation: 75,313 bytes

### 3. Revised PID baseline on 100 experiments

Command:

```powershell
py -3.11 repro_support\run_pid_baseline.py --limit 100
```

Result:

- Passed
- Success rate: 76.00%
- Overshoot rate: 27.3889%

### 4. Revised PID baseline on all 3000 experiments

Command:

```powershell
py -3.11 repro_support\run_pid_baseline.py
```

Result:

- Passed
- Detailed report written to `output/repro/pid_baseline_report.txt`
- Summary written to `output/repro/pid_baseline_summary.json`
- Success rate: 76.1667%
- Successful experiments: 2285 / 3000
- Average successful steps: 23.7046
- Standard deviation of successful steps: 10.8160
- Overshoot rate: 29.8760%

## Manuscript / SI reproducibility wording audit

The manuscript and SI working copies were resolved successfully on `Z:` after switching from the garbled path strings in the task brief to the actual Unicode path.

Relevant wording found:

- The manuscript already contains a placeholder `Data availability` heading rather than a completed repository-specific statement.
- The SI states that random processes were seeded with `seed = 555` for reproducibility.
- The SI states: `The raw code for the machine learning process is available at GitHub https://github.com/Wang-Group/pH-Titration.`

Recommended update:

- replace the generic / placeholder data-availability language with a repository-specific statement
- add a software-environment note distinguishing bundled runnable pieces from notebook sections that still require manual setup
- explicitly note that helper scripts and dependency notes were added for the revision

## What is reproducible now

- Inspection of released data structure
- Loading of bundled CSV and XLSX files
- Inspection of archived text outputs
- Generation of at least one representative figure from bundled data via relative paths
- Re-execution of the revised PID baseline as a standalone script
- Basic environment and path auditing

## What is partially reproducible now

- Notebook figure generation for bundled datasets
  The logic is visible and some parts can be adapted, but many notebook cells still depend on absolute paths or duplicated plotting variants.
- Bayesian / RL / imitation-learning workflow interpretation
  The code is present in `main_code3.ipynb`, but not yet extracted into a clean module or script pipeline.

## What is not fully reproducible from current materials

- End-to-end reruns of the PyTorch-based training and reinforcement-learning sections in the audit environment used here
- SHAP and some analysis cells that depend on packages absent from the tested environment
- Plotting cells that depend on extra external files not bundled in this repository copy, especially some UV-Vis, protein, and manuscript-figure export paths
- Exact historical software environment reconstruction

## Immediate fixes completed in this pass

- Added a real `README.md`
- Added `requirements.txt`
- Added `reproducibility_notes.md`
- Added a standalone revised PID baseline script
- Added a relative-path example plotting script
- Added a basic repository validation script
- Added manuscript/SI wording suggestions
- Documented the major remaining portability barriers explicitly

## Recommended immediate next fixes for revision

- Keep using the helper scripts and documentation added in this pass as the reproducibility-facing layer for reviewers and readers.
- If time permits, extract one more notebook figure workflow beyond wastewater into a standalone script, ideally for one of the main text figures that uses fully bundled data.
- Add a short reviewer-response paragraph pointing to the new README, environment notes, helper scripts, and path-limit documentation.

## Recommended post-submission improvements

- Refactor `main_code3.ipynb` into small Python modules or scripts for:
  - data generation
  - imitation learning
  - reinforcement learning
  - Bayesian online updating
  - analysis / interpretation
- Refactor `plot.ipynb` into named figure scripts with relative-path inputs.
- Replace machine-specific save paths with configuration-driven output directories.
- Preserve a pinned environment file from the actual training environment.
- Add unit tests for core pH calculation utilities and baseline controllers.

## Bottom line

The copied repository is now substantially easier to understand and partially rerun. The revision still does not provide a fully modular end-to-end training pipeline, but the most visible reviewer concerns about a template README and poor reproducibility support are now addressed concretely with documentation, helper scripts, dependency notes, smoke tests, and explicit disclosure of the remaining gaps.
