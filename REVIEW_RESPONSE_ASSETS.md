# Reviewer-Response Assets

This note summarizes the code and repo changes that were added while preparing the reviewer-response package.

## Main public code package

The main code/data package intended for GitHub is:

- `ph4github/`

This folder now includes:

- `README.md`
- `requirements.txt`
- `reproducibility_notes.md`
- `repro_support/` for lightweight reproducibility helpers
- `review_support/` for reviewer-response analysis scripts
- `tools/export_notebook_cells.py`
- `main_code3_cells/` generated per-cell Python exports from `main_code3.ipynb`

## Reviewer-response scripts added

These scripts were written or adapted during the reviewer-response work and are now placed under `ph4github/review_support/`:

- `expert_rule_baseline.py`
  Implements the human-like expert-rule baseline requested by the reviewer.
- `timing_comparison_benchmark.py`
  Benchmarks Bayesian, PID, expert-rule, imitation-learning, and reinforcement-learning controller decision latency.
- `theoretical_titration_plots.py`
  Generates representative theoretical titration curves for the mixed-acid systems discussed around Figure 2.

## Reproducibility helpers added

These scripts are now placed under `ph4github/repro_support/`:

- `run_basic_checks.py`
- `run_example_plots.py`
- `run_pid_baseline.py`

They were added to make the released package easier to inspect and partially rerun without opening the full monolithic notebooks first.

## Current release

This note describes historical helpers. The current controller, training,
data and verification entry points are documented in the top-level
[`README.md`](README.md). Internal peer-review working copies are not part
of the public reproducibility package.
