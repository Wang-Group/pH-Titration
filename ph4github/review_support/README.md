# Review Support Scripts

This folder contains scripts that were added or adapted while preparing the reviewer-response package.

## Scripts

- `expert_rule_baseline.py`
  Implements the human-like expert-rule baseline suggested by the reviewer.
- `timing_comparison_benchmark.py`
  Benchmarks controller decision latency relative to the physical stabilization delay.
- `theoretical_titration_plots.py`
  Generates representative theoretical titration curves for the Figure 2 mixed-acid systems.

## Output location

By default these scripts write results under:

- `output/reviewer_response/`

inside the `ph4github/` package.

## Scope

These scripts were created to support the revision and rebuttal process. They are useful for reproducing the additional analyses requested during peer review, but they do not by themselves replace the original main notebooks.
