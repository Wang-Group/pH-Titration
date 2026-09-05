# PF dose-shaping ablation for SI Table S6

This package supplies the source and results for SI Table S6 and Response
Table R7. It was imported from `PF_internal_rule_ablation_package.zip`
without changing the supplied source, task manifests, or raw results.
The original `SHA256SUMS.txt` covers 24 supplied files.
The local `.gitattributes` preserves their original line endings when cloned.

Five task sets (seeds 101, 202, 303, 404, and 555) contain 300 tasks each.
All six dose-rule variants were evaluated on every task, giving 9,000
task-controller outcomes. Every variant uses 1,000 particles and executes its
own dosing trajectory. The study's exact controller snapshot is in
`source/controllers_release/`; the generator is in `source/study_source/`.
These are not the older `reward_and_dose_ablation/eq_s5_*` results in the
parent directory. Those older results remain archived and must not replace S6.

## Verify the supplied results

From the repository root:

```powershell
python scripts/audit_pf_internal_rule_ablation.py
```

The audit checks file contents, 1,500 matched task identities, all 9,000
outcomes, per-set and aggregate statistics, and the five exact McNemar tests
with Holm correction. It tolerates only Git's CRLF-to-LF conversion when
checking source hashes. To additionally recompute the 20 paired continuous
tests with SciPy installed:

```powershell
python scripts/audit_pf_internal_rule_ablation.py --paired-continuous
```

## Rerun the simulations

From this package directory, install `source/requirements-simulation.txt`
and run into a new output directory:

```powershell
python source/pf_internal_rule_ablation.py --output-dir results/reproduced_s6 --seeds 101 202 303 404 555 --tasks-per-seed 300 --particles 1000 --workers 8
```

Keep the archived `results/formal_results/` unchanged. Timing may differ
between machines and is not a column of the published S6 table.

## Statistical definitions and rounding

`success` is the final unrounded equilibrium pH error at or below 0.10 pH.
The `strict_success` field is a separate 0.05-pH diagnostic, not the success
column of S6. Additions are averaged among successful tasks within each set;
overshoot counts, total volume, and final error are averaged over all tasks.
Reported means and sample SDs are then calculated across the five set-level
results. Paired continuous tests use all matched tasks, so their differences
in step count need not equal differences between successful-task means.

Round the SD directly to one significant digit, then round the mean to the
same decimal place. Do not round a two-decimal summary a second time.
In particular, the linear-mapping additions are
`4.830562284968709 ± 0.245070096633694`, displayed as `4.8 ± 0.2`.
Removing the required-volume term increases successful-task additions by
`0.256085915603348`, displayed as approximately `0.26` additions.
All five Holm-adjusted McNemar p-values are 1.0; these display corrections
do not change that result.
