# Local pH-response reconstruction near the control point

This block supplies the source and task-level data for the reported local
RMSE values **0.0399, 0.1280 and 0.2452 pH**. It contains the unchanged
contents of `local_response_0p0399_0p1280_0p2452_package.zip`, an explicitly
recovered dependency, and repository-level audit/replay entry points.

The complete independent replay in `verified_replay/` reproduced all
**1,500 tasks and 12,000 snapshots**, with no mismatched original fields
(numeric tolerance: relative `2e-10`, absolute `1e-11`). The replay also
records solution-state columns absent from the supplied endpoint CSV.

## Results and what they measure

| Signed additional-dose window | Mean RMSE (pH) | Sample SD across five task sets (pH) | Tasks with RMSE ≤0.10 pH |
|---|---:|---:|---:|
| ±0.10 mL | 0.0399429594 | 0.0103845394 | 92.4000% |
| ±0.50 mL | 0.1280199904 | 0.0194297698 | 78.7333% |
| ±1.00 mL | 0.2451563081 | 0.0243823209 | 67.8667% |

These are the `natural_control_end` rows of the supplied aggregate table.
Each value is first averaged over 300 tasks within each seed, then over the
five seed-level means. Sample SD uses five independent task sets (`ddof=1`),
not 12,000 independent experiments. Extra digits above are for tracing the
calculation; the four-decimal RMSE values reproduce the reported numbers.

For each current solution state, the script evaluates 81 equally spaced
signed additional doses in each window. Positive doses add 0.100 M base;
negative doses add 0.100 M acid, not negative liquid volume. Both additions
increase total solution volume. The true and fitted curves are each
centered on their own zero-addition pH:

```text
delta_true(v) = pH_true(v) - pH_true(0)
delta_fit(v)  = pH_fit(v)  - pH_fit(0)
RMSE = sqrt(mean((delta_fit(v) - delta_true(v))**2))
```

The fitted curve uses posterior-weighted mean concentration and pKa
parameters within the maximum-posterior model order. It is a parameter
plug-in curve, not a mean of all particle-predicted pH values. The metric
measures agreement in local pH changes after removing the intercept; it
does not measure unanchored absolute-pH prediction or chemical identification.

## Important checkpoint definition

The original field name `natural_control_end` is broader than a successful
natural stop. The diagnostic records the first state satisfying the **observed**
pH error ≤0.10 within 12 iterations. If none does, it records the last available
state. Among the 1,500 records used for the three values above:

- **1,404** meet the observed-pH criterion.
- **96** use the last state at iteration 12 without meeting that criterion
  (24, 19, 16, 21 and 16 in seeds 101, 202, 303, 404 and 555).

Thus these are **first-endpoint-or-12-iteration diagnostics over all tasks**,
not a successful-task-only result. The original labels and values are retained
for traceability, not silently corrected. The supplied discussion and PNG
use the shorter label “natural end”; read that label with this definition.

Fixed checkpoints at indices 0, 1, 2, 3, 5, 8 and 12 continue the diagnostic
after the first target encounter by clearing the controller's `done` flag.
They are not the normal deployment trajectory stopped at success. Also,
despite its name, the raw column `current_true_ph` stores the simulated pH
rounded to two decimal places; the local curve calculations themselves use
unrounded equilibrium pH.

## Tasks, PF implementation and package repair

The five locked manifests contain 300 tasks each (seeds 101, 202, 303, 404,
555; generator seeds 4,000,000 plus the benchmark seed). These are separate
from the principal five sets of 3,000 tasks. The study produces seven fixed
checkpoints and one terminal diagnostic per task: **12,000 snapshots from
1,500 tasks**. The controller is initialized with each task's recorded
initial volume and initial base moles.

The supplied controller, inference, chemistry and generator files match the
repository's archived `formal_pf_95_36_reproduction_20260901` versions after
newline normalization. This filter updates particle weights at every update,
but resamples a model bank only when its ESS is below half its particle count;
Liu–West regularization uses `h=0.15`. It uses 1,000 particles in total across
the three model banks.

The ZIP omitted `reference/original_bayesian_controller.py`, which its
controller imports. `dependency/reference/` restores that dependency from
the matching archived formal PF source. This addition is recorded separately
in `PROVENANCE.json`; it is not presented as an original ZIP member. No
supplied source, task file, numeric result or figure was altered.

`original_package/ANALYSIS_AND_DISCUSSION_CN.md` is an unchanged historical
discussion covering several studies. Only its local-response results are
audited by this block; it is not an updated account of other repository studies.

## Verify and reproduce

Use the repository commands below. The supplied `RUN_LOCAL_RESPONSE.cmd`
is retained as received and does not configure the separately recovered dependency.

From the repository root, use the read-only standard-library audit:

```powershell
python scripts/audit_pf_local_curves.py
```

It checks file integrity, task/checkpoint identities, all 40 per-seed and
eight aggregate summaries, the delivery-table copy, and the 96 fallback
records. The same audit is included in `scripts/verify_source.py`.

To replay all 1,500 tasks using the provided source and locked manifests:

```powershell
python scripts/reproduce_pf_local_curves.py --workers 8 --output runs/pf_local_curves
```

Use Python 3.11 and NumPy 2.4.6 (the repository's pinned numerical environment).
The command works in Windows PowerShell, needs no training or network access,
and requires a new output directory outside `evidence/`. It compares every
replayed original field with the supplied data and writes `REPLAY_AUDIT.json`.
Additional columns record the current solution volume, base/acid moles and
delivered-addition count so the local curves can be reconstructed later.
These extra columns do not change the original simulation or diagnostic.

All 12,000 replayed checkpoint indices matched the actual delivered-addition
count; no repeated no-dose iterations were found in this data set.

A short execution check on three locked tasks per seed is available:

```powershell
python scripts/reproduce_pf_local_curves.py --workers 4 --limit-tasks-per-seed 3 --output runs/pf_local_curves_smoke
```

This uses a subset of the original 300-task manifests, not a newly generated
three-task distribution. Smoke summaries must not replace the full results.
