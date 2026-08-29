# Matched PyMC/PF timing and posterior recovery

This block contains the matched single-step timing benchmark and the current
one-observation PF/PyMC posterior-recovery comparison. It is an online-call
benchmark designed around the experimental control sequence; it is **not** a
wall-clock measurement collected during the physical experiments. The current
PF timing and success values measured together during complete trajectories are
archived separately in `17_PF_CLOSED_LOOP_TIMING_100TASKS`.

All six methods used the same 100 locked simulation cases, comprising 20
prespecified task IDs from each of benchmark seeds 101, 202, 303, 404, and
555. Each case used the same 0.01 mL pre-dose and the same rounded first
post-dose pH observation. PF and PyMC recovery metrics were calculated from
the posterior snapshot produced inside that same timed call, so they describe
recovery after one post-dose observation rather than a full control trajectory.

## Timing scope

The measured interval begins when the new rounded pH observation enters the
controller and ends when the next action is returned. It includes state
bookkeeping, posterior or observation updating, and action selection. It
excludes Python startup, imports, checkpoint loading, controller construction,
task loading, chemical-transition calculation, liquid delivery, mixing,
sensor acquisition, and file I/O.

Every method ran in a fresh process pinned to logical CPU 2 with one numerical
thread. The PyMC implementation used sequential Monte Carlo sampling with 300
draws for each candidate model order (`K = 1, 2, 3`) and one chain.

| Method | Median wall time per step (ms) |
|---|---:|
| Imitation policy | 0.15495 |
| PPO policy | 0.15390 |
| PF, 1,000 particles | 22.99615 |
| PF, 10,000 particles | 101.45055 |
| PF, 100,000 particles | 900.93545 |
| PyMC, variable K | 14,407.37565 |

The approximately 20 s liquid-transfer, mixing, electrode-stabilization, and
pH-acquisition interval discussed in the response letter is experimental
context only and is not included in these computational timings.

The PF rows in this table are retained as the CPU-affinity-controlled
single-step protocol. They must not be substituted for the current pooled
complete-trajectory PF medians of 40.131, 93.046, and 594.127 ms in block 17.

## One-observation posterior recovery

| Method | Model-order accuracy | Median concentration relative error | Median matched pKa MAE | Median full-curve RMSE (pH) |
|---|---:|---:|---:|---:|
| PF, 1,000 particles | 33/100 (33.0%) | 45.20% | 0.674 | 3.180 |
| PyMC, variable K | 29/100 (29.0%) | 47.48% | 0.634 | 3.076 |

None of the paired differences was statistically significant. The exact
McNemar p value for model-order accuracy was 0.557. The paired Wilcoxon p
values were 0.635 for concentration error, 0.896 for matched pKa MAE, and
0.693 for full-curve RMSE.

## Files and provenance

- `results/CONTROLLED_RESULT_SUMMARY.csv`: timing summary for all six methods.
- `results/POSTERIOR_RECOVERY_SUMMARY.csv`: PF/PyMC recovery summary.
- `results/POSTERIOR_RECOVERY_TASK_RESULTS.csv`: task-level recovery metrics.
- `results/<method>/raw.csv`: exact task-level timing and posterior rows.
- `results/MATCHED_RUN_CONFIG.json`: task/input audit, paired tests, hashes,
  environment metadata, and method-level run configurations.
- `scripts/benchmark_controlled_observation_to_action_100tasks.py`: timing
  worker.
- `scripts/run_controlled_timing_100tasks.py`: isolated-process launcher.
- `scripts/finalize_matched_timing_recovery_100tasks.py`: result auditor and
  finalizer.

The worker uses the locked task manifests under
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/tasks`, the released
controllers and checkpoints, and the PyMC source plus tested environment under
`13_SOURCE_ARCHIVES/joint_parameter_bayesian_code_current`. The source and
checkpoints in those locations are byte-identical or text-identical after line
ending normalization to the copies used for the formal run.

Run the repository's non-mutating verification, which checks the released raw
row counts, method set, timing medians, recovery statistics, common-input audit,
and PyMC sampling configuration:

```powershell
python scripts/verify_source.py --skip-self-test
```

`scripts/finalize_matched_timing_recovery_100tasks.py` regenerates the derived
tables and audit metadata in the output directory supplied to it. Use it on a
copy of the released results if regeneration is desired.

To perform a new controlled run with the same formal settings, use a Python
3.11 PyMC environment satisfying
`13_SOURCE_ARCHIVES/joint_parameter_bayesian_code_current/requirements_tested.txt`:

```powershell
python scripts/run_controlled_timing_100tasks.py --output-dir runs/matched_timing_recovery_100tasks --repeats 1 --cpu-index 2 --draws 300 --chains 1 --pymc-python <path-to-pymc-python>
```
