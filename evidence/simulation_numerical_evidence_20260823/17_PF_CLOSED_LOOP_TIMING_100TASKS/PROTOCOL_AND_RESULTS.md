# PF closed-loop timing and outcome benchmark

This block archives the PF runs used for the current manuscript, Supporting
Information, and response-letter timing statements. The same complete
closed-loop run produced both the control outcome and the timing records; the
success rates and latencies are therefore paired at the task/trajectory level.

The cohort contains 100 locked simulation cases: 20 prespecified task IDs from
each of benchmark seeds 101, 202, 303, 404, and 555. The task manifests remain
in `01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/tasks`; they are not
duplicated here.

## Timing scope

For every post-dose cycle, the timer begins immediately before
`controller.observe()` receives the rounded pH observation and ends after
`controller.recommend()` returns the next action. Controller construction,
reset, the untimed bootstrap recommendation, chemical transition, pH solving,
sensor quantisation, dose clipping, and file I/O are excluded. The summary
below pools all recorded post-dose decision cycles from the 100 complete
trajectories at each particle count.

| PF particles | Tasks | Recorded cycles | Success (%) | Mean final absolute pH error | Median observation-to-action time (ms) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 100 | 592 | 97.0 | 0.0600 | 40.131 |
| 10,000 | 100 | 589 | 97.0 | 0.0640 | 93.046 |
| 100,000 | 100 | 577 | 97.0 | 0.0595 | 594.127 |

Success is based on the final unrounded simulator equilibrium pH with an
absolute-error tolerance of 0.10 pH. Increasing the particle count did not
improve success in this cohort. The 1,000-particle configuration was retained
as the practical online setting.

These PF timings are not an identical-call timing experiment with the neural
or PyMC results in block `16`. Imitation, PPO, and PyMC each used one matched
single-step call per task in a CPU-affinity-controlled process. The present PF
benchmark pooled every recorded cycle from complete trajectories, used
single-thread numerical settings, and did not impose that CPU-affinity
control. Cross-method comparisons are therefore matched-cohort practical
wall-time comparisons, not fully hardware-controlled head-to-head estimates.

## Files and regeneration

- `results/<method>/task_results.csv`: task-level outcomes from the timed run.
- `results/<method>/per_step_timing.csv`: every post-dose timing observation.
- `results/<method>/trajectories.jsonl`: complete action/observation traces.
- `results/<method>/closed_loop_summary.csv`: regenerated outcome summary.
- `results/<method>/timing_first_n_summary.csv`: regenerated first-*n* and
  all-cycle timing summaries.
- `results/PF_CLOSED_LOOP_OUTCOME_SUMMARY.csv` and
  `results/PF_CLOSED_LOOP_TIMING_SUMMARY.csv`: consolidated derived tables.
- `results/PUBLICATION_TIMING_SCOPE_SUMMARY.csv`: cross-method values with an
  explicit protocol scope and CPU-affinity-control field for every method.
- `results/RELEASE_VALIDATION.json`: row-level numerical audit.
- `scripts/benchmark_pf_first_n_step_full_stats.py`: runnable repository-relative
  benchmark implementation.
- `scripts/finalize_pf_closed_loop_timing_100tasks.py`: non-simulation auditor
  that regenerates the derived summaries from the released raw files.

To rerun the audit without changing the raw task, step, or trajectory files:

```powershell
python scripts/finalize_pf_closed_loop_timing_100tasks.py
```

To conduct a new benchmark run, choose a new output directory rather than
overwriting the released observations:

```powershell
python scripts/benchmark_pf_first_n_step_full_stats.py --output-dir runs/pf_closed_loop_timing_100tasks
```
