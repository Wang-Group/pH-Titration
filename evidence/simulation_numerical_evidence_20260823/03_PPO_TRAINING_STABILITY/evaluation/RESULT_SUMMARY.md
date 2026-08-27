# Secondary PPO stability screen (common held-out evaluation)

This archived screen is retained for transparency. It is separate from the
current primary five-training-seed x five-benchmark analysis reported in the
manuscript. In this screen, the PF teacher, selected PF-distilled imitation
checkpoint, and five independently trained PPO checkpoints were evaluated on
the same locked tasks within each suite.

## Nominal locked-set comparison

On the common nominal locked set of 1,000 tasks, the directly comparable
success rates were:

| Controller | Checkpoints/runs | Success (%) |
|---|---:|---:|
| PF-distilled imitation policy | 1 | 89.10 |
| PPO stability-screen checkpoints | 5 | 89.54 +/- 2.23 |
| PF teacher | 1 | 95.10 |

For PPO, `89.54 +/- 2.23%` is the mean +/- sample standard deviation across
the five independently trained checkpoints. The imitation and PF-teacher
values each come from one frozen controller evaluated on the same 1,000-task
manifest, so no between-training-seed standard deviation applies to those two
rows.

These three values may be compared within this secondary common-task screen.
They must not be pooled with, substituted for, or interpreted as a replication
of the primary `91.79 +/- 1.53%` result, which uses five PPO training seeds
crossed with five locked benchmark sets of 3,000 tasks each.

## Complete archived screen

| Suite | Method | Runs | Success (%) | Strict (%) | Severe failure (%) | Successful steps | Final error |
|---|---|---:|---:|---:|---:|---:|---:|
| close_pka | imitation | 1 | 85.00 | 32.33 | 3.67 | 8.88 | 0.1526 |
| close_pka | ppo | 5 | 88.20 +/- 1.94 | 35.47 +/- 4.52 | 4.13 +/- 0.80 | 7.30 +/- 1.10 | 0.1524 +/- 0.0259 |
| close_pka | teacher | 1 | 95.00 | 41.00 | 2.00 | 5.63 | 0.0798 |
| nominal_locked | imitation | 1 | 89.10 | 32.60 | 3.90 | 8.63 | 0.1382 |
| nominal_locked | ppo | 5 | 89.54 +/- 2.23 | 38.62 +/- 5.27 | 4.12 +/- 0.86 | 6.80 +/- 1.09 | 0.1423 +/- 0.0202 |
| nominal_locked | teacher | 1 | 95.10 | 42.20 | 1.80 | 4.87 | 0.0789 |
| wide_concentration | imitation | 1 | 85.67 | 33.33 | 3.67 | 8.89 | 0.1361 |
| wide_concentration | ppo | 5 | 86.67 +/- 2.24 | 36.27 +/- 3.04 | 5.33 +/- 1.13 | 7.39 +/- 0.76 | 0.1754 +/- 0.0295 |
| wide_concentration | teacher | 1 | 93.33 | 46.00 | 1.67 | 5.53 | 0.0744 |

PPO checkpoints were selected only from independent validation tasks. The
locked test tasks were evaluated after checkpoint selection. Task-level
results and paired tests are provided in the accompanying CSV files:

- `aggregate_summary.csv`: aggregate controller metrics by suite.
- `per_run_summary.csv`: individual PPO-checkpoint results.
- `all_task_results.csv`: task-level outcomes for all evaluated controllers.
- `paired_method_tests.csv`: paired statistical comparisons.
- `../common_locked_test_tasks.jsonl`: the common 1,000-task nominal manifest.
