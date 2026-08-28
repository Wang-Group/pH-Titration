# Matched timing and posterior recovery on 100 tasks

All six methods used the same 100 locked task cases, the same 0.01 mL pre-dose, and the same rounded first post-dose pH observation. PF and PyMC recovery metrics are taken directly from the posterior snapshot produced inside the corresponding timed observation-to-action call. Recovery therefore describes one-observation initialization, not a complete closed-loop trajectory.

## Posterior recovery

| Method | Model-order accuracy | Median concentration relative error | Median pKa MAE | Median full-curve RMSE (pH) |
|---|---:|---:|---:|---:|
| PF (1,000 particles) | 33/100 (33.0%) | 45.20% | 0.674 | 3.180 |
| PyMC (variable K, 300 draws per K) | 29/100 (29.0%) | 47.48% | 0.634 | 3.076 |

## Matched single-step timing

The measured interval was from the new rounded pH observation entering the controller to the next action being returned. Startup, imports, checkpoint loading, controller construction, task loading, chemical transition calculation, liquid delivery, mixing, sensor acquisition, and file I/O were excluded. All methods ran in fresh processes pinned to the same logical CPU with one numerical thread.

| Method | Median wall time per step (ms) |
|---|---:|
| Imitation policy | 0.15495 |
| PPO policy | 0.15390 |
| PF, 1,000 particles | 22.99615 |
| PF, 10,000 particles | 101.45055 |
| PF, 100,000 particles | 900.93545 |
| PyMC, variable K | 14,407.37565 |

The same task and input audit passed for all six methods. Exact task-level records are in `POSTERIOR_RECOVERY_TASK_RESULTS.csv` and each method's `raw.csv`; configuration, hashes, and paired tests are in `MATCHED_RUN_CONFIG.json`.
