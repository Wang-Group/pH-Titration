# Matched PF, imitation, and PPO evaluation

Five existing PF benchmark task sets were reused without rerunning PF. The PF task-level rows were read from the archived hybrid_full results, and the PF-distilled imitation policy and validation-selected PPO policy were evaluated on the identical tasks.

Protocol: 5 benchmark seeds x 1 tasks per seed; sensor resolution 0.01 pH; action classes 0.01-10.00 mL; true success evaluated from unquantized equilibrium pH.
The neural policies select volume only. The common external rule selects base below target and acid above target. The persistent post-overshoot cap is enabled: after a target crossing or increased absolute pH error, later volumes are capped at half the triggering delivered dose.

| Method | Success (%) | Strict (%) | Severe failure (%) | Successful steps | Total volume (mL) | Final error (pH) |
|---|---:|---:|---:|---:|---:|---:|
| PF teacher | 80.00 +/- 44.72 | 40.00 +/- 54.77 | 0.00 +/- 0.00 | 2.75 +/- 2.36 | 2.17 +/- 1.54 | 0.0673 +/- 0.0288 |
| PF-distilled imitation | 60.00 +/- 54.77 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 21.67 +/- 16.92 | 1.50 +/- 1.10 | 0.1125 +/- 0.0366 |
| PPO | 80.00 +/- 44.72 | 20.00 +/- 44.72 | 0.00 +/- 0.00 | 2.00 +/- 1.15 | 2.30 +/- 1.23 | 0.0773 +/- 0.0540 |

## Pooled paired success tests

- imitation_minus_pf_teacher: -20.00 percentage points; exact McNemar p=1; Holm-adjusted p=1.
- ppo_minus_pf_teacher: +0.00 percentage points; exact McNemar p=1; Holm-adjusted p=1.
- ppo_minus_imitation: +20.00 percentage points; exact McNemar p=1; Holm-adjusted p=1.

Per-seed task-level results and tests are exported separately. The pooled tests do not replace the mean +/- sample SD across the five benchmark seeds.
