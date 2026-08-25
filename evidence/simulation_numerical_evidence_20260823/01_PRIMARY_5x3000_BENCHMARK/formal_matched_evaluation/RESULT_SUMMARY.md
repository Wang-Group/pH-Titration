# Matched PF, imitation, and PPO evaluation

Five existing PF benchmark task sets were reused without rerunning PF. The PF task-level rows were read from the archived hybrid_full results, and the PF-distilled imitation policy and validation-selected PPO policy were evaluated on the identical tasks.

Protocol: 5 benchmark seeds x 3000 tasks per seed; sensor resolution 0.01 pH; action classes 0.01-10.00 mL; true success evaluated from unquantized equilibrium pH.
The neural policies select volume only. The common external rule selects base below target and acid above target. The persistent post-overshoot cap is enabled through the shared `controllers.controller_api.PersistentOvershootCap` implementation: after a target crossing or increased absolute pH error, later volumes are capped at half the triggering delivered dose.

| Method | Success (%) | Strict (%) | Severe failure (%) | Successful steps | Total volume (mL) | Final error (pH) |
|---|---:|---:|---:|---:|---:|---:|
| PF teacher | 95.36 +/- 0.59 | 42.49 +/- 1.46 | 1.30 +/- 0.29 | 4.84 +/- 0.09 | 7.82 +/- 0.09 | 0.0729 +/- 0.0039 |
| PF-distilled imitation | 89.17 +/- 0.51 | 35.62 +/- 1.08 | 2.21 +/- 0.32 | 7.83 +/- 0.16 | 7.47 +/- 0.11 | 0.0953 +/- 0.0046 |
| PPO | 93.95 +/- 0.63 | 45.29 +/- 1.09 | 1.79 +/- 0.31 | 5.20 +/- 0.10 | 7.52 +/- 0.11 | 0.0785 +/- 0.0045 |

## Pooled paired success tests

- imitation_minus_pf_teacher: -6.19 percentage points; exact McNemar p=3.26392e-166; Holm-adjusted p=9.79175e-166.
- ppo_minus_pf_teacher: -1.41 percentage points; exact McNemar p=7.85758e-18; Holm-adjusted p=7.85758e-18.
- ppo_minus_imitation: +4.77 percentage points; exact McNemar p=2.19497e-96; Holm-adjusted p=4.38994e-96.

Per-seed task-level results and tests are exported separately. The pooled tests do not replace the mean +/- sample SD across the five benchmark seeds.
