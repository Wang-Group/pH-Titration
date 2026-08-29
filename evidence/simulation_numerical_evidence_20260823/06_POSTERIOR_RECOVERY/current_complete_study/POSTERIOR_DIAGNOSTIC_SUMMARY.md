# PF posterior accuracy and curve-similarity study

Evaluation used 5 independent seeds, 300 tasks per seed, and 1000 particles.
The fixed-observation analysis continues the same hybrid-full PF controller to 12 observations even if it reaches the control target earlier. This keeps the task cohort identical at every observation count. Natural control endpoints are reported separately.
Complete response curves are evaluated over signed primary-titrant additions from -100 to +100 mL relative to the initial chemical state.

| Observations | Curve RMSE (pH) | Curve correlation | Concentration error (%) | K accuracy (%) | True-K probability | pKa MAE when K correct |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.3894 +/- 0.0589 | 0.9624 +/- 0.0032 | 72.28 +/- 3.74 | 42.33 +/- 3.33 | 0.333 +/- 0.000 | 1.5618 +/- 0.0322 |
| 1 | 1.2204 +/- 0.0471 | 0.9700 +/- 0.0023 | 50.65 +/- 4.01 | 44.13 +/- 2.75 | 0.370 +/- 0.004 | 1.0084 +/- 0.0974 |
| 2 | 1.1773 +/- 0.0527 | 0.9706 +/- 0.0023 | 47.77 +/- 3.51 | 46.80 +/- 2.18 | 0.388 +/- 0.005 | 0.8978 +/- 0.0782 |
| 3 | 1.1689 +/- 0.0451 | 0.9705 +/- 0.0022 | 47.49 +/- 3.49 | 44.87 +/- 2.18 | 0.393 +/- 0.008 | 0.8626 +/- 0.0978 |
| 5 | 1.1417 +/- 0.0400 | 0.9709 +/- 0.0018 | 47.34 +/- 3.96 | 45.60 +/- 2.66 | 0.400 +/- 0.006 | 0.8388 +/- 0.0568 |
| 8 | 1.1396 +/- 0.0438 | 0.9709 +/- 0.0020 | 47.12 +/- 4.57 | 45.53 +/- 1.71 | 0.407 +/- 0.006 | 0.8248 +/- 0.0602 |
| 12 | 1.1388 +/- 0.0415 | 0.9709 +/- 0.0017 | 46.85 +/- 4.04 | 45.87 +/- 2.18 | 0.408 +/- 0.006 | 0.8318 +/- 0.0620 |

## Natural control endpoint

Mean observations: 6.32 +/- 0.56.
Curve RMSE: 1.1370 +/- 0.0523 pH.
Concentration relative error: 47.33 +/- 3.91%.
Pair-count accuracy: 46.07 +/- 2.55%.
pKa MAE conditional on correct K: 0.8494 +/- 0.0751.
Control success: 95.80 +/- 1.92%.

Task-level posterior estimates, seed-level summaries, final subgroups, plots, and exact generated tasks accompany this report.
