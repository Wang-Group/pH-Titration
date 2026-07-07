# Q1b Quantitative Analysis

- Dataset: 20 simulated tasks from `experiment_summary.csv`.
- Step-budget view: metrics at `min(final step, 8)` to answer the reviewer's question about whether roughly eight titration measurements are sufficient.

## Headline observations

- Overall matched posterior pKa MAE changes from 0.974 initially to 0.968 by step <= 8, and 0.970 at the final step.
- By step <= 8, the fraction of tasks for which all true pKa values are recovered within +-1.0 pH units is 35.00%; at the final step it is 35.00%.
- The mean posterior pKa standard deviation is 0.200 initially, 0.188 by step <= 8, and 0.187 at the final step.

## Predictive usefulness of the latent state

- To quantify whether the Bayesian latent state remains chemically meaningful even when exact pKa recovery is imperfect, I also measured the one-step pH prediction error made by the current posterior model before each titration action.

| acid_type   | stage          |   transitions |   mean_prediction_abs_error |   median_prediction_abs_error |   prediction_error_le_0p1_pct |   prediction_error_le_0p2_pct |
|:------------|:---------------|--------------:|----------------------------:|------------------------------:|------------------------------:|------------------------------:|
| diprotic    | all_steps      |            59 |                      3.3495 |                        2.9015 |                          3.39 |                          8.47 |
| diprotic    | within_8_steps |            46 |                      3.081  |                        2.5704 |                          4.35 |                         10.87 |
| monoprotic  | all_steps      |            36 |                      3.4428 |                        1.8917 |                          8.33 |                         16.67 |
| monoprotic  | within_8_steps |            27 |                      3.0467 |                        0.8644 |                         11.11 |                         22.22 |
| triprotic   | all_steps      |            53 |                      1.1228 |                        1.0129 |                         26.42 |                         28.3  |
| triprotic   | within_8_steps |            46 |                      1.0577 |                        0.9173 |                         26.09 |                         26.09 |
| overall     | all_steps      |           148 |                      2.5748 |                        1.8496 |                         12.84 |                         17.57 |
| overall     | within_8_steps |           119 |                      2.2911 |                        1.2215 |                         14.29 |                         19.33 |

## Species-count caveat

- The current Bayesian controller keeps exactly three latent buffer slots throughout the episode. It does not maintain an explicit posterior over the number of species.
- As a diagnostic, I tracked the inferred mole fraction assigned to unmatched latent slots for monoprotic and diprotic tasks after optimal pKa matching. Large residual mass in unmatched slots means the model is not cleanly deactivating extra components.

## Correlations

| stage         |   corr_posterior_std_vs_pka_mae |   corr_posterior_std_vs_abs_pH_error |   corr_pka_mae_vs_abs_pH_error |
|:--------------|--------------------------------:|-------------------------------------:|-------------------------------:|
| final         |                         -0.075  |                              -0.2843 |                         0.1107 |
| initial       |                        nan      |                             nan      |                        -0.1598 |
| up_to_8_steps |                         -0.0923 |                               0.2205 |                        -0.3731 |

## Extra-slot diagnostic

| acid_type   | stage         |   experiments |   mean_unused_slot_mole_fraction |   median_unused_slot_mole_fraction |   unused_slot_fraction_lt_0p1_pct |   unused_slot_fraction_lt_0p2_pct |
|:------------|:--------------|--------------:|---------------------------------:|-----------------------------------:|----------------------------------:|----------------------------------:|
| diprotic    | final         |             7 |                           0.3493 |                             0.3832 |                                 0 |                                 0 |
| diprotic    | initial       |             7 |                           0.3536 |                             0.3829 |                                 0 |                                 0 |
| diprotic    | up_to_8_steps |             7 |                           0.3541 |                             0.3832 |                                 0 |                                 0 |
| monoprotic  | final         |             5 |                           0.605  |                             0.5448 |                                 0 |                                 0 |
| monoprotic  | initial       |             5 |                           0.531  |                             0.5396 |                                 0 |                                 0 |
| monoprotic  | up_to_8_steps |             5 |                           0.6056 |                             0.5448 |                                 0 |                                 0 |
