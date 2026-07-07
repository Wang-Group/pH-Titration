# Q1b Quantitative Analysis

- Dataset: 200 simulated tasks from `experiment_summary.csv`.
- Step-budget view: metrics at `min(final step, 8)` to answer the reviewer's question about whether roughly eight titration measurements are sufficient.

## Headline observations

- Overall matched posterior pKa MAE changes from 0.882 initially to 0.890 by step <= 8, and 0.890 at the final step.
- By step <= 8, the fraction of tasks for which all true pKa values are recovered within +-1.0 pH units is 43.50%; at the final step it is 44.00%.
- The mean posterior pKa standard deviation is 0.200 initially, 0.190 by step <= 8, and 0.192 at the final step.

## Predictive usefulness of the latent state

- To quantify whether the Bayesian latent state remains chemically meaningful even when exact pKa recovery is imperfect, I also measured the one-step pH prediction error made by the current posterior model before each titration action.

| acid_type   | stage          |   transitions |   mean_prediction_abs_error |   median_prediction_abs_error |   prediction_error_le_0p1_pct |   prediction_error_le_0p2_pct |
|:------------|:---------------|--------------:|----------------------------:|------------------------------:|------------------------------:|------------------------------:|
| diprotic    | all_steps      |           617 |                      3.1309 |                        2.6295 |                          3.08 |                          5.67 |
| diprotic    | within_8_steps |           421 |                      2.8502 |                        1.8223 |                          4.28 |                          7.6  |
| monoprotic  | all_steps      |           732 |                      3.8754 |                        3.4991 |                          3.14 |                          5.33 |
| monoprotic  | within_8_steps |           356 |                      3.526  |                        2.0829 |                          6.18 |                         10.67 |
| triprotic   | all_steps      |           579 |                      1.2299 |                        0.9793 |                         20.55 |                         24.87 |
| triprotic   | within_8_steps |           489 |                      1.1887 |                        1.0109 |                         18.61 |                         23.11 |
| overall     | all_steps      |          1928 |                      2.8427 |                        2.2249 |                          8.35 |                         11.31 |
| overall     | within_8_steps |          1266 |                      2.3985 |                        1.3386 |                         10.35 |                         14.45 |

## Species-count caveat

- The current Bayesian controller keeps exactly three latent buffer slots throughout the episode. It does not maintain an explicit posterior over the number of species.
- As a diagnostic, I tracked the inferred mole fraction assigned to unmatched latent slots for monoprotic and diprotic tasks after optimal pKa matching. Large residual mass in unmatched slots means the model is not cleanly deactivating extra components.

## Correlations

| stage         |   corr_posterior_std_vs_pka_mae |   corr_posterior_std_vs_abs_pH_error |   corr_pka_mae_vs_abs_pH_error |
|:--------------|--------------------------------:|-------------------------------------:|-------------------------------:|
| final         |                          0.0547 |                               0.0825 |                        -0.0365 |
| initial       |                        nan      |                             nan      |                         0.0785 |
| up_to_8_steps |                          0.0755 |                               0.0311 |                        -0.2763 |

## Extra-slot diagnostic

| acid_type   | stage         |   experiments |   mean_unused_slot_mole_fraction |   median_unused_slot_mole_fraction |   unused_slot_fraction_lt_0p1_pct |   unused_slot_fraction_lt_0p2_pct |
|:------------|:--------------|--------------:|---------------------------------:|-----------------------------------:|----------------------------------:|----------------------------------:|
| diprotic    | final         |            65 |                           0.3469 |                             0.3358 |                              7.69 |                             20    |
| diprotic    | initial       |            65 |                           0.3406 |                             0.32   |                              9.23 |                             21.54 |
| diprotic    | up_to_8_steps |            65 |                           0.347  |                             0.3358 |                              7.69 |                             20    |
| monoprotic  | final         |            58 |                           0.6429 |                             0.6755 |                              0    |                              1.72 |
| monoprotic  | initial       |            58 |                           0.6279 |                             0.6413 |                              0    |                              1.72 |
| monoprotic  | up_to_8_steps |            58 |                           0.6433 |                             0.6755 |                              0    |                              1.72 |
