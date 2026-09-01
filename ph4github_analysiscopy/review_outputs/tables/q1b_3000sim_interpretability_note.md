# Q1b Quantitative Analysis

- Dataset: 3000 simulated tasks from `experiment_summary.csv`.
- Step-budget view: metrics at `min(final step, 8)` to answer the reviewer's question about whether roughly eight titration measurements are sufficient.

## Headline observations

- Overall matched posterior pKa MAE changes from 0.878 initially to 0.884 by step <= 8, and 0.885 at the final step.
- By step <= 8, the fraction of tasks for which all true pKa values are recovered within +-1.0 pH units is 47.10%; at the final step it is 47.13%.
- The mean posterior pKa standard deviation is 0.200 initially, 0.190 by step <= 8, and 0.192 at the final step.

## Predictive usefulness of the latent state

- To quantify whether the Bayesian latent state remains chemically meaningful even when exact pKa recovery is imperfect, I also measured the one-step pH prediction error made by the current posterior model before each titration action.

| acid_type   | stage          |   transitions |   mean_prediction_abs_error |   median_prediction_abs_error |   prediction_error_le_0p1_pct |   prediction_error_le_0p2_pct |
|:------------|:---------------|--------------:|----------------------------:|------------------------------:|------------------------------:|------------------------------:|
| diprotic    | all_steps      |         10310 |                      2.9005 |                        2.2343 |                          3.06 |                          5.97 |
| diprotic    | within_8_steps |          6681 |                      2.4974 |                        1.5232 |                          3.7  |                          7.56 |
| monoprotic  | all_steps      |         13302 |                      3.7814 |                        3.4862 |                          2.53 |                          4.12 |
| monoprotic  | within_8_steps |          6386 |                      3.6114 |                        2.285  |                          4.2  |                          7.02 |
| triprotic   | all_steps      |          8457 |                      1.1989 |                        0.9582 |                         15.36 |                         19.47 |
| triprotic   | within_8_steps |          6309 |                      1.2029 |                        0.9954 |                         14.01 |                         18.32 |
| overall     | all_steps      |         32069 |                      2.8172 |                        2.0847 |                          6.08 |                          8.77 |
| overall     | within_8_steps |         19376 |                      2.443  |                        1.4449 |                          7.22 |                         10.88 |

## Species-count caveat

- The current Bayesian controller keeps exactly three latent buffer slots throughout the episode. It does not maintain an explicit posterior over the number of species.
- As a diagnostic, I tracked the inferred mole fraction assigned to unmatched latent slots for monoprotic and diprotic tasks after optimal pKa matching. Large residual mass in unmatched slots means the model is not cleanly deactivating extra components.

## Correlations

| stage         |   corr_posterior_std_vs_pka_mae |   corr_posterior_std_vs_abs_pH_error |   corr_pka_mae_vs_abs_pH_error |
|:--------------|--------------------------------:|-------------------------------------:|-------------------------------:|
| final         |                         -0.0051 |                               0.0624 |                         0.0003 |
| initial       |                        nan      |                             nan      |                        -0.0286 |
| up_to_8_steps |                          0.0303 |                               0.0122 |                        -0.1906 |

## Extra-slot diagnostic

| acid_type   | stage         |   experiments |   mean_unused_slot_mole_fraction |   median_unused_slot_mole_fraction |   unused_slot_fraction_lt_0p1_pct |   unused_slot_fraction_lt_0p2_pct |
|:------------|:--------------|--------------:|---------------------------------:|-----------------------------------:|----------------------------------:|----------------------------------:|
| diprotic    | final         |          1031 |                           0.3326 |                             0.3304 |                             11.83 |                             25.8  |
| diprotic    | initial       |          1031 |                           0.3353 |                             0.3325 |                             11.45 |                             25.61 |
| diprotic    | up_to_8_steps |          1031 |                           0.3336 |                             0.3307 |                             11.74 |                             25.41 |
| monoprotic  | final         |           994 |                           0.6682 |                             0.6683 |                              0    |                              1.11 |
| monoprotic  | initial       |           994 |                           0.6677 |                             0.6691 |                              0    |                              1.11 |
| monoprotic  | up_to_8_steps |           994 |                           0.6681 |                             0.6695 |                              0    |                              1.11 |
