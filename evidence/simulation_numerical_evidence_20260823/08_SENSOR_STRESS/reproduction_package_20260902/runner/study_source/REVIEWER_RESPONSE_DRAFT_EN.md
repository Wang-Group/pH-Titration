# Draft response to Reviewer 1: new PF, external-rule ablation, imitation, and PPO evidence

This document addresses the reviewer points that can be answered by the present simulation package. Page, figure, table, and section numbers must be updated after the revised manuscript is typeset. Claims that require wet-laboratory replication, PID retuning, unit correction, or literature verification are explicitly marked as remaining author actions.

## Major point 1: value of the reinforcement-learning stage

**Reviewer comment.** The reported 0.4 percentage-point gain did not demonstrate that reinforcement learning refined the imitation policy. The authors should either demonstrate a clear statistically significant regime or present RL as a marginal attempted refinement.

**Response.** We agree that the original single-seed point estimate was insufficient. We replaced the original REINFORCE refinement with PPO and retrained five independent PPO runs, each initialized from the exact same validation-selected imitation checkpoint. Checkpoints were selected only on independent PPO validation tasks. On 1,000 nominal locked tasks, the validation-selected PPO network (training seed 303) achieved 93.4% success versus 89.1% for imitation, a paired difference of +4.30 percentage points (exact McNemar p = 8.91 x 10^-7). It also reduced successful-task steps from 8.63 to 5.68 and false stops from 2.6% to 1.2%. On close-pKa and wide-concentration stress tasks, the paired success gains were +6.33 points (p = 0.00540) and +4.67 points (p = 0.0201), respectively.

We additionally performed a predefined intervention audit beyond the main locked evaluation: five independent evaluation seeds, 500 tasks per seed and perturbation, with identical tasks and random draws for imitation and all five PPO policies. The selected PPO improved nominal unseen success by +4.64 points (95% CI +3.40 to +5.88; p = 2.03 x 10^-13) and success under 0.05-pH sensor noise by +12.96 points (95% CI +10.75 to +15.17; p = 1.08 x 10^-29). However, it deteriorated under a severe sensor-response lag (response fraction 0.70) by -9.24 points (95% CI -11.61 to -6.87; p = 3.79 x 10^-14). Under the predefined combined unseen perturbation, the +1.48-point gain was not significant (95% CI -1.03 to +3.99; p = 0.261), although all five PPO seeds had positive task-level success differences and the mean was +2.06 +/- 1.54 points.

We have therefore revised the claim. PPO provides statistically supported, regime-specific refinement, particularly for nominal tasks and sensor noise, but the data do not support general robustness. The response-lag failure is now reported explicitly. We no longer use a small aggregate point estimate as evidence of universal RL benefit.

## Major point 2: multiple seeds and paired significance tests

**Reviewer comment.** Headline comparisons relied on one seed; repeat the benchmark across multiple seeds, report mean +/- SD, and use paired tests.

**Response.** We implemented a multi-seed protocol. The external-rule PF ablation uses five independent seeds (101, 202, 303, 404, and 555), 3,000 matched tasks per seed, three controllers per task, and 1,000 particles, yielding 45,000 task-level controller outcomes. The full PF controller achieved 95.36 +/- 0.59% success, compared with 73.63 +/- 0.57% without the overshoot cap and 80.11 +/- 1.01% for posterior-direct dosing. Severe failure rates were 1.30%, 20.89%, and 8.58%, respectively.

Imitation learning was repeated with three training seeds, and PPO with five training seeds and 100,000 interactions per seed. On the 1,000-task nominal locked suite, the five PPO runs averaged 89.54 +/- 2.23% success. We report all task-level outcomes, per-run summaries, exact paired McNemar tests, paired Wilcoxon tests for continuous metrics, Holm-adjusted p values where applicable, and checkpoint SHA-256 hashes. Checkpoint selection never used locked-test or intervention outcomes.

The original numerical inconsistency between text and figure counts should be removed by regenerating every manuscript table and figure from the exported CSV files in this package.

## Major point 3: validation and limits of Bayesian interpretability

**Reviewer comment.** The manuscript did not show convergence of inferred concentration and pKa values, and the no-prior-knowledge claim was too strong.

**Response.** We added posterior diagnostics on 1,500 independent tasks (five seeds x 300 tasks), each with 1,000 particles. The same task cohort is evaluated after 0, 1, 2, 3, 5, 8, and 12 observations, and at the natural control endpoint. The complete fitted response curve is compared with the true curve over signed -100 to +100 mL of 0.1 M titrant relative to the initial state. We report curve RMSE, MAE, correlation, R2, concentration error and approximate 95% coverage, effective-pair-count accuracy and true-model probability, and pKa error conditional on correct pair count.

At the natural endpoint (6.32 +/- 0.56 observations), control success was 95.80 +/- 1.92%, but curve RMSE remained 1.1370 +/- 0.0523 pH, concentration relative error was 47.33 +/- 3.91%, effective-pair-count accuracy was 46.07 +/- 2.55%, and pKa MAE conditional on correct pair count was 0.8494 +/- 0.0751. Thus effective closed-loop control does not imply accurate recovery of all chemical parameters. We have removed or qualified claims of full chemical identification and “no prior knowledge.” The revised interpretation is that the PF supplies a structured, uncertainty-aware latent representation useful for control under a declared prior family; it is not demonstrated here as a universally accurate chemical-identification method.

We additionally report the full task distribution rather than only seed-level means. Across the 1,500 strictly paired tasks, full-curve RMSE had mean 1.3894 pH and median 1.1334 [IQR 0.7696-1.7348] at the prior, versus mean 1.1388 and median 0.9643 [0.4225-1.6725] after 12 observations. The fraction with RMSE <= 0.5 pH increased from 9.87% to 28.07%. The paired 0-to-12 mean change was -0.2506 pH, the median change was -0.1811, and 60.0% of tasks improved (paired Wilcoxon p = 3.83 x 10^-19). All five benchmark seeds had a negative mean change; the across-seed change was -0.2506 +/- 0.0453 pH with a 95% t interval of -0.3069 to -0.1943 and an exact one-sided five-seed sign-flip p = 0.03125. The largest population gain occurred in the first observation and improvements largely plateaued after approximately five observations. We therefore do not claim monotonic convergence for every task: approximately 40% did not have a lower final RMSE.

The distributional analysis also identifies limitations hidden by aggregate means. At 12 observations, concentration-error median was 29.15% with IQR 8.87%-63.42%, indicating a long right tail. Conditional pKa MAE median improved from 1.551 to 0.543, but effective-pair-count accuracy increased only from 42.33% to 45.87%; conditional pKa accuracy is therefore reported together with K accuracy. At natural termination, curve fitting was harder for acid-direction than base-direction tasks (mean RMSE 1.399 versus 0.875 pH), for near targets than far targets (1.426 versus 0.617), and for overlapping-pKa systems than separated systems (1.700 versus 1.127). Finally, global curve RMSE and final local control error were only weakly associated (Spearman rho = 0.157). These data support a limited interpretability claim and explicitly separate control success, full-curve fitting, and parameter recovery.

Sensitivity to the exact user-supplied `pKa,ref` in the earlier formulation is not part of this new variable-K PF package because the new model uses explicit priors over K, concentration, and ordered pKa values. If the original `pKa,ref` formulation remains anywhere in the revised manuscript or SI, the author must either add a matched prior-sensitivity study or remove the stronger no-prior-knowledge wording.

## Minor point: dosing-rule heuristics and particle-filter framing

**Reviewer comment.** The hand-designed factors require ablation, and the method should be named as a sequential importance-resampling particle filter.

**Response.** We now explicitly describe the method as a sequential importance-resampling particle filter. We conducted a matched external-rule ablation. The complete hybrid controller achieved 95.36 +/- 0.59% success. Removing the overshoot-volume cap reduced success to 73.63 +/- 0.57%, increased severe failures from 1.30% to 20.89%, and increased mean dose from 7.82 to 58.64 mL. Posterior-direct dosing achieved 80.11 +/- 1.01% success and used 106.40 mL on average. These results show that external dose shaping is essential to the current controller rather than a decorative implementation detail. Accordingly, the best-performing full PF plus control logic was used as the imitation teacher.

The revised limitations should state that high control success belongs to the complete PF-plus-control architecture, not to posterior inference alone. Long-horizon sample impoverishment remains a general PF concern; the present control trajectories are short (natural endpoint 6.32 observations on average), and all formal runs used 1,000 particles.

## Major points 9 and 10: RL algorithm, learning curves, and measurement noise

**Reviewer comment.** REINFORCE was high variance and undertrained; justify the algorithm and quantify realistic measurement noise.

**Response.** We replaced REINFORCE with PPO using generalized advantage estimation, clipped policy updates, five independent seeds, and 100,000 environment interactions per seed. The critic was initialized independently while every actor started from the same imitation checkpoint. The imitation checkpoint remained an eligible zero-interaction checkpoint, so PPO could not replace it unless independent validation improved. Complete learning curves are provided for every seed. The selected PPO checkpoint was seed 303 at 100,023 interactions with 92.2% independent validation success, compared with 88.4% at its imitation initialization.

Training included measurement noise up to 0.02 pH, actuator log-noise SD 0.02, titrant-strength scaling 0.97-1.03, and sensor response fraction 0.95-1.00. The separate unseen audit used stronger perturbations. At sensor-noise SD 0.05 pH, selected PPO success was 75.96 +/- 1.86% versus 63.00 +/- 1.09% for imitation, with false stops reduced by 10.36 percentage points. Conversely, severe sensor-response lag exposed a failure mode. We now report both results and avoid claiming blanket robustness.

## Remaining reviewer items not resolved by this simulation package

The following points require manuscript editing, new physical experiments, baseline reruns, or verified literature work and must not be claimed as completed from these files alone:

- Correct the benchmark amount/concentration units and regenerate affected results if the implemented distribution differs from the manuscript.
- Provide replicated robotic experiments, recovery efficiency, uncertainty, operator details, and appropriately tempered human-comparison claims.
- Retune and document the PID baseline on the same development distribution using comparable effort.
- Add a simple adaptive-controller comparison on the real systems if claiming that Bayesian structure is necessary for out-of-distribution deployment.
- Explain and validate any wastewater correction factor.
- Verify Cu-SSA constants, Job's-method assumptions, and the cited autonomous-experimentation literature.
- Add the required author-contributions section and manuscript line/page references.
