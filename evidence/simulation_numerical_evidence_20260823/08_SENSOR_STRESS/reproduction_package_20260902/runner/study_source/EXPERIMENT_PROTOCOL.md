# Robust PF Distillation and PPO Refinement Protocol

## Scope

This package retrains a dosing-volume policy from the improved particle-filter controller and then refines the selected imitation policy with PPO. PPO is the only reinforcement-learning algorithm retained because the preceding matched screening identified it as the strongest and most stable candidate. The formal comparison is therefore particle-filter teacher versus imitation policy versus PPO-refined policy.

## Teacher and task distribution

The teacher is `pf_pka_conc_variable_k`, which jointly represents analyte concentration, ordered effective pKa values, and the number of effective pairs K in {1, 2, 3}. Nominal tasks use K probabilities 0.40, 0.35, and 0.25; analyte concentration is log-uniform over 0.03-0.18 M; initial volume is uniform over 8-16 mL; and 15% of multiprotic tasks contain overlapping pKa transitions. Initial neutralization states span the titration curve. The accepted task set uses fixed acid- and base-direction quotas: even-sized sets are exactly 1:1 and odd-sized sets differ by at most one task. An unreachable chemistry is resampled for the same direction slot, so reachability rejection cannot bias the accepted direction ratio. Targets are sampled from near, medium, and far pH shifts and must be reachable using no more than 30 mL of 0.1 M titrant.

Training, validation, PPO training, PPO validation, and locked testing use separate fixed seed spaces. The locked test is not used for checkpoint selection.

## Imitation data and model

The state is `[current pH, target pH, recent pH change, current pH error, previous requested dose]`. Teacher generation, neural-policy training, validation, and testing all use the same 0.01 pH sensor resolution. The network selects one of 1,000 volume classes from 0.01 to 10.00 mL. A common external direction rule adds base when current pH is below the target and acid otherwise.

Teacher trajectories are retained only when the endpoint absolute error is at most 0.10 pH, the trajectory uses at most 30 steps and 8 target crossings, dose overhead relative to the oracle volume is at most 20 mL, all values are finite, and a predefined quality score is at least 0.30. The score combines endpoint accuracy, step efficiency, target-crossing count, and dose overhead. Near-duplicate records are keyed by the state rounded to 0.001 plus the action class, and only the higher-quality record is retained. The generator automatically adds independent candidate-task batches until the minimum unique-state count is reached and coverage requirements are met for current and target pH, error, dose, concentration, initial volume, acid type, pair count, pKa family, and difficulty. Joint coverage is additionally required for direction by error, direction by pKa family, and direction by acid type by difficulty. The standard profile requires at least 60,000 training and 12,000 validation states, exceeding the 47,810 and 10,247 effective records in the archived imitation datasets.

On 25% of training decisions, a log-normal plus small additive perturbation is applied to the executed teacher dose while the unperturbed teacher dose remains the supervised label. This exposes the student to recovery states. Because the two directions can still produce different trajectory lengths, data generation requires at least half of the state target from each direction, and every imitation-learning epoch uses exactly equal numbers of acid- and base-direction states. Loss weights additionally reduce imbalance across direction, pH-error bin, dose bin, acid type, and teacher quality.

The imitation loss combines weighted cross-entropy with label smoothing and a SmoothL1 penalty on expected dose volume. Evaluation reports volume MAE and tolerance accuracy in addition to class accuracy. Each imitation seed is selected by validation closed-loop success, then final absolute pH error, then teacher-volume MAE. The final imitation checkpoint is chosen across the independent imitation training seeds by the same rule.

## PPO refinement

Each PPO actor starts from the selected imitation checkpoint. The critic is newly initialized. PPO uses generalized advantage estimation, clipped policy updates, a fixed environment-interaction budget, and moderate training-only randomization of observation noise, delivered volume, titrant strength, and sensor response. Nominal validation tasks select the checkpoint. The imitation starting point is itself an eligible checkpoint, so an RL run cannot replace it unless validation performance improves according to the predefined selection rule.

## Final evaluation

The particle-filter teacher, selected imitation policy, and independently trained PPO checkpoints are evaluated on the nominal locked test. All methods control from 0.01 pH sensor observations, but success, strict success, severe failure, and final error are computed from the underlying unquantized chemical pH; measured success and false stops are reported separately. Two additional suites examine close-pKa systems and a wider 0.015-0.25 M concentration range. The package exports task-level outcomes, per-run and aggregate summaries, Holm-adjusted paired exact McNemar tests among the teacher, imitation policy, and PPO policies, learning curves, and success-rate figures. Between-run standard deviations are reported only when more than one independent run is available.

The `quick` profile verifies implementation only. The `standard` or `full` profile must be used for scientific interpretation, and the profile and hardware metadata in `RUN_CONFIG.json` must accompany reported results.

The chemistry pH bisection uses 60 iterations, and task-oracle and controller-volume bisections use 32 iterations. These bounds are far tighter than the 0.01 pH sensor and 0.01 mL action resolutions; the exact iteration counts are recorded in `RUN_CONFIG.json`.

## External-rule selection and posterior diagnostics

Before neural training, a matched five-seed, 3,000-task-per-seed ablation selects the teacher controller. The formal package stops if `hybrid_full` is not the highest mean-success arm; this prevents silently training from a different policy than the documented teacher. The selected policy is the improved PF together with the complete inherited dose-control logic.

The posterior diagnostic suite follows the selected controller for fixed observation counts 0, 1, 2, 3, 5, 8, and 12 on the same task cohort, and records the natural control endpoint separately. It compares the fitted response curve from the posterior mean parameters with the true curve over signed -100 to +100 mL primary titrant relative to the initial state. It reports curve RMSE, MAE, correlation, R2, concentration error and approximate 95% coverage, effective-pair count accuracy and true-model probability, and pKa errors conditional on correct pair count.

The distribution analysis reuses these task-level posterior records without rerunning PF trajectories. For each fixed observation count it reports the mean, SD, median, IQR, 5th-95th percentiles, and fractions below predefined RMSE thresholds. Changes between checkpoints are strictly paired within the same simulated task and are evaluated descriptively with paired Wilcoxon and sign tests with Holm adjustment. Reproducibility is reported separately at the benchmark-seed level: the mean paired change is calculated within each seed, followed by a five-seed mean, SD, t interval, and exact one-sided sign-flip test. Task-level and seed-level inference are explicitly distinguished. Natural-end RMSE is stratified by dosing direction, target distance, true pair count, and pKa family. Full-curve RMSE, local endpoint control error, and chemical-parameter recovery are treated as separate outcomes.

## RL effectiveness audit

The main locked evaluation is supplemented by a predefined paired intervention audit. All PPO actors start from the exact same selected imitation checkpoint. Five independent evaluation seeds each contribute 500 unseen nominal tasks per intervention. Imitation and every PPO seed receive identical tasks and identical random draws under nominal conditions, actuator log-noise SD 0.10, titrant-strength scales 0.90 and 1.10, sensor-noise SD 0.05 pH, response fraction 0.70, and a combined unseen perturbation. These values exceed the PPO training randomization ranges.

The primary endpoint is the validation-selected PPO minus imitation true-success difference under the combined unseen perturbation. The audit reports a paired 95% confidence interval, exact McNemar test, Holm-adjusted secondary tests, all five PPO training-seed effects, overshoot-conditioned recovery, learning-curve before/after values, checkpoint hashes, and actor-parameter displacement from imitation. The intervention results are not used to choose the PPO checkpoint. Evidence is labeled supported only when the selected PPO improves the primary endpoint with p < 0.05, at least four of five PPO training seeds show a positive effect, and the selected checkpoint occurs after nonzero RL interaction. Otherwise the report states that effectiveness was not supported by the predefined criteria.
