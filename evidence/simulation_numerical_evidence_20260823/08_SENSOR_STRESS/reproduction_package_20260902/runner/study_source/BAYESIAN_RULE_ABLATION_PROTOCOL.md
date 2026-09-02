# Bayesian External-Rule Ablation Protocol

## Question

The improved `pf_pka_conc_variable_k` particle filter replaced the inference state but retained the submitted controller's dose-selection logic. This experiment tests whether the observed control performance depends on those inherited external dose rules.

## Matched controller arms

1. `hybrid_full`: the improved particle filter plus the inherited pH-rate, posterior-uncertainty, buffering-response, required-volume, and tanh dose shaping. After an overshoot or increased error, later candidate volumes are capped at half the dose that triggered the event.
2. `hybrid_no_overshoot_cap`: identical to `hybrid_full`, except the persistent post-overshoot candidate-volume cap is ignored. This isolates the specific rule highlighted by the reviewer.
3. `posterior_direct`: the particle-filter posterior equilibrium model directly supplies the root volume needed to reach the target. If the posterior target is outside the 0.01-10.00 mL action interval, the nearest boundary is used. No overshoot cap or heuristic dose-shaping factor is applied.

All arms retain the same necessary interface and safety conventions: primary 0.1 M titrant, base below the target and acid above the target, 0.01-10.00 mL action bounds, the same endpoint tolerance, and the same maximum number of steps. Secondary/dilute titrant switching is disabled in all arms. Particle count, particle seed, true task, and initial state are matched across the three policies. Decisions use pH quantized to 0.01, while endpoint success and error use unquantized true chemical pH. Measured success and false stops are retained separately.

## Formal design

The standard script generates five independent 3,000-task sets using seeds 101, 202, 303, 404, and 555. Each task is run under all three policies with 1,000 particles and the same filter seed. Thus, success/failure and continuous metrics are paired at the task level.

Reported metrics include success within 0.10 pH, strict success within 0.05 pH, severe failure above 0.50 pH error, successful and overall step counts, target crossings, total added volume, final absolute error, controller computation time, and activation of the overshoot threshold.

Exact McNemar tests compare paired success outcomes for every seed and for the pooled task set. Paired Wilcoxon tests compare step count, crossings, total volume, and final error. Holm adjustment is applied within each exported test family. Means and sample standard deviations across the five independently generated task sets are reported separately from pooled task-level tests.

## Running

Run `RUN_BAYESIAN_RULE_ABLATION_QUICK.cmd` first to verify the environment. Run `RUN_BAYESIAN_RULE_ABLATION.cmd` for the formal five-seed experiment. Completed seeds have independent completion markers and are skipped when the command is resumed.

## Teacher-selection rule

The controller arm with the highest mean success across the five independent task seeds is selected before neural training. If `hybrid_full` is not the winner, the integrated pipeline stops rather than silently training from a mismatched teacher implementation. The selected policy and all three mean success rates are written to `TEACHER_SELECTION.json`.

## Posterior diagnostics

The selected PF controller is additionally evaluated on five independent seeds and 300 tasks per seed. The same task cohort is continued to fixed observation counts 0, 1, 2, 3, 5, 8, and 12, even if the control target is reached earlier. This prevents survivor bias across observation counts. The natural control endpoint is recorded separately before any forced continuation.

At each checkpoint, the posterior mean concentration, MAP effective-pair count, MAP-model pKa means and uncertainties are recorded. The fitted complete response curve is compared with the true curve over signed 0.1 M titrant additions from -100 to +100 mL relative to the initial chemical state. Reported metrics include curve RMSE, MAE, correlation, R2, percentage within 0.25 pH, concentration relative error and approximate 95% coverage, pair-count accuracy and true-model probability, and pKa MAE/coverage conditional on correct pair count.
