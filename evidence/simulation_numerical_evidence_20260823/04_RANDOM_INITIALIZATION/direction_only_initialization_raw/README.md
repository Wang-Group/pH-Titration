# Direction-Only Initialization Study

This dataset compares policy initialization under a common nominal simulated
titration interface.

## Protocol

- evaluation seeds: `901, 902, 903, 904, 905`;
- tasks per evaluation seed: `3,000`;
- PF particles: `1,000`;
- action classes: `0.01` to `10.00` mL;
- maximum episode length: `50` actions;
- success: final true absolute pH error no greater than `0.10`;
- strict success: final true absolute pH error no greater than `0.05`;
- no stochastic domain randomization during this formal evaluation.

PPO conditions were trained with five model seeds (`101, 202, 303, 404,
555`) and evaluated on the same locked task sets. Thus one PPO condition has
5 model seeds x 5 evaluation sets x 3,000 tasks; the model-level summary first
aggregates within each frozen model and then reports the mean and sample
standard deviation across model seeds.

## Contents

- `RUN_CONFIG.json`: protocol and seed mapping;
- `MODEL_PROVENANCE.json`: checkpoint provenance and validation metadata;
- `locked_tasks/`: the five reusable task sets;
- `task_results/`: compressed task-level result tables;
- `per_model_seed_summary.csv`: model-level summary;
- `aggregate_summary.csv`: across-model summary;
- `paired_success_vs_pf.csv`: paired comparisons with posterior-direct PF;
- `paired_rl_effect_tests.csv`: paired comparisons against each initialization.

The locked task sets are intentionally shared across conditions. Reuse is part
of the matched comparison design and must not be described as independent
replication of the task generator.

