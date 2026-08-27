# PPO step-cost sensitivity materials

This block reports the requested local coefficient range from 0 to 0.01: `0`, `0.0025`, `0.005`, and `0.01`. Each coefficient has one stochastic PPO retraining run using nominal seed 303, the same frozen imitation checkpoint, identical 5,000-task training and 500-task validation sets, and a 100,000-interaction budget.

The original validation-selected PPO also used step cost `0.005`. It is kept separate from the independently retrained `0.005` checkpoint because the two runs followed different stochastic training trajectories and selected different networks.

Key files:

- `candidate_validation_summary.csv`: validation-only checkpoint-selection results.
- `evaluation_full_5x3000/RESULT_SUMMARY.md`: held-out five-set benchmark summary.
- `evaluation_full_5x3000/tasks/`: task-level outcomes for all five reported networks.
- `step_cost_*/seed_303/`: checkpoints, learning curves, and training/validation task manifests.
- `../../../../scripts/run_ppo_step_cost_tuning.py`: validation aggregation.
- `../../../../scripts/evaluate_ppo_step_cost_tuning_full_benchmark.py`: locked evaluation and aggregation.

The default reward remains the full reward with `--step-cost 0.005`. The screen is exploratory because it contains one retraining run per coefficient; it should not be interpreted as a causal hyperparameter comparison.
