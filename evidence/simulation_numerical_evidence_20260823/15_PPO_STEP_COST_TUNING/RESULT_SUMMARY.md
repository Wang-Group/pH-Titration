# PPO step-cost sensitivity screen (0 to 0.01)

Four coefficients were retrained with nominal training seed 303 under the standard PPO protocol. Checkpoint selection used only the common 500-task validation set; the locked benchmark was not used for selection.

| Step cost | Validation success (%) | Validation final error | Validation steps | Best checkpoint | Best step |
|---:|---:|---:|---:|---|---:|
| 0.0000 | 88.40 | 0.1148 | 12.18 | imitation_start | 0 |
| 0.0025 | 89.20 | 0.1278 | 9.98 | ppo_final_batch | 100005 |
| 0.0050 | 90.80 | 0.1253 | 9.80 | ppo | 60000 |
| 0.0100 | 88.40 | 0.1148 | 12.18 | imitation_start | 0 |

The repeated 0.005 run was the best retraining within this local screen. It is not the independently validation-selected original PPO checkpoint, which used the same coefficient but followed a different stochastic training trajectory.

This is a one-retraining-run-per-coefficient sensitivity screen. It does not isolate a causal coefficient effect or estimate a universally optimal coefficient.
