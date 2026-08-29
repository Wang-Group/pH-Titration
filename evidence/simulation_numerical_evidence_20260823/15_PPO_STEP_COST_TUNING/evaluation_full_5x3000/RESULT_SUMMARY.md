# PPO step-cost sensitivity: complete 5 x 3,000 benchmark

The independently validation-selected original PPO and four retrained step-cost checkpoints were evaluated on the same five locked benchmark sets (3,000 tasks per set; 15,000 tasks per checkpoint). The benchmark was used for held-out reporting, not checkpoint selection.

| Network | Step cost | Success mean +/- SD (%) | Successful additions | Overshoots | Total volume (mL) | Final error (pH) |
|---|---:|---:|---:|---:|---:|---:|
| original validation-selected PPO | 0.005 | 93.95 +/- 0.63 | 5.20 | 3.05 | 7.52 | 0.0785 |
| retrained checkpoint | 0 | 89.17 +/- 0.51 | 7.83 | 2.93 | 7.47 | 0.0953 |
| retrained checkpoint | 0.0025 | 88.66 +/- 0.44 | 6.45 | 2.82 | 7.01 | 0.1030 |
| independently retrained 0.005 checkpoint | 0.005 | 91.86 +/- 0.56 | 6.49 | 2.70 | 7.12 | 0.0961 |
| retrained checkpoint | 0.01 | 89.17 +/- 0.51 | 7.83 | 2.93 | 7.47 | 0.0953 |

None of the retrained checkpoints outperformed the original validation-selected PPO. The two 0.005 rows are intentionally retained: they use the same reward coefficient but correspond to different stochastic training trajectories and selected checkpoints. Their difference demonstrates that this one-run-per-coefficient screen cannot attribute performance differences solely to step cost.

Task-level results, benchmark-seed summaries, paired comparisons, checkpoints, and task manifests are included in this directory.
