# Direction-assisted volume-policy RL comparison

This report is generated from the saved task-level outputs.

## Protocol

The neural actor receives current pH, target pH, measured pH change, current-minus-target error, and the last requested volume. It selects only one of 1,000 dosing volumes from 0.01 to 10.00 mL.
A common external rule selects base when measured pH is below the target and acid otherwise. The titrant concentration is fixed at 0.1 M. No overshoot-based action masking, automatic dose reduction, or dilute-titrant switching is used.
This allocation matches the deployed 1,000-output policy: the algorithms are compared as volume policies inside the same direction-assisted controller.

Algorithms: ppo, a2c, reinforce; initialization modes: imitation and random; seeds: [101, 202, 303, 404, 555].
Each run used 25,000 environment interactions, a 5,000-task training pool, and 1,000 held-out nominal test tasks.
For each seed, both initialization modes used the identical generated training and test tasks, reward, optimizer settings, and evaluation action rule.

## Aggregate results (mean +/- SD across training seeds)

| Algorithm | Initialization | Success (%) | Strict +/-0.05 (%) | Severe failure (%) | Steps | Volume (mL) |
|---|---|---:|---:|---:|---:|---:|
| A2C | imitation | 85.72 +/- 2.99 | 33.14 +/- 2.67 | 8.86 +/- 3.06 | 13.91 +/- 1.84 | 19.16 +/- 1.17 |
| A2C | random | 53.00 +/- 6.91 | 21.84 +/- 6.69 | 34.72 +/- 5.40 | 33.00 +/- 2.98 | 28.04 +/- 9.85 |
| PPO | imitation | 88.12 +/- 0.47 | 35.48 +/- 2.49 | 6.40 +/- 0.55 | 12.63 +/- 0.23 | 19.78 +/- 0.35 |
| PPO | random | 43.00 +/- 13.77 | 17.08 +/- 9.18 | 44.38 +/- 18.16 | 35.37 +/- 5.27 | 38.15 +/- 31.96 |
| REINFORCE | imitation | 72.38 +/- 10.46 | 26.74 +/- 4.48 | 22.72 +/- 10.80 | 20.02 +/- 4.95 | 16.06 +/- 2.25 |
| REINFORCE | random | 40.56 +/- 13.99 | 17.12 +/- 9.42 | 48.34 +/- 15.08 | 36.90 +/- 4.93 | 52.27 +/- 63.21 |

## Initialization-paired tests

The task-level McNemar test is conditional on the frozen model from each seed. The seed-level difference and sign-flip test treat the 5 training run(s) as the independent units.

| Algorithm | Random - imitation success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |
|---|---:|---|---:|---:|
| A2C | -32.72 | [-36.76, -28.44] | 0 | 0.0625 |
| PPO | -45.12 | [-56.88, -35.28] | 0 | 0.0625 |
| REINFORCE | -31.82 | [-45.74, -17.90] | 0 | 0.0625 |

## Algorithm-paired tests

Differences are reported as algorithm B minus algorithm A in success-rate percentage points. Task-level tests are conditional on the frozen trained models; seed-level results use the 5 independent training run(s).

| Initialization | Algorithm A | Algorithm B | B - A success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |
|---|---|---|---:|---|---:|---:|
| imitation | A2C | PPO | 2.40 | [0.46, 4.80] | 8.474e-24 | 0.0625 |
| imitation | A2C | REINFORCE | -13.34 | [-23.28, -2.52] | 2.204e-129 | 0.125 |
| imitation | PPO | REINFORCE | -15.74 | [-24.18, -7.64] | 1.016e-207 | 0.0625 |
| random | A2C | PPO | -10.00 | [-23.98, 0.64] | 1.384e-39 | 0.25 |
| random | A2C | REINFORCE | -12.44 | [-26.54, -2.36] | 7.215e-80 | 0.0625 |
| random | PPO | REINFORCE | -2.44 | [-8.16, 5.04] | 0.000149 | 0.5625 |

## Interpretation guardrail

The random-initialization condition is an independent training control, not a claim that random initialization is universally inferior. Conclusions should be based on paired seed-level differences, uncertainty, learning curves, and the prespecified protocol. These results evaluate volume-policy learning with a shared external direction rule; they are not policy-only direction-and-volume control results.
