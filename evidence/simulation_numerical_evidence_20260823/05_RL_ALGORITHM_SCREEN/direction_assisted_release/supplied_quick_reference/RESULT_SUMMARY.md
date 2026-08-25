# Direction-assisted volume-policy RL comparison

This report is generated from the saved task-level outputs.

## Protocol

The neural actor receives current pH, target pH, measured pH change, current-minus-target error, and the last requested volume. It selects only one of 1,000 dosing volumes from 0.01 to 10.00 mL.
A common external rule selects base when measured pH is below the target and acid otherwise. The titrant concentration is fixed at 0.1 M. No overshoot-based action masking, automatic dose reduction, or dilute-titrant switching is used.
This allocation matches the deployed 1,000-output policy: the algorithms are compared as volume policies inside the same direction-assisted controller.

Algorithms: ppo, a2c, reinforce; initialization modes: imitation and random; seeds: [101].
Each run used 50 environment interactions, a 20-task training pool, and 10 held-out nominal test tasks.
For each seed, both initialization modes used the identical generated training and test tasks, reward, optimizer settings, and evaluation action rule.

## Aggregate results (mean +/- SD across training seeds)

| Algorithm | Initialization | Success (%) | Strict +/-0.05 (%) | Severe failure (%) | Steps | Volume (mL) |
|---|---|---:|---:|---:|---:|---:|
| A2C | imitation | 90.00 +/- 0.00 | 30.00 +/- 0.00 | 10.00 +/- 0.00 | 15.00 +/- 0.00 | 29.96 +/- 0.00 |
| A2C | random | 10.00 +/- 0.00 | 0.00 +/- 0.00 | 90.00 +/- 0.00 | 45.20 +/- 0.00 | 7.68 +/- 0.00 |
| PPO | imitation | 90.00 +/- 0.00 | 40.00 +/- 0.00 | 10.00 +/- 0.00 | 14.20 +/- 0.00 | 29.76 +/- 0.00 |
| PPO | random | 10.00 +/- 0.00 | 0.00 +/- 0.00 | 90.00 +/- 0.00 | 45.20 +/- 0.00 | 7.68 +/- 0.00 |
| REINFORCE | imitation | 90.00 +/- 0.00 | 70.00 +/- 0.00 | 10.00 +/- 0.00 | 14.00 +/- 0.00 | 29.81 +/- 0.00 |
| REINFORCE | random | 10.00 +/- 0.00 | 0.00 +/- 0.00 | 90.00 +/- 0.00 | 45.20 +/- 0.00 | 7.68 +/- 0.00 |

## Initialization-paired tests

The task-level McNemar test is conditional on the frozen model from each seed. The seed-level difference and sign-flip test treat the 1 training run(s) as the independent units.

| Algorithm | Random - imitation success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |
|---|---:|---|---:|---:|
| A2C | -80.00 | [-80.00, -80.00] | 0.02344 | 1 |
| PPO | -80.00 | [-80.00, -80.00] | 0.02344 | 1 |
| REINFORCE | -80.00 | [-80.00, -80.00] | 0.02344 | 1 |

## Algorithm-paired tests

Differences are reported as algorithm B minus algorithm A in success-rate percentage points. Task-level tests are conditional on the frozen trained models; seed-level results use the 1 independent training run(s).

| Initialization | Algorithm A | Algorithm B | B - A success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |
|---|---|---|---:|---|---:|---:|
| imitation | A2C | PPO | 0.00 | [0.00, 0.00] | 1 | 1 |
| imitation | A2C | REINFORCE | 0.00 | [0.00, 0.00] | 1 | 1 |
| imitation | PPO | REINFORCE | 0.00 | [0.00, 0.00] | 1 | 1 |
| random | A2C | PPO | 0.00 | [0.00, 0.00] | 1 | 1 |
| random | A2C | REINFORCE | 0.00 | [0.00, 0.00] | 1 | 1 |
| random | PPO | REINFORCE | 0.00 | [0.00, 0.00] | 1 | 1 |

## Interpretation guardrail

The random-initialization condition is an independent training control, not a claim that random initialization is universally inferior. Conclusions should be based on paired seed-level differences, uncertainty, learning curves, and the prespecified protocol. These results evaluate volume-policy learning with a shared external direction rule; they are not policy-only direction-and-volume control results.
