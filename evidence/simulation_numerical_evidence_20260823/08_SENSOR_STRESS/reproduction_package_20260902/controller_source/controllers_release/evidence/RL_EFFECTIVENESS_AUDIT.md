# RL effectiveness audit beyond the main locked evaluation

Protocol: 5 independent evaluation seeds x 500 tasks per perturbation = 2500 paired tasks per suite. Every PPO network starts from the exact same selected imitation checkpoint. The audit uses perturbations stronger than the PPO training randomization and identical task/random draws for imitation and PPO.

Primary endpoint: selected PPO versus imitation success under `combined_unseen`.

| Endpoint | Result |
|---|---:|
| Validation-selected PPO seed | 303 |
| Success difference | +1.48 percentage points |
| 95% CI | [-1.03, +3.99] |
| Exact paired McNemar p | 0.261276 |
| PPO training seeds with positive effect | 5/5 |
| Mean effect across PPO seeds | +2.06 +/- 1.54 percentage points |
| Selected checkpoint environment steps | 100023 |
| Predefined evidence conclusion | not_supported_by_predefined_criteria |

Important secondary regimes:

| Regime | Success difference (pp) | 95% CI | Exact McNemar p | Interpretation |
|---|---:|---:|---:|---|
| Nominal unseen tasks | +4.64 | [+3.40, +5.88] | 2.03249e-13 | supported improvement |
| Sensor noise SD 0.05 pH | +12.96 | [+10.75, +15.17] | 1.08034e-29 | supported improvement |
| Sensor response fraction 0.70 | -9.24 | [-11.61, -6.87] | 3.78523e-14 | significant deterioration |
| Combined unseen perturbation | +1.48 | [-1.03, +3.99] | 0.261276 | no significant success gain |

The evidence therefore supports regime-specific PPO refinement, especially under sensor noise, but does not support a claim of general robustness. Response lag is a documented failure mode and the combined-intervention primary endpoint is not significant.

This audit is a causal before/after test of PPO refinement from a shared imitation initialization, not a new checkpoint-selection set. The selected PPO seed was fixed by the original independent validation results before these intervention outcomes were read. All per-task outcomes, all five PPO seeds, paired tests, learning dynamics, checkpoint hashes, and parameter-change measurements are included in the CSV files.

The environment still applies the shared acid/base direction rule; the neural network fully determines dose volume. Therefore this audit supports only volume-policy refinement within that disclosed hybrid control architecture.
