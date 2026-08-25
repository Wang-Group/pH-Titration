# Final Analysis of the Fixed-3000 Confirmatory Experiments

## 1. Executive conclusion

The results support a scenario-dependent conclusion rather than a claim that RL universally outperforms Bayesian control:

- In the nominal/common environment, no RL candidate achieved a confirmatory success-rate advantage over `bayesian_common`; Bayesian remains a strong baseline.
- Under `close_random_actuator`, SAC and TD3 showed reproducible success-rate gains in both the paired fixed-3000 design and the 25-cell crossed-seed replication. SAC is the more balanced candidate, whereas TD3 is more aggressive.
- Across the extended stress scenarios, the benefit is regime-dependent. RL is much stronger under `high_conc_under`, `large_volume_drift`, `partial_response`, and `partial_bias`, while Bayesian remains highly competitive under `close_pka` and `out_of_range`.

The defensible manuscript claim is therefore: robust RL provides reproducible gains in selected disturbance/model-mismatch regimes, while Bayesian remains a strong baseline in the nominal regime and in some chemistry-shift regimes.

## 2. Runtime and validation

- Project: `rl_bayesian_fixed3000_confirmatory_20260724`
- Runtime: 64-bit Python 3.11 in the project `.venv`
- Main dependencies: NumPy, SciPy, CPU PyTorch, and Matplotlib
- `PACKAGE_VALIDATION_FIXED3000.json`: `PASS`
- `run_fixed3000.py`, `analyze_fixed3000.py`, `evaluate_candidates.py`, and `challenge_common.py` passed `py_compile`
- No core algorithm logic was changed in this round. The operational setup used the 64-bit Python 3.11 environment with the required scientific dependencies installed.

## 3. Experimental design

### Paired fixed-3000 confirmation

Each scenario used 3000 fixed tasks and five train/evaluation seed shards. All candidate methods and the native Bayesian baseline were evaluated. The primary scenarios were `nominal` and `close_random_actuator`; each candidate used 500 particles, five workers, and 10,000 bootstrap iterations.

### 25-cell crossed-seed replication

For `close_random_actuator`, SAC, TD3, and Bayesian were evaluated over all 5×5 train-seed × evaluation-seed combinations (25 cells), with 20,000 bootstrap iterations. This directly tests whether the gain is an artifact of one paired seed alignment.

### Extended stress scenarios

Core methods were evaluated on 3000 fixed tasks in eight additional scenarios: `high_conc_under`, `large_volume_drift`, `close_pka`, `out_of_range`, `tetra_noise`, `noise_010`, `partial_response`, and `partial_bias`.

## 4. Key results

### 4.1 Primary confirmation

| Scenario / method (vs Bayesian) | Success-rate difference | 95% cluster CI | Additional observations |
|---|---:|---:|---|
| nominal / SAC history-robust | -0.2267 pp | [-0.38, -0.0733] | Not a clear win |
| nominal / TD3 filtered-robust | +0.0333 pp | [-0.04, 0.1067] | Not a clear win |
| nominal / PPO residual imitation | +0.0533 pp | [-0.02, 0.1267] | Not a clear win |
| close_random_actuator / SAC history-robust | +4.92 pp | [4.6667, 5.1533] | Positive in 5/5 clusters; steps -22.78%, volume +13.24%, overshoot +18.17% |
| close_random_actuator / TD3 filtered-robust | +4.8467 pp | [4.5933, 5.0733] | Positive in 5/5 clusters; steps -15.22%, volume +45.52%, overshoot +44.16% |

Both SAC and TD3 meet the `clear_success_win` criterion in `close_random_actuator`. TD3, however, incurs substantially larger titrant-volume and overshoot costs, making SAC the more balanced robust candidate.

### 4.2 Crossed-seed replication

| Method | Success rate | Difference vs Bayesian | Positive cells | Decision |
|---|---:|---:|---:|---|
| Bayesian common | 94.90% | — | — | Baseline |
| SAC history-robust | 99.84% | +4.94 pp | 25/25 | Clear win; MO win |
| TD3 filtered-robust | 99.8107% | +4.9107 pp | 25/25 | Clear win; MO win |

Imitation, submitted RL, and PPO reference were clearly negative relative to Bayesian in this crossed-seed check. The 25/25 positive cells show that the SAC/TD3 gain is not driven by a single paired-seed coincidence.

### 4.3 Extended stress scenarios

| Scenario | Bayesian | SAC | TD3 | Interpretation |
|---|---:|---:|---:|---|
| high_conc_under | 7.833% | 98.887% | 98.880% | Large RL advantage; clear + MO |
| large_volume_drift | 19.353% | 87.080% | 86.087% | Large RL advantage; clear + MO |
| close_pka | 98.787% | 98.947% | 98.973% | Small improvement only; not a headline win |
| out_of_range | 96.000% | 95.793% | 96.140% | Bayesian remains strong; TD3 gain is marginal |
| tetra_noise | 86.953% | 90.273% | 89.613% | Success improves, but not the multi-objective winner |
| noise_010 | 63.893% | 66.973% | 66.020% | Several clear success wins, not clear MO wins |
| partial_response | 32.313% | 40.933% | 40.047% | Multiple RL methods clear; SAC/TD3 often MO wins |
| partial_bias | 20.160% | 26.273% | 26.260% | Multiple RL methods clear + MO |

The decision summary records `confirmed_success_wins = 79` and `multiobjective_tradeoff_wins = 61`. These counts span methods and scenarios and must not be interpreted as universal RL superiority.

## 5. Discussion and recommended wording

1. Use the actuator-randomized setting as the primary evidence for robust-RL gains, while reporting the cost differences between SAC and TD3.
2. Keep Bayesian as a strong nominal/common-environment baseline; avoid wording such as “RL universally outperforms Bayesian.”
3. Describe the extended results as regime-dependent robustness: RL helps most under concentration underestimation, volume drift, partial response, and bias, whereas Bayesian is already strong near pKa and out-of-range conditions.
4. Include the 25-cell crossed-seed replication to demonstrate that the actuator-randomized gain is not a single random pairing artifact.
5. Report success rate jointly with steps, total titrant volume, and overshoot; success rate alone hides the more aggressive resource profile of TD3.

## 6. Reproducibility

Each result directory contains `settings.json`, `RUN_COMPLETE.txt`, aggregate summaries, paired tests, per-task results, shards, and automatically generated Chinese/English reports. For review materials, cite the relevant `DECISION_SUMMARY.json` and CSV files rather than a single headline number.
