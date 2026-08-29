# fixed-3000 确认性实验：最终分析

## 1. 结论先行

这轮结果支持一个“场景依赖”的结论，而不是“RL 全面优于 Bayesian”：

- 在 nominal/common environment 中，没有任何 RL 候选相对 `bayesian_common` 获得确认性的成功率优势；Bayesian 仍是强基线。
- 在 `close_random_actuator` 扰动下，SAC 和 TD3 的成功率优势在 paired fixed-3000 设计和 25-cell crossed-seed 复核中都可重复。SAC 的代价更平衡，TD3 更激进。
- 在扩展压力场景中，RL 的收益依赖于失配类型：`high_conc_under`、`large_volume_drift`、`partial_response`、`partial_bias` 等场景收益很大；`close_pka` 和 `out_of_range` 中 Bayesian 仍然很有竞争力。

因此，适合写入论文的主张是：robust RL 在特定扰动/模型失配 regime 中带来可重复收益，而 Bayesian 在 nominal 及部分 chemistry-shift 场景下仍是强基线。

## 2. 运行环境与验证

- 项目：`rl_bayesian_fixed3000_confirmatory_20260724`
- Python：64-bit Python 3.11，项目内 `.venv`
- 主要依赖：NumPy、SciPy、PyTorch（CPU）、Matplotlib
- `PACKAGE_VALIDATION_FIXED3000.json`：`PASS`
- `run_fixed3000.py`、`analyze_fixed3000.py`、`evaluate_candidates.py`、`challenge_common.py` 已通过 `py_compile`
- 本轮没有改动核心算法逻辑；修复的是运行环境选择和依赖问题，确保脚本使用 64-bit `.venv` Python。

## 3. 实验设计

### 主确认实验（paired fixed-3000）

每个场景固定 3000 个任务，使用 5 个训练/评估 seed shard；比较全部候选方法和原生 Bayesian 基线。主场景为 `nominal` 和 `close_random_actuator`，每个候选使用 500 particles，5 workers，bootstrap 10000 次。

### 交叉种子复核（25-cell crossed-seed）

在 `close_random_actuator` 下，对 SAC、TD3 与 Bayesian 进行 5×5 训练 seed × 评估 seed 交叉组合，共 25 个 cells；bootstrap 20000 次，用于检验优势是否只是配对 seed 的偶然结果。

### 扩展压力场景

固定 3000 个任务，在以下 8 个场景中运行 core methods：`high_conc_under`、`large_volume_drift`、`close_pka`、`out_of_range`、`tetra_noise`、`noise_010`、`partial_response`、`partial_bias`。

## 4. 关键结果

### 4.1 主确认实验

| 场景 / 方法（相对 Bayesian） | 成功率差 | 95% cluster CI | 额外观察 |
|---|---:|---:|---|
| nominal / SAC history-robust | -0.2267 pp | [-0.38, -0.0733] | 非 clear win |
| nominal / TD3 filtered-robust | +0.0333 pp | [-0.04, 0.1067] | 非 clear win |
| nominal / PPO residual imitation | +0.0533 pp | [-0.02, 0.1267] | 非 clear win |
| close_random_actuator / SAC history-robust | +4.92 pp | [4.6667, 5.1533] | 5/5 clusters 为正；steps -22.78%，volume +13.24%，overshoot +18.17% |
| close_random_actuator / TD3 filtered-robust | +4.8467 pp | [4.5933, 5.0733] | 5/5 clusters 为正；steps -15.22%，volume +45.52%，overshoot +44.16% |

SAC 和 TD3 在 `close_random_actuator` 下均满足 `clear_success_win`；但 TD3 的 titrant volume 和 overshoot 代价明显更高，因此多目标权衡上 SAC 更适合作为稳健候选。

### 4.2 交叉种子复核

| 方法 | success rate | 相对 Bayesian | 正向 cells | 决策 |
|---|---:|---:|---:|---|
| Bayesian common | 94.90% | — | — | 基线 |
| SAC history-robust | 99.84% | +4.94 pp | 25/25 | clear win，MO win |
| TD3 filtered-robust | 99.8107% | +4.9107 pp | 25/25 | clear win，MO win |

imitation、submitted RL 和 PPO reference 在该交叉复核中相对 Bayesian 为明显负向。25/25 全部正向说明 SAC/TD3 的 actuator-randomized 优势不是单一配对 seed 造成的。

### 4.3 扩展压力场景

| 场景 | Bayesian | SAC | TD3 | 解读 |
|---|---:|---:|---:|---|
| high_conc_under | 7.833% | 98.887% | 98.880% | RL 大幅胜出，clear + MO |
| large_volume_drift | 19.353% | 87.080% | 86.087% | RL 大幅胜出，clear + MO |
| close_pka | 98.787% | 98.947% | 98.973% | 仅轻微改善，不能宣传为显著胜出 |
| out_of_range | 96.000% | 95.793% | 96.140% | Bayesian 仍很强；TD3 仅微小提升 |
| tetra_noise | 86.953% | 90.273% | 89.613% | 成功率改善，但多目标代价不占优 |
| noise_010 | 63.893% | 66.973% | 66.020% | 多个候选 clear success win，非明显 MO win |
| partial_response | 32.313% | 40.933% | 40.047% | 多个 RL 候选 clear；SAC/TD3 多数 MO win |
| partial_bias | 20.160% | 26.273% | 26.260% | 多个 RL 候选 clear + MO |

总体决策文件记录 `confirmed_success_wins = 79`、`multiobjective_tradeoff_wins = 61`；这些计数跨越多个候选和场景，不能解读为 RL 在所有场景都胜出。

## 5. 讨论与论文措辞建议

1. 将 actuator-randomized 场景作为 RL 稳健性收益的主要证据，并同时报告 SAC 与 TD3 的代价差异。
2. 保留 Bayesian 作为 nominal/common environment 的强基线，不把主张写成 universal superiority。
3. 将扩展场景结果解释为 regime-dependent robustness：RL 对浓度低估、体积漂移、部分响应和偏差更有帮助，但对接近 pKa 或 out-of-range 场景，Bayesian 已经很强。
4. 将 25-cell crossed-seed 复核作为独立稳健性分析记录，以说明结果不是单一随机配对的产物。
5. 报告成功率、steps、总滴定体积和 overshoot 的联合权衡；只报告成功率会掩盖 TD3 更激进的资源代价。

## 6. 可复现性

每个结果目录均包含 `settings.json`、`RUN_COMPLETE.txt`、聚合统计、paired tests、逐任务结果、shards 以及中英文自动报告。建议审稿材料中引用本包内的 `DECISION_SUMMARY.json` 和相应 CSV，而不是仅引用单一汇总数字。
