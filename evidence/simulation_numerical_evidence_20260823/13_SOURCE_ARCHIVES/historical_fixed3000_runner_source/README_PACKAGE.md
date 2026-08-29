# Fixed-3000 Confirmatory Results Package

本包对应 `rl_bayesian_fixed3000_confirmatory_20260724` 的完整重跑结果。

## 包含内容

- `results_fixed3000_primary/`：nominal 与 close_random_actuator 的 paired fixed-3000 主确认实验
- `results_fixed3000_crossed_winner/`：close_random_actuator 的 25-cell crossed-seed 复核
- `results_fixed3000_extended/`：8 个扩展压力场景
- `FINAL_ANALYSIS_CN.md` / `FINAL_ANALYSIS_EN.md`：完整分析与讨论
- `REVIEWER_RESPONSE_DRAFT_EN.md` / `REVIEWER_RESPONSE_DRAFT_CN.md`：审稿回复草稿
- `PACKAGE_VALIDATION_FIXED3000.json`：环境验证结果（PASS）

## 读取顺序

1. 先看 `FINAL_ANALYSIS_EN.md` 或 `FINAL_ANALYSIS_CN.md`。
2. 再查看三个结果目录中的 `DECISION_SUMMARY.json`、`aggregate_summary.csv` 和 `paired_tests.csv`。
3. 需要逐任务审计时，使用各目录的 `per_task_results.csv` 与 `shards/`。

## 解释口径

不要把结果表述成“RL 全面优于 Bayesian”。正确结论是：SAC/TD3 在 actuator-randomized 场景中具有可重复优势；扩展场景的优势具有 regime dependence；Bayesian 在 nominal 和部分 chemistry-shift 场景中仍然是强基线。

## 完整性

三个正式结果目录均包含 `RUN_COMPLETE.txt`，且没有残留 `.tmp` shard。smoke-test 输出和 `.venv` 不包含在交付包中。
