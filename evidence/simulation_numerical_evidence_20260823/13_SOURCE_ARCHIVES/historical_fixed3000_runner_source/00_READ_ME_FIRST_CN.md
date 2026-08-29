# 旧版源码说明（本轮请先阅读 README_FIXED3000_CN.md）

> 本目录已经升级为固定 3,000 项任务的多种子确认实验包。实际运行请使用 `01_CHECK_ENV.cmd`、`02_RUN_PRIMARY_FIXED3000.cmd`、`03_RUN_CROSSED_WINNER_FIXED3000.cmd` 或 `RUN_RECOMMENDED.cmd`。下文保留为上一轮算法设计和源码背景。

本包用于回答两个不同问题：

1. 新 RL 方法能否在成功率上明确超过 Bayesian controller？
2. 如果成功率不能超过，能否在成功率不劣的前提下减少过冲、步骤、加液量或计算时间？

所有候选算法、场景和判定标准都在运行前固定。程序会报告全部结果，不会只挑选 RL 表现较好的场景。Bayesian + residual PPO 是迁移性探索结果：residual 策略围绕 imitation 教师训练，不能表述为已经沿 Bayesian 轨迹重新训练的独立 hybrid。

## 九种候选算法

- `ppo_nominal`：从 imitation 权重初始化，在名义任务上用 success-first reward 训练的离散 PPO。
- `ppo_robust`：在浓度、体积、泵误差、噪声、漂移、迟滞和酸类型随机化条件下训练的 PPO。
- `a2c_robust`：相同域随机化条件下的 A2C，用于判断优化器差异。
- `ppo_history_robust`：输入当前状态和前三步历史的 PPO，针对噪声、漂移和不完全响应。
- `sac_history_robust`：连续加液动作的 SAC，避免 1,000 类离散动作限制。
- `ppo_residual_robust`：学习对基础控制器建议量乘以 0.25-2.0 的修正倍率。评估时同时测试 imitation + residual PPO 和 Bayesian + residual PPO。
- `ppo_filtered_robust`：PPO 使用中位数、波动、趋势和加液历史等稳健滤波特征，针对传感器噪声和漂移。
- `ppo_conservative_robust`：在同一滤波状态上使用更强的过冲、误停和严重误差惩罚，优先降低尾部失败。
- `td3_filtered_robust`：TD3 连续体积控制器，使用目标动作平滑和双 Q 网络，避免离散 0.01 mL 动作空间。

已有 imitation、submitted RL 和上一轮 PPO 会作为参考策略一起评估。

## 评价角度

- 真实成功率和 measured/true false stopping。
- 成功任务步数、全部任务步数及 95th percentile。
- 过冲率和每项任务过冲次数。
- 酸碱总加液体积及 95th percentile。
- 最终绝对 pH 误差及 95th percentile。
- 每项任务决策时间。
- 严格成功率（绝对误差 <= 0.05 pH）、宽松成功率（<= 0.20 pH）和严重失败率（> 0.50 pH）。
- 最终误差、加液量和步数的 CVaR95，以及决策时间 P95。
- 按酸类型、升/降 pH 方向和初始误差难度分层的结果。
- 配对 McNemar 检验、Holm 多重校正、paired bootstrap 95% CI。
- 按评估种子聚类的 paired bootstrap 95% CI，减少同一种子内任务相关性造成的过度置信。
- 步数、过冲、体积、误差和计算时间的配对 Wilcoxon 检验。

## 预先确定的胜出标准

成功率明确胜出必须同时满足：

- 相对 Bayesian 至少提高 1.0 percentage point；
- 配对 McNemar 检验经 Holm 校正后 p < 0.05。

多目标 trade-off 胜出必须同时满足：

- 成功率差异的 seed-clustered paired-bootstrap 95% CI 下限不低于 -0.5 percentage points；
- 步数、过冲或加液量至少一项改善 10%，且配对检验经校正后 p < 0.05；
- 最终绝对误差没有明显恶化。

第二种结果只能支持“效率或稳定性 trade-off 更好”，不能写成成功率超过 Bayesian。

## 推荐运行顺序

1. 解压到本地磁盘，不要直接在 ZIP、微信缓存或网络盘中运行。
2. 双击 `01_CHECK_ENV.cmd`。
3. 双击 `02_RUN_QUICK_TEST.cmd`，确认九种算法都能完成训练、评估和出图。
4. 双击 `03_RUN_SCREENING.cmd`。这是推荐先跑的正式筛选，使用 3 个训练种子。
5. 只有 screening 出现可信候选时，再双击 `04_RUN_FULL_CONFIRMATORY.cmd`，使用 5 个独立训练种子和更大的确认性任务集。

程序支持断点续跑。训练模型和验证行都存在时会自动跳过。需要强制重跑某一模式时，在 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_challenge.ps1 -Mode Screen -Force
```

## 预计耗时

- Quick：通常 2-10 分钟。
- Screen：根据 CPU/GPU 和机器性能，通常约 3-12 小时。
- Full：可能需要 12-48 小时。Bayesian 粒子评估主要依赖多核 CPU，RL 训练可使用 CUDA。

不要因为 Full 耗时较长而删除某些表现不好的算法或场景；完整报告全部预定义结果对审稿回复更重要。

## 环境

- Windows 10/11，64 位。
- 64 位 Python 3.10 或更新版本。
- NumPy、SciPy、PyTorch、Matplotlib。
- 缺少依赖时脚本会创建 `.venv` 并联网安装。
- 建议预留至少 10 GB 磁盘空间。

## 关键输出

Screen 输出位于 `results_screen`，Full 输出位于 `results_full`。最重要的文件是：

- `CHALLENGE_REPORT.md`：直接阅读的结论。
- `DECISION_SUMMARY.json`：机器可读的胜出判定。
- `evaluation/aggregate_summary.csv`：各场景、各方法多指标汇总。
- `evaluation/stratified_summary.csv`：按酸类型、调节方向和初始难度分层的结果。
- `evaluation/paired_tests.csv`：相对 Bayesian 的全部配对统计。
- `evaluation/per_task_results.csv`：任务级原始结果。
- `training/learning_curves.csv`：所有算法学习曲线数据。
- `learning_curves.png`、`scenario_success_heatmap.png`、`nominal_pareto.png`。
- `run_challenge.log` 和 `RUN_COMPLETE_*.txt`。

运行结束后压缩整个 `results_screen` 或 `results_full` 文件夹带回。

## 科学解释限制

- 名义任务使用原始 Bayesian 模拟流程作为主基线。
- 失配场景使用 Bayesian controller 与其他策略共享同一个受扰动物理环境，因此结果应描述为 common-environment comparison。
- 单个失配场景胜出只支持该场景下的 regime-specific gain。
- 本包用于寻找可复现的改进，不保证 RL 一定超过 Bayesian。
