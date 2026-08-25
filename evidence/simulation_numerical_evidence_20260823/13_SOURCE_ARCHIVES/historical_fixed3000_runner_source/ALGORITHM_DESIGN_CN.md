# RL 与 Bayesian 对比实验设计说明

## 研究问题

这套程序不是只寻找一个表现最好的 RL 模型，而是预先固定多个算法、多个失配场景和多个评价角度，回答：

1. 是否存在某个明确的模型失配 regime，使 RL 相对 Bayesian 的成功率提高至少 1.0 个百分点，并且配对 McNemar 检验经 Holm 校正后 `p < 0.05`。
2. 如果成功率没有提高，是否能在成功率不劣的情况下减少过冲、步数、加液量、尾部误差或计算时间。

## 候选算法

| 候选 | 核心变化 | 主要假设 |
| --- | --- | --- |
| `ppo_nominal` | imitation 初始化的离散 PPO | 名义环境中的局部策略优化 |
| `ppo_robust` | 域随机化 PPO | 浓度、体积、执行器、噪声和酸类型变化 |
| `a2c_robust` | 同一域随机化下的 A2C | 判断优化器差异而非状态差异 |
| `ppo_history_robust` | 当前状态加前三步原始历史 | 处理迟滞、漂移和部分响应 |
| `sac_history_robust` | 连续体积 SAC | 避免 1000 类离散动作限制 |
| `ppo_residual_robust` | 对 imitation 动作乘以 0.25--2.0 修正倍率 | 学习小幅安全修正 |
| `ppo_filtered_robust` | 中位数、方差、趋势和加液历史特征 | 对噪声和传感器漂移更稳健 |
| `ppo_conservative_robust` | 滤波特征加重过冲、误停和严重误差惩罚 | 优先减少尾部失败 |
| `td3_filtered_robust` | 滤波特征上的连续 TD3、双 Q 和目标平滑 | 连续动作和更平滑的体积控制 |

`Bayesian + transferred residual PPO` 只作为探索性 hybrid 输出：residual 模型围绕 imitation 教师训练，直接迁移到 Bayesian 的结果不能写成“已在 Bayesian 轨迹上公平再训练”的结论。

## 预定义场景

名义场景之外，确认性面板包含：分析物浓度高/低、初始体积大/小、滴定剂浓度变化、执行器欠量/过量/随机误差、观测噪声 0.05 和 0.10、传感器偏置、漂移、部分响应、四元酸、超出初始化 pKa 范围、接近 pKa、以及多个复合失配场景。所有方法在同一任务和同一受扰动物理环境中配对运行。

## 输出指标

- true success、measured success 和 false stop；
- 默认 `+/-0.10` pH 成功率，以及严格 `+/-0.05`、宽松 `+/-0.20` 成功率；
- 严重失败率（最终真实误差 `> 0.50` pH）；
- 成功任务步数、全部任务步数、P95 和 CVaR95；
- 过冲率、过冲次数、总加液量、加液量 P95/CVaR95；
- 最终绝对误差、最终误差 CVaR95 和有符号偏差；
- 每项任务决策时间均值和 P95；
- 按酸类型、升/降 pH 方向和初始误差难度的分层统计。

## 统计判定

每个评估种子生成一组固定任务，所有方法共享同一任务键。成功率使用 exact McNemar 检验并进行 Holm 校正；连续指标使用配对 Wilcoxon 检验。成功率差异同时给出普通 paired bootstrap 和按评估种子聚类的 paired bootstrap 95% CI。报告中的多目标判定使用聚类 CI。

推荐的 Full 结果报告方式是 `mean +/- seed SD`，而不是把同一个种子下的所有任务当成独立重复实验。只有达到预注册门槛才称为 clear win；单一压力场景的改善只能写成 regime-specific gain。

## 运行顺序

1. 双击 `01_CHECK_ENV.cmd`。
2. 双击 `02_RUN_QUICK_TEST.cmd`，只检查流程，不解释科学结果。
3. 双击 `03_RUN_SCREENING.cmd`，先用 3 个训练种子筛选候选。
4. 只有 screening 显示可复现候选时，再双击 `04_RUN_FULL_CONFIRMATORY.cmd`。

主要结果在 `results_screen` 或 `results_full`：

- `CHALLENGE_REPORT.md`：人读报告；
- `DECISION_SUMMARY.json`：clear win、trade-off、严格阈值和严重失败判定；
- `evaluation/aggregate_summary.csv`：各场景、各方法的均值和 seed SD；
- `evaluation/paired_tests.csv`：配对统计、Holm 校正和聚类 CI；
- `evaluation/stratified_summary.csv`：分层结果；
- `evaluation/per_task_results.csv`：任务级原始结果。

上一轮独立校正基准已经显示 Bayesian 在名义任务上约为 `98.02%`，RL 约为 `94.08%`；因此新包最值得观察的是高噪声、漂移、偏置和部分响应场景，而不是预设 RL 一定能在名义成功率上反超。
