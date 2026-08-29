# 固定 3,000 项任务多随机种子确认实验包

## 为什么还需要这一轮

上一轮 `close_random_actuator` 中，SAC 的成功率为 99.8%，Bayesian 为 95.4%，因此 SAC 在该特定场景中确实更高，而且配对检验显著。

不能直接写成“RL 全面优于 Bayesian”，原因是：

1. nominal 条件下 SAC 没有显著超过 Bayesian；
2. 上一轮每个评估种子生成了不同的任务池，压力场景每个种子只有 200 项；
3. 训练种子和评估种子一一绑定，不能完全排除种子组合的偶然性；
4. 压力场景 Bayesian 使用 100 个粒子，而 nominal 使用 500 个；
5. nominal 的 Bayesian 原生环境与 RL 共同环境没有进行桥接验证。

本包记录上述固定任务规模和交叉种子实验，用于评估不同协议下的控制表现。

## 推荐运行顺序

1. 双击 `01_CHECK_ENV.cmd`。
2. 第一次使用建议双击 `05_RUN_SMOKE_TEST.cmd`，只运行极小数据检查流程。
3. 双击 `02_RUN_PRIMARY_FIXED3000.cmd`，完成主要确认实验。
4. 双击 `03_RUN_CROSSED_WINNER_FIXED3000.cmd`，完成 SAC 胜出场景的 5x5 交叉种子复核。
5. `04_RUN_EXTENDED_FIXED3000.cmd` 是可选的长时间扩展实验。

也可以直接双击 `RUN_RECOMMENDED.cmd`，它会依次执行环境检查、主要确认实验和交叉种子复核。

如果环境检查失败，双击 `00_INSTALL_ENV.cmd` 创建本地 `.venv`。安装需要联网，且 PyTorch 下载体积较大。

## 三组实验分别回答什么

### `02_RUN_PRIMARY_FIXED3000.cmd`

- 场景：`nominal`、`close_random_actuator`
- 每个场景：固定同一套 3,000 项任务
- 种子：5 个训练种子与 5 个扰动种子配对
- 方法：Bayesian、Imitation、submitted RL、previous PPO 和全部 9 个新候选算法
- Bayesian 粒子：统一为 500
- nominal 同时运行 `bayesian_original` 和 `bayesian_common`，检查适配器是否改变基线

这是回复审稿人时最重要的一组结果。

### `03_RUN_CROSSED_WINNER_FIXED3000.cmd`

- 场景：`close_random_actuator`
- 固定 3,000 项任务
- 5 个训练种子 x 5 个扰动种子，共 25 个组合
- 方法：Bayesian、Imitation、submitted RL、previous PPO、SAC、TD3
- Bayesian 粒子：500

这组实验用于确认 SAC 的优势不是某一对训练/评估种子碰巧造成的。统计报告会分别重采样训练种子和扰动种子。

### `04_RUN_EXTENDED_FIXED3000.cmd`

- 场景：`high_conc_under`、`large_volume_drift`、`close_pka`、`out_of_range`、`tetra_noise`、`noise_010`、`partial_response`、`partial_bias`
- 每个场景固定 3,000 项任务
- 5 个配对种子
- 运行主要算法和对照方法

这组数据用于评估模型失配、pKa 敏感性、噪声和传感器响应下的控制表现；其协议与主基准独立。

## 输出文件

每个结果目录都会自动生成：

- `FIXED3000_REPORT_CN.md`：中文自包含报告；
- `FIXED3000_REPORT_EN.md`：可用于审稿回复的英文表格和统计说明；
- `aggregate_summary.csv`：均值 +/- 种子组合标准差；
- `paired_tests.csv`：pooled McNemar、Holm 校正、种子置信区间和胜出判定；
- `per_cluster_paired_tests.csv`：每个种子组合在相同 3,000 项任务上的 exact McNemar 检验；
- `per_cluster_summary.csv`：每个种子组合的全部指标；
- `stratified_summary.csv`：酸类型、调节方向和难度分层；
- `per_task_results.csv`：合并后的任务级结果；
- `shards/`：可恢复的分片结果；
- `DECISION_SUMMARY.json`：明确胜出和多目标权衡结论。

## 中断恢复

每个场景和种子组合完成后立即写入一个原子分片。电脑重启或程序中断后，重新双击相同的 `.cmd` 文件即可；完整分片会自动跳过。

不要手动修改 `settings.json` 或 `shards` 中已完成的 CSV。如果需要全新重跑，使用新的输出目录名称。

## 胜出判据

只有同时满足以下条件才标记为 `clear_success_win=True`：

- 成功率提高至少 1.0 个百分点；
- pooled exact McNemar 经全局 Holm 校正后 `p<0.05`；
- 种子聚类或双向重采样 95% CI 下界大于 0；
- 至少 80% 的种子组合中改进方向为正。

因此报告不会仅凭合并后的巨大样本量把极小差异写成算法优势。

## 环境要求

- Windows 10/11
- Python 3.10-3.12，64 位
- `numpy`、`scipy`、`torch`、`matplotlib`
- 推荐至少 16 GB 内存和 5 个以上 CPU 逻辑核心
- 不需要 GPU

主实验和交叉实验可能运行数小时，具体取决于 CPU。运行时可以正常查看日志，但不要同时启动两个完整实验，以免内存和 CPU 竞争。
