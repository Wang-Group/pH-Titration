# 源代码归档说明

本目录保存按协议分组的分析源代码：

- `major_reviewer_evidence_source/`：PID 调参、主多种子基准、Bayesian 稳健性、RL 算法/奖励分析和配对统计检验源代码；其结果与当前证据块中的汇总表对应。
- `joint_parameter_bayesian_code_current/`：PF、PyMC、参数恢复和多种子分析的独立源代码包。
- `primary_locked_benchmark_source/`：主 5×3000 benchmark 的任务生成器、环境、评估脚本和运行依赖。
- `formal_pf_95_36_reproduction_20260901/`：正式 PF `95.36 ± 0.59%` 结果的完整历史运行器、缺失依赖补全说明及跨机器验证脚本；验证时逐字节核对任务清单，并逐字段核对除硬件相关计时外的任务结果。
- `historical_baseline_runner_20260817/`：simple-rule、prespecified PID 和 tuned PID runner 及其参数定义。
- `historical_fixed3000_runner_source/`：fixed-3000 运行器和候选模型训练源代码，用于独立辅助协议。

`historical_fixed3000_runner_source` 用于独立辅助协议，不用于生成当前主表；当前主结果以 `01_PRIMARY_5x3000_BENCHMARK` 的锁定汇总、配对检验和验证记录为准。

所有当前主结果和各项扩展结果仍按 `00_INDEX_AND_PROTOCOLS` 的统计单位定义使用；不同协议不应合并计算。
公开的 `controllers.RobustPFController` 是后续部署接口，不能替代上述历史运行器来复现论文主 PF 结果；后续 posterior-to-control 消融也应从正式历史运行器派生，只替换被检验的 posterior 使用方式。
