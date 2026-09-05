# PF 内部剂量塑形规则消融实验

本目录是 Table S6 对应的完整代码与结果包。

## 实验配置

- 5 个 benchmark seed：101、202、303、404、555
- 每个 seed 300 个任务
- 每个任务 1000 个粒子
- 6 个变体，共 5 x 300 x 6 = 9000 个 episode

## 变体映射

| 代码标识 | 表格名称 |
| --- | --- |
| `full` | Full rule |
| `no_ph_rate_bonus` | Without response bonus |
| `no_uncertainty_factor` | Without uncertainty factor |
| `no_buffering_factor` | Without B |
| `no_required_volume_term` | Without 0.1Vreq term |
| `linear_clip_instead_of_tanh` | Linear mapping instead of tanh |

## 目录

- `source/pf_internal_rule_ablation.py`：主实验脚本
- `source/controllers_release/`：脚本使用的控制器与化学模型源码
- `source/study_source/task_distribution.py`：任务生成源码
- `source/RUN_INTERNAL_RULE_ABLATION.cmd`：Windows 运行命令
- `source/requirements-simulation.txt`：Python 依赖
- `results/formal_results/`：完整正式结果，包括逐任务 CSV、逐 seed 汇总、统计检验、5 个 seed 的任务 JSONL 和完成标记
- `results/publication_tables/`：用于论文表格的汇总表和检验表
- `docs/ANALYSIS_AND_DISCUSSION_CN.md`：中文分析与讨论

## 复现

在本目录下安装 Python 依赖后运行：

```powershell
python -m pip install -r source/requirements-simulation.txt
python source/pf_internal_rule_ablation.py --output-dir results/reproduced_new_pf_internal_rule_ablation --seeds 101 202 303 404 555 --tasks-per-seed 300 --particles 1000 --workers 8
```

原始结果已经保存在 `results/formal_results/`，复现时请使用一个新的空输出目录。

## 论文表格字段

`aggregate_summary.csv` 中的字段对应关系：

- `success_rate_percent_mean/sd` -> Success (%)
- `successful_steps_mean_mean/sd` -> Steps
- `crossings_mean_mean/sd` -> Overshoot count
- `total_volume_mean_ml_mean/sd` -> Total dose (mL)
- `final_abs_error_mean_mean/sd` -> Final absolute error (pH)

表格中的 `均值 +/- 标准差` 是跨 5 个 seed 的结果；用户给出的数值与该文件四舍五入后完全一致。
