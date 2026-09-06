# 局部响应拟合（0.0399 / 0.1280 / 0.2452）原始材料包

## 数字对应关系

在 `ANALYSIS_AND_DISCUSSION_CN.md` 第 3 节：

> 自然停止点时，+/-0.10、+/-0.50、+/-1.00 mL 窗口的平均 RMSE 分别为 **0.0399**、**0.1280** 和 **0.2452 pH**；RMSE <=0.10 pH 的任务比例分别为 92.4%、78.7% 和 67.9%。

精确来源：`formal_results/new_pf_local_response/aggregate_summary.csv` 中 `checkpoint_type=natural_control_end` 行：

- `local_rmse_0p1_ml_mean_mean` ≈ 0.03994296 → **0.0399**
- `local_rmse_0p5_ml_mean_mean` ≈ 0.12801999 → **0.1280**
- `local_rmse_1p0_ml_mean_mean` ≈ 0.24515631 → **0.2452**

单位：pH。

## 窗口与曲线定义

- 窗口：`WINDOWS_ML = (0.10, 0.50, 1.00)` mL（脚本中硬编码）
- 网格：每个窗口上 `np.linspace(-window, window, 81)` 的剂量增量（mL）
- 真值曲线与拟合曲线均由 `chemistry_model.response_curve(...)` 在**当前溶液状态**（volume / base_moles / acid_moles）下计算
- **对齐当前 pH（锚定）**：
  ```python
  center = len(grid) // 2
  true_delta = true_curve - true_curve[center]
  fitted_delta = fitted_curve - fitted_curve[center]
  residual = fitted_delta - true_delta
  local_rmse = sqrt(mean(residual**2))
  ```
  即两条曲线都减去中心点（当前决策点）的 pH，只比较相对形状/斜率，不惩罚已知的截距误差。

`LOCAL_RESPONSE_COMPLETE.json` 明确记录：
```json
"anchoring": "truth and PF curves are anchored at the current decision point"
```

## 本包内容

| 路径 | 说明 |
|------|------|
| `pf_local_response_diagnostics.py` | 原始诊断脚本（生成上述数字的完整逻辑） |
| `RUN_LOCAL_RESPONSE.cmd` | 运行入口 |
| `study_source/chemistry_model.py` | `response_curve` 与 `SolutionState` |
| `study_source/particle_controllers.py` | `JointInferenceController` |
| `study_source/particle_inference.py` | 粒子滤波推理依赖 |
| `study_source/task_distribution.py` | 任务生成 `generate_tasks` / `save_tasks` |
| `formal_results/new_pf_local_response/` | 正式运行输出 |
|   `aggregate_summary.csv` | 聚合统计（含 0.0399/0.1280/0.2452） |
|   `all_local_response_rows.csv` | 逐任务、逐 checkpoint 完整输出（约 4.3 MB） |
|   `per_seed_summary.csv` | 按种子汇总 |
|   `seed_*_tasks.jsonl` | 各种子的输入任务列表（5×300） |
|   `LOCAL_RESPONSE_COMPLETE.json` | 运行元数据 + anchoring 说明 |
| `FINAL_DELIVERY/tables/...` | 交付用聚合表副本 |
| `FINAL_DELIVERY/figures/local_response_rmse_overview.png` | 结果图 |
| `ANALYSIS_AND_DISCUSSION_CN.md` | 讨论文本（含上述数字引用） |

## 运行规模

- 5 个 benchmark seeds：101, 202, 303, 404, 555
- 每种子 300 个任务
- 粒子数默认 1000
- 在观测数 0/1/2/3/5/8/12 以及自然停止点（`|current_ph - target_ph| ≤ 0.10`）处评估三个窗口

## 复现要点

脚本会把生成的任务写入 `seed_*_tasks.jsonl`，并把所有快照写入 `all_local_response_rows.csv`。若要重新跑，需保证 `study_source` 在 `sys.path` 中（脚本已处理），且输出目录为空。
