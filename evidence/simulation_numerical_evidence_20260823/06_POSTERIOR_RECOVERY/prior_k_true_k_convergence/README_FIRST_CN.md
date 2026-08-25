# 新 PF: 先验 K x 真实 K 后验曲线收敛分析

## 研究问题

本包检验变量 K 粒子滤波器在九种组合下的滴定曲线收敛:

- 先验 K=1, 真实 K=1/2/3
- 先验 K=2, 真实 K=1/2/3
- 先验 K=3, 真实 K=1/2/3

同时专门筛选了六种错误先验 K 稳定更新到正确 K 的案例。这里的"稳定纠正"定义为: 从某一步开始，后续直到 step 12 的后验点估计 K 均等于真实 K。

## 正式协议

- 对指定先验 K 设置 `P(K_prior)=0.80`，另外两个 K 各为 0.10。
- 使用 5 个独立 benchmark seed: 101、202、303、404、555。
- 每个 seed 300 个任务，共 1500 个基础任务。
- 每个任务分别运行三种 K 先验，共 4500 个任务-先验组合。
- 每个组合记录 step 0 到 step 12，共 58,500 条结果。
- PF 使用 1000 个粒子。
- 三种先验复用相同任务和相同 PF 随机粒子种子，属于配对比较。

注意: 原始均匀 K 先验 `[1/3, 1/3, 1/3]` 在 step 0 出现并列，点估计会因并列规则显示为 K=1，不能用于严谨的九分类。因此本分析使用上述 0.80/0.10/0.10 的可识别先验协议。

## 主要结论

1. 九个类别从 step 0 到 step 12 的平均曲线 RMSE 均下降，说明 PF 能持续改善滴定曲线拟合。
2. 正确强先验的 step 12 K 准确率约为 92%-94%。
3. 错误强先验在 12 步内通常不会恢复正确 K，稳定纠正率仅为 4.90%-19.06%。
4. 曲线 RMSE 改善不等于 K 恢复正确。错误 K 模型也可能通过参数补偿得到更接近真实曲线的预测。
5. 六种错误先验方向都找到了稳定纠正案例，并展示了从 step 0 到 step 12 的完整后验曲线和 95% 后验可信区间。

## step 12 结果

| 先验 K | 真实 K | 任务数 | step 0 RMSE | step 12 RMSE | step 12 K 准确率 | 稳定纠正/保持正确 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 635 | 0.9988 | 0.6300 | 94.47% | 93.54% |
| 1 | 2 | 498 | 1.4126 | 1.0306 | 6.02% | 6.02% |
| 1 | 3 | 367 | 2.0312 | 1.5763 | 12.10% | 11.72% |
| 2 | 1 | 635 | 1.7682 | 1.1800 | 14.46% | 14.49% |
| 2 | 2 | 498 | 1.3001 | 0.9259 | 91.98% | 91.16% |
| 2 | 3 | 367 | 1.5277 | 1.0972 | 5.05% | 4.90% |
| 3 | 1 | 635 | 2.5194 | 1.5537 | 19.12% | 19.06% |
| 3 | 2 | 498 | 1.7330 | 1.2938 | 6.66% | 6.63% |
| 3 | 3 | 367 | 1.5215 | 1.0825 | 92.02% | 91.01% |

## 区间定义

- 九宫格 RMSE 和 K 准确率图: 先对每个 benchmark seed 求均值，再对 5 个独立 seed 的均值计算 t 型 95% 置信区间。
- 九宫格平均滴定曲线图: 阴影表示跨任务平均曲线的 95% 置信区间。
- 具体任务图: 对变量 K、浓度和 pKa 的 PF 联合后验抽样 200 条曲线，阴影为逐体积点的 2.5%-97.5% 分位，应称为 95% 后验可信区间。

## 案例选择规则

- 九类代表案例按最终 RMSE 最接近该类别中位数自动选择，不是人工挑选最好结果。
- 六类错误到正确案例先选择稳定纠正步最接近该方向中位数的任务，再选择最终 RMSE 最接近候选任务中位数的任务。

## 文件导航

- `results/figures/rmse_convergence_by_prior_k_and_true_k.png`: 九类 RMSE 随 step 变化。
- `results/figures/k_accuracy_by_prior_k_and_true_k.png`: 九类 K 准确率随 step 变化。
- `results/figures/mean_curve_convergence_by_prior_k_and_true_k.png`: 九类平均曲线在 step 0/4/8/12 的收敛。
- `results/figures/category_examples/`: 九类各一个代表任务，逐步曲线和 95% 后验可信区间。
- `results/figures/wrong_to_correct_examples/`: 六类错误 K 稳定纠正案例的完整 step 0-12 图。
- `results/figures/wrong_to_correct_examples_overview.png`: 六类纠正案例的先验、纠正时刻和最终状态总览。
- `results/*.csv`: 汇总统计和案例索引。
- `raw_data/all_prior_k_posterior_rows.csv`: 58,500 行正式运行结果。
- `tasks/`: 五个 seed 的原始任务定义。
- `source/`: 本分析和 PF 运行所需源码。
- `SHA256SUMS.txt`: 包内文件 SHA-256 清单。

## 复现

在包根目录使用 Python 3，并安装 `source/requirements.txt` 后运行:

```powershell
$env:PYTHONPATH = "$PWD\source"
python source\posterior_prior_k_grid.py --source-task-dir tasks --output-dir reproduced_raw --seeds 101 202 303 404 555 --prior-k 1 2 3 --prior-strength 0.8 --particles 1000 --checkpoints 0 1 2 3 4 5 6 7 8 9 10 11 12
python source\analyze_prior_k_true_k_convergence.py --result-dir reproduced_raw --source-task-dir tasks --output-dir reproduced_analysis --particles 1000 --prior-strength 0.8 --posterior-draws 200
```

正式结果完成标记分别位于 `raw_data/PRIOR_K_GRID_COMPLETE.json` 和 `results/PRIOR_K_TRUE_K_ANALYSIS_COMPLETE.json`。
