# 新 PF 与新 PPO 控制器代码包

本包提供与正式评价结果对应的两个控制器：

1. `RobustPFController`：可变 K、联合浓度/pKa 的序贯重要性重采样粒子滤波器，加上正式消融胜出的完整体积整形和持久过冲上限。
2. `PPOVolumeController`：从同一模仿检查点出发训练的 PPO，使用独立验证集选中的训练种子 303、100023 次环境交互检查点。

## 最重要的接口约定

两个控制器均使用相同循环：

1. 调用 `reset(...)` 初始化。
2. 调用 `recommend()` 得到试剂方向和请求体积。
3. 实际执行动作并等待测量稳定。
4. 调用 `observe(measured_ph, actual_volume_ml, reagent)` 反馈新测量和实际送液体积。
5. 重复，直到 `recommend()` 返回 `stop=True`。

动作体积为 0.01-10.00 mL，正式实验使用 0.1 M 主滴定剂和 0.01 pH 测量分辨率。`reagent` 为 `acid` 或 `base`。

PPO 网络只决定体积。酸/碱方向由共同规则决定：当前测量 pH 低于目标时加碱，否则加酸。这一点必须在论文、软件说明和硬件接口中披露。

## PF 示例

```python
from new_pf_controller import RobustPFController

controller = RobustPFController(particles=1000, seed=101)
controller.reset(
    initial_measured_ph=4.36,
    target_ph=7.72,
    initial_volume_ml=8.80,
    initial_base_moles=0.0006067,
)

while True:
    action = controller.recommend()
    if action.stop:
        break

    # 在设备上执行 action.reagent 和 action.volume_ml。
    measured_ph = read_ph_from_instrument()
    actual_volume_ml = read_delivered_volume()
    controller.observe(measured_ph, actual_volume_ml, action.reagent)

print(controller.status())
```

`initial_base_moles` 是样品制备或既往加液过程中已知的初始强碱摩尔数，不是真实 pKa 或真实分析物浓度。若样品没有预加碱，使用 0。若初始状态包含已知强酸，可同时传 `initial_acid_moles`。

PF 的 `status()` 和动作诊断包含：

- 浓度后验均值和标准差；
- K=1/2/3 模型概率及当前 MAP K；
- pKa 后验均值和标准差；
- 有效样本量；
- posterior equilibrium 所需体积、tanh 整形后的理想体积；
- 持久过冲体积上限。

## PPO 示例

```python
from new_rl_controller import PPOVolumeController

controller = PPOVolumeController(
    "models/ppo_seed_303.pth",
    device="auto",
)
controller.reset(initial_measured_ph=4.36, target_ph=7.72)

while True:
    action = controller.recommend()
    if action.stop:
        break

    measured_ph = read_ph_from_instrument()
    actual_volume_ml = read_delivered_volume()
    controller.observe(measured_ph, actual_volume_ml, action.reagent)
```

PPO 输入状态固定为：

```text
[当前测量 pH, 目标 pH, 最近一次测量变化, 当前 pH-目标 pH, 上次请求体积]
```

部署采用确定性 `argmax`，不是随机采样。构造控制器时默认校验正式权重的文件 SHA-256 和 actor 张量 SHA-256，权重被替换或损坏会直接报错。

## 正式定量结果

标称锁定的 1000 个任务：

| 方法 | 成功率 | 严格成功率 | 严重失败率 | 成功任务步数 | 最终误差 |
|---|---:|---:|---:|---:|---:|
| 新 PF + 完整控制规则 | 95.10% | 42.20% | 1.80% | 4.87 | 0.0789 pH |
| 选定 PPO 303 | 93.40% | 44.90% | 2.90% | 5.68 | 0.1163 pH |

PF 在正式 5 种子 x 每种子 3000 个任务的消融中成功率为 95.36 +/- 0.59%。选定 PPO 相对模仿学习在标称锁定任务中提高 4.30 个百分点，精确配对 McNemar p=8.91e-7。

PPO 并非普遍鲁棒：在 0.05 pH 传感器噪声下相对模仿提高 12.96 点，但在响应比例 0.70 的严重滞后下下降 9.24 点；联合未见扰动下 +1.48 点且 p=0.261。

## 文件

- `new_pf_controller.py`：PF 控制器统一 API。
- `new_rl_controller.py`：PPO 控制器统一 API和权重校验。
- `controller_api.py`：动作结构、pH 量化和公共常量。
- `particle_inference.py`：可变 K 联合粒子滤波器。
- `chemistry_model.py`：化学平衡求解器。
- `models.py`：PPO actor 网络和状态归一化。
- `controller_example.py`：交互式调用示例。
- `controller_package_self_test.py`：解压后独立运行的闭环自检。
- `MODEL_CARD.json`：来源、正式指标、限制和模型哈希。
- `models/ppo_seed_303.pth`：验证集选中的正式 PPO 权重。
- `SHA256SUMS.txt`：逐文件哈希清单。

## 安装和自检

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe controller_package_self_test.py
```

也可以在 Windows 上运行 `RUN_SELF_TEST.cmd`。

## 使用边界

- 这是模拟验证的研究控制器代码，不是经过硬件安全认证的医疗、工业或无人值守控制软件。
- 硬件层必须自行实现泵限位、最大累计体积、传感器异常、通信超时、人工急停和容器容量保护。
- `observe()` 应使用实际送液体积；若泵没有体积回读，至少使用经过独立校准的体积估计。
- 应在每次加液后等待系统充分混合和传感器稳定，否则 PPO 已知的响应滞后失败模式可能被触发。
- PPO 的正式结论对应 0.1 M 滴定剂及声明的任务分布。改变浓度、动作范围、传感器分辨率或状态定义需要重新评价，通常也需要重新训练。
- PF 的高成功率属于 PF 后验、体积整形和过冲上限组成的完整控制器，不能归因于后验推断单独完成。
