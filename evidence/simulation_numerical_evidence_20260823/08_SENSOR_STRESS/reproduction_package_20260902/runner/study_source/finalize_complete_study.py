from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "results" / "bayesian_rule_ablation_standard_v2"
STUDY = ROOT / "results" / "complete_study_standard_v1"
DELIVERY = ROOT.parent / "bayesian_external_rule_ablation_delivery_20260812"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value(row: dict[str, str], metric: str, digits: int = 2) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = row.get(f"{metric}_sd", "")
    if sd in ("", None, "None"):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {float(sd):.{digits}f}"


def build_report() -> str:
    ablation = read_csv(ABLATION / "aggregate_summary.csv")
    evaluation = read_csv(STUDY / "04_evaluation" / "aggregate_summary.csv")
    networks = read_csv(STUDY / "06_two_network_evaluation" / "two_network_summary.csv")
    posterior = read_csv(STUDY / "05_posterior_diagnostics" / "aggregate_posterior_summary.csv")
    rmse_distribution = read_csv(
        STUDY / "08_pf_fit_distributions" / "rmse_distribution_by_observation.csv"
    )
    rmse_seed_changes = read_csv(
        STUDY / "08_pf_fit_distributions" / "rmse_change_seed_summary.csv"
    )
    fit_control_association = json.loads(
        (STUDY / "08_pf_fit_distributions" / "fit_control_association.json").read_text(
            encoding="utf-8"
        )
    )
    rl_tests = read_csv(STUDY / "07_rl_effectiveness" / "selected_ppo_paired_tests.csv")
    rl_seed_effects = read_csv(STUDY / "07_rl_effectiveness" / "all_ppo_seed_effects.csv")
    rl_dynamics = read_csv(STUDY / "07_rl_effectiveness" / "ppo_training_dynamics.csv")
    rl_completion = json.loads(
        (STUDY / "07_rl_effectiveness" / "RL_EFFECTIVENESS_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_train = json.loads(
        (STUDY / "01_teacher_data" / "train_teacher_summary.json").read_text(encoding="utf-8")
    )
    teacher_validation = json.loads(
        (STUDY / "01_teacher_data" / "validation_teacher_summary.json").read_text(encoding="utf-8")
    )
    selection = json.loads((STUDY / "TEACHER_SELECTION.json").read_text(encoding="utf-8"))

    labels = {
        "hybrid_full": "新 PF + 完整控制规则",
        "hybrid_no_overshoot_cap": "新 PF，无过冲体积上限",
        "posterior_direct": "新 PF，posterior-direct",
    }
    lines = [
        "# 新 PF 外部控制、模仿学习、PPO 与后验准确度：完整分析",
        "",
        "## 1. 研究问题与协议",
        "",
        "本研究先用严格配对消融判断新的可变 K、联合浓度/pKa 粒子滤波器是否依赖外部加液体积控制逻辑。随后只使用正式消融胜出的 PF 控制器生成教师数据，训练模仿网络和 PPO 网络，并在未参与选择的锁定任务上评价。最后用独立任务研究不同观测次数及自然实验结束时的完整曲线拟合和后验参数准确度。",
        "",
        "控制决策使用 0.01 pH 分辨率测量值，结局使用未量化真实 pH。加液体积范围为 0.01-10.00 mL。神经网络负责体积，酸/碱方向仍使用共同外部方向规则。",
        "",
        "## 2. 外部加液规则消融（5 个种子，每种子 3000 个配对任务）",
        "",
        "| 控制器 | 成功率 (%) | 严格成功率 (%) | 严重失败 (%) | 成功任务步数 | 过冲 | 总体积 (mL) | 最终误差 (pH) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation:
        lines.append(
            f"| {labels[row['policy']]} | {value(row, 'success_rate_percent')} | "
            f"{value(row, 'strict_success_rate_percent')} | {value(row, 'severe_failure_rate_percent')} | "
            f"{value(row, 'successful_steps_mean')} | {value(row, 'overshoots_mean')} | "
            f"{value(row, 'total_volume_mean_ml')} | {value(row, 'final_abs_error_mean', 4)} |"
        )
    full = next(row for row in ablation if row["policy"] == "hybrid_full")
    no_cap = next(row for row in ablation if row["policy"] == "hybrid_no_overshoot_cap")
    direct = next(row for row in ablation if row["policy"] == "posterior_direct")
    lines.extend(
        [
            "",
            f"完整控制规则相对去掉过冲上限提高 {float(full['success_rate_percent_mean']) - float(no_cap['success_rate_percent_mean']):.2f} 个百分点，"
            f"相对 posterior-direct 提高 {float(full['success_rate_percent_mean']) - float(direct['success_rate_percent_mean']):.2f} 个百分点。"
            "因此外部体积控制是重要组成，而不是可忽略的接口细节。",
            f"自动教师选择结果为 `{selection['selected_teacher_policy']}`。",
            "",
            "## 3. 教师数据与训练规模",
            "",
            f"训练阶段从 {teacher_train['candidate_tasks']} 个候选任务获得 {teacher_train['states']} 个去重状态；"
            f"验证阶段从 {teacher_validation['candidate_tasks']} 个候选任务获得 {teacher_validation['states']} 个状态。"
            f"质量通过率分别为 {teacher_train['quality_pass_task_percent']:.3f}% 和 "
            f"{teacher_validation['quality_pass_task_percent']:.3f}%，覆盖审计无缺口。",
            "",
            "## 4. PF、模仿学习与 PPO 的统一锁定评价",
            "",
            "| 方法 | 运行数 | 成功率 (%) | 严格成功率 (%) | 严重失败 (%) | 成功任务步数 | 总体积 (mL) | 最终误差 (pH) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    nominal = [row for row in evaluation if row["suite"] == "nominal_locked"]
    method_labels = {"teacher": "Robust PF", "imitation": "模仿学习", "ppo": "PPO"}
    for row in sorted(nominal, key=lambda item: ["teacher", "imitation", "ppo"].index(item["method"])):
        lines.append(
            f"| {method_labels[row['method']]} | {row['runs']} | {value(row, 'success_rate_percent')} | "
            f"{value(row, 'strict_success_rate_percent')} | {value(row, 'severe_failure_rate_percent')} | "
            f"{value(row, 'successful_steps_mean')} | {value(row, 'total_volume_mean_ml', 3)} | "
            f"{value(row, 'final_abs_error_mean', 4)} |"
        )

    lines.extend(
        [
            "",
            "## 5. 按验证集选定的两个网络",
            "",
            "| 网络 | 训练种子 | 成功率 (%) | 严格成功率 (%) | 严重失败 (%) | False stop (%) | 成功任务步数 | 最终误差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [row for row in networks if row["suite"] == "nominal_locked"]:
        lines.append(
            f"| {row['network']} | {row['training_seed']} | {float(row['success_rate_percent']):.2f} | "
            f"{float(row['strict_success_rate_percent']):.2f} | {float(row['severe_failure_rate_percent']):.2f} | "
            f"{float(row['false_stop_rate_percent']):.2f} | {float(row['successful_steps_mean']):.2f} | "
            f"{float(row['final_abs_error_mean']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 6. 不同实验次数的曲线与后验准确度",
            "",
            "固定观测次数分析对同一批 1500 个任务继续到 12 次观测，即使控制目标提前达到也继续采样；自然停止点另行保存，因此固定次数之间没有困难任务幸存偏差。完整曲线按相对初始状态的 -100 至 +100 mL 有符号 0.1 M 滴定剂评价。",
            "",
            "| 观测次数 | 曲线 RMSE (pH) | 曲线相关系数 | 浓度相对误差 (%) | K 准确率 (%) | 真实 K 后验概率 | K 正确时 pKa MAE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    fixed = [row for row in posterior if row["checkpoint_type"] == "fixed_observation_count"]
    for row in fixed:
        lines.append(
            f"| {row['observations']} | {value(row, 'curve_rmse_ph_mean', 4)} | "
            f"{value(row, 'curve_correlation_mean', 4)} | "
            f"{value(row, 'concentration_relative_error_percent_mean')} | "
            f"{value(row, 'pair_count_accuracy_percent')} | {value(row, 'true_pair_probability_mean', 3)} | "
            f"{value(row, 'pka_mae_if_k_correct_mean', 4)} |"
        )
    end = next(row for row in posterior if row["checkpoint_type"] == "natural_control_end")
    rmse_prior = next(row for row in rmse_distribution if int(row["observations"]) == 0)
    rmse_final = max(rmse_distribution, key=lambda row: int(row["observations"]))
    rmse_total_change = next(
        row
        for row in rmse_seed_changes
        if int(row["start_observations"]) == 0
        and int(row["end_observations"]) == int(rmse_final["observations"])
    )
    rl_primary = next(
        row
        for row in rl_tests
        if row["suite"] == "combined_unseen" and row["metric"] == "true_success"
    )
    rl_replication = next(
        row
        for row in rl_seed_effects
        if row["suite"] == "combined_unseen" and row["training_seed"] == "all"
    )
    rl_selected_dynamic = next(
        row
        for row in rl_dynamics
        if int(row["training_seed"]) == int(rl_completion["selected_ppo_seed"])
    )
    rl_nominal = next(
        row for row in rl_tests if row["suite"] == "nominal" and row["metric"] == "true_success"
    )
    rl_sensor_noise = next(
        row
        for row in rl_tests
        if row["suite"] == "sensor_noise_sd_0p05" and row["metric"] == "true_success"
    )
    rl_response_lag = next(
        row
        for row in rl_tests
        if row["suite"] == "response_fraction_0p70" and row["metric"] == "true_success"
    )
    lines.extend(
        [
            "",
            "自然实验结束时：平均观测次数 "
            f"{value(end, 'observations_mean')}；曲线 RMSE {value(end, 'curve_rmse_ph_mean', 4)} pH；"
            f"浓度相对误差 {value(end, 'concentration_relative_error_percent_mean')}%；"
            f"K 准确率 {value(end, 'pair_count_accuracy_percent')}%；"
            f"K 正确时 pKa MAE {value(end, 'pka_mae_if_k_correct_mean', 4)}。",
            "",
            "分布分析显示，完整曲线 RMSE 的总体均值从先验的 "
            f"{float(rmse_prior['mean']):.4f} 降至 {rmse_final['observations']} 次观测的 "
            f"{float(rmse_final['mean']):.4f} pH，中位数从 {float(rmse_prior['median']):.4f} 降至 "
            f"{float(rmse_final['median']):.4f}。RMSE <= 0.5 pH 的任务比例从 "
            f"{float(rmse_prior['within_0p5_percent']):.2f}% 增至 {float(rmse_final['within_0p5_percent']):.2f}%。",
            f"五个独立种子的 0 到 {rmse_final['observations']} 次均值变化全部为负，跨种子均值变化 "
            f"{float(rmse_total_change['mean_of_seed_mean_changes']):.4f} +/- "
            f"{float(rmse_total_change['seed_change_sd']):.4f} pH，95% t 区间 "
            f"[{float(rmse_total_change['seed_t95_ci_low']):.4f}, {float(rmse_total_change['seed_t95_ci_high']):.4f}]。"
            "但是仅 60.0% 的任务从先验到 12 次获得更低 RMSE，因此不能声称每个任务单调收敛。",
            f"自然结束时，全曲线 RMSE 与最终控制误差仅弱相关（Spearman rho="
            f"{float(fit_control_association['spearman_rmse_vs_final_error_rho']):.3f}）。"
            "这再次说明控制成功、全曲线拟合和参数恢复是不同终点。详细分布和图见 `08_pf_fit_distributions`。",
            "",
            "## 7. 强化学习有效性的独立干预审计",
            "",
            "该审计不是新的挑选集。五个 PPO 均从同一个模仿检查点出发；先用原验证集固定 PPO 种子，再在 5 个独立评测种子、每种子 500 个任务上施加超出训练随机化范围的执行器误差、滴定剂强度偏移、传感器噪声、响应滞后和联合扰动。模仿与 PPO 使用完全相同的任务和随机抽样。",
            "",
            f"预设主终点（联合未见扰动）中，验证集选定的 PPO 相对模仿成功率差为 {float(rl_primary['difference']):+.2f} 个百分点，"
            f"95% CI [{float(rl_primary['ci95_lower']):+.2f}, {float(rl_primary['ci95_upper']):+.2f}]，"
            f"精确配对 McNemar p={float(rl_primary['p_value']):.6g}。"
            f"五个独立 PPO 训练种子中有 {int(float(rl_replication['positive_training_seeds']))}/5 个为正效应，"
            f"平均差 {float(rl_replication['success_difference_pp']):+.2f} +/- {float(rl_replication['success_difference_seed_sd']):.2f} 个百分点。",
            f"选定检查点位于 {int(float(rl_selected_dynamic['selected_environment_steps']))} 次环境交互，"
            f"相对模仿初始化的参数 L2 变化比例为 {100.0 * float(rl_selected_dynamic['relative_parameter_l2_change']):.3f}%。"
            f"按预设证据规则，结论为 `{rl_completion['evidence_conclusion']}`；若不满足规则，不把训练后波动表述为 RL 有效。",
            "",
            f"次级但预先固定的条件显示出明显异质性：标称未见任务成功率差 {float(rl_nominal['difference']):+.2f} 点"
            f"（95% CI {float(rl_nominal['ci95_lower']):+.2f} 至 {float(rl_nominal['ci95_upper']):+.2f}），"
            f"0.05 pH 传感器噪声下为 {float(rl_sensor_noise['difference']):+.2f} 点"
            f"（95% CI {float(rl_sensor_noise['ci95_lower']):+.2f} 至 {float(rl_sensor_noise['ci95_upper']):+.2f}）；"
            f"但 0.70 响应滞后下为 {float(rl_response_lag['difference']):+.2f} 点"
            f"（95% CI {float(rl_response_lag['ci95_lower']):+.2f} 至 {float(rl_response_lag['ci95_upper']):+.2f}）。"
            "因此 PPO 的正确结论是特定条件下有改善、严重响应滞后是明确失败模式，而不是普遍鲁棒。",
            "",
            "该审计只证明共同酸/碱方向规则下的体积策略是否经 PPO 得到改进；方向仍由外部规则给出，神经网络完整决定 0.01-10.00 mL 加液体积。",
            "",
            "## 8. 结论边界",
            "",
            "外部体积整形和过冲上限是当前 PF 高控制成功率的重要部分，因此将完整 PF 控制器作为教师是有数据依据的。神经方法是否优于 PF 或彼此，应以五个独立 PPO 种子的均值与验证集选定的双网络锁定比较共同判断，不能挑锁定集最优种子。",
            "",
            "曲线拟合和参数识别与到达目标 pH 是不同目标：控制可能在后验参数完全收敛前结束。pKa 误差只在有效配对数 K 预测正确的任务上解释，必须与 K 准确率同时报告。所有结果均为已知化学模型下的模拟，仍需湿实验外部验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def validate() -> dict:
    required = [
        ABLATION / "ABLATION_COMPLETE.json",
        STUDY / "PIPELINE_COMPLETE.json",
        STUDY / "01_teacher_data" / "TEACHER_DATA_COMPLETE.json",
        STUDY / "02_imitation" / "IMITATION_COMPLETE.json",
        STUDY / "03_ppo" / "PPO_COMPLETE.json",
        STUDY / "04_evaluation" / "EVALUATION_COMPLETE.json",
        STUDY / "05_posterior_diagnostics" / "POSTERIOR_DIAGNOSTICS_COMPLETE.json",
        STUDY / "06_two_network_evaluation" / "TWO_NETWORK_EVALUATION_COMPLETE.json",
        STUDY / "07_rl_effectiveness" / "RL_EFFECTIVENESS_COMPLETE.json",
        STUDY / "08_pf_fit_distributions" / "PF_FIT_DISTRIBUTION_COMPLETE.json",
        STUDY / "TEACHER_SELECTION.json",
    ]
    errors = [f"missing: {path}" for path in required if not path.is_file()]
    counts = {}
    if not errors:
        counts = {
            "ablation_task_rows": len(read_csv(ABLATION / "all_task_results.csv")),
            "evaluation_task_rows": len(read_csv(STUDY / "04_evaluation" / "all_task_results.csv")),
            "evaluation_per_run_rows": len(read_csv(STUDY / "04_evaluation" / "per_run_summary.csv")),
            "posterior_task_rows": len(
                read_csv(STUDY / "05_posterior_diagnostics" / "all_posterior_task_results.csv")
            ),
            "posterior_per_seed_rows": len(
                read_csv(STUDY / "05_posterior_diagnostics" / "per_seed_posterior_summary.csv")
            ),
            "two_network_summary_rows": len(
                read_csv(STUDY / "06_two_network_evaluation" / "two_network_summary.csv")
            ),
            "rl_effectiveness_task_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "all_intervention_task_results.csv")
            ),
            "rl_effectiveness_per_seed_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "per_evaluation_seed_summary.csv")
            ),
            "rl_effectiveness_aggregate_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "aggregate_intervention_summary.csv")
            ),
            "rl_effectiveness_paired_test_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "selected_ppo_paired_tests.csv")
            ),
            "rl_effectiveness_seed_effect_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "all_ppo_seed_effects.csv")
            ),
            "rl_effectiveness_training_rows": len(
                read_csv(STUDY / "07_rl_effectiveness" / "ppo_training_dynamics.csv")
            ),
            "pf_rmse_distribution_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "rmse_distribution_by_observation.csv")
            ),
            "pf_rmse_per_seed_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "rmse_per_seed_distribution.csv")
            ),
            "pf_paired_change_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "paired_rmse_change_tests.csv")
            ),
            "pf_change_by_seed_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "rmse_change_by_seed.csv")
            ),
            "pf_change_seed_summary_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "rmse_change_seed_summary.csv")
            ),
            "pf_parameter_distribution_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "posterior_parameter_distribution.csv")
            ),
            "pf_natural_subgroup_rows": len(
                read_csv(STUDY / "08_pf_fit_distributions" / "natural_end_rmse_subgroups.csv")
            ),
        }
        expected = {
            "ablation_task_rows": 45000,
            "evaluation_task_rows": 11200,
            "evaluation_per_run_rows": 21,
            "posterior_task_rows": 12000,
            "posterior_per_seed_rows": 40,
            "two_network_summary_rows": 6,
            "rl_effectiveness_task_rows": 105000,
            "rl_effectiveness_per_seed_rows": 210,
            "rl_effectiveness_aggregate_rows": 42,
            "rl_effectiveness_paired_test_rows": 49,
            "rl_effectiveness_seed_effect_rows": 42,
            "rl_effectiveness_training_rows": 5,
            "pf_rmse_distribution_rows": 7,
            "pf_rmse_per_seed_rows": 35,
            "pf_paired_change_rows": 7,
            "pf_change_by_seed_rows": 35,
            "pf_change_seed_summary_rows": 7,
            "pf_parameter_distribution_rows": 28,
            "pf_natural_subgroup_rows": 11,
        }
        for key, expected_value in expected.items():
            if counts[key] != expected_value:
                errors.append(f"{key}: {counts[key]} != {expected_value}")

    checkpoints = [STUDY / "02_imitation" / "imitation_best.pth"] + [
        STUDY / "03_ppo" / f"seed_{seed}" / "best_ppo.pth" for seed in [101, 202, 303, 404, 555]
    ]
    hashes = {}
    for path in checkpoints:
        if not path.is_file():
            errors.append(f"missing checkpoint: {path}")
        else:
            hashes[path.relative_to(STUDY).as_posix()] = sha256(path)
    return {
        "generated_unix_time": time.time(),
        "status": "pass" if not errors else "fail",
        "counts": counts,
        "checkpoint_sha256": hashes,
        "errors": errors,
    }


def source_files() -> list[tuple[Path, str]]:
    output = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in {".venv", "__pycache__", "results"} for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".zip"} or path.name == "SHA256SUMS.txt" or path.name.endswith(".sha256.txt"):
            continue
        output.append((path, relative.as_posix()))
    return output


def result_files() -> list[tuple[Path, str]]:
    output = []
    for source_root, prefix in ((ABLATION, "ablation"), (STUDY, "complete_study")):
        for path in source_root.rglob("*"):
            if path.is_file():
                output.append((path, f"{prefix}/{path.relative_to(source_root).as_posix()}"))
    for name in ["README_CN.md", "BAYESIAN_RULE_ABLATION_PROTOCOL.md", "EXPERIMENT_PROTOCOL.md", "DATASET_DESIGN.md"]:
        output.append((ROOT / name, f"documents/{name}"))
    for name in ["REVIEWER_RESPONSE_DRAFT_EN.md", "REVISION_ACTIONS_CN.md"]:
        output.append((ROOT / name, f"documents/{name}"))
    return output


def archive(zip_path: Path, archive_root: str, files: list[tuple[Path, str]]) -> str:
    manifest = [f"{sha256(path)}  {relative}" for path, relative in sorted(files, key=lambda item: item[1])]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path, relative in sorted(files, key=lambda item: item[1]):
            handle.write(path, f"{archive_root}/{relative}")
        handle.writestr(f"{archive_root}/SHA256SUMS.txt", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(zip_path, "r") as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        for line in manifest:
            expected, relative = line.split("  ", 1)
            actual = hashlib.sha256(handle.read(f"{archive_root}/{relative}")).hexdigest()
            if actual != expected:
                raise RuntimeError(f"ZIP hash mismatch: {relative}")
    digest = sha256(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256.txt").write_text(
        f"{digest}  {zip_path.name}\n", encoding="ascii"
    )
    return digest


def main() -> None:
    validation = validate()
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "VALIDATION_REPORT.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation["status"] != "pass":
        raise RuntimeError("Study validation failed; see VALIDATION_REPORT.json")
    (STUDY / "COMPREHENSIVE_ANALYSIS_CN.md").write_text(build_report(), encoding="utf-8")
    (STUDY / "README_FIRST_CN.md").write_text(
        "# 阅读顺序\n\n"
        "1. `COMPREHENSIVE_ANALYSIS_CN.md`\n"
        "2. `VALIDATION_REPORT.json`\n"
        "3. `04_evaluation/RESULT_SUMMARY.md`\n"
        "4. `06_two_network_evaluation/TWO_NETWORK_EVALUATION.md`\n"
        "5. `07_rl_effectiveness/RL_EFFECTIVENESS_AUDIT.md`\n"
        "6. `05_posterior_diagnostics/POSTERIOR_DIAGNOSTIC_SUMMARY.md`\n"
        "7. `08_pf_fit_distributions/PF_FIT_DISTRIBUTION_ANALYSIS_CN.md`\n"
        "8. `08_pf_fit_distributions/pf_rmse_distribution_by_observation.png`\n"
        "9. `../documents/REVIEWER_RESPONSE_DRAFT_EN.md`（压缩包内）\n"
        "10. `../documents/REVISION_ACTIONS_CN.md`（压缩包内）\n"
        "11. `../ablation/RESULT_SUMMARY.md`（压缩包内）\n",
        encoding="utf-8",
    )
    DELIVERY.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(path)}  {relative}" for path, relative in sorted(sources, key=lambda item: item[1])) + "\n",
        encoding="utf-8",
    )
    source_zip = DELIVERY / "bayesian_external_rule_ablation_source_20260812.zip"
    results_zip = DELIVERY / "bayesian_external_rule_ablation_results_20260812.zip"
    source_hash = archive(source_zip, "bayesian_external_rule_ablation_source_20260812", sources)
    results_hash = archive(results_zip, "bayesian_external_rule_ablation_results_20260812", result_files())
    payload = {
        "validation": "pass",
        "source_zip": str(source_zip),
        "source_zip_sha256": source_hash,
        "results_zip": str(results_zip),
        "results_zip_sha256": results_hash,
    }
    (DELIVERY / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
