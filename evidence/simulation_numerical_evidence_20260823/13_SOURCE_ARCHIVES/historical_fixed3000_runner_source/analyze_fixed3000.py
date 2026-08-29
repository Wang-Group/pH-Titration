from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from benchmark_core import exact_mcnemar, holm_adjust


BOOLEAN_FIELDS = {
    "true_success",
    "success_005",
    "success_020",
    "severe_failure_050",
    "measured_success",
}
CONTINUOUS_METRICS = ["steps", "overshoots", "total_added_ml", "final_abs_error"]
PRIMARY_METHODS = {
    "sac_history_robust",
    "td3_filtered_robust",
    "ppo_residual_imitation",
    "ppo_history_robust",
    "ppo_filtered_robust",
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def upper_cvar(values: list[float], alpha: float = 0.95) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(math.ceil((1.0 - alpha) * ordered.size)))
    return float(np.mean(ordered[-count:]))


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    successful = [row for row in rows if as_bool(row["true_success"])]
    steps = [float(row["steps"]) for row in rows]
    volumes = [float(row["total_added_ml"]) for row in rows]
    errors = [float(row["final_abs_error"]) for row in rows]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * len(successful) / len(rows),
        "strict_success_005_percent": 100.0 * sum(as_bool(row["success_005"]) for row in rows) / len(rows),
        "wide_success_020_percent": 100.0 * sum(as_bool(row["success_020"]) for row in rows) / len(rows),
        "severe_failure_050_percent": 100.0 * sum(as_bool(row["severe_failure_050"]) for row in rows) / len(rows),
        "false_stop_rate_percent": 100.0 * sum(
            as_bool(row["measured_success"]) and not as_bool(row["true_success"])
            for row in rows
        ) / len(rows),
        "all_steps_mean": float(np.mean(steps)),
        "steps_p95": float(np.percentile(steps, 95)),
        "steps_cvar95": upper_cvar(steps),
        "overshoots_per_task_mean": float(np.mean([float(row["overshoots"]) for row in rows])),
        "total_added_ml_mean": float(np.mean(volumes)),
        "total_added_ml_p95": float(np.percentile(volumes, 95)),
        "total_added_ml_cvar95": upper_cvar(volumes),
        "final_abs_error_mean": float(np.mean(errors)),
        "final_abs_error_p95": float(np.percentile(errors, 95)),
        "final_abs_error_cvar95": upper_cvar(errors),
        "decision_time_ms_mean": float(np.mean([float(row["decision_time_ms"]) for row in rows])),
    }


def cluster_ci(
    cluster_values: dict[tuple[int, int], float],
    design: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = sorted(cluster_values)
    if not keys:
        return math.nan, math.nan
    samples = np.empty(iterations, dtype=float)
    if design == "cross":
        train_seeds = sorted({key[0] for key in keys})
        eval_seeds = sorted({key[1] for key in keys})
        for index in range(iterations):
            selected_train = rng.choice(train_seeds, size=len(train_seeds), replace=True)
            selected_eval = rng.choice(eval_seeds, size=len(eval_seeds), replace=True)
            values = [cluster_values[(int(train), int(evaluate))] for train in selected_train for evaluate in selected_eval]
            samples[index] = float(np.mean(values))
    else:
        values = np.asarray([cluster_values[key] for key in keys], dtype=float)
        indices = rng.integers(0, len(values), size=(iterations, len(values)))
        samples = values[indices].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    differences = a - b
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue)


def load_scenario_rows(shards: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in shards:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def aggregate_cluster_summaries(cluster_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cluster_rows:
        grouped[(str(row["scenario"]), str(row["method"]))].append(row)
    metric_names = [
        "success_rate_percent",
        "strict_success_005_percent",
        "wide_success_020_percent",
        "severe_failure_050_percent",
        "false_stop_rate_percent",
        "all_steps_mean",
        "steps_p95",
        "steps_cvar95",
        "overshoots_per_task_mean",
        "total_added_ml_mean",
        "total_added_ml_p95",
        "total_added_ml_cvar95",
        "final_abs_error_mean",
        "final_abs_error_p95",
        "final_abs_error_cvar95",
        "decision_time_ms_mean",
    ]
    output: list[dict[str, Any]] = []
    for (scenario, method), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "clusters": len(rows),
            "tasks_per_cluster": int(rows[0]["tasks"]),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_cluster_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def comparisons_for(methods: set[str]) -> list[tuple[str, str]]:
    comparisons: list[tuple[str, str]] = []
    if "bayesian_common" in methods:
        for method in sorted(methods - {"bayesian_common", "bayesian_original"}):
            comparisons.append(("bayesian_common", method))
    for comparator in ("imitation", "ppo_reference"):
        if comparator not in methods:
            continue
        for method in sorted(PRIMARY_METHODS & methods):
            if method != comparator:
                comparisons.append((comparator, method))
    if "bayesian_original" in methods and "bayesian_common" in methods:
        comparisons.append(("bayesian_original", "bayesian_common"))
    return comparisons


def compare_methods(
    scenario: str,
    rows: list[dict[str, str]],
    design: str,
    iterations: int,
    comparison_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_method: dict[str, dict[tuple[int, int, int], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["train_seed"]), int(row["eval_seed"]), int(row["task_id"]))
        by_method[str(row["method"])][key] = row
    tests: list[dict[str, Any]] = []
    cluster_tests: list[dict[str, Any]] = []
    for comparator, method in comparisons_for(set(by_method)):
        base = by_method[comparator]
        candidate = by_method[method]
        keys = sorted(set(base) & set(candidate))
        if not keys:
            continue
        b_success = np.asarray([as_bool(base[key]["true_success"]) for key in keys], dtype=bool)
        m_success = np.asarray([as_bool(candidate[key]["true_success"]) for key in keys], dtype=bool)
        b_strict = np.asarray([as_bool(base[key]["success_005"]) for key in keys], dtype=bool)
        m_strict = np.asarray([as_bool(candidate[key]["success_005"]) for key in keys], dtype=bool)
        b_severe = np.asarray([as_bool(base[key]["severe_failure_050"]) for key in keys], dtype=bool)
        m_severe = np.asarray([as_bool(candidate[key]["severe_failure_050"]) for key in keys], dtype=bool)
        mcnemar = exact_mcnemar(b_success.tolist(), m_success.tolist())
        strict_test = exact_mcnemar(b_strict.tolist(), m_strict.tolist())
        severe_test = exact_mcnemar(b_severe.tolist(), m_severe.tolist())

        clusters = sorted({(key[0], key[1]) for key in keys})
        cluster_differences: dict[tuple[int, int], float] = {}
        for train_seed, eval_seed in clusters:
            cluster_keys = [key for key in keys if key[0] == train_seed and key[1] == eval_seed]
            base_values = [as_bool(base[key]["true_success"]) for key in cluster_keys]
            method_values = [as_bool(candidate[key]["true_success"]) for key in cluster_keys]
            difference = 100.0 * float(np.mean(np.asarray(method_values, dtype=float) - np.asarray(base_values, dtype=float)))
            cluster_differences[(train_seed, eval_seed)] = difference
            cluster_mcnemar = exact_mcnemar(base_values, method_values)
            cluster_tests.append(
                {
                    "scenario": scenario,
                    "comparator": comparator,
                    "method": method,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "matched_tasks": len(cluster_keys),
                    "comparator_success_percent": 100.0 * float(np.mean(base_values)),
                    "method_success_percent": 100.0 * float(np.mean(method_values)),
                    "success_difference_pp": difference,
                    **cluster_mcnemar,
                }
            )
        ci_low, ci_high = cluster_ci(
            cluster_differences,
            design,
            iterations,
            20260724 + comparison_index * 17,
        )

        metrics: dict[str, Any] = {}
        for metric in CONTINUOUS_METRICS:
            base_values = np.asarray([float(base[key][metric]) for key in keys])
            method_values = np.asarray([float(candidate[key][metric]) for key in keys])
            metrics[f"{metric}_mean_difference"] = float(np.mean(method_values - base_values))
            metrics[f"{metric}_relative_change_percent"] = 100.0 * (
                float(np.mean(method_values)) - float(np.mean(base_values))
            ) / max(abs(float(np.mean(base_values))), 1e-12)
            metrics[f"{metric}_p_value"] = safe_wilcoxon(method_values, base_values)

        tests.append(
            {
                "scenario": scenario,
                "comparator": comparator,
                "method": method,
                "matched_rows": len(keys),
                "clusters": len(clusters),
                "tasks_per_cluster": len(keys) // len(clusters),
                "comparator_success_percent": 100.0 * float(np.mean(b_success)),
                "method_success_percent": 100.0 * float(np.mean(m_success)),
                "success_difference_pp": 100.0 * float(np.mean(m_success.astype(float) - b_success.astype(float))),
                "success_difference_cluster_ci95_low": ci_low,
                "success_difference_cluster_ci95_high": ci_high,
                "positive_clusters": sum(value > 0 for value in cluster_differences.values()),
                "zero_clusters": sum(value == 0 for value in cluster_differences.values()),
                "negative_clusters": sum(value < 0 for value in cluster_differences.values()),
                "success_p_value_exact_pooled": mcnemar["p_value_exact_two_sided"],
                "success_a_success_b_fail": mcnemar["a_success_b_fail"],
                "success_a_fail_b_success": mcnemar["a_fail_b_success"],
                "strict_success_005_difference_pp": 100.0 * float(np.mean(m_strict.astype(float) - b_strict.astype(float))),
                "strict_success_005_p_value_exact_pooled": strict_test["p_value_exact_two_sided"],
                "severe_failure_050_difference_pp": 100.0 * float(np.mean(m_severe.astype(float) - b_severe.astype(float))),
                "severe_failure_050_p_value_exact_pooled": severe_test["p_value_exact_two_sided"],
                **metrics,
            }
        )
    return tests, cluster_tests


def add_holm_and_decisions(tests: list[dict[str, Any]]) -> None:
    if not tests:
        return
    fields = [
        "success_p_value_exact_pooled",
        "strict_success_005_p_value_exact_pooled",
        "severe_failure_050_p_value_exact_pooled",
        *[f"{metric}_p_value" for metric in CONTINUOUS_METRICS],
    ]
    for field in fields:
        adjusted = holm_adjust([float(row[field]) for row in tests])
        for row, value in zip(tests, adjusted):
            row[f"{field}_holm"] = value
    for row in tests:
        consistency = float(row["positive_clusters"]) / max(1, int(row["clusters"]))
        row["clear_success_win"] = bool(
            float(row["success_difference_pp"]) >= 1.0
            and float(row["success_p_value_exact_pooled_holm"]) < 0.05
            and float(row["success_difference_cluster_ci95_low"]) > 0.0
            and consistency >= 0.80
        )
        noninferior = float(row["success_difference_cluster_ci95_low"]) >= -0.5
        efficient = any(
            float(row[f"{metric}_relative_change_percent"]) <= -10.0
            and float(row[f"{metric}_p_value_holm"]) < 0.05
            for metric in ("steps", "overshoots", "total_added_ml")
        )
        row["multiobjective_tradeoff_win"] = bool(
            noninferior and efficient and float(row["final_abs_error_mean_difference"]) <= 0.01
        )


def add_cluster_holm(cluster_tests: list[dict[str, Any]]) -> None:
    if not cluster_tests:
        return
    adjusted = holm_adjust([float(row["p_value_exact_two_sided"]) for row in cluster_tests])
    for row, value in zip(cluster_tests, adjusted):
        row["p_value_exact_two_sided_holm_global"] = value


def format_pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def build_reports(
    output_dir: Path,
    settings: dict[str, Any],
    aggregate: list[dict[str, Any]],
    tests: list[dict[str, Any]],
) -> None:
    confirmed = [row for row in tests if row["clear_success_win"]]
    chinese = [
        "# 固定 3,000 项任务多种子确认性报告",
        "",
        "## 实验设计",
        "",
        f"- 设计：`{settings['design']}`",
        f"- 每个场景固定任务数：{settings['task_count']}",
        f"- 固定任务种子：{settings['task_seed']}",
        f"- 训练种子：{settings['train_seeds']}",
        f"- 扰动/评估种子：{settings['eval_seeds']}",
        f"- Bayesian 粒子数：{settings['particles']}",
        "- 所有方法在每个场景、每个种子组合上共享完全相同的任务和扰动随机数。",
        "",
        "## 聚合结果（均值 +/- 种子组合标准差）",
        "",
    ]
    english = [
        "# Fixed-3,000 Multi-seed Confirmatory Report",
        "",
        "## Design",
        "",
        f"- Design: `{settings['design']}`",
        f"- Fixed tasks per scenario: {settings['task_count']}",
        f"- Fixed task seed: {settings['task_seed']}",
        f"- Training seeds: {settings['train_seeds']}",
        f"- Evaluation/disturbance seeds: {settings['eval_seeds']}",
        f"- Bayesian particles: {settings['particles']}",
        "- Every method used the same tasks and disturbance random numbers within each seed combination.",
        "",
        "## Aggregate results (mean +/- seed-combination SD)",
        "",
    ]
    for scenario in settings["scenarios"]:
        rows = sorted(
            [row for row in aggregate if row["scenario"] == scenario],
            key=lambda row: float(row["success_rate_percent_mean"]),
            reverse=True,
        )
        chinese.extend(
            [
                f"### {scenario}",
                "",
                "| 方法 | 成功率 (%) | 严格 +/-0.05 (%) | 严重失败 (%) | 步数 | 加液量 (mL) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        english.extend(
            [
                f"### {scenario}",
                "",
                "| Method | Success (%) | Strict +/-0.05 (%) | Severe failure (%) | Steps | Added volume (mL) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            values = (
                row["method"],
                format_pm(float(row["success_rate_percent_mean"]), float(row["success_rate_percent_cluster_sd"])),
                format_pm(float(row["strict_success_005_percent_mean"]), float(row["strict_success_005_percent_cluster_sd"])),
                format_pm(float(row["severe_failure_050_percent_mean"]), float(row["severe_failure_050_percent_cluster_sd"])),
                format_pm(float(row["all_steps_mean_mean"]), float(row["all_steps_mean_cluster_sd"])),
                format_pm(float(row["total_added_ml_mean_mean"]), float(row["total_added_ml_mean_cluster_sd"])),
            )
            line = f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} |"
            chinese.append(line)
            english.append(line)
        chinese.append("")
        english.append("")

    chinese.extend(["## 满足严格判据的成功率胜出", ""])
    english.extend(["## Success-rate wins meeting the strict decision rule", ""])
    if confirmed:
        header_cn = "| 场景 | 方法 | 对照 | 差值 (pp) | 种子聚类 95% CI | Holm p | 正向种子组合 |"
        header_en = "| Scenario | Method | Comparator | Difference (pp) | Seed-cluster 95% CI | Holm p | Positive seed cells |"
        separator = "|---|---|---|---:|---:|---:|---:|"
        chinese.extend([header_cn, separator])
        english.extend([header_en, separator])
        for row in confirmed:
            line = (
                f"| {row['scenario']} | {row['method']} | {row['comparator']} | "
                f"{float(row['success_difference_pp']):.2f} | "
                f"[{float(row['success_difference_cluster_ci95_low']):.2f}, "
                f"{float(row['success_difference_cluster_ci95_high']):.2f}] | "
                f"{float(row['success_p_value_exact_pooled_holm']):.3g} | "
                f"{row['positive_clusters']}/{row['clusters']} |"
            )
            chinese.append(line)
            english.append(line)
    else:
        chinese.append("没有比较同时满足效应量、Holm 校正、聚类置信区间和种子一致性判据。")
        english.append("No comparison met the effect-size, Holm, cluster-CI, and seed-consistency criteria simultaneously.")

    chinese.extend(
        [
            "",
            "## 统计说明",
            "",
            f"- 每个种子组合都单独输出了基于同一 {settings['task_count']} 项任务的 exact McNemar 检验。",
            "- 汇总表同时给出 pooled McNemar-Holm 结果和按种子组合重采样的置信区间。",
            "- `cross` 设计的置信区间分别重采样训练种子和扰动种子，避免把 25 个组合当作完全独立样本。",
            "- `clear_success_win` 要求成功率至少提高 1.0 pp、Holm p<0.05、聚类 CI 下界>0，且至少 80% 的种子组合方向一致。",
            "- `bayesian_original` 仅用于 nominal 原生环境与共同环境适配器的桥接检查；算法公平对比应以 `bayesian_common` 为基线。",
        ]
    )
    english.extend(
        [
            "",
            "## Statistical notes",
            "",
            f"- An exact McNemar test on the same {settings['task_count']} tasks is reported separately for every seed combination.",
            "- The summary includes pooled McNemar-Holm results and seed-cluster bootstrap confidence intervals.",
            "- For the crossed design, training and disturbance seeds are resampled separately rather than treating all 25 cells as independent.",
            "- A clear win requires at least +1.0 pp, Holm p<0.05, a cluster-CI lower bound above zero, and a positive direction in at least 80% of seed cells.",
            "- `bayesian_original` is only a nominal bridge check; fair common-environment comparisons use `bayesian_common`.",
        ]
    )
    (output_dir / "FIXED3000_REPORT_CN.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    (output_dir / "FIXED3000_REPORT_EN.md").write_text("\n".join(english) + "\n", encoding="utf-8")


def analyze_directory(output_dir: Path, iterations: int | None = None) -> None:
    output_dir = output_dir.resolve()
    settings = json.loads((output_dir / "settings.json").read_text(encoding="utf-8"))
    iterations = int(iterations or settings.get("bootstrap_iterations", 10000))
    shard_dir = output_dir / "shards"
    all_cluster_rows: list[dict[str, Any]] = []
    all_tests: list[dict[str, Any]] = []
    all_cluster_tests: list[dict[str, Any]] = []
    all_strata: list[dict[str, Any]] = []
    comparison_index = 0

    for scenario in settings["scenarios"]:
        shards = sorted(shard_dir.glob(f"{scenario}__train*__eval*.csv"))
        expected_shards = len(settings["seed_pairs"])
        if len(shards) != expected_shards:
            raise RuntimeError(
                f"Scenario {scenario} has {len(shards)} shards; expected {expected_shards}."
            )
        rows = load_scenario_rows(shards)
        grouped: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(int(row["train_seed"]), int(row["eval_seed"]), str(row["method"]))].append(row)
        for (train_seed, eval_seed, method), subset in sorted(grouped.items()):
            all_cluster_rows.append(
                {
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "method": method,
                    **summarize(subset),
                }
            )

        strata: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            strata[(str(row["method"]), str(row["acid_type"]), str(row["direction"]), str(row["difficulty_bin"]))].append(row)
        for (method, acid_type, direction, difficulty), subset in sorted(strata.items()):
            summary = summarize(subset)
            all_strata.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "acid_type": acid_type,
                    "direction": direction,
                    "difficulty_bin": difficulty,
                    "tasks": summary["tasks"],
                    "success_rate_percent": summary["success_rate_percent"],
                    "strict_success_005_percent": summary["strict_success_005_percent"],
                    "severe_failure_050_percent": summary["severe_failure_050_percent"],
                    "final_abs_error_mean": summary["final_abs_error_mean"],
                    "all_steps_mean": summary["all_steps_mean"],
                    "total_added_ml_mean": summary["total_added_ml_mean"],
                }
            )

        tests, cluster_tests = compare_methods(
            scenario,
            rows,
            str(settings["design"]),
            iterations,
            comparison_index,
        )
        comparison_index += len(tests)
        all_tests.extend(tests)
        all_cluster_tests.extend(cluster_tests)

    aggregate = aggregate_cluster_summaries(all_cluster_rows)
    add_holm_and_decisions(all_tests)
    add_cluster_holm(all_cluster_tests)
    write_csv(output_dir / "per_cluster_summary.csv", all_cluster_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "paired_tests.csv", all_tests)
    write_csv(output_dir / "per_cluster_paired_tests.csv", all_cluster_tests)
    write_csv(output_dir / "stratified_summary.csv", all_strata)
    build_reports(output_dir, settings, aggregate, all_tests)
    decision = {
        "design": settings["design"],
        "task_count": settings["task_count"],
        "scenarios": settings["scenarios"],
        "confirmed_success_wins": [row for row in all_tests if row["clear_success_win"]],
        "multiobjective_tradeoff_wins": [row for row in all_tests if row["multiobjective_tradeoff_win"]],
    }
    (output_dir / "DECISION_SUMMARY.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed-3000 confirmatory shards.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    args = parser.parse_args()
    analyze_directory(args.output_dir, args.bootstrap_iterations)


if __name__ == "__main__":
    main()
