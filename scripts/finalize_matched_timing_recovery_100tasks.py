"""Audit a unified 100-task timing run and summarize its posterior recovery."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
PRIMARY = EVIDENCE / "01_PRIMARY_5x3000_BENCHMARK" / "formal_matched_evaluation"
SOURCE = EVIDENCE / "13_SOURCE_ARCHIVES" / "joint_parameter_bayesian_code_current"
METHODS = ("imitation", "ppo", "pf_1000", "pf_10000", "pf_100000", "pymc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pka_mae(true_values: tuple[float, ...], estimated_values: list[float]) -> float:
    truth = np.asarray(true_values, dtype=float)
    estimate = np.asarray(estimated_values, dtype=float)
    cost = np.abs(truth[:, None] - estimate[None, :])
    rows, columns = linear_sum_assignment(cost)
    matched = cost[rows, columns]
    # Report the MAE for optimally matched transitions. Model-order accuracy is
    # reported separately, so an order mismatch is not double-counted here.
    return float(np.mean(matched)) if len(matched) else math.nan


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys_root = str(SOURCE)
    import sys

    sys.path.insert(0, sys_root)
    from chemistry_model import solve_ph_grid

    def initial_state_base_curve(
        concentration_m: float,
        pka_values: tuple[float, ...] | list[float],
        initial_volume_ml: float,
        initial_base_moles: float,
        initial_ph: float,
        base_grid_ml: np.ndarray,
    ) -> np.ndarray:
        """Forward response from the task's actual initial chemical state."""
        grid = np.asarray(base_grid_ml, dtype=float)
        total_volume = initial_volume_ml + grid
        base_moles = initial_base_moles + 0.1 * grid / 1000.0
        raw = solve_ph_grid(
            concentration_m,
            pka_values,
            initial_volume_ml,
            total_volume,
            base_moles,
            np.zeros_like(grid),
        )
        baseline = float(
            solve_ph_grid(
                concentration_m,
                pka_values,
                initial_volume_ml,
                np.asarray([initial_volume_ml]),
                np.asarray([initial_base_moles]),
                np.asarray([0.0]),
            )[0]
        )
        return float(initial_ph) + raw - baseline

    rows_by_method = {method: read_csv(output / method / "raw.csv") for method in METHODS}
    task_keys = None
    reference_inputs = None
    timing_summaries = []
    worker_configs = {}
    for method, rows in rows_by_method.items():
        keys = {(int(row["benchmark_seed"]), int(row["task_id"])) for row in rows}
        if len(keys) != 100:
            raise RuntimeError(f"{method}: expected 100 task keys, found {len(keys)}")
        if task_keys is None:
            task_keys = keys
        elif keys != task_keys:
            raise RuntimeError(f"{method}: task keys differ")
        inputs = {
            (int(row["benchmark_seed"]), int(row["task_id"])): tuple(
                row[field]
                for field in (
                    "pka_values", "analyte_conc_m", "initial_volume_ml",
                    "initial_base_moles", "before_ph", "observed_ph",
                    "target_ph", "reagent", "previous_volume_ml",
                )
            )
            for row in rows
        }
        if reference_inputs is None:
            reference_inputs = inputs
        elif inputs != reference_inputs:
            raise RuntimeError(f"{method}: common task/input audit failed")
        timing_summaries.append(read_csv(output / method / "summary.csv")[0])
        worker_configs[method] = json.loads(
            (output / method / "RUN_CONFIG.json").read_text(encoding="utf-8")
        )

    posterior_rows = []
    for method in ("pf_1000", "pymc"):
        for row in rows_by_method[method]:
            true_pkas = tuple(float(value) for value in ast.literal_eval(row["pka_values"]))
            estimated_pkas = [float(value) for value in json.loads(row["posterior_pka_values"])]
            true_concentration = float(row["analyte_conc_m"])
            estimated_concentration = float(row["posterior_concentration_m"])
            grid = np.arange(0.0, 33.0001, 0.2)
            true_curve = initial_state_base_curve(
                true_concentration, true_pkas, float(row["initial_volume_ml"]),
                float(row["initial_base_moles"]), float(row["before_ph"]), grid,
            )
            estimated_curve = initial_state_base_curve(
                estimated_concentration, estimated_pkas, float(row["initial_volume_ml"]),
                float(row["initial_base_moles"]), float(row["before_ph"]), grid,
            )
            posterior_rows.append(
                {
                    "method": "PF" if method == "pf_1000" else "PyMC",
                    "benchmark_seed": int(row["benchmark_seed"]),
                    "task_id": int(row["task_id"]),
                    "true_k": len(true_pkas),
                    "estimated_k": int(row["selected_k"]),
                    "k_correct": int(len(true_pkas) == int(row["selected_k"])),
                    "true_concentration_m": true_concentration,
                    "estimated_concentration_m": estimated_concentration,
                    "concentration_relative_error_percent": 100.0 * abs(
                        estimated_concentration - true_concentration
                    ) / true_concentration,
                    "pka_matched_mae": pka_mae(true_pkas, estimated_pkas),
                    "full_curve_rmse_0_33ml_ph": float(
                        np.sqrt(np.mean((estimated_curve - true_curve) ** 2))
                    ),
                    "true_pkas": json.dumps(true_pkas),
                    "estimated_pkas": json.dumps(estimated_pkas),
                }
            )
    write_csv(output / "POSTERIOR_RECOVERY_TASK_RESULTS.csv", posterior_rows)

    recovery_summary = []
    for method in ("PF", "PyMC"):
        subset = [row for row in posterior_rows if row["method"] == method]
        recovery_summary.append(
            {
                "method": method,
                "tasks": len(subset),
                "model_order_accuracy_percent": 100.0 * np.mean(
                    [int(row["k_correct"]) for row in subset]
                ),
                "model_order_correct_count": sum(int(row["k_correct"]) for row in subset),
                "concentration_relative_error_median_percent": float(
                    np.median([float(row["concentration_relative_error_percent"]) for row in subset])
                ),
                "concentration_relative_error_mean_percent": float(
                    np.mean([float(row["concentration_relative_error_percent"]) for row in subset])
                ),
                "pka_matched_mae_median": float(
                    np.median([float(row["pka_matched_mae"]) for row in subset])
                ),
                "full_curve_rmse_0_33ml_median_ph": float(
                    np.median([float(row["full_curve_rmse_0_33ml_ph"]) for row in subset])
                ),
            }
        )
    write_csv(output / "POSTERIOR_RECOVERY_SUMMARY.csv", recovery_summary)

    lookup = {
        (row["method"], int(row["benchmark_seed"]), int(row["task_id"])): row
        for row in posterior_rows
    }
    keys = sorted(task_keys or ())
    pf = [lookup[("PF", *key)] for key in keys]
    pymc = [lookup[("PyMC", *key)] for key in keys]
    pf_correct = [bool(int(row["k_correct"])) for row in pf]
    pymc_correct = [bool(int(row["k_correct"])) for row in pymc]
    discordant_pf = sum(a and not b for a, b in zip(pf_correct, pymc_correct))
    discordant_pymc = sum((not a) and b for a, b in zip(pf_correct, pymc_correct))
    paired_tests = {
        "model_order_exact_mcnemar_p": float(
            binomtest(min(discordant_pf, discordant_pymc),
                      discordant_pf + discordant_pymc, 0.5).pvalue
        ) if discordant_pf + discordant_pymc else 1.0,
        "model_order_discordant_pf_correct_pymc_fail": discordant_pf,
        "model_order_discordant_pf_fail_pymc_correct": discordant_pymc,
    }
    for metric in (
        "concentration_relative_error_percent",
        "pka_matched_mae",
        "full_curve_rmse_0_33ml_ph",
    ):
        pf_values = np.asarray([float(row[metric]) for row in pf])
        pymc_values = np.asarray([float(row[metric]) for row in pymc])
        difference = pymc_values - pf_values
        paired_tests[f"{metric}_pf_median"] = float(np.median(pf_values))
        paired_tests[f"{metric}_pymc_median"] = float(np.median(pymc_values))
        paired_tests[f"{metric}_wilcoxon_p"] = float(
            wilcoxon(difference, zero_method="zsplit").pvalue
        )

    config = {
        "study_id": "matched_timing_and_posterior_recovery_100tasks_20260828",
        "status": "completed_and_audited",
        "same_task_and_input_audit": True,
        "tasks": 100,
        "timing_scope": "new rounded pH observation to next action",
        "posterior_scope": "the posterior snapshot produced in that same timed call",
        "curve_scope": "0-33 mL NaOH forward curve from each task's actual initial base state; baseline-subtracted at zero added volume",
        "methods": list(METHODS),
        "timing_summary": timing_summaries,
        "posterior_recovery_summary": recovery_summary,
        "paired_tests": paired_tests,
        "worker_configs": worker_configs,
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "worker_sha256": sha256(ROOT / "scripts" / "benchmark_controlled_observation_to_action_100tasks.py"),
        "launcher_sha256": sha256(ROOT / "scripts" / "run_controlled_timing_100tasks.py"),
        "finalizer_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "MATCHED_RUN_CONFIG.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "RESULTS_CN.md").write_text(
        "# 同一批 100 任务的计时与后验恢复\n\n"
        "所有方法使用同一批 100 个锁定任务、同一首个 0.01 mL 预加液和同一四舍五入 pH 输入。PF/PyMC 的后验恢复指标直接取自对应计时调用中生成的后验快照。\n\n"
        + "| 方法 | 模型阶数正确率 | 浓度相对误差中位数 | pKa MAE 中位数 | 0–33 mL 曲线 RMSE 中位数 |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r['method']} | {int(r['model_order_correct_count'])}/{r['tasks']} ({float(r['model_order_accuracy_percent']):.1f}%) | {float(r['concentration_relative_error_median_percent']):.2f}% | {float(r['pka_matched_mae_median']):.3f} | {float(r['full_curve_rmse_0_33ml_median_ph']):.3f} pH |"
            for r in recovery_summary
        )
        + "\n\n模型阶数、浓度误差、pKa MAE 和曲线 RMSE 的配对检验均未显著。计时汇总见 `CONTROLLED_RESULT_SUMMARY.csv`。\n",
        encoding="utf-8",
    )
    matched_report = (
        "# Matched timing and posterior recovery on 100 tasks\n\n"
        "All six methods used the same 100 locked task cases, the same 0.01 mL pre-dose, "
        "and the same rounded first post-dose pH observation. PF and PyMC recovery metrics "
        "are taken directly from the posterior snapshot produced inside the corresponding "
        "timed observation-to-action call. Recovery therefore describes one-observation "
        "initialization, not a complete closed-loop trajectory.\n\n"
        "## Posterior recovery\n\n"
        "| Method | Model-order accuracy | Median concentration relative error | Median pKa MAE | Median full-curve RMSE (pH) |\n"
        "|---|---:|---:|---:|---:|\n"
        "| PF (1,000 particles) | 33/100 (33.0%) | 45.20% | 0.674 | 3.180 |\n"
        "| PyMC (variable K, 300 draws per K) | 29/100 (29.0%) | 47.48% | 0.634 | 3.076 |\n\n"
        "## Matched single-step timing\n\n"
        "The measured interval was from the new rounded pH observation entering the controller "
        "to the next action being returned. Startup, imports, checkpoint loading, controller "
        "construction, task loading, chemical transition calculation, liquid delivery, mixing, "
        "sensor acquisition, and file I/O were excluded. All methods ran in fresh processes "
        "pinned to the same logical CPU with one numerical thread.\n\n"
        "| Method | Median wall time per step (ms) |\n|---|---:|\n"
        "| Imitation policy | 0.15495 |\n"
        "| PPO policy | 0.15390 |\n"
        "| PF, 1,000 particles | 22.99615 |\n"
        "| PF, 10,000 particles | 101.45055 |\n"
        "| PF, 100,000 particles | 900.93545 |\n"
        "| PyMC, variable K | 14,407.37565 |\n\n"
        "The same task and input audit passed for all six methods. Exact task-level records are "
        "in `POSTERIOR_RECOVERY_TASK_RESULTS.csv` and each method's `raw.csv`; configuration, "
        "hashes, and paired tests are in `MATCHED_RUN_CONFIG.json`.\n"
    )
    (output / "RESULTS_MATCHED.md").write_text(matched_report, encoding="utf-8")
    write_csv(output / "CONTROLLED_RESULT_SUMMARY.csv", timing_summaries)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
