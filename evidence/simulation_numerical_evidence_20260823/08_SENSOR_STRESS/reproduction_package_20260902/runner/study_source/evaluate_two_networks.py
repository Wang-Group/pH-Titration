from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
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


def select_ppo_run(ppo_complete: dict) -> dict:
    def score(run: dict):
        metrics = run["best_validation"]
        return (
            float(metrics["success_rate_percent"]),
            float(metrics["strict_success_rate_percent"]),
            -float(metrics["severe_failure_rate_percent"]),
            -float(metrics["final_abs_error_mean"]),
        )

    return max(ppo_complete["runs"], key=score)


def exact_mcnemar(reference: list[int], comparison: list[int]) -> tuple[int, int, float]:
    reference_only = sum(bool(a) and not bool(b) for a, b in zip(reference, comparison))
    comparison_only = sum(not bool(a) and bool(b) for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(reference_only, discordant, 0.5).pvalue)
    return reference_only, comparison_only, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked evaluation of the selected imitation and PPO networks")
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = args.pipeline_dir / "04_evaluation"
    ppo_dir = args.pipeline_dir / "03_ppo"
    imitation_path = args.pipeline_dir / "02_imitation" / "imitation_best.pth"
    ppo_complete = json.loads((ppo_dir / "PPO_COMPLETE.json").read_text(encoding="utf-8"))
    selected_run = select_ppo_run(ppo_complete)
    selected_seed = int(selected_run["training_seed"])
    selected_ppo_path = ppo_dir / f"seed_{selected_seed}" / "best_ppo.pth"

    task_rows = read_csv(evaluation_dir / "all_task_results.csv")
    per_run = read_csv(evaluation_dir / "per_run_summary.csv")
    summary_rows = []
    test_rows = []
    continuous_metrics = ["steps", "overshoots", "total_volume_ml", "final_abs_error"]
    for suite in sorted({row["suite"] for row in task_rows}):
        imitation_summary = next(
            row for row in per_run if row["suite"] == suite and row["method"] == "imitation"
        )
        ppo_summary = next(
            row
            for row in per_run
            if row["suite"] == suite
            and row["method"] == "ppo"
            and int(row["training_seed"]) == selected_seed
        )
        summary_rows.extend(
            [
                {
                    "suite": suite,
                    "network": "imitation",
                    "training_seed": 0,
                    "selection_basis": "best imitation validation checkpoint",
                    **{key: value for key, value in imitation_summary.items() if key not in {"suite", "method", "training_seed"}},
                    "checkpoint_sha256": sha256(imitation_path),
                },
                {
                    "suite": suite,
                    "network": "ppo",
                    "training_seed": selected_seed,
                    "selection_basis": "best PPO validation success, then strict/severe/error tie-breaks",
                    **{key: value for key, value in ppo_summary.items() if key not in {"suite", "method", "training_seed"}},
                    "checkpoint_sha256": sha256(selected_ppo_path),
                },
            ]
        )
        imitation = {
            int(row["task_id"]): row
            for row in task_rows
            if row["suite"] == suite and row["method"] == "imitation"
        }
        ppo = {
            int(row["task_id"]): row
            for row in task_rows
            if row["suite"] == suite
            and row["method"] == "ppo"
            and int(row["training_seed"]) == selected_seed
        }
        keys = sorted(set(imitation) & set(ppo))
        imitation_success = [int(imitation[key]["true_success"]) for key in keys]
        ppo_success = [int(ppo[key]["true_success"]) for key in keys]
        imitation_only, ppo_only, p_value = exact_mcnemar(imitation_success, ppo_success)
        test_rows.append(
            {
                "suite": suite,
                "metric": "true_success",
                "comparison": "ppo_minus_imitation",
                "selected_ppo_seed": selected_seed,
                "paired_tasks": len(keys),
                "imitation_only_success": imitation_only,
                "ppo_only_success": ppo_only,
                "difference": 100.0 * (float(np.mean(ppo_success)) - float(np.mean(imitation_success))),
                "difference_unit": "percentage_points",
                "test": "exact_mcnemar",
                "p_value": p_value,
            }
        )
        for metric in continuous_metrics:
            imitation_values = np.asarray([float(imitation[key][metric]) for key in keys])
            ppo_values = np.asarray([float(ppo[key][metric]) for key in keys])
            differences = ppo_values - imitation_values
            if np.allclose(differences, 0.0):
                statistic, continuous_p = 0.0, 1.0
            else:
                result = wilcoxon(ppo_values, imitation_values, zero_method="wilcox", method="auto")
                statistic, continuous_p = float(result.statistic), float(result.pvalue)
            test_rows.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "comparison": "ppo_minus_imitation",
                    "selected_ppo_seed": selected_seed,
                    "paired_tasks": len(keys),
                    "difference": float(np.mean(differences)),
                    "difference_unit": "raw_metric_units",
                    "test": "paired_wilcoxon",
                    "statistic": statistic,
                    "p_value": continuous_p,
                }
            )

    write_csv(args.output_dir / "two_network_summary.csv", summary_rows)
    write_csv(args.output_dir / "two_network_paired_tests.csv", test_rows)
    nominal = [row for row in summary_rows if row["suite"] == "nominal_locked"]
    lines = [
        "# Selected imitation and PPO network evaluation",
        "",
        f"The PPO network was selected from independent validation results: seed {selected_seed}, "
        f"validation success {selected_run['best_validation']['success_rate_percent']:.2f}% at "
        f"{selected_run['best_environment_steps']} interactions.",
        "",
        "| Network | Locked success (%) | Strict (%) | Severe failure (%) | False stop (%) | Successful steps | Volume (mL) | Final error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in nominal:
        lines.append(
            f"| {row['network']} | {float(row['success_rate_percent']):.2f} | "
            f"{float(row['strict_success_rate_percent']):.2f} | {float(row['severe_failure_rate_percent']):.2f} | "
            f"{float(row['false_stop_rate_percent']):.2f} | {float(row['successful_steps_mean']):.2f} | "
            f"{float(row['total_volume_mean_ml']):.3f} | {float(row['final_abs_error_mean']):.4f} |"
        )
    nominal_success_test = next(
        row for row in test_rows if row["suite"] == "nominal_locked" and row["metric"] == "true_success"
    )
    lines.extend(
        [
            "",
            f"Selected PPO minus imitation locked success difference: {nominal_success_test['difference']:+.2f} percentage points; "
            f"exact paired McNemar p={nominal_success_test['p_value']:.6g}.",
            "",
            "The locked tasks were not used for network or checkpoint selection. Stress-suite rows and all paired continuous tests are included in the CSV files.",
        ]
    )
    (args.output_dir / "TWO_NETWORK_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "TWO_NETWORK_EVALUATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "selected_ppo_seed": selected_seed,
                "selected_ppo_validation": selected_run["best_validation"],
                "imitation_sha256": sha256(imitation_path),
                "ppo_sha256": sha256(selected_ppo_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Two-network evaluation complete: selected PPO seed {selected_seed}", flush=True)


if __name__ == "__main__":
    main()
