from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_core import exact_mcnemar, holm_adjust
from evaluate_candidates import clustered_bootstrap_difference, safe_wilcoxon


CANDIDATES = ("sac_history_robust", "td3_filtered_robust")
COMPARATORS = ("imitation", "submitted_rl", "ppo_reference")
BOOLEAN_FIELDS = ("true_success", "success_005", "severe_failure_050")
CONTINUOUS_FIELDS = ("steps", "overshoots", "total_added_ml", "final_abs_error")


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_selected(path: Path) -> dict[tuple[str, str], dict[tuple[int, int], dict]]:
    selected = set(CANDIDATES) | set(COMPARATORS)
    grouped: dict[tuple[str, str], dict[tuple[int, int], dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in selected:
                continue
            for field in BOOLEAN_FIELDS:
                row[field] = parse_bool(row[field])
            for field in CONTINUOUS_FIELDS:
                row[field] = float(row[field])
            key = (int(row["eval_seed"]), int(row["task_id"]))
            grouped.setdefault((row["scenario"], method), {})[key] = row
    return grouped


def paired_rows(grouped: dict, bootstrap_iterations: int) -> list[dict]:
    scenarios = sorted({scenario for scenario, _ in grouped})
    output: list[dict] = []
    for scenario in scenarios:
        for candidate in CANDIDATES:
            candidate_rows = grouped[(scenario, candidate)]
            for comparator in COMPARATORS:
                comparator_rows = grouped[(scenario, comparator)]
                keys = sorted(set(candidate_rows) & set(comparator_rows))
                clusters = np.asarray([key[0] for key in keys], dtype=int)
                c_success = np.asarray([candidate_rows[key]["true_success"] for key in keys], dtype=bool)
                r_success = np.asarray([comparator_rows[key]["true_success"] for key in keys], dtype=bool)
                differences = 100.0 * (c_success.astype(float) - r_success.astype(float))
                ci_low, ci_high = clustered_bootstrap_difference(
                    differences,
                    clusters,
                    bootstrap_iterations,
                    50_000 + len(output),
                )
                success_test = exact_mcnemar(r_success.tolist(), c_success.tolist())
                strict_test = exact_mcnemar(
                    [comparator_rows[key]["success_005"] for key in keys],
                    [candidate_rows[key]["success_005"] for key in keys],
                )
                severe_test = exact_mcnemar(
                    [comparator_rows[key]["severe_failure_050"] for key in keys],
                    [candidate_rows[key]["severe_failure_050"] for key in keys],
                )
                item = {
                    "scenario": scenario,
                    "candidate": candidate,
                    "comparator": comparator,
                    "matched_tasks": len(keys),
                    "success_difference_pp": float(differences.mean()),
                    "success_difference_cluster_ci95_low": ci_low,
                    "success_difference_cluster_ci95_high": ci_high,
                    "success_p_value_exact": success_test["p_value_exact_two_sided"],
                    "strict_success_005_difference_pp": 100.0
                    * np.mean(
                        [
                            candidate_rows[key]["success_005"] - comparator_rows[key]["success_005"]
                            for key in keys
                        ]
                    ),
                    "strict_success_005_p_value_exact": strict_test["p_value_exact_two_sided"],
                    "severe_failure_050_difference_pp": 100.0
                    * np.mean(
                        [
                            candidate_rows[key]["severe_failure_050"]
                            - comparator_rows[key]["severe_failure_050"]
                            for key in keys
                        ]
                    ),
                    "severe_failure_050_p_value_exact": severe_test["p_value_exact_two_sided"],
                }
                for field in CONTINUOUS_FIELDS:
                    candidate_values = np.asarray([candidate_rows[key][field] for key in keys], dtype=float)
                    comparator_values = np.asarray([comparator_rows[key][field] for key in keys], dtype=float)
                    item[f"{field}_mean_difference"] = float(np.mean(candidate_values - comparator_values))
                    item[f"{field}_p_value"] = safe_wilcoxon(candidate_values, comparator_values)
                output.append(item)

    adjusted = {
        "success": holm_adjust([row["success_p_value_exact"] for row in output]),
        "strict": holm_adjust([row["strict_success_005_p_value_exact"] for row in output]),
        "severe": holm_adjust([row["severe_failure_050_p_value_exact"] for row in output]),
    }
    for field in CONTINUOUS_FIELDS:
        adjusted[field] = holm_adjust([row[f"{field}_p_value"] for row in output])
    for index, row in enumerate(output):
        row["success_p_value_holm"] = adjusted["success"][index]
        row["strict_success_005_p_value_holm"] = adjusted["strict"][index]
        row["severe_failure_050_p_value_holm"] = adjusted["severe"][index]
        for field in CONTINUOUS_FIELDS:
            row[f"{field}_p_value_holm"] = adjusted[field][index]
        row["clear_success_gain_over_existing"] = bool(
            row["success_difference_pp"] >= 1.0 and row["success_p_value_holm"] < 0.05
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory paired comparison of Full SAC/TD3 results to existing learned policies.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results_full")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    output_dir = args.results_dir / "additional_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_selected(args.results_dir / "evaluation" / "per_task_results.csv")
    rows = paired_rows(grouped, args.bootstrap_iterations)
    write_csv(output_dir / "new_candidates_vs_existing.csv", rows)
    summary = {
        "status": "exploratory_post_hoc",
        "family_size": len(rows),
        "candidates": list(CANDIDATES),
        "comparators": list(COMPARATORS),
        "clear_success_gains": [row for row in rows if row["clear_success_gain_over_existing"]],
    }
    (output_dir / "new_candidates_vs_existing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"comparisons": len(rows), "clear_success_gains": len(summary["clear_success_gains"])}, indent=2))


if __name__ == "__main__":
    main()
