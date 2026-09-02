from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCK = (
    ROOT
    / "evidence"
    / "simulation_numerical_evidence_20260823"
    / "18_CONTROLLER_REPRESENTATION_FACTORIAL"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def lf_normalized_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_hash_table(name: str) -> int:
    rows = read_csv(BLOCK / name)
    if not rows:
        raise SystemExit(f"{name} is empty")
    for row in rows:
        path = BLOCK / row["path"]
        if not path.is_file():
            raise SystemExit(f"Missing archived file: {path.relative_to(ROOT)}")
        observed = lf_normalized_sha256(path)
        if observed != row["sha256_lf_normalized"]:
            raise SystemExit(
                f"Newline-normalized SHA-256 mismatch for {path.relative_to(ROOT)}: "
                f"{observed}"
            )
        if name == "MANIFEST_SHA256.csv":
            task_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
            if task_count != 3000:
                raise SystemExit(
                    f"Locked manifest {path.relative_to(ROOT)} contains {task_count} tasks"
                )
    return len(rows)


def assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"{label}: observed {observed}, expected {expected}")


def decimal_places(text: str) -> int:
    text = text.strip()
    if "e" in text.lower():
        return max(0, -int(text.lower().split("e", 1)[1]))
    return len(text.split(".", 1)[1]) if "." in text else 0


def rounded_value_matches(raw: float, displayed: str, label: str) -> None:
    places = decimal_places(displayed)
    displayed_value = float(displayed)
    tolerance = 0.5 * (10.0 ** (-places)) + 1e-12
    assert_close(raw, displayed_value, tolerance, label)


def verify_publication_tables() -> dict[str, int | float]:
    results = BLOCK / "results"
    s14 = read_csv(results / "table_s14_posterior_to_control.csv")
    expected_s14 = {
        "weighted_parameters_primary": (95.4, 0.6, 4.84, 0.09, 1.81, 0.03, 52.9, 0.6),
        "map_k_posterior_predictive": (95.4, 0.5, 4.84, 0.09, 1.68, 0.04, 101.0, 2.0),
        "full_posterior_predictive": (95.3, 0.5, 4.8, 0.1, 1.64, 0.03, 269.0, 5.0),
    }
    s14_fields = (
        "success_percent_mean",
        "success_percent_sample_sd",
        "successful_additions_mean",
        "successful_additions_sample_sd",
        "curve_rmse_ph_mean",
        "curve_rmse_ph_sample_sd",
        "median_decision_time_ms_mean",
        "median_decision_time_ms_sample_sd",
    )
    if {row["posterior_to_control_strategy"] for row in s14} != set(expected_s14):
        raise SystemExit("Table S14 strategy set is invalid")
    for row in s14:
        observed = tuple(float(row[field]) for field in s14_fields)
        expected = expected_s14[row["posterior_to_control_strategy"]]
        if observed != expected:
            raise SystemExit(f"Table S14 mismatch for {row['posterior_to_control_strategy']}")

    s15 = read_csv(results / "table_s15_pf_representation.csv")
    expected_s15 = {
        "effective_k_protic_k1_3": (95.36, 95.23, 1.814, 2.057),
        "two_independent_monoacids": (95.69, 95.70, 1.715, 1.443),
        "independent_components_j1_3": (95.01, 94.91, 1.797, 1.495),
    }
    s15_fields = (
        "effective_k_protic_pf_success_percent",
        "independent_component_pf_success_percent",
        "effective_k_protic_curve_rmse_ph",
        "independent_component_curve_rmse_ph",
    )
    if {row["ground_truth_domain"] for row in s15} != set(expected_s15):
        raise SystemExit("Table S15 domain set is invalid")
    max_endpoint_difference = 0.0
    for row in s15:
        observed = tuple(float(row[field]) for field in s15_fields)
        expected = expected_s15[row["ground_truth_domain"]]
        if observed != expected:
            raise SystemExit(f"Table S15 mismatch for {row['ground_truth_domain']}")
        max_endpoint_difference = max(max_endpoint_difference, abs(observed[0] - observed[1]))
    assert_close(max_endpoint_difference, 0.13, 1e-12, "maximum PF representation difference")

    raw_rows = read_csv(results / "family_method_summary.csv")
    if len(raw_rows) != 15:
        raise SystemExit(f"Expected 15 policy family-domain rows, found {len(raw_rows)}")
    raw_by_key = {(row["domain"], row["family"]): row for row in raw_rows}
    if len(raw_by_key) != 15:
        raise SystemExit("Policy family-domain keys are not unique")

    domain_map = {
        "effective_k_protic_k1_3": "sequential_k123",
        "two_independent_monoacids": "fixed_two_independent",
        "independent_components_j1_3": "independent_j123",
    }
    s16 = read_csv(results / "table_s16_policy_families.csv")
    if len(s16) != 15:
        raise SystemExit(f"Expected 15 Table S16 rows, found {len(s16)}")
    for displayed in s16:
        raw = raw_by_key[(domain_map[displayed["evaluation_domain"]], displayed["family"])]
        comparisons = (
            ("imitation_success_percent_mean", "imitation_success_percent_mean"),
            ("imitation_success_percent_sample_sd", "imitation_success_percent_benchmark_sd"),
            ("ppo_success_percent_mean", "ppo_success_rate_percent_training_seed_mean"),
            ("ppo_success_percent_sample_sd", "ppo_success_rate_percent_training_seed_sample_sd"),
            (
                "ppo_minus_imitation_percentage_points",
                "ppo_minus_imitation_success_percentage_points_mean",
            ),
        )
        for displayed_field, raw_field in comparisons:
            rounded_value_matches(
                float(raw[raw_field]),
                displayed[displayed_field],
                f"Table S16 {displayed['evaluation_domain']} {displayed['family']} {displayed_field}",
            )
        if int(displayed["ppo_training_seeds_above_imitation"]) != int(
            raw["ppo_training_seeds_above_imitation"]
        ):
            raise SystemExit(
                f"Table S16 seed-count mismatch for {displayed['evaluation_domain']} "
                f"{displayed['family']}"
            )

    ppo_mean_above_imitation = sum(
        float(row["ppo_success_rate_percent_training_seed_mean"])
        > float(row["imitation_success_percent_mean"])
        for row in raw_rows
    )
    ppo_seed_domain_wins = sum(int(row["ppo_training_seeds_above_imitation"]) for row in raw_rows)
    if ppo_mean_above_imitation != 12:
        raise SystemExit(f"Expected 12 of 15 PPO means above imitation, found {ppo_mean_above_imitation}")
    if ppo_seed_domain_wins != 45:
        raise SystemExit(f"Expected 45 of 75 PPO seed-domain wins, found {ppo_seed_domain_wins}")

    per_policy = read_csv(results / "per_policy_summary.csv")
    per_cell = read_csv(results / "per_evaluation_cell_summary.csv")
    if len(per_policy) != 90:
        raise SystemExit(f"Expected 90 per-policy rows, found {len(per_policy)}")
    if len(per_cell) != 450:
        raise SystemExit(f"Expected 450 evaluation-cell rows, found {len(per_cell)}")
    completion = json.loads((results / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    if completion.get("status") != "PASS" or completion.get("evaluation_cells") != 450:
        raise SystemExit("Policy-factorial completion record is invalid")
    if completion.get("task_policy_outcomes") != 1_350_000:
        raise SystemExit("Policy-factorial task-outcome count is invalid")

    return {
        "table_s14_rows": len(s14),
        "table_s15_rows": len(s15),
        "table_s16_rows": len(s16),
        "ppo_mean_above_imitation": ppo_mean_above_imitation,
        "ppo_seed_domain_wins": ppo_seed_domain_wins,
        "maximum_pf_endpoint_difference_points": max_endpoint_difference,
    }


def main() -> int:
    report = {
        "status": "PASS",
        "manifest_files_verified": verify_hash_table("MANIFEST_SHA256.csv"),
        "source_files_verified": verify_hash_table("SOURCE_SHA256.csv"),
        **verify_publication_tables(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
