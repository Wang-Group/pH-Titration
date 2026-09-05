"""Read-only audit of the five primary PPO checkpoints on the locked benchmark.

Uses only the standard library. Recalculates statistics from task-level outcomes;
does not import the neural evaluator, train models, or modify released results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/simulation_numerical_evidence_20260823"
BLOCK = EVIDENCE / "19_PRIMARY_PPO_FIVE_SEED_REEVALUATION"
FORMAL = EVIDENCE / "01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation"
SEEDS = (101, 202, 303, 404, 555)
METRICS = (
    "success_rate_percent", "strict_success_rate_percent",
    "severe_failure_rate_percent", "false_stop_rate_percent", "steps_mean",
    "successful_steps_mean", "overshoots_mean", "total_volume_mean_ml",
    "final_abs_error_mean", "cap_activation_rate_percent",
)
PROTOCOL = {
    "success_tolerance_ph": 0.10, "max_additions": 50,
    "max_total_dose_ml": 50, "observation_rounding_ph": 0.01,
    "action_range_ml": [0.01, 10.0], "persistent_overshoot_cap": True,
    "evaluation": "argmax, no retraining", "training_seeds": list(SEEDS),
    "benchmark_seeds": list(SEEDS),
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-11,
                         abs_tol=1e-12), f"{label}: {actual} != {expected}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_bytes(path):
    data = Path(path).read_bytes()
    return data if Path(path).suffix == ".pth" else data.replace(b"\r\n", b"\n")


def verify_inputs(block=BLOCK):
    """Check existing weights, tasks and evaluator against the archived inputs."""
    records = read_json(Path(block) / "INPUT_PROVENANCE.json")["files"]
    require(len(records) == 16, "Expected five checkpoints, five manifests and six source files")
    for row in records:
        path = (ROOT / row["path"]).resolve()
        require(path.is_relative_to(ROOT), "Input path escapes repository")
        require(hashlib.sha256(canonical_bytes(path)).hexdigest() == row["sha256"],
                f"Input hash mismatch: {row['path']}")
    return len(records)


def verify_archive(block=BLOCK):
    block = Path(block).resolve()
    entries = read_csv(block / "MANIFEST_SHA256.csv")
    actual_files = {p.relative_to(block).as_posix() for p in block.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                    and p.name != "MANIFEST_SHA256.csv"}
    require(len(entries) == len(actual_files), "Archive file count mismatch")
    require({r["path"] for r in entries} == actual_files, "Archive manifest coverage mismatch")
    for row in entries:
        path = (block / row["path"]).resolve()
        require(path.is_relative_to(block), "Archive path escapes block")
        data = canonical_bytes(path)
        require(len(data) == int(row["bytes"]) and
                hashlib.sha256(data).hexdigest() == row["sha256"],
                f"Archive hash mismatch: {row['path']}")
    return len(entries)


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(int(row[f]) for f in fields)
        require(key not in result, f"Duplicate key {key} in {fields}")
        result[key] = row
    return result


def compare_fields(actual, reference, label, fields=None):
    """Compare specified/shared CSV fields, with strict numeric tolerances."""
    for key in (fields if fields is not None else actual.keys() & reference.keys()):
        a, b = actual[key], reference[key]
        try:
            float(a), float(b)
        except (ValueError, TypeError):
            require(str(a) == str(b), f"{label}/{key}: {a!r} != {b!r}")
        else:
            close(a, b, f"{label}/{key}")


def validate_task_row(row, task, training_seed, benchmark_seed):
    require(row["method"] == "ppo", "Unexpected method")
    require(int(row["training_seed"]) == training_seed and
            int(row["benchmark_seed"]) == benchmark_seed, "Wrong seed labels")
    for field in ("acid_type", "difficulty", "direction", "pka_family"):
        require(row[field] == task[field], f"Task metadata mismatch: {field}")
    for field, expected in {
        "task_seed": task["seed"], "task_id": task["task_id"],
        "true_pair_count": len(task["pka_values"]), "initial_ph": task["initial_ph"],
        "target_ph": task["target_ph"], "true_concentration_m": task["analyte_conc_m"],
    }.items():
        close(row[field], expected, f"Task {field}")
    error = abs(float(row["final_true_ph"]) - task["target_ph"])
    measured_error = abs(float(row["final_measured_ph"]) - task["target_ph"])
    close(row["final_abs_error"], error, "Final absolute error")
    expected_flags = {
        "true_success": error <= .10, "strict_success": error <= .05,
        "severe_failure": error > .50, "measured_success": measured_error <= .10,
        "false_stop": measured_error <= .10 and error > .10,
    }
    for key, value in expected_flags.items():
        require(int(row[key]) == int(value), f"Incorrect {key} for task {task['task_id']}")
    steps, overshoots = int(row["steps"]), int(row["overshoots"])
    require(0 <= overshoots <= steps <= 50, "Invalid addition/crossing counts")
    for field in ("overshoot_cap_events", "overshoot_cap_applied_steps"):
        require(0 <= int(row[field]) <= steps, f"Invalid {field}")
    acid, base = float(row["acid_added_ml"]), float(row["base_added_ml"])
    require(acid >= 0 and base >= 0 and acid + base <= 50 + 1e-8,
            "Invalid total delivered volume")
    close(row["total_volume_ml"], acid + base, "Dose sum")
    observed = float(row["final_measured_ph"])
    close(observed * 100, round(observed * 100), "Observation resolution")
    require(0 <= observed <= 14 and math.isfinite(error), "Nonfinite/invalid pH")
    reason = ("initial_success" if steps == 0 and measured_error <= .1 else
              "measured_success" if measured_error <= .1 else
              "max_steps" if steps == 50 else
              "dose_limit" if acid + base >= 50 - 1e-9 else "running")
    require(reason != "running" and row["stop_reason"] == reason, "Inconsistent stop reason")


def summarize(rows):
    successful = [r for r in rows if int(r["true_success"])]
    result = {"tasks": len(rows)}
    for metric, field in (
        ("success_rate_percent", "true_success"),
        ("strict_success_rate_percent", "strict_success"),
        ("severe_failure_rate_percent", "severe_failure"),
        ("false_stop_rate_percent", "false_stop"),
    ):
        result[metric] = 100 * st.mean(int(r[field]) for r in rows)
    for metric, field in (
        ("steps_mean", "steps"), ("overshoots_mean", "overshoots"),
        ("total_volume_mean_ml", "total_volume_ml"),
        ("final_abs_error_mean", "final_abs_error"),
    ):
        result[metric] = st.mean(float(r[field]) for r in rows)
    require(bool(successful), "No successful tasks in this cell")
    result["successful_steps_mean"] = st.mean(int(r["steps"]) for r in successful)
    result["cap_activation_rate_percent"] = 100 * st.mean(
        int(r["overshoot_cap_events"]) > 0 for r in rows)
    return result


def audit(results=None, block=BLOCK):
    block = Path(block).resolve()
    released = (block / "results").resolve()
    results = released if results is None else Path(results).resolve()
    archive_files = verify_archive(block)
    input_files = verify_inputs(block)
    config = read_json(results / "RUN_CONFIG.json")
    for field, expected in PROTOCOL.items():
        require(config[field] == expected, f"Unexpected protocol setting: {field}")
    count = config["tasks_per_cell"]
    require(type(count) is int and 1 <= count <= 3000, "Invalid task count")
    if results == released:
        require(count == 3000, "Released evaluation is incomplete")
    # A new run must identify the same exact tasks/checkpoints, not just the same seeds.
    convention = config.get("input_hash_convention", "original_run_bytes")
    require(convention in ("lf_normalized_text", "original_run_bytes"), "Unknown input hash convention")
    provenance = read_json(block / "INPUT_PROVENANCE.json")["files"]
    for field in ("task_manifests", "checkpoints", "source_hashes"):
        expected_inputs = {}
        for record in provenance:
            if record["kind"] == field:
                key = record["path"] if field == "source_hashes" else Path(record["path"]).name
                expected_inputs[key] = record["sha256" if convention == "lf_normalized_text"
                                              else "original_run_sha256"]
        actual_inputs = {key.replace("\\", "/"): value for key, value in config[field].items()}
        require(actual_inputs == expected_inputs, f"Changed input identity: {field}")
    cells = indexed(read_csv(results / "per_cell_summary.csv"),
                    ("training_seed", "benchmark_seed"))
    require(set(cells) == {(a, b) for a in SEEDS for b in SEEDS}, "Expected all 25 cells")
    five = indexed(read_csv(results / "per_training_seed_summary.csv"), ("training_seed",))
    require(set(five) == {(s,) for s in SEEDS}, "Expected five training-seed summaries")
    paired = indexed(read_csv(results / "paired_set_success_vs_imitation.csv"),
                     ("training_seed", "benchmark_seed"))
    require(set(paired) == set(cells), "Incomplete imitation comparisons")
    reference_rows = read_csv(FORMAL / "all_task_results.csv")
    references = {method: indexed([r for r in reference_rows if r["method"] == method],
                                  ("benchmark_seed", "task_seed", "task_id"))
                  for method in ("ppo", "imitation")}
    tasks = {}
    for seed in SEEDS:
        records = [json.loads(line) for line in
                   (FORMAL / "tasks" / f"seed_{seed}_tasks.jsonl").read_text().splitlines()
                   if line.strip()]
        require(len(records) == 3000, "Incomplete benchmark manifest")
        tasks[seed] = indexed(records[:count], ("seed", "task_id"))
    calculated = {}
    higher = 0
    for (a, b), archived_summary in cells.items():
        path = results / f"ppo_{a}_benchmark_{b}.csv"
        rows = read_csv(path)
        keyed = indexed(rows, ("task_seed", "task_id"))
        require(set(keyed) == set(tasks[b]), f"Wrong task identities: PPO {a}, benchmark {b}")
        released_rows = indexed(read_csv(released / path.name), ("task_seed", "task_id"))
        imitation_success = []
        for key, row in keyed.items():
            task = tasks[b][key]
            validate_task_row(row, task, a, b)
            if results != released:
                compare_fields(row, released_rows[key], "Re-evaluation vs released row",
                               fields=released_rows[key].keys())
            ref_key = (b, *key)
            if a == 303:
                compare_fields(row, references["ppo"][ref_key], "Selected model vs original")
            im = references["imitation"][ref_key]
            im_error = abs(float(im["final_true_ph"]) - task["target_ph"])
            require(int(im["true_success"]) == int(im_error <= .1), "Imitation success mismatch")
            imitation_success.append(int(im_error <= .1))
        summary = summarize(rows)
        calculated[(a, b)] = summary
        detail = read_json(path.with_suffix(".json"))
        for key, value in summary.items():
            close(archived_summary[key], value, f"Per-cell {a}/{b}/{key}")
            close(detail[key], value, f"Cell JSON {a}/{b}/{key}")
        for record in (archived_summary, detail):
            require(record["method"] == "ppo" and int(record["training_seed"]) == a
                    and int(record["benchmark_seed"]) == b, "Cell summary labels differ")
        im_rate = 100 * st.mean(imitation_success)
        difference = summary["success_rate_percent"] - im_rate
        higher += difference > 0
        for key, value in {"ppo_success_percent": summary["success_rate_percent"],
                           "imitation_success_percent": im_rate, "difference_pp": difference}.items():
            close(paired[(a, b)][key], value, f"Paired comparison {a}/{b}/{key}")
        if a == 303:
            require(detail["selected_model_reference_check"]["mismatched_fields"] == 0
                    and detail["selected_model_reference_check"]["tasks_compared"] == count,
                    "Incorrect selected-model reference check")
    means = []
    for seed in SEEDS:
        record = five[(seed,)]
        require(int(record["evaluations"]) == 5 * count and int(record["benchmark_sets"]) == 5,
                "Training-seed summary has wrong statistical units")
        for metric in METRICS:
            values = [calculated[(seed, b)][metric] for b in SEEDS]
            close(record[metric + "_mean"], st.mean(values), f"Seed {seed}/{metric}/mean")
            close(record[metric + "_sd"], st.stdev(values), f"Seed {seed}/{metric}/sample SD")
        means.append(st.mean(calculated[(seed, b)]["success_rate_percent"] for b in SEEDS))
    completion = read_json(results / "COMPLETE.json")
    expected_completion = {
        "evaluations": 25 * count, "cells": 25,
        "ppo_higher_than_imitation_cells": higher,
        "training_seed_success_mean_percent": st.mean(means),
        "training_seed_success_sd_percent": st.stdev(means),
    }
    require(completion["status"] == "COMPLETE", "Run not complete")
    for key, value in expected_completion.items():
        close(completion[key], value, f"Completion {key}")
    for actual, expected in zip(completion["training_seed_success_range_percent"],
                                (min(means), max(means)), strict=True):
        close(actual, expected, "Training-seed range")
    checks = completion["selected_model_reference_checks"]
    require(len(checks) == 5 and all(r["tasks_compared"] == count and
            r["mismatched_fields"] == 0 for r in checks), "Incorrect completion reference checks")
    if results == released:
        historical = read_json(released / "INDEPENDENT_AUDIT.json")
        close(historical["overall_mean"], st.mean(means), "Historical audit mean")
        close(historical["overall_sample_sd"], st.stdev(means), "Historical audit SD")
        require(historical["raw_rows_rechecked"] == 75000 and
                historical["selected_303_reference_mismatches"] == 0, "Historical audit mismatch")
    return {"status": "PASS", "archive_files_verified": archive_files,
            "input_files_verified": input_files, **expected_completion,
            "selected_model_rows_compared": 5 * count, "selected_model_mismatched_fields": 0,
            "new_results_match_released_rows": True if results != released else None,
            "scope": "full benchmark" if count == 3000 else "smoke test only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Audit a new run against the released rows")
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2))


if __name__ == "__main__":
    main()
