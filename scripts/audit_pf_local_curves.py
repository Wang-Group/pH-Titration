"""Read-only arithmetic and provenance audit of the local-response RMSE study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "evidence/simulation_numerical_evidence_20260823/06_POSTERIOR_RECOVERY/local_response_reproduction_20260906"
RESULTS = BLOCK / "original_package/formal_results/new_pf_local_response"
SEEDS = (101, 202, 303, 404, 555)
CHECKPOINTS = (0, 1, 2, 3, 5, 8, 12)
WINDOWS = ("0p1", "0p5", "1p0")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=1e-11),
            f"{label}: {actual} != {expected}")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_bytes(path):
    data = Path(path).read_bytes()
    return data if Path(path).suffix.lower() in (".png", ".zip") else data.replace(b"\r\n", b"\n")


def verify_files(block=BLOCK):
    block = Path(block).resolve()
    manifest = read_csv(block / "MANIFEST_SHA256.csv")
    actual = {p.relative_to(block).as_posix() for p in block.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST_SHA256.csv"}
    require(len(manifest) == len(actual) and {r["path"] for r in manifest} == actual,
            "Local-response archive manifest coverage mismatch")
    for entry in manifest:
        path = (block / entry["path"]).resolve()
        require(path.is_relative_to(block), "Manifest path escapes archive")
        data = canonical_bytes(path)
        require(len(data) == int(entry["bytes"]) and
                hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"Local-response file hash mismatch: {entry['path']}")
    provenance = read_json(block / "PROVENANCE.json")
    for entry in provenance["supplied_files"]:
        path = block / "original_package" / entry["path"]
        require(hashlib.sha256(canonical_bytes(path)).hexdigest() == entry["normalized_sha256"],
                f"Supplied file modified: {entry['path']}")
    for entry in provenance["dependency_files"]:
        data = canonical_bytes(block / entry["path"])
        require(hashlib.sha256(data).hexdigest() == entry["normalized_sha256"], "Dependency modified")
    return len(manifest)


def row_key(row):
    return (int(row["benchmark_seed"]), int(row["task_id"]),
            row["checkpoint_type"], int(row["observations"]))


def indexed(rows):
    result = {row_key(row): row for row in rows}
    require(len(result) == len(rows), "Duplicate task/checkpoint identity")
    return result


def summary_group_key(row):
    return (row["checkpoint_type"], "natural" if row["checkpoint_type"] == "natural_control_end"
            else str(row["observations"]))


def seed_summary(rows):
    result = {"tasks": len(rows)}
    for window in WINDOWS:
        metric = f"local_rmse_{window}_ml"
        values = [float(r[metric]) for r in rows]
        result[metric + "_mean"] = st.mean(values)
        result[metric + "_median"] = st.median(values)
        for cutoff, tag in ((.05, "0p05"), (.1, "0p10")):
            result[metric + f"_le_{tag}_percent"] = 100 * st.mean(v <= cutoff for v in values)
    return result


def check_summaries(rows, results):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["benchmark_seed"]), *summary_group_key(row))].append(row)
    expected = {key: seed_summary(group) for key, group in groups.items()}
    summaries = read_csv(Path(results) / "per_seed_summary.csv")
    keys = [(int(r["benchmark_seed"]), *summary_group_key(r)) for r in summaries]
    require(len(set(keys)) == len(keys) and set(keys) == set(expected), "Per-seed summary groups differ")
    for row, key in zip(summaries, keys, strict=True):
        for metric, value in expected[key].items():
            close(row[metric], value, f"Per-seed {key}/{metric}")
    aggregate_groups = defaultdict(list)
    for (seed, kind, obs), summary in expected.items():
        aggregate_groups[(kind, obs)].append(summary)
    aggregates = read_csv(Path(results) / "aggregate_summary.csv")
    keys = [summary_group_key(r) for r in aggregates]
    require(len(set(keys)) == len(keys) and set(keys) == set(aggregate_groups), "Aggregate groups differ")
    for row, key in zip(aggregates, keys, strict=True):
        summaries = aggregate_groups[key]
        require(int(row["seed_runs"]) == len(summaries), "Wrong aggregate replication unit")
        for metric in summaries[0]:
            values = [s[metric] for s in summaries]
            close(row[metric + "_mean"], st.mean(values), f"Aggregate {key}/{metric}/mean")
            close(row[metric + "_sd"], st.stdev(values), f"Aggregate {key}/{metric}/sample SD")
    return expected


def audit():
    files = verify_files()
    rows = read_csv(RESULTS / "all_local_response_rows.csv")
    keyed = indexed(rows)
    require(len(rows) == 12000, "Expected 12,000 snapshots, not 12,000 independent tasks")
    tasks = {}
    for seed in SEEDS:
        records = [json.loads(line) for line in (RESULTS / f"seed_{seed}_tasks.jsonl").read_text().splitlines()
                   if line.strip()]
        require(len(records) == 300, "Expected 300 tasks per seed")
        require(Counter(t["direction"] for t in records) == {"acid": 150, "base": 150}, "Direction imbalance")
        for task in records:
            key = (seed, task["task_id"])
            require(key not in tasks, "Duplicate locked task")
            tasks[key] = task
    task_groups = defaultdict(list)
    for row in rows:
        seed, task_id, kind, observations = row_key(row)
        task = tasks[(seed, task_id)]
        require(int(row["task_seed"]) == task["seed"] == 4000000 + seed, "Incorrect task seed")
        for field in ("direction", "difficulty", "pka_family"):
            require(row[field] == task[field], f"Task metadata differs: {field}")
        close(row["target_ph"], task["target_ph"], "Target pH")
        require(json.loads(row["true_pka_json"]) == task["pka_values"], "True pKa differs")
        require(int(row["true_pair_count"]) == len(task["pka_values"]), "True model order differs")
        pka = json.loads(row["estimated_pka_json"])
        require(len(pka) == int(row["estimated_pair_count"]) in (1, 2, 3), "Invalid posterior model order")
        require(pka == sorted(pka) and all(1 <= x <= 10 for x in pka), "Invalid posterior pKa")
        require(.02 <= float(row["estimated_concentration_m"]) <= .30, "Invalid inferred concentration")
        require(0 <= observations <= 12, "Invalid observation counter")
        ph = float(row["current_true_ph"])
        close(ph * 100, round(ph * 100), "Legacy current_true_ph is rounded to 0.01")
        require(0 <= ph <= 14, "Invalid pH")
        for window in WINDOWS:
            rmse, mae, maximum = (float(row[f"local_{m}_{window}_ml"]) for m in ("rmse", "mae", "max_abs"))
            require(all(math.isfinite(x) for x in (rmse, mae, maximum)) and
                    0 <= mae <= rmse + 1e-12 and rmse <= maximum + 1e-12, "Invalid curve-error metrics")
        task_groups[(seed, task_id)].append(row)
    fallback = []
    for key, group in task_groups.items():
        fixed = {int(r["observations"]): r for r in group if r["checkpoint_type"] == "fixed_observation_count"}
        natural = [r for r in group if r["checkpoint_type"] == "natural_control_end"]
        require(set(fixed) == set(CHECKPOINTS) and len(group) == 8 and len(natural) == 1,
                "Expected seven fixed checkpoints and one terminal diagnostic per task")
        terminal = natural[0]
        observation = int(terminal["observations"])
        for i, record in fixed.items():
            if 0 < i < observation:
                require(abs(float(record["current_true_ph"]) - float(record["target_ph"])) > .1,
                        "Terminal record skips an earlier recorded threshold crossing")
        if observation in fixed:
            compare_fields(terminal, fixed[observation], exclude={"checkpoint_type"})
        if abs(float(terminal["current_true_ph"]) - float(terminal["target_ph"])) > .1:
            require(observation == 12, "Fallback is not the last diagnostic iteration")
            fallback.append(terminal)
    require(set(task_groups) == set(tasks), "Missing task snapshots")
    expected = check_summaries(rows, RESULTS)
    copy = BLOCK / "original_package/FINAL_DELIVERY/tables/new_pf_local_response__aggregate_summary.csv"
    require(canonical_bytes(copy) == canonical_bytes(RESULTS / "aggregate_summary.csv"), "Delivery table differs")
    config = read_json(RESULTS / "LOCAL_RESPONSE_COMPLETE.json")
    require(config["seeds"] == list(SEEDS) and config["tasks_per_seed"] == 300 and config["particles"] == 1000
            and config["windows_ml"] == [.1, .5, 1.], "Unexpected run metadata")
    means, sds = [], []
    for window in WINDOWS:
        values = [expected[(s, "natural_control_end", "natural")][f"local_rmse_{window}_ml_mean"] for s in SEEDS]
        means.append(st.mean(values))
        sds.append(st.stdev(values))
    require([f"{v:.4f}" for v in means] == ["0.0399", "0.1280", "0.2452"], "Reported RMSE values differ")
    require(len(fallback) == 96, "Terminal fallback count changed")
    report = {"status": "PASS", "manifest_files_verified": files, "unique_tasks": 1500,
            "snapshots": len(keyed), "per_seed_summary_rows": len(expected), "aggregate_summary_rows": 8,
            "terminal_rmse_means_ph": means, "terminal_rmse_sample_sds_ph": sds,
            "terminal_threshold_met": 1500 - len(fallback), "terminal_horizon_fallbacks": len(fallback),
            "fallbacks_by_seed": dict(Counter(int(r["benchmark_seed"]) for r in fallback)),
            "interpretation": "Anchored local pH-change curves; first observed endpoint within 12 iterations or final available state"}
    report["independent_replay"] = compare_replay(BLOCK / "verified_replay")
    require(report["independent_replay"]["replayed_tasks"] == 1500, "Archived replay is incomplete")
    return report


def compare_fields(actual, reference, exclude=frozenset()):
    for field, value in reference.items():
        if field in exclude:
            continue
        require(field in actual, f"Missing replay field: {field}")
        if field.endswith("_json"):
            a, b = json.loads(actual[field]), json.loads(value)
            require(len(a) == len(b), "Replay pKa dimension differs")
            for x, y in zip(a, b, strict=True):
                close(x, y, "Replay pKa")
        else:
            try:
                float(actual[field]), float(value)
            except (ValueError, TypeError):
                require(str(actual[field]) == str(value), f"Replay text differs: {field}")
            else:
                close(actual[field], value, f"Replay {field}")


def compare_replay(directory):
    directory = Path(directory)
    config = read_json(directory / "REPLAY_CONFIG.json")
    count = int(config["tasks_per_seed"])
    require(1 <= count <= 300 and config["particles"] == 1000 and config["seeds"] == list(SEEDS), "Wrong replay scope")
    require(config["status"] == "COMPLETE" and config["tasks"] == count * 5 and
            config["snapshots"] == count * 40 and config["windows_ml"] == [.1, .5, 1.]
            and config["grid_points_per_window"] == 81, "Invalid replay completion metadata")
    reference = indexed(read_csv(RESULTS / "all_local_response_rows.csv"))
    rows = read_csv(directory / "all_local_response_rows.csv")
    expected_keys = {key for key in reference if key[1] <= count}
    actual = indexed(rows)
    require(set(actual) == expected_keys and len(rows) == count * 5 * 8, "Replay task/checkpoint coverage differs")
    repeated = 0
    for key, row in actual.items():
        compare_fields(row, reference[key])
        require(float(row["state_total_volume_ml"]) > 0 and float(row["state_base_moles"]) >= 0
                and float(row["state_acid_moles"]) >= 0, "Invalid replay solution state")
        steps = int(row["controller_steps"])
        require(0 <= steps <= int(row["observations"]), "Invalid delivered-addition count")
        repeated += steps != int(row["observations"])
    check_summaries(rows, directory)
    return {"status": "PASS", "replayed_tasks": count * 5, "matched_snapshots": len(rows),
            "mismatched_fields": 0, "state_columns_recorded": True,
            "snapshots_with_observation_counter_different_from_delivered_steps": repeated,
            "scope": "full study" if count == 300 else "smoke test only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, help="Also compare a fresh replay with all matching archived records")
    args = parser.parse_args()
    report = audit()
    if args.replay:
        report["replay"] = compare_replay(args.replay)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
