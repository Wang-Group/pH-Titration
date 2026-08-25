from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate merged direction-assisted RL results.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--algorithms", nargs="+", default=["ppo", "a2c", "reinforce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--eval-tasks", type=int, default=1_000)
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--random-actor-seed-offset", type=int, default=9_000_000)
    args = parser.parse_args()

    root = args.results_dir.resolve()
    errors: list[str] = []
    initializations = ["imitation", "random"]
    expected = {
        (algorithm, initialization, seed)
        for algorithm in args.algorithms
        for initialization in initializations
        for seed in args.seeds
    }
    task_rows = read_rows(root / "task_results.csv")
    curve_rows = read_rows(root / "learning_curves.csv")
    observed = {(row["algorithm"], row["initialization"], int(row["seed"])) for row in task_rows}
    if observed != expected:
        errors.append(f"condition mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")

    rows_by_condition: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    signatures_by_seed: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for row in task_rows:
        condition = (row["algorithm"], row["initialization"], int(row["seed"]))
        rows_by_condition[condition].append(row)
        expected_success = int(float(row["final_abs_error"]) <= 0.1)
        if int(row["true_success"]) != expected_success:
            errors.append(f"success flag mismatch: {condition}, task {row['task_id']}")
        signatures_by_seed[int(row["seed"])].add(
            (row["task_id"], row["task_seed"], row["acid_type"], row["pka_values"], row["initial_ph"], row["target_ph"])
        )
    for condition in sorted(expected):
        if len(rows_by_condition[condition]) != args.eval_tasks:
            errors.append(f"{condition}: expected {args.eval_tasks} task rows, found {len(rows_by_condition[condition])}")
    for seed in args.seeds:
        if len(signatures_by_seed[seed]) != args.eval_tasks:
            errors.append(f"seed {seed}: paired task signature count is {len(signatures_by_seed[seed])}")

    curves_by_condition: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in curve_rows:
        curves_by_condition[(row["algorithm"], row["initialization"], int(row["seed"]))].append(row)
    for condition in sorted(expected):
        steps = sorted(int(row["environment_steps"]) for row in curves_by_condition[condition])
        if not steps or steps[0] != 0:
            errors.append(f"{condition}: learning curve does not start at zero")
        elif steps[-1] < args.train_steps or steps[-1] > args.train_steps + 49:
            errors.append(f"{condition}: final interaction count {steps[-1]} is outside expected range")

    random_hashes: dict[int, set[str]] = defaultdict(set)
    imitation_hashes: set[str] = set()
    model_hashes: dict[str, str] = {}
    elapsed_seconds: dict[str, float] = {}
    for algorithm, initialization, seed in sorted(expected):
        run_name = f"{algorithm}_{initialization}_seed{seed}"
        run_dir = root / "runs" / run_name
        config_path = run_dir / "run_config.json"
        complete_path = run_dir / "COMPLETE.json"
        model_path = run_dir / "final_model.pth"
        required = (config_path, complete_path, model_path, run_dir / "learning_curve.csv", run_dir / "task_results.csv")
        for path in required:
            if not path.exists():
                errors.append(f"missing run artifact: {path}")
        if not config_path.exists() or not complete_path.exists() or not model_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_fields = {
            "protocol_id": "direction_assisted_volume_only_v1",
            "external_direction_rule": "base if measured_ph < target_ph else acid",
            "policy_action": "1000 volume classes from 0.01 to 10.00 mL",
            "titrant_concentration_m": 0.1,
            "overshoot_action_mask": False,
            "automatic_titrant_switching": False,
            "eval_interval": args.eval_interval,
        }
        for key, expected_value in expected_fields.items():
            if config.get(key) != expected_value:
                errors.append(f"{run_name}: {key}={config.get(key)!r}, expected {expected_value!r}")
        if float(config.get("reward_config", {}).get("dense_lambda", 0.0)) <= 0:
            errors.append(f"{run_name}: dense progress reward is not positive")
        if int(config.get("torch_threads", -1)) != 1:
            errors.append(f"{run_name}: torch_threads was not fixed at 1")
        actor_seed = config.get("random_actor_seed")
        if initialization == "random":
            expected_seed = seed + args.random_actor_seed_offset
            if actor_seed != expected_seed:
                errors.append(f"{run_name}: random actor seed {actor_seed!r}, expected {expected_seed}")
            random_hashes[seed].add(str(config.get("initial_actor_sha256", "")))
        else:
            if actor_seed is not None:
                errors.append(f"{run_name}: imitation run should record random_actor_seed=null")
            imitation_hashes.add(str(config.get("initial_actor_sha256", "")))
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        elapsed_seconds[run_name] = float(completion["elapsed_seconds"])
        model_hashes[run_name] = sha256(model_path)
    for seed in args.seeds:
        if len(random_hashes[seed]) != 1:
            errors.append(f"seed {seed}: random actor initialization differs across algorithms")
    if len(imitation_hashes) != 1:
        errors.append("imitation actor initialization hash differs across runs")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "results_dir": str(root),
        "protocol_id": "direction_assisted_volume_only_v1",
        "condition_count": len(observed),
        "task_row_count": len(task_rows),
        "learning_curve_row_count": len(curve_rows),
        "expected_task_rows": len(expected) * args.eval_tasks,
        "model_count": len(model_hashes),
        "model_sha256": model_hashes,
        "elapsed_seconds_by_run": elapsed_seconds,
        "random_initial_actor_sha256_by_seed": {str(seed): sorted(values) for seed, values in random_hashes.items()},
        "imitation_initial_actor_sha256": sorted(imitation_hashes),
        "control_interpretation": "Neural policy selects volume only; acid/base direction is externally supplied.",
    }
    output = root / "RESULT_VALIDATION_STRICT.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "condition_count", "task_row_count", "model_count", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
