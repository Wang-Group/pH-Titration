from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from direction_assisted_rl_comparison import (
    PROTOCOL_ID,
    build_report,
    plot_results,
    read_csv,
    summarize_all,
    validate_results,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed direction-assisted RL shards.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", default=["ppo", "a2c", "reinforce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--training-pool-size", type=int, default=5_000)
    parser.add_argument("--eval-tasks", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo-batch-steps", type=int, default=2_048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    initializations = ["imitation", "random"]
    expected = {
        (algorithm, initialization, seed)
        for algorithm in args.algorithms
        for initialization in initializations
        for seed in args.seeds
    }
    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"No shard directories found in {args.shards_dir}")

    all_task_rows: list[dict[str, object]] = []
    all_curve_rows: list[dict[str, object]] = []
    source_shards: list[str] = []
    for shard in shard_dirs:
        task_rows = read_csv(shard / "task_results.csv")
        curve_rows = read_csv(shard / "learning_curves.csv")
        if not task_rows:
            raise RuntimeError(f"Shard has no task results: {shard}")
        all_task_rows.extend(task_rows)
        all_curve_rows.extend(curve_rows)
        source_shards.append(str(shard.resolve()))

    observed = {
        (str(row["algorithm"]), str(row["initialization"]), int(row["seed"]))
        for row in all_task_rows
    }
    if observed != expected:
        raise RuntimeError(
            f"Condition mismatch; missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for condition in sorted(expected):
        count = sum(
            (str(row["algorithm"]), str(row["initialization"]), int(row["seed"])) == condition
            for row in all_task_rows
        )
        if count != args.eval_tasks:
            raise RuntimeError(f"Expected {args.eval_tasks} task rows for {condition}, found {count}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ("runs", "task_manifests"):
        target = output / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    write_csv(output / "task_results.csv", all_task_rows)
    write_csv(output / "learning_curves.csv", all_curve_rows)

    for shard in shard_dirs:
        for source in (shard / "runs").glob("*"):
            target = output / "runs" / source.name
            if target.exists():
                raise RuntimeError(f"Duplicate run directory: {target}")
            shutil.copytree(source, target)
        for source in (shard / "task_manifests").glob("*.csv"):
            target = output / "task_manifests" / source.name
            if target.exists() and source.read_bytes() != target.read_bytes():
                raise RuntimeError(f"Task manifest mismatch: {source.name}")
            if not target.exists():
                shutil.copy2(source, target)

    summarize_all(all_task_rows, all_curve_rows, output)
    seed_summary = read_csv(output / "seed_summary.csv")
    plot_results(output, all_curve_rows, seed_summary)
    report_args = SimpleNamespace(
        algorithms=args.algorithms,
        seeds=args.seeds,
        train_steps=args.train_steps,
        training_pool_size=args.training_pool_size,
        eval_tasks=args.eval_tasks,
    )
    build_report(output, report_args)
    validation_args = SimpleNamespace(
        algorithms=args.algorithms,
        seeds=args.seeds,
        train_steps=args.train_steps,
        eval_tasks=args.eval_tasks,
        eval_interval=args.eval_interval,
    )
    validate_results(output, validation_args, all_curve_rows, all_task_rows)
    settings = {
        "protocol_id": PROTOCOL_ID,
        "merged_from": source_shards,
        "algorithms": args.algorithms,
        "initializations": initializations,
        "seeds": args.seeds,
        "train_steps": args.train_steps,
        "training_pool_size": args.training_pool_size,
        "eval_tasks": args.eval_tasks,
        "eval_interval": args.eval_interval,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "ppo_batch_steps": args.ppo_batch_steps,
        "ppo_epochs": args.ppo_epochs,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "condition_count": len(observed),
        "task_row_count": len(all_task_rows),
        "curve_row_count": len(all_curve_rows),
        "policy_action": "1000 volume classes from 0.01 to 10.00 mL",
        "external_direction_rule": "base if measured_ph < target_ph else acid",
        "titrant_concentration_m": 0.1,
        "overshoot_action_mask": False,
        "automatic_titrant_switching": False,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "MERGE_COMPLETE.txt").write_text(
        "All expected algorithm/initialization/seed conditions validated and merged.\n",
        encoding="utf-8",
    )
    print(json.dumps({"conditions": len(observed), "task_rows": len(all_task_rows), "curve_rows": len(all_curve_rows)}, indent=2))


if __name__ == "__main__":
    main()
