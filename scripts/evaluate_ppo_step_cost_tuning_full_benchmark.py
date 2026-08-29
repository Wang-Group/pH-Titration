from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_ppo_reward_ablation_pilot import (
    FORMAL_BLOCK,
    SUMMARY_METRICS,
    exact_mcnemar,
    load_checkpoint,
    read_csv,
    rollout,
    sha256,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "simulation_numerical_evidence_20260823"
TRAINING_BLOCK = EVIDENCE / "02_TEACHER_AND_IMITATION"
TUNING_BLOCK = EVIDENCE / "15_PPO_STEP_COST_TUNING"
DEFAULT_OUTPUT = TUNING_BLOCK / "evaluation_full_5x3000"
BENCHMARK_SEEDS = (101, 202, 303, 404, 555)
CANDIDATES = (0.0, 0.0025, 0.005, 0.01)
LABELS = ("original_full", *(f"step_cost_{value:g}".replace(".", "p") for value in CANDIDATES))


def checkpoint_path(label: str, training_seed: int) -> Path:
    if label == "original_full":
        return TRAINING_BLOCK / "checkpoints" / f"principal_ppo_seed_{training_seed}.pth"
    return TUNING_BLOCK / label / f"seed_{training_seed}" / "best_ppo.pth"


def task_path(output_dir: Path, label: str, benchmark_seed: int) -> Path:
    return output_dir / "tasks" / label / f"benchmark_seed_{benchmark_seed}_task_results.csv"


def evaluate_label(
    output_dir: Path,
    label: str,
    training_seed: int,
    benchmark_seeds: tuple[int, ...],
    tasks_per_seed: int,
    device: torch.device,
    resume: bool,
) -> None:
    checkpoint = checkpoint_path(label, training_seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    actor, normalizer, metadata = load_checkpoint(checkpoint, device)
    summaries = []
    for benchmark_seed in benchmark_seeds:
        result_path = task_path(output_dir, label, benchmark_seed)
        if resume and result_path.exists():
            rows = read_csv(result_path)
            for row in rows:
                for key in (
                    "task_seed", "task_id", "true_pair_count", "true_success",
                    "strict_success", "severe_failure", "measured_success",
                    "false_stop", "steps", "overshoots", "overshoot_cap_events",
                    "overshoot_cap_applied_steps",
                ):
                    row[key] = int(float(row[key]))
            print(f"reused {label} benchmark seed {benchmark_seed}: {len(rows)}", flush=True)
        else:
            from task_distribution import load_tasks

            tasks = load_tasks(FORMAL_BLOCK / "tasks" / f"seed_{benchmark_seed}_tasks.jsonl")[:tasks_per_seed]
            if len(tasks) != tasks_per_seed:
                raise ValueError(f"Expected {tasks_per_seed} tasks for benchmark seed {benchmark_seed}")
            rows = []
            for index, task in enumerate(tasks, 1):
                row = rollout(actor, normalizer, task, device)
                row.update({
                    "label": label,
                    "training_seed": training_seed,
                    "benchmark_seed": benchmark_seed,
                    "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_reward_variant": metadata.get("reward_variant", "full"),
                    "step_cost": metadata.get("step_cost", "0.005" if label == "original_full" else "unknown"),
                })
                rows.append(row)
                if index % 500 == 0 or index == len(tasks):
                    print(f"evaluation {label} benchmark seed {benchmark_seed}: {index}/{len(tasks)}", flush=True)
            write_csv(result_path, rows)
        summary = summarize(rows, label, training_seed)
        summary["label"] = label
        summary["benchmark_seed"] = benchmark_seed
        summaries.append(summary)
        write_csv(output_dir / "tasks" / label / "per_benchmark_seed_summary.csv", summaries)
        write_csv(output_dir / "tasks" / label / "all_task_results.csv", [
            row
            for seed in benchmark_seeds[: len(summaries)]
            for row in read_csv(task_path(output_dir, label, seed))
        ])
    (output_dir / "tasks" / label / "COMPLETE.json").write_text(
        json.dumps({
            "label": label,
            "training_seed": training_seed,
            "benchmark_seeds": list(benchmark_seeds),
            "tasks_per_seed": tasks_per_seed,
            "tasks_total": len(benchmark_seeds) * tasks_per_seed,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_metadata": metadata,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def paired_success(full_rows: list[dict], comparison_rows: list[dict], label: str, benchmark_seed: str) -> dict:
    full_lookup = {(int(row["task_seed"]), int(row["task_id"])): row for row in full_rows}
    comparison_lookup = {(int(row["task_seed"]), int(row["task_id"])): row for row in comparison_rows}
    keys = sorted(set(full_lookup) & set(comparison_lookup))
    full_success = [int(full_lookup[key]["true_success"]) for key in keys]
    comparison_success = [int(comparison_lookup[key]["true_success"]) for key in keys]
    full_only, comparison_only, p_value = exact_mcnemar(full_success, comparison_success)
    return {
        "label": label,
        "training_seed": int(full_rows[0]["training_seed"]),
        "benchmark_seed": benchmark_seed,
        "paired_tasks": len(keys),
        "full_only_success": full_only,
        "candidate_only_success": comparison_only,
        "success_difference_pp": 100.0 * (np.mean(comparison_success) - np.mean(full_success)),
        "exact_mcnemar_p": p_value,
    }


def aggregate(output_dir: Path, training_seed: int, benchmark_seeds: tuple[int, ...], tasks_per_seed: int) -> None:
    summaries_by_label = {}
    rows_by_label_seed = {}
    for label in LABELS:
        summaries = read_csv(output_dir / "tasks" / label / "per_benchmark_seed_summary.csv")
        for row in summaries:
            row["training_seed"] = int(row["training_seed"])
            row["benchmark_seed"] = int(row["benchmark_seed"])
        summaries_by_label[label] = summaries
        for benchmark_seed in benchmark_seeds:
            rows_by_label_seed[(label, benchmark_seed)] = read_csv(task_path(output_dir, label, benchmark_seed))

    pooled = []
    mean_sd = []
    for label in LABELS:
        rows = [row for seed in benchmark_seeds for row in rows_by_label_seed[(label, seed)]]
        summary = summarize(rows, label, training_seed)
        summary["label"] = label
        summary["benchmark_seed"] = "pooled"
        summary["tasks_total"] = len(rows)
        pooled.append(summary)
        row = {"label": label, "training_seed": training_seed, "benchmark_seeds": len(benchmark_seeds), "tasks_total": len(rows)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(item[metric]) for item in summaries_by_label[label]], dtype=float)
            row[f"{metric}_mean_across_benchmark_seeds"] = float(np.mean(values))
            row[f"{metric}_sd_across_benchmark_seeds"] = float(np.std(values, ddof=1))
        mean_sd.append(row)
    write_csv(output_dir / "pooled_summary.csv", pooled)
    write_csv(output_dir / "benchmark_seed_mean_sd_summary.csv", mean_sd)

    full_rows = [row for seed in benchmark_seeds for row in rows_by_label_seed[("original_full", seed)]]
    full_summary = next(row for row in pooled if row["label"] == "original_full")
    paired = []
    differences = []
    for label in LABELS[1:]:
        candidate_rows = [row for seed in benchmark_seeds for row in rows_by_label_seed[(label, seed)]]
        paired.append(paired_success(full_rows, candidate_rows, label, "pooled"))
        candidate_summary = next(row for row in pooled if row["label"] == label)
        difference = {"label": label, "training_seed": training_seed, "benchmark_seed": "pooled"}
        for metric in SUMMARY_METRICS:
            difference[f"{metric}_difference"] = float(candidate_summary[metric]) - float(full_summary[metric])
        differences.append(difference)
        for benchmark_seed in benchmark_seeds:
            paired.append(paired_success(
                rows_by_label_seed[("original_full", benchmark_seed)],
                rows_by_label_seed[(label, benchmark_seed)],
                label,
                str(benchmark_seed),
            ))
    write_csv(output_dir / "paired_success_vs_original_full.csv", paired)
    write_csv(output_dir / "paired_summary_differences.csv", differences)

    mean_sd_lookup = {row["label"]: row for row in mean_sd}
    lines = [
        "# PPO step-cost tuning: complete 5 x 3,000 benchmark",
        "",
        f"The original full-reward PPO and four step-cost candidates were evaluated with training seed {training_seed} on benchmark seeds {', '.join(map(str, benchmark_seeds))}, {tasks_per_seed:,} tasks per seed.",
        "The benchmark was used for held-out reporting, not for selecting the step-cost coefficient.",
        "",
        "| Network | Pooled success (%) | Success mean +/- SD (%) | Successful additions | Overshoots | Total volume (mL) | Final error (pH) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = next(item for item in pooled if item["label"] == label)
        spread = mean_sd_lookup[label]
        if label == "original_full":
            display = "original full"
        else:
            value = label.removeprefix("step_cost_").replace("p", ".")
            display = f"step cost {value}"
        lines.append(
            f"| {display} | {float(row['success_rate_percent']):.2f} | "
            f"{float(spread['success_rate_percent_mean_across_benchmark_seeds']):.2f} +/- {float(spread['success_rate_percent_sd_across_benchmark_seeds']):.2f} | "
            f"{float(row['successful_steps_mean']):.2f} | {float(row['overshoots_mean']):.2f} | "
            f"{float(row['total_volume_mean_ml']):.2f} | {float(row['final_abs_error_mean']):.4f} |"
        )
    lines.extend(["", "Paired success tests versus the original full-reward PPO:", "", "| Network | Difference (percentage points) | Exact McNemar p |", "|---|---:|---:|"])
    for row in paired:
        if row["benchmark_seed"] == "pooled":
            lines.append(f"| {row['label']} | {float(row['success_difference_pp']):.2f} | {float(row['exact_mcnemar_p']):.3g} |")
    lines.extend(["", "This compares one trained model seed across five benchmark seeds; it is not a five-training-seed hyperparameter selection study."])
    (output_dir / "RESULT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "RUN_CONFIG.json").write_text(json.dumps({
        "study_id": "ppo_step_cost_tuning_full_5x3000",
        "training_seed": training_seed,
        "benchmark_seeds": list(benchmark_seeds),
        "tasks_per_seed": tasks_per_seed,
        "networks": list(LABELS),
        "tasks_total_per_network": len(benchmark_seeds) * tasks_per_seed,
        "selection_data": "500-task validation set from tuning study",
        "benchmark_used_for_selection": False,
        "formal_evaluation_stop_operator": "<=",
        "formal_evaluation_tolerance_ph": 0.10,
        "persistent_post_overshoot_cap": True,
    }, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate step-cost tuning candidates on the complete benchmark")
    parser.add_argument("--phase", choices=("evaluate", "aggregate"), default="evaluate")
    parser.add_argument("--label", choices=LABELS, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-seed", type=int, default=303)
    parser.add_argument("--benchmark-seeds", nargs="+", type=int, default=list(BENCHMARK_SEEDS))
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate" and args.label is None:
        raise ValueError("--label is required for evaluation")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_seeds = tuple(args.benchmark_seeds)
    if args.phase == "evaluate":
        evaluate_label(output_dir, args.label, args.training_seed, benchmark_seeds, args.tasks_per_seed, torch.device(args.device), args.resume)
    else:
        aggregate(output_dir, args.training_seed, benchmark_seeds, args.tasks_per_seed)


if __name__ == "__main__":
    main()
