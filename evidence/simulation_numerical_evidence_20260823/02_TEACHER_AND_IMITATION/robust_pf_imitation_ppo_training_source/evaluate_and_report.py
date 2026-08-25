from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest

from models import ActorCritic, StateNormalizer, load_actor_checkpoint
from particle_controllers import build_controller
from policy_evaluation import evaluate_actor, summarize_rows
from task_distribution import generate_tasks, load_tasks, save_tasks


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar(a, b):
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    b_only = sum(not bool(x) and bool(y) for x, y in zip(a, b))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    return a_only, b_only, p


def teacher_episode(task, particles, seed):
    seed = int(seed) % (2**32 - 1)
    np.random.seed(seed)
    controller = build_controller("pf_pka_conc_variable_k", particles, (seed + 17) % (2**32 - 1))
    controller.initialize_task(task)
    controller.base_added_moles = task.initial_base_moles
    controller.acid_added_moles = 0.0
    controller.base_volume = 0.0
    controller.acid_volume = 0.0
    controller.total_volume = task.initial_volume_ml
    controller.previous_total_volume = task.initial_volume_ml
    controller.current_ph = task.initial_ph
    controller.previous_ph = task.initial_ph
    controller.last_measured_ph = task.initial_ph
    controller.prev_measured_ph = task.initial_ph
    controller.last_action_volume = 0.0
    controller.done = False
    overshoots = 0
    while not controller.done:
        controller.use_secondary_reagents = False
        action, _ = controller.select_best_action()
        direction = "base" if controller.current_ph < controller.target_ph else "acid"
        reagent = "Dilute base 1" if direction == "base" else "Dilute acid 1"
        current_ph, _, done, info = controller.step((reagent, float(action[1])), mode="Simulate")
        overshoots += int(bool(info.get("crossed_target", False)))
        controller.update_posteriors((reagent, float(action[1])), current_ph)
        if done:
            break
    error = abs(controller.current_ph - controller.target_ph)
    return {
        "task_seed": task.seed,
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "true_concentration_m": task.analyte_conc_m,
        "initial_volume_ml": task.initial_volume_ml,
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "true_success": int(error <= 0.10),
        "strict_success": int(error <= 0.05),
        "severe_failure": int(error > 0.50),
        "measured_success": int(error <= 0.10),
        "false_stop": 0,
        "steps": controller.steps_taken,
        "overshoots": overshoots,
        "final_abs_error": error,
        "total_volume_ml": controller.acid_volume + controller.base_volume,
        "acid_added_ml": controller.acid_volume,
        "base_added_ml": controller.base_volume,
        "final_true_ph": controller.current_ph,
        "final_measured_ph": controller.current_ph,
        "stop_reason": "success" if error <= 0.10 else "max_steps",
    }


def load_ppo_actor(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer = StateNormalizer(
        np.asarray(payload["state_mean"], dtype=np.float32),
        np.asarray(payload["state_std"], dtype=np.float32),
    )
    return model.actor, normalizer, payload


def aggregate_seed_rows(seed_rows):
    metrics = [
        "success_rate_percent",
        "strict_success_rate_percent",
        "severe_failure_rate_percent",
        "successful_steps_mean",
        "overshoots_mean",
        "total_volume_mean_ml",
        "final_abs_error_mean",
    ]
    output = []
    for suite in sorted({row["suite"] for row in seed_rows}):
        for method in sorted({row["method"] for row in seed_rows if row["suite"] == suite}):
            subset = [row for row in seed_rows if row["suite"] == suite and row["method"] == method]
            result = {"suite": suite, "method": method, "runs": len(subset)}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                finite = values[np.isfinite(values)]
                result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                result[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
            output.append(result)
    return output


def paired_tests(task_rows, ppo_seeds):
    output = []
    for suite in sorted({row["suite"] for row in task_rows}):
        teacher = {
            int(row["task_id"]): int(row["true_success"])
            for row in task_rows if row["suite"] == suite and row["method"] == "teacher"
        }
        imitation = {
            int(row["task_id"]): int(row["true_success"])
            for row in task_rows if row["suite"] == suite and row["method"] == "imitation"
        }
        keys = sorted(set(teacher) & set(imitation))
        teacher_values = [teacher[key] for key in keys]
        imitation_values = [imitation[key] for key in keys]
        reference_only, comparison_only, p_value = exact_mcnemar(teacher_values, imitation_values)
        output.append(
            {
                "suite": suite,
                "comparison": "imitation_minus_teacher",
                "training_seed": 0,
                "paired_tasks": len(keys),
                "reference_only_success": reference_only,
                "comparison_only_success": comparison_only,
                "success_difference_pp": 100.0 * (
                    float(np.mean(imitation_values)) - float(np.mean(teacher_values))
                ),
                "exact_mcnemar_p": p_value,
            }
        )
        seed_differences = {"ppo_minus_imitation": [], "ppo_minus_teacher": []}
        for seed in ppo_seeds:
            ppo = {
                int(row["task_id"]): int(row["true_success"])
                for row in task_rows
                if row["suite"] == suite and row["method"] == "ppo" and int(row["training_seed"]) == seed
            }
            for comparison, reference in (
                ("ppo_minus_imitation", imitation),
                ("ppo_minus_teacher", teacher),
            ):
                keys = sorted(set(reference) & set(ppo))
                reference_values = [reference[key] for key in keys]
                ppo_values = [ppo[key] for key in keys]
                reference_only, comparison_only, p_value = exact_mcnemar(reference_values, ppo_values)
                difference = 100.0 * (float(np.mean(ppo_values)) - float(np.mean(reference_values)))
                seed_differences[comparison].append(difference)
                output.append(
                    {
                        "suite": suite,
                        "comparison": comparison,
                        "training_seed": seed,
                        "paired_tasks": len(keys),
                        "reference_only_success": reference_only,
                        "comparison_only_success": comparison_only,
                        "success_difference_pp": difference,
                        "exact_mcnemar_p": p_value,
                    }
                )
        for comparison, differences in seed_differences.items():
            values = np.asarray(differences, dtype=float)
            observed = abs(float(np.mean(values)))
            permutations = [
                abs(float(np.mean(values * np.asarray(signs))))
                for signs in itertools.product([-1.0, 1.0], repeat=len(values))
            ]
            output.append(
                {
                    "suite": suite,
                    "comparison": f"seed_level_{comparison}",
                    "training_seed": "all",
                    "paired_tasks": len(values),
                    "success_difference_pp": float(np.mean(values)),
                    "success_difference_seed_sd": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    ),
                    "exact_sign_flip_p": float(np.mean(np.asarray(permutations) >= observed - 1e-12)),
                }
            )
    test_indices = [index for index, row in enumerate(output) if row.get("exact_mcnemar_p") is not None]
    ordered = sorted(test_indices, key=lambda index: float(output[index]["exact_mcnemar_p"]))
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(output[index]["exact_mcnemar_p"]))
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = running
    return output


def plot_results(aggregate, ppo_root, output_dir):
    suites = sorted({row["suite"] for row in aggregate})
    methods = ["teacher", "imitation", "ppo"]
    labels = ["PF teacher", "Imitation", "PPO"]
    colors = ["#3D7A5C", "#557FA3", "#C98232"]
    fig, axes = plt.subplots(1, len(suites), figsize=(4.0 * len(suites), 4.2), squeeze=False)
    for ax, suite in zip(axes[0], suites):
        subset = {row["method"]: row for row in aggregate if row["suite"] == suite}
        means = [subset[method]["success_rate_percent_mean"] for method in methods]
        sds = [subset[method]["success_rate_percent_sd"] or 0.0 for method in methods]
        ax.bar(range(3), means, yerr=sds, color=colors, capsize=4)
        ax.set_xticks(range(3), labels, rotation=15)
        ax.set_ylim(0, 100)
        ax.set_title(suite)
        ax.set_ylabel("Success (%)")
        ax.grid(axis="y", alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_dir / "locked_test_success.png", dpi=260)
    fig.savefig(output_dir / "locked_test_success.svg")
    plt.close(fig)

    curves = []
    for path in sorted(ppo_root.glob("seed_*/learning_curve.csv")):
        seed = int(path.parent.name.split("_")[-1])
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                curves.append({"seed": seed, **row})
    if curves:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        checkpoints = sorted({int(float(row["environment_steps"])) for row in curves})
        means = []
        sds = []
        for checkpoint in checkpoints:
            values = [
                float(row["success_rate_percent"])
                for row in curves if int(float(row["environment_steps"])) == checkpoint
            ]
            means.append(float(np.mean(values)))
            sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        means = np.asarray(means)
        sds = np.asarray(sds)
        ax.plot(checkpoints, means, marker="o", color="#C98232")
        ax.fill_between(checkpoints, means - sds, means + sds, color="#C98232", alpha=0.18)
        ax.set_xlabel("Environment interactions")
        ax.set_ylabel("Validation success (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2, linestyle=":")
        fig.tight_layout()
        fig.savefig(output_dir / "ppo_learning_curve.png", dpi=260)
        fig.savefig(output_dir / "ppo_learning_curve.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Locked-test evaluation and report")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--imitation-checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppo-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--stress-tasks", type=int, default=300)
    parser.add_argument("--teacher-particles", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    suites = {
        "nominal_locked": load_tasks(args.data_dir / "test_tasks.jsonl"),
        "close_pka": generate_tasks(910_001, args.stress_tasks, "stress_close_pka", "close_pka"),
        "wide_concentration": generate_tasks(920_001, args.stress_tasks, "stress_wide_concentration", "wide_concentration"),
    }
    for name, tasks in suites.items():
        if name != "nominal_locked":
            save_tasks(args.output_dir / f"{name}_tasks.jsonl", tasks)

    imitation_actor, imitation_normalizer, _ = load_actor_checkpoint(args.imitation_checkpoint, device)
    ppo_models = {
        seed: load_ppo_actor(args.ppo_dir / f"seed_{seed}" / "best_ppo.pth", device)
        for seed in args.ppo_seeds
    }
    task_rows = []
    seed_rows = []
    for suite_name, tasks in suites.items():
        teacher_rows = []
        for index, task in enumerate(tasks, 1):
            teacher_rows.append(
                teacher_episode(task, args.teacher_particles, task.seed * 1_000_003 + task.task_id)
            )
            if index % 100 == 0:
                print(f"teacher {suite_name}: {index}/{len(tasks)}", flush=True)
        for row in teacher_rows:
            row.update({"suite": suite_name, "method": "teacher", "training_seed": 0})
        task_rows.extend(teacher_rows)
        seed_rows.append({"suite": suite_name, "method": "teacher", "training_seed": 0, **summarize_rows(teacher_rows)})

        imitation_rows = evaluate_actor(imitation_actor, imitation_normalizer, tasks, device, seed_offset=1_100_000)
        for row in imitation_rows:
            row.update({"suite": suite_name, "method": "imitation", "training_seed": 0})
        task_rows.extend(imitation_rows)
        seed_rows.append({"suite": suite_name, "method": "imitation", "training_seed": 0, **summarize_rows(imitation_rows)})

        for seed, (actor, normalizer, _) in ppo_models.items():
            rows = evaluate_actor(actor, normalizer, tasks, device, seed_offset=seed * 20_000_033)
            for row in rows:
                row.update({"suite": suite_name, "method": "ppo", "training_seed": seed})
            task_rows.extend(rows)
            seed_rows.append({"suite": suite_name, "method": "ppo", "training_seed": seed, **summarize_rows(rows)})

    aggregate = aggregate_seed_rows(seed_rows)
    tests = paired_tests(task_rows, args.ppo_seeds)
    write_csv(args.output_dir / "all_task_results.csv", task_rows)
    write_csv(args.output_dir / "per_run_summary.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    write_csv(args.output_dir / "paired_method_tests.csv", tests)
    plot_results(aggregate, args.ppo_dir, args.output_dir)

    checkpoint_word = "checkpoint" if len(args.ppo_seeds) == 1 else "checkpoints"
    report = [
        "# Robust PF teacher -> imitation -> PPO",
        "",
        f"The PF teacher, selected imitation checkpoint, and {len(args.ppo_seeds)} independently trained PPO {checkpoint_word} were evaluated on locked tasks.",
        "",
        "| Suite | Method | Runs | Success (%) | Strict (%) | Severe failure (%) | Successful steps | Final error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def formatted(mean, sd, digits):
        if not math.isfinite(mean):
            return "n/a"
        if sd is None:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} +/- {sd:.{digits}f}"

    for row in aggregate:
        report.append(
            f"| {row['suite']} | {row['method']} | {row['runs']} | "
            f"{formatted(row['success_rate_percent_mean'], row['success_rate_percent_sd'], 2)} | "
            f"{formatted(row['strict_success_rate_percent_mean'], row['strict_success_rate_percent_sd'], 2)} | "
            f"{formatted(row['severe_failure_rate_percent_mean'], row['severe_failure_rate_percent_sd'], 2)} | "
            f"{formatted(row['successful_steps_mean_mean'], row['successful_steps_mean_sd'], 2)} | "
            f"{formatted(row['final_abs_error_mean_mean'], row['final_abs_error_mean_sd'], 4)} |"
        )
    report.extend(
        [
            "",
            "PPO checkpoints were selected only from independent validation tasks. The locked test tasks were evaluated after checkpoint selection.",
            "Task-level results and paired tests are provided in the accompanying CSV files.",
        ]
    )
    (args.output_dir / "RESULT_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "EVALUATION_COMPLETE.json").write_text(
        json.dumps({"suites": {key: len(value) for key, value in suites.items()}, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print("Evaluation complete", flush=True)


if __name__ == "__main__":
    main()
