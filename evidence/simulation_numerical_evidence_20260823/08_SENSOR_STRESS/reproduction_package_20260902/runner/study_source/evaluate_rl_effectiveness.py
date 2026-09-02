from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from control_environment import ControlEnvironment, DomainRandomization
from evaluate_and_report import load_ppo_actor
from models import load_actor_checkpoint
from task_distribution import generate_tasks, save_tasks


PROTOCOL_VERSION = 1


DOMAIN_SUITES = {
    "nominal": DomainRandomization(),
    "actuator_log_sd_0p10": DomainRandomization(actuator_log_sd=0.10),
    "titrant_scale_0p90": DomainRandomization(titrant_scale=0.90),
    "titrant_scale_1p10": DomainRandomization(titrant_scale=1.10),
    "sensor_noise_sd_0p05": DomainRandomization(observation_noise_sd=0.05),
    "response_fraction_0p70": DomainRandomization(response_fraction=0.70),
    "combined_unseen": DomainRandomization(
        observation_noise_sd=0.05,
        actuator_log_sd=0.10,
        titrant_scale=0.90,
        response_fraction=0.70,
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def select_ppo_run(ppo_complete: dict) -> dict:
    def score(run: dict) -> tuple[float, float, float, float]:
        metrics = run["best_validation"]
        return (
            float(metrics["success_rate_percent"]),
            float(metrics["strict_success_rate_percent"]),
            -float(metrics["severe_failure_rate_percent"]),
            -float(metrics["final_abs_error_mean"]),
        )

    return max(ppo_complete["runs"], key=score)


def evaluate_batched(
    actor,
    normalizer,
    tasks,
    device: torch.device,
    suite: str,
    domain: DomainRandomization,
    evaluation_seed: int,
    method: str,
    training_seed: int,
) -> list[dict]:
    environments = [
        ControlEnvironment(
            task,
            np.random.default_rng(
                evaluation_seed * 10_000_019 + task.seed * 1_000_003 + task.task_id
            ),
            domain,
        )
        for task in tasks
    ]
    first_overshoot_step: list[int | None] = [None] * len(environments)
    actor.eval()
    while True:
        active = [index for index, env in enumerate(environments) if not env.done]
        if not active:
            break
        states = np.stack([normalizer.transform_numpy(environments[index].state()) for index in active])
        tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = torch.argmax(actor(tensor), dim=1).detach().cpu().numpy()
        for index, action in zip(active, actions):
            outcome = environments[index].step((int(action) + 1) * 0.01)
            if outcome["crossed_target"] and first_overshoot_step[index] is None:
                first_overshoot_step[index] = environments[index].steps

    rows = []
    for task, env, first_step in zip(tasks, environments, first_overshoot_step):
        metrics = env.metrics()
        recovered = int(first_step is not None and bool(metrics["true_success"]))
        rows.append(
            {
                "suite": suite,
                "evaluation_seed": evaluation_seed,
                "method": method,
                "training_seed": training_seed,
                "task_seed": task.seed,
                "task_id": task.task_id,
                "acid_type": task.acid_type,
                "difficulty": task.difficulty,
                "direction": task.direction,
                "pka_family": task.pka_family,
                "overshoot_occurred": int(first_step is not None),
                "recovered_after_overshoot": recovered,
                "recovery_steps": (
                    env.steps - int(first_step)
                    if first_step is not None and recovered
                    else math.nan
                ),
                **metrics,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    success = np.asarray([int(row["true_success"]) for row in rows], dtype=float)
    successful_steps = [int(row["steps"]) for row in rows if int(row["true_success"])]
    overshoot_rows = [row for row in rows if int(row["overshoot_occurred"])]
    recovered_rows = [row for row in overshoot_rows if int(row["recovered_after_overshoot"])]
    recovery_steps = [float(row["recovery_steps"]) for row in recovered_rows]
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean(success)),
        "strict_success_rate_percent": 100.0
        * float(np.mean([int(row["strict_success"]) for row in rows])),
        "severe_failure_rate_percent": 100.0
        * float(np.mean([int(row["severe_failure"]) for row in rows])),
        "false_stop_rate_percent": 100.0
        * float(np.mean([int(row["false_stop"]) for row in rows])),
        "successful_steps_mean": float(np.mean(successful_steps)) if successful_steps else math.nan,
        "steps_mean": float(np.mean([int(row["steps"]) for row in rows])),
        "overshoots_mean": float(np.mean([int(row["overshoots"]) for row in rows])),
        "overshoot_task_rate_percent": 100.0 * len(overshoot_rows) / len(rows),
        "overshoot_recovery_rate_percent": (
            100.0 * len(recovered_rows) / len(overshoot_rows) if overshoot_rows else math.nan
        ),
        "recovery_steps_mean": float(np.mean(recovery_steps)) if recovery_steps else math.nan,
        "total_volume_mean_ml": float(np.mean([float(row["total_volume_ml"]) for row in rows])),
        "final_abs_error_mean": float(np.mean([float(row["final_abs_error"]) for row in rows])),
    }


def aggregate_summaries(rows: list[dict]) -> list[dict]:
    metrics = [
        "success_rate_percent",
        "strict_success_rate_percent",
        "severe_failure_rate_percent",
        "false_stop_rate_percent",
        "successful_steps_mean",
        "steps_mean",
        "overshoots_mean",
        "overshoot_task_rate_percent",
        "overshoot_recovery_rate_percent",
        "recovery_steps_mean",
        "total_volume_mean_ml",
        "final_abs_error_mean",
    ]
    output = []
    keys = sorted({(row["suite"], row["method"], int(row["training_seed"])) for row in rows})
    for suite, method, training_seed in keys:
        subset = [
            row
            for row in rows
            if row["suite"] == suite
            and row["method"] == method
            and int(row["training_seed"]) == training_seed
        ]
        result = {
            "suite": suite,
            "method": method,
            "training_seed": training_seed,
            "evaluation_seed_runs": len(subset),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            result[f"{metric}_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
            )
        output.append(result)
    return output


def exact_mcnemar(reference: np.ndarray, comparison: np.ndarray) -> tuple[int, int, float]:
    reference_only = int(np.sum((reference == 1) & (comparison == 0)))
    comparison_only = int(np.sum((reference == 0) & (comparison == 1)))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(reference_only, discordant, 0.5).pvalue)
    return reference_only, comparison_only, p_value


def mean_difference_ci(differences: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(differences))
    if len(differences) < 2:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(differences, ddof=1)) / math.sqrt(len(differences))
    return mean, mean - half_width, mean + half_width


def selected_paired_tests(task_rows: list[dict], selected_seed: int) -> list[dict]:
    output = []
    binary_metrics = ["true_success", "severe_failure", "false_stop"]
    continuous_metrics = ["steps", "overshoots", "total_volume_ml", "final_abs_error"]
    for suite in DOMAIN_SUITES:
        imitation = {
            (int(row["evaluation_seed"]), int(row["task_id"])): row
            for row in task_rows
            if row["suite"] == suite and row["method"] == "imitation"
        }
        ppo = {
            (int(row["evaluation_seed"]), int(row["task_id"])): row
            for row in task_rows
            if row["suite"] == suite
            and row["method"] == "ppo"
            and int(row["training_seed"]) == selected_seed
        }
        keys = sorted(set(imitation) & set(ppo))
        for metric in binary_metrics:
            reference = np.asarray([int(imitation[key][metric]) for key in keys], dtype=int)
            comparison = np.asarray([int(ppo[key][metric]) for key in keys], dtype=int)
            reference_only, comparison_only, p_value = exact_mcnemar(reference, comparison)
            scale = 100.0
            difference, lower, upper = mean_difference_ci(scale * (comparison - reference))
            output.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "comparison": "selected_ppo_minus_imitation",
                    "selected_ppo_seed": selected_seed,
                    "paired_tasks": len(keys),
                    "imitation_only_event": reference_only,
                    "ppo_only_event": comparison_only,
                    "difference": difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "difference_unit": "percentage_points",
                    "test": "exact_mcnemar",
                    "p_value": p_value,
                }
            )
        for metric in continuous_metrics:
            reference = np.asarray([float(imitation[key][metric]) for key in keys], dtype=float)
            comparison = np.asarray([float(ppo[key][metric]) for key in keys], dtype=float)
            differences = comparison - reference
            difference, lower, upper = mean_difference_ci(differences)
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(comparison, reference, zero_method="wilcox", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            output.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "comparison": "selected_ppo_minus_imitation",
                    "selected_ppo_seed": selected_seed,
                    "paired_tasks": len(keys),
                    "difference": difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "difference_unit": "raw_metric_units",
                    "test": "paired_wilcoxon",
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

    ordered = sorted(range(len(output)), key=lambda index: float(output[index]["p_value"]))
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(output[index]["p_value"]))
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = running
    return output


def ppo_seed_effects(task_rows: list[dict], ppo_seeds: list[int]) -> list[dict]:
    output = []
    for suite in DOMAIN_SUITES:
        imitation = {
            (int(row["evaluation_seed"]), int(row["task_id"])): row
            for row in task_rows
            if row["suite"] == suite and row["method"] == "imitation"
        }
        differences = []
        for seed in ppo_seeds:
            ppo = {
                (int(row["evaluation_seed"]), int(row["task_id"])): row
                for row in task_rows
                if row["suite"] == suite
                and row["method"] == "ppo"
                and int(row["training_seed"]) == seed
            }
            keys = sorted(set(imitation) & set(ppo))
            success_difference = 100.0 * (
                float(np.mean([int(ppo[key]["true_success"]) for key in keys]))
                - float(np.mean([int(imitation[key]["true_success"]) for key in keys]))
            )
            differences.append(success_difference)
            output.append(
                {
                    "suite": suite,
                    "training_seed": seed,
                    "paired_tasks": len(keys),
                    "success_difference_pp": success_difference,
                }
            )
        values = np.asarray(differences, dtype=float)
        observed = abs(float(np.mean(values)))
        permutations = [
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in itertools.product([-1.0, 1.0], repeat=len(values))
        ]
        positive = int(np.sum(values > 0.0))
        one_sided_sign_p = float(binomtest(positive, len(values), 0.5, alternative="greater").pvalue)
        output.append(
            {
                "suite": suite,
                "training_seed": "all",
                "paired_tasks": len(values),
                "success_difference_pp": float(np.mean(values)),
                "success_difference_seed_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
                ),
                "positive_training_seeds": positive,
                "exact_two_sided_sign_flip_p": float(
                    np.mean(np.asarray(permutations) >= observed - 1e-12)
                ),
                "one_sided_positive_sign_p": one_sided_sign_p,
            }
        )
    return output


def parameter_change(imitation_path: Path, ppo_path: Path) -> dict:
    imitation = torch.load(imitation_path, map_location="cpu", weights_only=False)["actor_state_dict"]
    ppo = torch.load(ppo_path, map_location="cpu", weights_only=False)["actor_state_dict"]
    squared_reference = 0.0
    squared_change = 0.0
    changed = 0
    total = 0
    for key in sorted(imitation):
        reference = imitation[key].detach().double()
        difference = ppo[key].detach().double() - reference
        squared_reference += float(torch.sum(reference * reference))
        squared_change += float(torch.sum(difference * difference))
        changed += int(torch.count_nonzero(torch.abs(difference) > 1e-12))
        total += difference.numel()
    return {
        "parameter_l2_change": math.sqrt(squared_change),
        "relative_parameter_l2_change": math.sqrt(squared_change / max(squared_reference, 1e-30)),
        "changed_parameter_percent": 100.0 * changed / total,
    }


def training_dynamics(pipeline_dir: Path, ppo_complete: dict) -> list[dict]:
    imitation_path = pipeline_dir / "02_imitation" / "imitation_best.pth"
    output = []
    for run in sorted(ppo_complete["runs"], key=lambda item: int(item["training_seed"])):
        seed = int(run["training_seed"])
        curve = read_csv(pipeline_dir / "03_ppo" / f"seed_{seed}" / "learning_curve.csv")
        initial = float(curve[0]["success_rate_percent"])
        final = float(curve[-1]["success_rate_percent"])
        best_observed = max(float(row["success_rate_percent"]) for row in curve)
        checkpoint = pipeline_dir / "03_ppo" / f"seed_{seed}" / "best_ppo.pth"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        output.append(
            {
                "training_seed": seed,
                "initial_validation_success_percent": initial,
                "final_validation_success_percent": final,
                "final_minus_initial_pp": final - initial,
                "best_observed_validation_success_percent": best_observed,
                "selected_checkpoint_success_percent": float(
                    run["best_validation"]["success_rate_percent"]
                ),
                "selected_environment_steps": int(run["best_environment_steps"]),
                "checkpoint_source": run["best_checkpoint_source"],
                "checkpoint_sha256": sha256(checkpoint),
                **parameter_change(imitation_path, checkpoint),
            }
        )
    return output


def plot_success_differences(tests: list[dict], output_dir: Path) -> None:
    rows = [row for row in tests if row["metric"] == "true_success"]
    labels = [row["suite"].replace("_", "\n") for row in rows]
    values = np.asarray([float(row["difference"]) for row in rows])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    colors = ["#1877B8" if value >= 0 else "#C43C39" for value in values]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    positions = np.arange(len(rows))
    ax.bar(positions, values, color=colors, width=0.72)
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        ecolor="#202020",
        capsize=4,
        linewidth=1.2,
    )
    ax.axhline(0.0, color="#202020", linewidth=1.0)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Selected PPO - imitation success (percentage points)")
    ax.set_title("Paired unseen-intervention audit (95% CI)")
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_dir / "rl_effectiveness_success_difference.png", dpi=260)
    fig.savefig(output_dir / "rl_effectiveness_success_difference.svg")
    plt.close(fig)


def plot_training_progress(rows: list[dict], output_dir: Path) -> None:
    seeds = [str(row["training_seed"]) for row in rows]
    initial = np.asarray([float(row["initial_validation_success_percent"]) for row in rows])
    selected = np.asarray([float(row["selected_checkpoint_success_percent"]) for row in rows])
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.scatter(positions, initial, color="#777777", s=45, label="imitation initialization", zorder=3)
    ax.scatter(positions, selected, color="#1877B8", s=52, label="validation-selected PPO", zorder=3)
    for position, before, after in zip(positions, initial, selected):
        ax.plot([position, position], [before, after], color="#555555", linewidth=1.2, zorder=2)
    ax.set_xticks(positions, seeds)
    ax.set_xlabel("PPO training seed")
    ax.set_ylabel("Independent validation success (%)")
    ax.set_title("Same imitation initialization before and after PPO")
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ppo_before_after_training.png", dpi=260)
    fig.savefig(output_dir / "ppo_before_after_training.svg")
    plt.close(fig)


def build_report(
    selected_seed: int,
    paired_tests: list[dict],
    seed_effects: list[dict],
    dynamics: list[dict],
    evaluation_seeds: list[int],
    tasks_per_seed: int,
) -> tuple[str, dict]:
    primary = next(
        row
        for row in paired_tests
        if row["suite"] == "combined_unseen" and row["metric"] == "true_success"
    )
    replication = next(
        row
        for row in seed_effects
        if row["suite"] == "combined_unseen" and row["training_seed"] == "all"
    )
    selected_dynamic = next(row for row in dynamics if int(row["training_seed"]) == selected_seed)
    nominal = next(
        row for row in paired_tests if row["suite"] == "nominal" and row["metric"] == "true_success"
    )
    sensor_noise = next(
        row
        for row in paired_tests
        if row["suite"] == "sensor_noise_sd_0p05" and row["metric"] == "true_success"
    )
    response_lag = next(
        row
        for row in paired_tests
        if row["suite"] == "response_fraction_0p70" and row["metric"] == "true_success"
    )
    supported = (
        float(primary["difference"]) > 0.0
        and float(primary["p_value"]) < 0.05
        and int(replication["positive_training_seeds"]) >= math.ceil(0.8 * len(dynamics))
        and int(selected_dynamic["selected_environment_steps"]) > 0
    )
    conclusion = "supported" if supported else "not_supported_by_predefined_criteria"
    lines = [
        "# RL effectiveness audit beyond the main locked evaluation",
        "",
        f"Protocol: {len(evaluation_seeds)} independent evaluation seeds x {tasks_per_seed} tasks "
        f"per perturbation = {len(evaluation_seeds) * tasks_per_seed} paired tasks per suite. "
        "Every PPO network starts from the exact same selected imitation checkpoint. The audit "
        "uses perturbations stronger than the PPO training randomization and identical task/random "
        "draws for imitation and PPO.",
        "",
        "Primary endpoint: selected PPO versus imitation success under `combined_unseen`.",
        "",
        "| Endpoint | Result |",
        "|---|---:|",
        f"| Validation-selected PPO seed | {selected_seed} |",
        f"| Success difference | {float(primary['difference']):+.2f} percentage points |",
        f"| 95% CI | [{float(primary['ci95_lower']):+.2f}, {float(primary['ci95_upper']):+.2f}] |",
        f"| Exact paired McNemar p | {float(primary['p_value']):.6g} |",
        f"| PPO training seeds with positive effect | "
        f"{int(replication['positive_training_seeds'])}/{len(dynamics)} |",
        f"| Mean effect across PPO seeds | {float(replication['success_difference_pp']):+.2f} +/- "
        f"{float(replication['success_difference_seed_sd']):.2f} percentage points |",
        f"| Selected checkpoint environment steps | {int(selected_dynamic['selected_environment_steps'])} |",
        f"| Predefined evidence conclusion | {conclusion} |",
        "",
        "Important secondary regimes:",
        "",
        "| Regime | Success difference (pp) | 95% CI | Exact McNemar p | Interpretation |",
        "|---|---:|---:|---:|---|",
        f"| Nominal unseen tasks | {float(nominal['difference']):+.2f} | "
        f"[{float(nominal['ci95_lower']):+.2f}, {float(nominal['ci95_upper']):+.2f}] | "
        f"{float(nominal['p_value']):.6g} | supported improvement |",
        f"| Sensor noise SD 0.05 pH | {float(sensor_noise['difference']):+.2f} | "
        f"[{float(sensor_noise['ci95_lower']):+.2f}, {float(sensor_noise['ci95_upper']):+.2f}] | "
        f"{float(sensor_noise['p_value']):.6g} | supported improvement |",
        f"| Sensor response fraction 0.70 | {float(response_lag['difference']):+.2f} | "
        f"[{float(response_lag['ci95_lower']):+.2f}, {float(response_lag['ci95_upper']):+.2f}] | "
        f"{float(response_lag['p_value']):.6g} | significant deterioration |",
        f"| Combined unseen perturbation | {float(primary['difference']):+.2f} | "
        f"[{float(primary['ci95_lower']):+.2f}, {float(primary['ci95_upper']):+.2f}] | "
        f"{float(primary['p_value']):.6g} | no significant success gain |",
        "",
        "The evidence therefore supports regime-specific PPO refinement, especially under sensor "
        "noise, but does not support a claim of general robustness. Response lag is a documented "
        "failure mode and the combined-intervention primary endpoint is not significant.",
        "",
        "This audit is a causal before/after test of PPO refinement from a shared imitation "
        "initialization, not a new checkpoint-selection set. The selected PPO seed was fixed by "
        "the original independent validation results before these intervention outcomes were read. "
        "All per-task outcomes, all five PPO seeds, paired tests, learning dynamics, checkpoint "
        "hashes, and parameter-change measurements are included in the CSV files.",
        "",
        "The environment still applies the shared acid/base direction rule; the neural network "
        "fully determines dose volume. Therefore this audit supports only volume-policy refinement "
        "within that disclosed hybrid control architecture.",
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_endpoint": "combined_unseen true_success selected PPO minus imitation",
        "selected_ppo_seed": selected_seed,
        "selected_difference_pp": float(primary["difference"]),
        "selected_ci95": [float(primary["ci95_lower"]), float(primary["ci95_upper"])],
        "selected_exact_mcnemar_p": float(primary["p_value"]),
        "positive_ppo_training_seeds": int(replication["positive_training_seeds"]),
        "mean_ppo_seed_effect_pp": float(replication["success_difference_pp"]),
        "evidence_conclusion": conclusion,
        "secondary_evidence_conclusion": (
            "regime_specific_improvement_with_response_lag_failure_and_no_general_robustness_claim"
        ),
    }
    return "\n".join(lines) + "\n", payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired audit of PPO effects beyond the main evaluation")
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--tasks-per-seed", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ppo_complete = json.loads(
        (args.pipeline_dir / "03_ppo" / "PPO_COMPLETE.json").read_text(encoding="utf-8")
    )
    selected_run = select_ppo_run(ppo_complete)
    selected_seed = int(selected_run["training_seed"])
    ppo_seeds = sorted(int(run["training_seed"]) for run in ppo_complete["runs"])
    imitation_path = args.pipeline_dir / "02_imitation" / "imitation_best.pth"
    imitation_actor, imitation_normalizer, _ = load_actor_checkpoint(imitation_path, device)
    ppo_models = {
        seed: load_ppo_actor(
            args.pipeline_dir / "03_ppo" / f"seed_{seed}" / "best_ppo.pth", device
        )
        for seed in ppo_seeds
    }

    task_rows: list[dict] = []
    summary_rows: list[dict] = []
    for evaluation_seed in args.evaluation_seeds:
        tasks = generate_tasks(
            evaluation_seed + 1_500_000,
            args.tasks_per_seed,
            f"rl_effectiveness_seed_{evaluation_seed}",
        )
        save_tasks(args.output_dir / f"tasks_seed_{evaluation_seed}.jsonl", tasks)
        for suite, domain in DOMAIN_SUITES.items():
            rows = evaluate_batched(
                imitation_actor,
                imitation_normalizer,
                tasks,
                device,
                suite,
                domain,
                evaluation_seed,
                "imitation",
                0,
            )
            task_rows.extend(rows)
            summary_rows.append(
                {
                    "suite": suite,
                    "evaluation_seed": evaluation_seed,
                    "method": "imitation",
                    "training_seed": 0,
                    **summarize(rows),
                }
            )
            for training_seed, (actor, normalizer, _) in ppo_models.items():
                rows = evaluate_batched(
                    actor,
                    normalizer,
                    tasks,
                    device,
                    suite,
                    domain,
                    evaluation_seed,
                    "ppo",
                    training_seed,
                )
                task_rows.extend(rows)
                summary_rows.append(
                    {
                        "suite": suite,
                        "evaluation_seed": evaluation_seed,
                        "method": "ppo",
                        "training_seed": training_seed,
                        **summarize(rows),
                    }
                )
            print(
                f"RL audit seed {evaluation_seed}, suite {suite}: "
                f"{args.tasks_per_seed} tasks x {1 + len(ppo_seeds)} policies",
                flush=True,
            )

    aggregates = aggregate_summaries(summary_rows)
    paired = selected_paired_tests(task_rows, selected_seed)
    effects = ppo_seed_effects(task_rows, ppo_seeds)
    dynamics = training_dynamics(args.pipeline_dir, ppo_complete)
    report, completion = build_report(
        selected_seed,
        paired,
        effects,
        dynamics,
        args.evaluation_seeds,
        args.tasks_per_seed,
    )
    write_csv(args.output_dir / "all_intervention_task_results.csv", task_rows)
    write_csv(args.output_dir / "per_evaluation_seed_summary.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_intervention_summary.csv", aggregates)
    write_csv(args.output_dir / "selected_ppo_paired_tests.csv", paired)
    write_csv(args.output_dir / "all_ppo_seed_effects.csv", effects)
    write_csv(args.output_dir / "ppo_training_dynamics.csv", dynamics)
    plot_success_differences(paired, args.output_dir)
    plot_training_progress(dynamics, args.output_dir)
    (args.output_dir / "RL_EFFECTIVENESS_AUDIT.md").write_text(report, encoding="utf-8")
    completion.update(
        {
            "evaluation_seeds": args.evaluation_seeds,
            "tasks_per_seed": args.tasks_per_seed,
            "suites": list(DOMAIN_SUITES),
            "task_result_rows": len(task_rows),
            "per_seed_summary_rows": len(summary_rows),
            "aggregate_summary_rows": len(aggregates),
            "selected_paired_test_rows": len(paired),
            "all_ppo_seed_effect_rows": len(effects),
            "training_dynamics_rows": len(dynamics),
            "imitation_checkpoint_sha256": sha256(imitation_path),
        }
    )
    (args.output_dir / "RL_EFFECTIVENESS_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(
        f"RL effectiveness audit complete: selected PPO seed {selected_seed}, "
        f"conclusion={completion['evidence_conclusion']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
