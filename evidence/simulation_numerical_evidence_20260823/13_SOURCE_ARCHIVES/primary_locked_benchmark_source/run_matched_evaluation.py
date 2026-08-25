from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

PACKAGE_ROOT = Path(__file__).resolve().parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from controllers.controller_api import PersistentOvershootCap

from control_environment import ControlEnvironment
from models import VolumeActor, StateNormalizer
from task_distribution import ControlTask, load_tasks


BENCHMARK_SEEDS = (101, 202, 303, 404, 555)
METHODS = ("pf_teacher", "imitation", "ppo")
COMPARISONS = (
    ("pf_teacher", "imitation"),
    ("pf_teacher", "ppo"),
    ("imitation", "ppo"),
)
METRICS = (
    "success_rate_percent",
    "strict_success_rate_percent",
    "severe_failure_rate_percent",
    "false_stop_rate_percent",
    "steps_mean",
    "successful_steps_mean",
    "overshoots_mean",
    "total_volume_mean_ml",
    "final_abs_error_mean",
)
EXPECTED_CHECKPOINT_SHA256 = {
    "imitation": "71ae0176252d55c2a30b5d69afc6033ee1b0f4ac045e05b689ae7031c488adcc",
    "ppo": "bafd85f896945245f4a2275764ee74cfb458aae78cbe91f5c17396c24fd22f1c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    actor = VolumeActor().to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    normalizer = StateNormalizer(
        np.asarray(payload["state_mean"], dtype=np.float32),
        np.asarray(payload["state_std"], dtype=np.float32),
    )
    actor.eval()
    return actor, normalizer, payload.get("metadata", {})


def actor_volume(actor, normalizer, state: np.ndarray, device: torch.device) -> float:
    normalized = normalizer.transform_numpy(state)
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = actor(tensor)
        action_class = int(torch.argmax(logits, dim=1).item())
    return round((action_class + 1) * 0.01, 2)


def standardize_pf_row(row: dict[str, str], benchmark_seed: int) -> dict:
    numeric_int = (
        "task_seed", "task_id", "true_pair_count", "true_success", "strict_success",
        "severe_failure", "measured_success", "false_stop", "steps", "overshoots",
        "overshoot_threshold_activations",
    )
    numeric_float = (
        "initial_ph", "target_ph", "true_concentration_m", "total_volume_ml",
        "final_abs_error", "final_true_ph", "final_measured_ph",
    )
    output = {
        "benchmark_seed": int(benchmark_seed),
        "method": "pf_teacher",
        "acid_type": row["acid_type"],
        "difficulty": row["difficulty"],
        "direction": row["direction"],
        "pka_family": row["pka_family"],
    }
    for key in numeric_int:
        output[key] = int(row[key])
    for key in numeric_float:
        output[key] = float(row[key])
    output["overshoot_cap_events"] = int(row["overshoot_threshold_activations"])
    output["overshoot_cap_applied_steps"] = ""
    output["overshoot_cap_final_ml"] = ""
    return output


def load_pf_reference(path: Path, tasks: list[ControlTask], benchmark_seed: int) -> list[dict]:
    rows = [row for row in read_csv(path) if row.get("policy") == "hybrid_full"]
    if len(rows) < len(tasks):
        raise ValueError(f"{path} has {len(rows)} hybrid_full rows; expected at least {len(tasks)}")
    by_key = {(int(row["task_seed"]), int(row["task_id"])): row for row in rows}
    output = []
    for task in tasks:
        key = (task.seed, task.task_id)
        if key not in by_key:
            raise ValueError(f"PF reference is missing task {key} in {path}")
        row = standardize_pf_row(by_key[key], benchmark_seed)
        if abs(row["initial_ph"] - task.initial_ph) > 1e-9 or abs(row["target_ph"] - task.target_ph) > 1e-9:
            raise ValueError(f"PF/task mismatch for {key}")
        if abs(row["true_concentration_m"] - task.analyte_conc_m) > 1e-12:
            raise ValueError(f"PF concentration mismatch for {key}")
        output.append(row)
    return output


def rollout_network(
    actor,
    normalizer,
    task: ControlTask,
    device: torch.device,
    seed: int,
    use_overshoot_cap: bool = True,
) -> dict:
    env = ControlEnvironment(task, np.random.default_rng(seed))
    overshoot_cap = PersistentOvershootCap(enabled=use_overshoot_cap)
    while not env.done:
        before_ph = float(env.measured_ph)
        requested = actor_volume(actor, normalizer, env.state(), device)
        executed_request, _ = overshoot_cap.apply(requested)
        info = env.step(executed_request)
        after_ph = float(env.measured_ph)
        overshoot_cap.update(
            before_ph,
            after_ph,
            task.target_ph,
            float(info.get("actual_volume_ml", executed_request)),
        )
    metrics = env.metrics()
    output = {
        "benchmark_seed": int(seed),
        "method": "network",
        "task_seed": int(task.seed),
        "task_id": int(task.task_id),
        "acid_type": task.acid_type,
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        "true_concentration_m": task.analyte_conc_m,
        "overshoot_cap_events": overshoot_cap.events,
        "overshoot_cap_applied_steps": overshoot_cap.applied_steps,
        "overshoot_cap_final_ml": "" if overshoot_cap.cap_ml is None else overshoot_cap.cap_ml,
    }
    output.update(metrics)
    return output


def summarize(rows: list[dict], method: str, benchmark_seed: int) -> dict:
    subset = [row for row in rows if row["method"] == method]
    successful_steps = [float(row["steps"]) for row in subset if int(row["true_success"])]
    return {
        "benchmark_seed": benchmark_seed,
        "method": method,
        "tasks": len(subset),
        "success_rate_percent": 100.0 * float(np.mean([int(row["true_success"]) for row in subset])),
        "strict_success_rate_percent": 100.0 * float(np.mean([int(row["strict_success"]) for row in subset])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([int(row["severe_failure"]) for row in subset])),
        "false_stop_rate_percent": 100.0 * float(np.mean([int(row["false_stop"]) for row in subset])),
        "steps_mean": float(np.mean([float(row["steps"]) for row in subset])),
        "successful_steps_mean": float(np.mean(successful_steps)) if successful_steps else math.nan,
        "overshoots_mean": float(np.mean([float(row["overshoots"]) for row in subset])),
        "total_volume_mean_ml": float(np.mean([float(row["total_volume_ml"]) for row in subset])),
        "final_abs_error_mean": float(np.mean([float(row["final_abs_error"]) for row in subset])),
        "cap_activation_rate_percent": 100.0 * float(np.mean([
            int(float(row.get("overshoot_cap_events", 0)) > 0) for row in subset
        ])),
    }


def aggregate(seed_rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        subset = [row for row in seed_rows if row["method"] == method]
        row = {"method": method, "runs": len(subset)}
        for metric in METRICS + ("cap_activation_rate_percent",):
            values = np.asarray([float(item[metric]) for item in subset], dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        output.append(row)
    return output


def exact_mcnemar(reference: list[int], comparison: list[int]) -> tuple[int, int, float]:
    reference_only = sum(bool(a) and not bool(b) for a, b in zip(reference, comparison))
    comparison_only = sum(not bool(a) and bool(b) for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(reference_only, discordant, 0.5).pvalue)
    return reference_only, comparison_only, p_value


def holm(rows: list[dict], key: str = "p_value") -> None:
    valid = [i for i, row in enumerate(rows) if row.get(key) not in (None, "")]
    ordered = sorted(valid, key=lambda i: float(rows[i][key]))
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * float(rows[index][key]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def paired_success_tests(rows: list[dict]) -> list[dict]:
    output = []
    scopes = sorted({row["benchmark_seed"] for row in rows}) + ["pooled"]
    for scope in scopes:
        scoped = rows if scope == "pooled" else [row for row in rows if row["benchmark_seed"] == scope]
        lookup = {
            method: {(int(row["task_seed"]), int(row["task_id"])): int(row["true_success"])
                     for row in scoped if row["method"] == method}
            for method in METHODS
        }
        scope_rows = []
        for reference, comparison in COMPARISONS:
            keys = sorted(set(lookup[reference]) & set(lookup[comparison]))
            ref_values = [lookup[reference][key] for key in keys]
            cmp_values = [lookup[comparison][key] for key in keys]
            ref_only, cmp_only, p_value = exact_mcnemar(ref_values, cmp_values)
            row = {
                "scope": scope,
                "comparison": f"{comparison}_minus_{reference}",
                "paired_tasks": len(keys),
                "reference_only_success": ref_only,
                "comparison_only_success": cmp_only,
                "success_difference_pp": 100.0 * (np.mean(cmp_values) - np.mean(ref_values)),
                "p_value": p_value,
            }
            scope_rows.append(row)
            output.append(row)
        holm(scope_rows)
        for row in scope_rows:
            matching = next(item for item in output if item is row)
            matching["holm_adjusted_p"] = row["holm_adjusted_p"]
    return output


def paired_continuous_tests(rows: list[dict]) -> list[dict]:
    output = []
    metrics = ("steps", "overshoots", "total_volume_ml", "final_abs_error")
    for reference, comparison in COMPARISONS:
        lookup = {
            method: {(int(row["task_seed"]), int(row["task_id"])): row
                     for row in rows if row["method"] == method}
            for method in METHODS
        }
        keys = sorted(set(lookup[reference]) & set(lookup[comparison]))
        scope_rows = []
        for metric in metrics:
            ref_values = np.asarray([float(lookup[reference][key][metric]) for key in keys])
            cmp_values = np.asarray([float(lookup[comparison][key][metric]) for key in keys])
            differences = cmp_values - ref_values
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(cmp_values, ref_values, zero_method="wilcox", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            scope_rows.append({
                "comparison": f"{comparison}_minus_{reference}",
                "metric": metric,
                "paired_tasks": len(keys),
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            })
        holm(scope_rows)
        output.extend(scope_rows)
    return output


def fmt(mean: float, sd: float, digits: int = 2) -> str:
    if not math.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def write_report(output_dir: Path, config: dict, aggregate_rows: list[dict], success_tests: list[dict]) -> None:
    lines = [
        "# Matched PF, imitation, and PPO evaluation",
        "",
        "Five existing PF benchmark task sets were reused without rerunning PF. The PF task-level rows were read from the archived hybrid_full results, and the PF-distilled imitation policy and validation-selected PPO policy were evaluated on the identical tasks.",
        "",
        f"Protocol: {len(config['benchmark_seeds'])} benchmark seeds x {config['tasks_per_seed']} tasks per seed; sensor resolution 0.01 pH; action classes 0.01-10.00 mL; true success evaluated from unquantized equilibrium pH.",
        "The neural policies select volume only. The common external rule selects base below target and acid above target. The persistent post-overshoot cap is enabled: after a target crossing or increased absolute pH error, later volumes are capped at half the triggering delivered dose.",
        "",
        "| Method | Success (%) | Strict (%) | Severe failure (%) | Successful steps | Total volume (mL) | Final error (pH) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        label = {"pf_teacher": "PF teacher", "imitation": "PF-distilled imitation", "ppo": "PPO"}[row["method"]]
        lines.append(
            f"| {label} | {fmt(row['success_rate_percent_mean'], row['success_rate_percent_sd'])} | "
            f"{fmt(row['strict_success_rate_percent_mean'], row['strict_success_rate_percent_sd'])} | "
            f"{fmt(row['severe_failure_rate_percent_mean'], row['severe_failure_rate_percent_sd'])} | "
            f"{fmt(row['successful_steps_mean_mean'], row['successful_steps_mean_sd'])} | "
            f"{fmt(row['total_volume_mean_ml_mean'], row['total_volume_mean_ml_sd'])} | "
            f"{fmt(row['final_abs_error_mean_mean'], row['final_abs_error_mean_sd'], 4)} |"
        )
    lines.extend(["", "## Pooled paired success tests", ""])
    for row in success_tests:
        if row["scope"] == "pooled":
            lines.append(
                f"- {row['comparison']}: {row['success_difference_pp']:+.2f} percentage points; "
                f"exact McNemar p={row['p_value']:.6g}; Holm-adjusted p={row['holm_adjusted_p']:.6g}."
            )
    lines.extend([
        "",
        "Per-seed task-level results and tests are exported separately. The pooled tests do not replace the mean +/- sample SD across the five benchmark seeds.",
    ])
    (output_dir / "RESULT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate imitation and PPO on the existing PF 5 x 3000 tasks")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--imitation-checkpoint", type=Path, default=None)
    parser.add_argument("--ppo-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tasks-per-seed", type=int, default=3000)
    parser.add_argument("--no-overshoot-cap", action="store_true", help="Run the original direction-only neural protocol")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    output_dir = (args.output_dir or package_dir / "results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    imitation_path = (args.imitation_checkpoint or package_dir / "models" / "imitation_best.pth").resolve()
    ppo_path = (args.ppo_checkpoint or package_dir / "models" / "ppo_seed_303.pth").resolve()
    if not imitation_path.is_file():
        raise FileNotFoundError(f"Missing imitation checkpoint: {imitation_path}")
    if not ppo_path.is_file():
        raise FileNotFoundError(f"Missing PPO checkpoint: {ppo_path}")
    actual_hashes = {"imitation": sha256(imitation_path), "ppo": sha256(ppo_path)}
    for method, expected in EXPECTED_CHECKPOINT_SHA256.items():
        if actual_hashes[method] != expected:
            raise RuntimeError(f"Unexpected {method} checkpoint hash: {actual_hashes[method]}")

    config = {
        "benchmark_seeds": list(BENCHMARK_SEEDS),
        "tasks_per_seed": int(args.tasks_per_seed),
        "particles_reused_for_pf": 1000,
        "neural_protocol": (
            "direction_only"
            if args.no_overshoot_cap
            else "direction_plus_persistent_post_overshoot_cap"
        ),
        "overshoot_cap_enabled": not args.no_overshoot_cap,
        "sensor_resolution_ph": 0.01,
        "action_volume_range_ml": [0.01, 10.0],
        "imitation_checkpoint_sha256": actual_hashes["imitation"],
        "ppo_checkpoint_sha256": actual_hashes["ppo"],
    }
    config_path = output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Existing output uses a different configuration: {config_path}")
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume or choose a new output directory: {output_dir}")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    imitation_actor, imitation_normalizer, imitation_meta = load_checkpoint(imitation_path, device)
    ppo_actor, ppo_normalizer, ppo_meta = load_checkpoint(ppo_path, device)
    all_rows: list[dict] = []
    seed_summaries: list[dict] = []
    for benchmark_seed in BENCHMARK_SEEDS:
        task_path = package_dir / "tasks" / f"seed_{benchmark_seed}_tasks.jsonl"
        pf_path = package_dir / "pf_reference" / f"seed_{benchmark_seed}_task_results.csv"
        tasks = load_tasks(task_path)[: args.tasks_per_seed]
        pf_rows = load_pf_reference(pf_path, tasks, benchmark_seed)
        result_path = output_dir / f"seed_{benchmark_seed}_task_results.csv"
        complete_path = output_dir / f"seed_{benchmark_seed}_COMPLETE.json"
        if args.resume and result_path.exists() and complete_path.exists():
            rows = read_csv(result_path)
            for row in rows:
                for key in ("benchmark_seed", "task_seed", "task_id", "true_pair_count", "true_success", "strict_success", "severe_failure", "measured_success", "false_stop", "steps", "overshoots", "overshoot_cap_events", "overshoot_cap_applied_steps"):
                    if key in row and row[key] != "":
                        row[key] = int(float(row[key]))
                for key in ("initial_ph", "target_ph", "true_concentration_m", "total_volume_ml", "final_abs_error", "final_true_ph", "final_measured_ph", "overshoot_cap_final_ml"):
                    if key in row and row[key] != "":
                        row[key] = float(row[key])
            all_rows.extend(rows)
            seed_summaries.extend(summarize(rows, method, benchmark_seed) for method in METHODS)
            print(f"seed {benchmark_seed}: reused existing result", flush=True)
            continue

        rows = list(pf_rows)
        for index, task in enumerate(tasks, 1):
            common_seed = int(task.seed * 1_000_003 + task.task_id)
            imitation_row = rollout_network(
                imitation_actor, imitation_normalizer, task, device, common_seed,
                use_overshoot_cap=not args.no_overshoot_cap,
            )
            imitation_row["benchmark_seed"] = benchmark_seed
            imitation_row["method"] = "imitation"
            ppo_row = rollout_network(
                ppo_actor, ppo_normalizer, task, device, common_seed,
                use_overshoot_cap=not args.no_overshoot_cap,
            )
            ppo_row["benchmark_seed"] = benchmark_seed
            ppo_row["method"] = "ppo"
            rows.extend((imitation_row, ppo_row))
            if index % 250 == 0 or index == len(tasks):
                print(f"seed {benchmark_seed}: {index}/{len(tasks)} tasks", flush=True)
        write_csv(result_path, rows)
        complete_path.write_text(json.dumps({"benchmark_seed": benchmark_seed, "tasks": len(tasks)}, indent=2), encoding="utf-8")
        all_rows.extend(rows)
        seed_summaries.extend(summarize(rows, method, benchmark_seed) for method in METHODS)

    aggregate_rows = aggregate(seed_summaries)
    success_tests = paired_success_tests(all_rows)
    continuous_tests = paired_continuous_tests(all_rows)
    write_csv(output_dir / "per_seed_summary.csv", seed_summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(output_dir / "paired_success_tests.csv", success_tests)
    write_csv(output_dir / "paired_continuous_tests.csv", continuous_tests)
    write_csv(output_dir / "all_task_results.csv", all_rows)
    write_report(output_dir, config, aggregate_rows, success_tests)
    (output_dir / "EVALUATION_COMPLETE.json").write_text(
        json.dumps({"config": config, "aggregate": aggregate_rows, "imitation_metadata": imitation_meta, "ppo_metadata": ppo_meta}, indent=2),
        encoding="utf-8",
    )
    print(f"Completed matched evaluation: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
