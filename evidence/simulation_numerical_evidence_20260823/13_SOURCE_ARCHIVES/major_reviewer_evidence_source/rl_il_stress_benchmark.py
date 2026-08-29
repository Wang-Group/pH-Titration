from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from benchmark_core import (
    EpisodeResult,
    NeuralVolumePolicy,
    StressScenario,
    exact_mcnemar,
    generate_tasks,
    holm_adjust,
    portable_settings,
    run_neural_policy,
    summarize_results,
)


SCENARIOS = {
    "nominal": StressScenario("nominal"),
    "analyte_0p03M": StressScenario("analyte_0p03M", analyte_conc_scale=0.30),
    "analyte_0p30M": StressScenario("analyte_0p30M", analyte_conc_scale=3.00),
    "initial_volume_5mL": StressScenario("initial_volume_5mL", initial_volume_scale=5.0 / 11.0),
    "initial_volume_25mL": StressScenario("initial_volume_25mL", initial_volume_scale=25.0 / 11.0),
    "titrant_0p05M": StressScenario("titrant_0p05M", titrant_conc_scale=0.50),
    "titrant_0p15M": StressScenario("titrant_0p15M", titrant_conc_scale=1.50),
    "actuator_under_25pct": StressScenario("actuator_under_25pct", actuator_scale=0.75),
    "actuator_over_25pct": StressScenario("actuator_over_25pct", actuator_scale=1.25),
    "actuator_random_15pct": StressScenario("actuator_random_15pct", actuator_log_sd=0.15),
    "measurement_noise_0p03": StressScenario("measurement_noise_0p03", measurement_noise_sd=0.03),
    "measurement_noise_0p05": StressScenario("measurement_noise_0p05", measurement_noise_sd=0.05),
    "measurement_noise_0p10": StressScenario("measurement_noise_0p10", measurement_noise_sd=0.10),
    "sensor_bias_sd_0p10": StressScenario("sensor_bias_sd_0p10", sensor_bias_sd=0.10),
    "sensor_drift_0p01_step": StressScenario("sensor_drift_0p01_step", drift_sd_per_step=0.01),
    "partial_response_60pct": StressScenario("partial_response_60pct", response_fraction=0.60),
    "tetraprotic": StressScenario("tetraprotic", acid_family="tetraprotic"),
    "outside_pka_range": StressScenario("outside_pka_range", acid_family="outside_pka_range"),
    "close_pka": StressScenario("close_pka", acid_family="close_pka"),
}

CORE_SCENARIOS = [
    "nominal",
    "analyte_0p03M",
    "analyte_0p30M",
    "titrant_0p05M",
    "titrant_0p15M",
    "actuator_under_25pct",
    "actuator_over_25pct",
    "measurement_noise_0p05",
    "partial_response_60pct",
    "tetraprotic",
    "outside_pka_range",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(results: list[EpisodeResult]) -> list[dict]:
    rows: list[dict] = []
    for scenario in sorted({row.scenario for row in results}):
        for method in ("imitation", "rl"):
            seed_summaries = []
            for seed in sorted({row.seed for row in results}):
                subset = [row for row in results if row.scenario == scenario and row.method == method and row.seed == seed]
                if subset:
                    seed_summaries.append(summarize_results(subset))
            result: dict[str, float | int | str] = {"scenario": scenario, "method": method, "seeds": len(seed_summaries)}
            for metric in ("success_rate_percent", "successful_steps_mean", "overshoot_rate_percent", "final_abs_error_mean", "false_stop_rate_percent"):
                values = [float(item[metric]) for item in seed_summaries if not math.isnan(float(item[metric]))]
                result[f"{metric}_mean"] = statistics.mean(values) if values else math.nan
                result[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            rows.append(result)
    return rows


def paired_tests(results: list[EpisodeResult]) -> list[dict]:
    tests: list[dict] = []
    for scenario in sorted({row.scenario for row in results}):
        subset = [row for row in results if row.scenario == scenario]
        imitation = {(row.seed, row.task_id): row for row in subset if row.method == "imitation"}
        rl = {(row.seed, row.task_id): row for row in subset if row.method == "rl"}
        keys = sorted(set(imitation) & set(rl))
        stats = exact_mcnemar(
            [imitation[key].true_success for key in keys],
            [rl[key].true_success for key in keys],
        )
        imitation_rate = 100.0 * sum(imitation[key].true_success for key in keys) / len(keys)
        rl_rate = 100.0 * sum(rl[key].true_success for key in keys) / len(keys)
        tests.append(
            {
                "scenario": scenario,
                "matched_tasks": len(keys),
                "imitation_success_percent": imitation_rate,
                "rl_success_percent": rl_rate,
                "rl_minus_imitation_pp": rl_rate - imitation_rate,
                **stats,
            }
        )
    adjusted = holm_adjust([float(row["p_value_exact_two_sided"]) for row in tests])
    for row, p_value in zip(tests, adjusted):
        row["p_value_holm_across_scenarios"] = p_value
        row["predeclared_clear_rl_gain"] = (
            float(row["rl_minus_imitation_pp"]) >= 1.0 and p_value < 0.05
        )
    return tests


def write_report(path: Path, summary: list[dict], tests: list[dict]) -> None:
    lookup = {(row["scenario"], row["method"]): row for row in summary}
    lines = [
        "# RL versus imitation stress benchmark",
        "",
        "All scenarios in this file were evaluated and reported. A scenario is flagged as a clear RL gain only when the pooled paired success-rate difference is at least 1.0 percentage point and the exact McNemar p-value remains below 0.05 after Holm correction across all tested scenarios.",
        "",
        "| Scenario | Imitation success, mean +/- seed SD (%) | RL success, mean +/- seed SD (%) | RL-IL (pp) | Holm p | Clear gain |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for test in tests:
        scenario = str(test["scenario"])
        il = lookup[(scenario, "imitation")]
        rl = lookup[(scenario, "rl")]
        lines.append(
            f"| {scenario} | {il['success_rate_percent_mean']:.2f} +/- {il['success_rate_percent_seed_sd']:.2f} | "
            f"{rl['success_rate_percent_mean']:.2f} +/- {rl['success_rate_percent_seed_sd']:.2f} | "
            f"{test['rl_minus_imitation_pp']:.2f} | {test['p_value_holm_across_scenarios']:.4g} | "
            f"{'yes' if test['predeclared_clear_rl_gain'] else 'no'} |"
        )
    candidates = [row for row in tests if row["predeclared_clear_rl_gain"]]
    lines.extend(["", "## Decision", ""])
    if candidates:
        names = ", ".join(str(row["scenario"]) for row in candidates)
        lines.append(f"The predeclared criterion identified the following regimes with a clear RL gain: {names}.")
    else:
        lines.append("No tested regime met the predeclared criterion. The manuscript should therefore describe RL as an attempted refinement with marginal or regime-dependent gains, rather than as a demonstrated general improvement over imitation learning.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predefined stress suite for paired RL-versus-imitation evaluation.")
    parser.add_argument("--imitation-weights", type=Path, required=True)
    parser.add_argument("--rl-weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--tasks-per-seed", type=int, default=1000)
    parser.add_argument("--scenario-set", choices=["core", "full"], default="core")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "rl_il_stress")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imitation = NeuralVolumePolicy(args.imitation_weights.resolve(), args.device)
    rl = NeuralVolumePolicy(args.rl_weights.resolve(), args.device)
    names = args.scenarios or (CORE_SCENARIOS if args.scenario_set == "core" else list(SCENARIOS))
    results: list[EpisodeResult] = []

    for name in names:
        scenario = SCENARIOS[name]
        for seed in args.seeds:
            tasks = generate_tasks(seed, args.tasks_per_seed, scenario)
            for index, task in enumerate(tasks, 1):
                common_seed = seed * 1_000_003 + task.task_id
                results.append(run_neural_policy(imitation, task, scenario, "imitation", common_seed))
                results.append(run_neural_policy(rl, task, scenario, "rl", common_seed))
                if index % 250 == 0:
                    print(f"{name}, seed {seed}: {index}/{len(tasks)}")

    write_csv(args.output_dir / "per_task_results.csv", [row.to_dict() for row in results])
    summary = aggregate(results)
    tests = paired_tests(results)
    write_csv(args.output_dir / "aggregate_summary.csv", summary)
    write_csv(args.output_dir / "paired_tests.csv", tests)
    write_report(args.output_dir / "report.md", summary, tests)
    (args.output_dir / "settings.json").write_text(
        json.dumps(portable_settings(vars(args), base), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
