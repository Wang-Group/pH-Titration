from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiment_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "reviewer_response" / "expert_rule"

TITRANT_CONC = 0.1
INITIAL_ACID_VOL = 11.0
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.1
MIN_VOLUME = 0.01
MAX_VOLUME = 3.0
STEP_RESOLUTION = 0.01


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quantize_volume(volume: float, resolution: float = STEP_RESOLUTION) -> float:
    units = round(volume / resolution)
    return round(max(MIN_VOLUME, units * resolution), 3)


def get_acid_charge_factor(pH: float, pKas: list[float]) -> float:
    H = 10 ** (-pH)
    Kas = [10 ** (-pk) for pk in sorted(pKas)]
    n = len(Kas)

    coeffs = [1.0]
    current = 1.0
    for K in Kas:
        current *= K
        coeffs.append(current)

    terms = [coeffs[i] * (H ** (n - i)) for i in range(n + 1)]
    denominator = sum(terms)
    return sum(i * terms[i] for i in range(n + 1)) / denominator


def charge_balance_equation(pH: float, c_A: float, c_Na: float, c_HCl: float, pKas: list[float]) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    acid_neg_charge = c_A * get_acid_charge_factor(pH, pKas)
    return H + c_Na - OH - c_HCl - acid_neg_charge


def solve_pH(base_vol: float, acid_vol: float, pKas: list[float]) -> float:
    total_vol_L = (INITIAL_ACID_VOL + base_vol + acid_vol) / 1000
    c_A = (INITIAL_ACID_VOL * 0.1 / 1000) / total_vol_L
    c_Na = (base_vol * TITRANT_CONC / 1000) / total_vol_L
    c_HCl = (acid_vol * TITRANT_CONC / 1000) / total_vol_L

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge_balance_equation(mid, c_A, c_Na, c_HCl, pKas) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


class TitrationEnv:
    def __init__(self) -> None:
        self.pKas: list[float] = []
        self.target_ph = 0.0
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = 0.0
        self.steps = 0

    def reset_state(self, pKas: list[float], target_ph: float) -> float:
        self.pKas = pKas if isinstance(pKas, list) else [pKas]
        self.target_ph = target_ph
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = solve_pH(0.0, 0.0, self.pKas)
        self.steps = 0
        return self.current_ph

    def step(self, volume: float) -> tuple[float, str, float, bool]:
        prev_ph = self.current_ph
        reagent = "Base" if self.current_ph < self.target_ph else "Acid"

        if reagent == "Base":
            self.base_added += volume
        else:
            self.acid_added += volume

        self.current_ph = solve_pH(self.base_added, self.acid_added, self.pKas)
        self.steps += 1

        overshoot = (
            (prev_ph < self.target_ph and self.current_ph > self.target_ph)
            or (prev_ph > self.target_ph and self.current_ph < self.target_ph)
        )
        return self.current_ph, reagent, volume, overshoot


class ExpertRuleController:
    """
    Human-like heuristic controller:
    1. choose a coarse aliquot from the current pH error bucket,
    2. once both sides of the setpoint have been observed, bisect the bracket in net titrant volume,
    3. switch to small aliquots near the setpoint.
    """

    def __init__(
        self,
        max_volume: float = MAX_VOLUME,
        min_volume: float = MIN_VOLUME,
        resolution: float = STEP_RESOLUTION,
        overshoot_factor: float = 0.5,
        slope_safety: float = 0.85,
    ) -> None:
        self.max_volume = max_volume
        self.min_volume = min_volume
        self.resolution = resolution
        self.overshoot_factor = overshoot_factor
        self.slope_safety = slope_safety
        self.reset()

    def reset(self) -> None:
        self.last_error: float | None = None
        self.last_volume: float | None = None
        self.last_reagent: str | None = None
        self.last_delta_ph: float | None = None
        self.last_overshot = False
        self.net_titrant_ml = 0.0
        self.lower_bracket_ml: float | None = None
        self.upper_bracket_ml: float | None = None

    def _bucket_volume(self, abs_error: float) -> float:
        if abs_error > 6.0:
            return 3.00
        if abs_error > 4.0:
            return 2.50
        if abs_error > 2.0:
            return 2.00
        if abs_error > 1.0:
            return 1.00
        if abs_error > 0.5:
            return 0.50
        if abs_error > 0.2:
            return 0.20
        return 0.05

    def initialize_bracket(self, current_ph: float, target_ph: float) -> None:
        self.net_titrant_ml = 0.0
        if current_ph < target_ph:
            self.lower_bracket_ml = 0.0
            self.upper_bracket_ml = None
        elif current_ph > target_ph:
            self.lower_bracket_ml = None
            self.upper_bracket_ml = 0.0
        else:
            self.lower_bracket_ml = 0.0
            self.upper_bracket_ml = 0.0

    def suggest_volume(self, current_ph: float, target_ph: float) -> float:
        error = target_ph - current_ph
        abs_error = abs(error)
        has_bracket = self.lower_bracket_ml is not None and self.upper_bracket_ml is not None

        if has_bracket:
            target_net_ml = (self.lower_bracket_ml + self.upper_bracket_ml) / 2
            delta_ml = target_net_ml - self.net_titrant_ml
            if abs(delta_ml) < self.min_volume:
                delta_ml = self.min_volume if error >= 0 else -self.min_volume
            volume = abs(delta_ml)
        else:
            volume = self._bucket_volume(abs_error)
            if self.last_overshot and self.last_volume is not None:
                volume = max(self.min_volume, self.last_volume * self.overshoot_factor)

        if abs_error <= 0.30:
            volume = min(volume, 0.10)
        if abs_error <= 0.15:
            volume = min(volume, 0.03)

        volume = clamp(volume, self.min_volume, self.max_volume)
        return quantize_volume(volume, self.resolution)

    def observe(
        self,
        prev_ph: float,
        new_ph: float,
        target_ph: float,
        reagent: str,
        volume: float,
        overshot: bool,
    ) -> None:
        if reagent == "Base":
            self.net_titrant_ml += volume
        else:
            self.net_titrant_ml -= volume

        if new_ph < target_ph:
            if self.lower_bracket_ml is None:
                self.lower_bracket_ml = self.net_titrant_ml
            else:
                self.lower_bracket_ml = max(self.lower_bracket_ml, self.net_titrant_ml)
        elif new_ph > target_ph:
            if self.upper_bracket_ml is None:
                self.upper_bracket_ml = self.net_titrant_ml
            else:
                self.upper_bracket_ml = min(self.upper_bracket_ml, self.net_titrant_ml)

        self.last_error = target_ph - new_ph
        self.last_volume = volume
        self.last_reagent = reagent
        self.last_delta_ph = abs(new_ph - prev_ph)
        self.last_overshot = overshot


@dataclass
class ExperimentResult:
    experiment_id: str
    acid_type: str
    target_ph: float
    success: bool
    steps: int
    overshoots: int
    final_ph: float
    base_added_ml: float
    acid_added_ml: float
    mean_decision_ms: float


def parse_pkas(raw_value: str) -> list[float]:
    parsed = ast.literal_eval(raw_value)
    if isinstance(parsed, list):
        return [float(x) for x in parsed]
    return [float(parsed)]


def load_experiments(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


def get_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"Missing expected columns: {names}")


def summarize_results(results: list[ExperimentResult]) -> dict[str, float]:
    success_steps = [result.steps for result in results if result.success]
    total_steps = sum(result.steps for result in results)
    total_overshoots = sum(result.overshoots for result in results)
    success_count = len(success_steps)
    total_experiments = len(results)
    success_rate = 100.0 * success_count / total_experiments if total_experiments else 0.0
    avg_success_steps = statistics.mean(success_steps) if success_steps else math.nan
    std_success_steps = statistics.stdev(success_steps) if len(success_steps) > 1 else 0.0
    overshoot_rate = 100.0 * total_overshoots / total_steps if total_steps else 0.0
    mean_decision_ms = statistics.mean(result.mean_decision_ms for result in results) if results else 0.0

    return {
        "total_experiments": total_experiments,
        "successful_experiments": success_count,
        "success_rate": success_rate,
        "avg_success_steps": avg_success_steps,
        "std_success_steps": std_success_steps,
        "total_steps": total_steps,
        "total_overshoots": total_overshoots,
        "overshoot_rate": overshoot_rate,
        "mean_controller_decision_ms": mean_decision_ms,
    }


def write_csv(results: list[ExperimentResult], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Experiment",
                "Acid_Type",
                "Target_pH",
                "Success",
                "Steps",
                "Overshoots",
                "Final_pH",
                "Base_Added_mL",
                "Acid_Added_mL",
                "Mean_Decision_ms",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.experiment_id,
                    result.acid_type,
                    f"{result.target_ph:.2f}",
                    "Yes" if result.success else "No",
                    result.steps,
                    result.overshoots,
                    f"{result.final_ph:.2f}",
                    f"{result.base_added_ml:.3f}",
                    f"{result.acid_added_ml:.3f}",
                    f"{result.mean_decision_ms:.6f}",
                ]
            )


def write_report(results: list[ExperimentResult], summary: dict[str, float], output_txt: Path) -> None:
    with output_txt.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Expert-rule titration baseline report\n")
        handle.write("=" * 60 + "\n")
        handle.write(
            "Rule set: bucketed aliquots by pH error until the setpoint is bracketed, "
            "then bisection in net titrant volume, with small aliquots near the setpoint.\n\n"
        )

        for result in results:
            handle.write(
                f"Experiment ID: {result.experiment_id} | Acid type: {result.acid_type} | "
                f"Target pH: {result.target_ph:.2f}\n"
            )
            handle.write(
                f"  Final pH: {result.final_ph:.2f} | Success: {'Yes' if result.success else 'No'} | "
                f"Steps: {result.steps} | Overshoots: {result.overshoots}\n"
            )
            handle.write(
                f"  Added volumes: base {result.base_added_ml:.3f} mL, acid {result.acid_added_ml:.3f} mL | "
                f"Decision time: {result.mean_decision_ms:.6f} ms/step\n"
            )
            handle.write("-" * 60 + "\n")

        handle.write("\nSummary statistics\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Total experiments: {summary['total_experiments']}\n")
        handle.write(f"Successful experiments: {summary['successful_experiments']}\n")
        handle.write(f"Success rate: {summary['success_rate']:.2f}%\n")
        handle.write(f"Successful steps: {summary['avg_success_steps']:.2f} +/- {summary['std_success_steps']:.2f}\n")
        handle.write(f"Total steps: {summary['total_steps']}\n")
        handle.write(f"Total overshoots: {summary['total_overshoots']}\n")
        handle.write(f"Overshoot rate: {summary['overshoot_rate']:.2f}%\n")
        handle.write(f"Mean controller decision time: {summary['mean_controller_decision_ms']:.6f} ms\n")
        handle.write("=" * 60 + "\n")


def run_all_experiments(
    input_csv: Path,
    output_txt: Path,
    output_csv: Path,
    output_json: Path,
    limit: int | None = None,
) -> dict[str, float]:
    env = TitrationEnv()
    controller = ExpertRuleController()
    rows = load_experiments(input_csv, limit=limit)
    results: list[ExperimentResult] = []

    for row in rows:
        experiment_id = get_field(row, "Experiment")
        acid_type = get_field(row, "Acid_Type", "Acid type")
        pKas = parse_pkas(get_field(row, "Acid_Params", "Acid params"))
        target_ph = float(get_field(row, "Target_pH", "Target pH", "Target p h"))

        current_ph = env.reset_state(pKas, target_ph)
        controller.reset()
        controller.initialize_bracket(current_ph, target_ph)
        overshoots = 0
        decision_times_ms: list[float] = []
        success = False

        while True:
            t0 = time.perf_counter()
            volume = controller.suggest_volume(current_ph, target_ph)
            decision_times_ms.append((time.perf_counter() - t0) * 1000)

            prev_ph = current_ph
            current_ph, reagent, added_volume, overshot = env.step(volume)
            controller.observe(prev_ph, current_ph, target_ph, reagent, added_volume, overshot)

            if overshot:
                overshoots += 1

            if abs(current_ph - target_ph) <= SUCCESS_THRESHOLD:
                success = True
                break
            if env.steps >= MAX_STEPS:
                break

        results.append(
            ExperimentResult(
                experiment_id=experiment_id,
                acid_type=acid_type,
                target_ph=target_ph,
                success=success,
                steps=env.steps,
                overshoots=overshoots,
                final_ph=current_ph,
                base_added_ml=env.base_added,
                acid_added_ml=env.acid_added,
                mean_decision_ms=statistics.mean(decision_times_ms) if decision_times_ms else 0.0,
            )
        )

    summary = summarize_results(results)
    write_csv(results, output_csv)
    write_report(results, summary, output_txt)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Detailed report written to: {output_txt}")
    print(f"Per-experiment CSV written to: {output_csv}")
    print(f"Summary JSON written to: {output_json}")
    print(f"Success rate: {summary['success_rate']:.2f}%")
    print(f"Successful steps: {summary['avg_success_steps']:.2f} +/- {summary['std_success_steps']:.2f}")
    print(f"Overshoot rate: {summary['overshoot_rate']:.2f}%")
    print(f"Mean controller decision time: {summary['mean_controller_decision_ms']:.6f} ms")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the expert-rule titration baseline on simulated experiments.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with simulated titration tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for report files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of experiments to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all_experiments(
        input_csv=args.input,
        output_txt=args.output_dir / "expert_rule_report.txt",
        output_csv=args.output_dir / "expert_rule_results.csv",
        output_json=args.output_dir / "expert_rule_summary.json",
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
