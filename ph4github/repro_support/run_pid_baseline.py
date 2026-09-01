from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TITRANT_CONC = 0.1
INITIAL_ACID_VOL = 11.0
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.1
MIN_VOLUME = 0.01
MAX_VOLUME = 3.00


def get_acid_charge_factor(ph_value: float, pka_values: list[float]) -> float:
    hydrogen = 10 ** (-ph_value)
    kas = [10 ** (-pka) for pka in sorted(pka_values)]
    order = len(kas)
    coeffs = [1.0]
    current = 1.0
    for equilibrium_constant in kas:
        current *= equilibrium_constant
        coeffs.append(current)
    terms = [coeffs[index] * (hydrogen ** (order - index)) for index in range(order + 1)]
    denominator = sum(terms)
    return sum(index * terms[index] for index in range(order + 1)) / denominator


def charge_balance_equation(ph_value: float, c_acid: float, c_base: float, c_hcl: float, pka_values: list[float]) -> float:
    hydrogen = 10 ** (-ph_value)
    hydroxide = 1e-14 / hydrogen
    acid_negative_charge = c_acid * get_acid_charge_factor(ph_value, pka_values)
    return hydrogen + c_base - hydroxide - c_hcl - acid_negative_charge


def solve_ph(base_volume: float, acid_volume: float, pka_values: list[float]) -> float:
    total_volume_l = (INITIAL_ACID_VOL + base_volume + acid_volume) / 1000
    c_acid = (INITIAL_ACID_VOL * 0.1 / 1000) / total_volume_l
    c_base = (base_volume * TITRANT_CONC / 1000) / total_volume_l
    c_hcl = (acid_volume * TITRANT_CONC / 1000) / total_volume_l

    lower = 0.0
    upper = 14.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        if charge_balance_equation(midpoint, c_acid, c_base, c_hcl, pka_values) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return round((lower + upper) / 2, 2)


class TitrationEnv:
    def __init__(self) -> None:
        self.pka_values: list[float] = []
        self.target_ph = 0.0
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = 0.0
        self.steps = 0

    def reset_state(self, pka_values: list[float], target_ph: float) -> float:
        self.pka_values = pka_values if isinstance(pka_values, list) else [pka_values]
        self.target_ph = target_ph
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = solve_ph(0, 0, self.pka_values)
        self.steps = 0
        return self.current_ph

    def step(self, volume: float) -> tuple[float, str, float, bool]:
        previous_ph = self.current_ph
        reagent = "Base" if self.current_ph < self.target_ph else "Acid"

        if reagent == "Base":
            self.base_added += volume
        else:
            self.acid_added += volume

        self.current_ph = solve_ph(self.base_added, self.acid_added, self.pka_values)
        self.steps += 1

        overshot = (
            (previous_ph < self.target_ph and self.current_ph > self.target_ph)
            or (previous_ph > self.target_ph and self.current_ph < self.target_ph)
        )
        return self.current_ph, reagent, volume, overshot


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AdaptivePIDController:
    def __init__(
        self,
        kp: float = 0.32,
        ki: float = 0.012,
        kd: float = 0.08,
        integral_limit: float = 12.0,
        output_limit: float = MAX_VOLUME,
        overshoot_decay: float = 0.10,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.overshoot_decay = overshoot_decay
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error: float | None = None

    def get_volume(self, current_ph: float, target_ph: float) -> float:
        error = target_ph - current_ph
        if self.previous_error is not None and error * self.previous_error < 0:
            self.integral *= self.overshoot_decay

        self.integral = clamp(self.integral + error, -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error

        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        volume = clamp(abs(control_signal), MIN_VOLUME, self.output_limit)
        return round(volume, 3)


@dataclass
class ExperimentResult:
    success: bool
    steps: int
    overshoots: int


def parse_pkas(raw_value: str) -> list[float]:
    parsed = ast.literal_eval(raw_value)
    if isinstance(parsed, list):
        return [float(value) for value in parsed]
    return [float(parsed)]


def load_experiments(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is None:
        return rows
    return rows[:limit]


def get_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError("Missing expected columns: {}".format(names))


def run_all_experiments(
    csv_path: Path,
    report_path: Path,
    limit: int | None = None,
    kp: float = 0.32,
    ki: float = 0.012,
    kd: float = 0.08,
    integral_limit: float = 12.0,
    output_limit: float = MAX_VOLUME,
    overshoot_decay: float = 0.10,
) -> dict[str, float | int]:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    environment = TitrationEnv()
    controller = AdaptivePIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        integral_limit=integral_limit,
        output_limit=output_limit,
        overshoot_decay=overshoot_decay,
    )
    rows = load_experiments(csv_path, limit=limit)

    results: list[ExperimentResult] = []
    total_steps = 0
    total_overshoots = 0

    with report_path.open("w", encoding="utf-8", newline="\n") as output_handle:
        output_handle.write("Titration experiment detailed report (revised PID)\n")
        output_handle.write("=" * 60 + "\n\n")

        for row in rows:
            experiment_id = get_field(row, "Experiment")
            acid_type = get_field(row, "Acid_Type", "Acid type")
            pka_values = parse_pkas(get_field(row, "Acid_Params", "Acid params"))
            target_ph = float(get_field(row, "Target_pH", "Target pH", "Target p h"))

            current_ph = environment.reset_state(pka_values, target_ph)
            controller.reset()

            output_handle.write(
                f"Experiment ID: {experiment_id} | Acid type: {acid_type} | Target pH: {target_ph:.2f}\n"
            )
            output_handle.write(f"Initial pH: {current_ph:.2f}\n")

            success = False
            overshoot_count = 0

            while True:
                volume = controller.get_volume(current_ph, target_ph)
                current_ph, reagent, added_volume, overshot = environment.step(volume)

                if overshot:
                    total_overshoots += 1
                    overshoot_count += 1

                output_handle.write(
                    f"  Step {environment.steps:02d}: add {reagent} {added_volume:.3f} mL -> current pH: {current_ph:.2f}"
                    + (" [overshoot]" if overshot else "")
                    + "\n"
                )

                if abs(current_ph - target_ph) <= SUCCESS_THRESHOLD:
                    success = True
                    break
                if environment.steps >= MAX_STEPS:
                    break

            total_steps += environment.steps
            results.append(ExperimentResult(success=success, steps=environment.steps, overshoots=overshoot_count))
            output_handle.write(
                f"Conclusion: {'success' if success else 'fail'} | Final pH: {current_ph:.2f} | Steps: {environment.steps}\n"
            )
            output_handle.write("-" * 40 + "\n\n")

        successful_steps = [item.steps for item in results if item.success]
        successful_experiments = len(successful_steps)
        total_experiments = len(results)
        success_rate = (successful_experiments / total_experiments) * 100 if total_experiments else 0.0
        average_success_steps = statistics.mean(successful_steps) if successful_steps else 0.0
        std_success_steps = statistics.stdev(successful_steps) if len(successful_steps) > 1 else 0.0
        overshoot_rate = (total_overshoots / total_steps) * 100 if total_steps else 0.0

        output_handle.write("\n" + "=" * 60 + "\n")
        output_handle.write("Summary statistics\n")
        output_handle.write("-" * 60 + "\n")
        output_handle.write(f"Total experiments: {total_experiments}\n")
        output_handle.write(f"Successful experiments: {successful_experiments}\n")
        output_handle.write(f"Success rate: {success_rate:.2f}%\n")
        output_handle.write(f"Successful steps: {average_success_steps:.2f} +/- {std_success_steps:.2f}\n")
        output_handle.write(f"Total steps: {total_steps}\n")
        output_handle.write(f"Total overshoots: {total_overshoots}\n")
        output_handle.write(f"Overshoot rate: {overshoot_rate:.2f}%\n")
        output_handle.write("=" * 60 + "\n")

    return {
        "success_rate": round(success_rate, 4),
        "avg_success_steps": round(average_success_steps, 4),
        "std_success_steps": round(std_success_steps, 4),
        "overshoot_rate": round(overshoot_rate, 4),
        "total_experiments": total_experiments,
        "successful_experiments": successful_experiments,
        "total_steps": total_steps,
        "total_overshoots": total_overshoots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the extracted revised PID titration baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiment_summary.csv",
        help="Input CSV with simulated titration experiments.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output" / "repro" / "pid_baseline_report.txt",
        help="Detailed text report output path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "output" / "repro" / "pid_baseline_summary.json",
        help="Machine-readable summary output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of experiments to run.",
    )
    parser.add_argument("--kp", type=float, default=0.32, help="Proportional gain.")
    parser.add_argument("--ki", type=float, default=0.012, help="Integral gain.")
    parser.add_argument("--kd", type=float, default=0.08, help="Derivative gain.")
    parser.add_argument(
        "--integral-limit",
        type=float,
        default=12.0,
        help="Absolute clamp for the integral term.",
    )
    parser.add_argument(
        "--output-limit",
        type=float,
        default=MAX_VOLUME,
        help="Maximum dosing volume per step.",
    )
    parser.add_argument(
        "--overshoot-decay",
        type=float,
        default=0.10,
        help="Integral decay factor after an error sign flip.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_all_experiments(
        args.input,
        args.report,
        limit=args.limit,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        integral_limit=args.integral_limit,
        output_limit=args.output_limit,
        overshoot_decay=args.overshoot_decay,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Report written to: {args.report}")
    print(f"Summary written to: {args.summary_json}")
    print(
        "Successful experiments: {}/{} | Success rate: {:.2f}% | Overshoot rate: {:.2f}%".format(
            summary["successful_experiments"],
            summary["total_experiments"],
            summary["success_rate"],
            summary["overshoot_rate"],
        )
    )


if __name__ == "__main__":
    main()
