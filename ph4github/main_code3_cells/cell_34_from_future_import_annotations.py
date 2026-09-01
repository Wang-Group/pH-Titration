# Source notebook: main_code3.ipynb
# Raw notebook cell index: 34
# Code-cell export index: 34
# First non-empty line: from __future__ import annotations
from __future__ import annotations

import argparse
import ast
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TITRANT_CONC = 0.1
INITIAL_ACID_VOL = 11.0
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.1
MIN_VOLUME = 0.01
MAX_VOLUME = 3.00


def get_acid_charge_factor(pH: float, pKas: list[float]) -> float:
    H = 10 ** (-pH)
    Kas = [10 ** (-pk) for pk in sorted(pKas)]
    n = len(Kas)
    coeffs = [1.0]
    curr = 1.0
    for K in Kas:
        curr *= K
        coeffs.append(curr)
    terms = [coeffs[i] * (H ** (n - i)) for i in range(n + 1)]
    D = sum(terms)
    return sum(i * terms[i] for i in range(n + 1)) / D


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
        self.current_ph = solve_pH(0, 0, self.pKas)
        self.steps = 0
        return self.current_ph

    def step(self, volume: float):
        prev_ph = self.current_ph
        reagent = "Base" if self.current_ph < self.target_ph else "Acid"

        if reagent == "Base":
            self.base_added += volume
        else:
            self.acid_added += volume

        self.current_ph = solve_pH(self.base_added, self.acid_added, self.pKas)
        self.steps += 1

        is_overshoot = (
            (prev_ph < self.target_ph and self.current_ph > self.target_ph)
            or (prev_ph > self.target_ph and self.current_ph < self.target_ph)
        )
        return self.current_ph, reagent, volume, is_overshoot


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
        self.prev_error: float | None = None

    def get_volume(self, current_ph: float, target_ph: float) -> float:
        error = target_ph - current_ph

        if self.prev_error is not None and error * self.prev_error < 0:
            self.integral *= self.overshoot_decay

        self.integral = clamp(self.integral + error, -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.prev_error is None else error - self.prev_error
        self.prev_error = error

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
        return [float(x) for x in parsed]
    return [float(parsed)]


def load_experiments(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.DictReader(f))
    if limit is not None:
        return reader[:limit]
    return reader


def get_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"Missing expected columns: {names}")


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
) -> dict[str, float]:
    env = TitrationEnv()
    pid = AdaptivePIDController(
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

    with report_path.open("w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("Titration experiment detailed report (revised PID)\n")
        out_f.write("=" * 60 + "\n\n")

        for row in rows:
            exp_id = get_field(row, "Experiment")
            acid_type = get_field(row, "Acid_Type", "Acid type")
            pKas = parse_pkas(get_field(row, "Acid_Params", "Acid params"))
            target_ph = float(get_field(row, "Target_pH", "Target pH", "Target p h"))

            curr_ph = env.reset_state(pKas, target_ph)
            pid.reset()

            out_f.write(f"Experiment ID: {exp_id} | Acid type: {acid_type} | Target pH: {target_ph:.2f}\n")
            out_f.write(f"Initial pH: {curr_ph:.2f}\n")

            success = False
            overshoot_count = 0

            while True:
                vol = pid.get_volume(curr_ph, target_ph)
                curr_ph, reagent, added_vol, overshot = env.step(vol)

                if overshot:
                    total_overshoots += 1
                    overshoot_count += 1

                out_f.write(
                    f"  Step {env.steps:02d}: add {reagent} {added_vol:.3f} mL -> current pH: {curr_ph:.2f}"
                    + (" [overshoot]" if overshot else "")
                    + "\n"
                )

                if abs(curr_ph - target_ph) <= SUCCESS_THRESHOLD:
                    success = True
                    break
                if env.steps >= MAX_STEPS:
                    break

            total_steps += env.steps
            results.append(ExperimentResult(success=success, steps=env.steps, overshoots=overshoot_count))
            out_f.write(
                f"Conclusion: {'success' if success else 'fail'} | Final pH: {curr_ph:.2f} | Steps: {env.steps}\n"
            )
            out_f.write("-" * 40 + "\n\n")

        success_steps = [r.steps for r in results if r.success]
        success_count = len(success_steps)
        total_exps = len(results)
        success_rate = (success_count / total_exps) * 100 if total_exps else 0.0
        avg_success_steps = statistics.mean(success_steps) if success_steps else 0.0
        std_success_steps = statistics.stdev(success_steps) if len(success_steps) > 1 else 0.0
        overshoot_rate = (total_overshoots / total_steps) * 100 if total_steps else 0.0

        out_f.write("\n" + "=" * 60 + "\n")
        out_f.write("Summary statistics\n")
        out_f.write("-" * 60 + "\n")
        out_f.write(f"Total experiments: {total_exps}\n")
        out_f.write(f"Successful experiments: {success_count}\n")
        out_f.write(f"Success rate: {success_rate:.2f}%\n")
        out_f.write(f"Successful steps: {avg_success_steps:.2f} +/- {std_success_steps:.2f}\n")
        out_f.write(f"Total steps: {total_steps}\n")
        out_f.write(f"Total overshoots: {total_overshoots}\n")
        out_f.write(f"Overshoot rate: {overshoot_rate:.2f}%\n")
        out_f.write("=" * 60 + "\n")

    print(f"Report written to: {report_path}")
    print(f"Successful experiments: {success_count}/{total_exps}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Successful steps: {avg_success_steps:.2f} +/- {std_success_steps:.2f}")
    print(f"Overshoot rate: {overshoot_rate:.2f}%")

    return {
        "success_rate": success_rate,
        "avg_success_steps": avg_success_steps,
        "std_success_steps": std_success_steps,
        "overshoot_rate": overshoot_rate,
        "total_experiments": total_exps,
        "successful_experiments": success_count,
        "total_steps": total_steps,
        "total_overshoots": total_overshoots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the revised PID titration baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ph4github") / "experiment_summary.csv",
        help="Input CSV with 3000 simulated experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ph4github") / "PIDexperiment_report_revised.txt",
        help="Detailed log output path.",
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
    run_all_experiments(
        args.input,
        args.output,
        limit=args.limit,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        integral_limit=args.integral_limit,
        output_limit=args.output_limit,
        overshoot_decay=args.overshoot_decay,
    )


if __name__ == "__main__":
    main()
