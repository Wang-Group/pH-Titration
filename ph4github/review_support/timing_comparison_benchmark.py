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
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_CSV = ROOT / "experiment_summary.csv"
DEFAULT_IL_WEIGHTS = WORKSPACE_ROOT / "volume_regressor_best_big_discrete_new1.pth"
DEFAULT_RL_WEIGHTS = WORKSPACE_ROOT / "volume_regressor_best_big_discrete_new1_trained-1.pth"
OUTPUT_DIR = ROOT / "output" / "reviewer_response" / "timing"
STABILIZATION_TIME_S = 20.0

SIMPLE_TITRANT_CONC = 0.1
SIMPLE_INITIAL_ACID_VOL = 11.0
SIMPLE_MAX_STEPS = 50
SIMPLE_SUCCESS_THRESHOLD = 0.1
SIMPLE_MIN_VOLUME = 0.01
SIMPLE_MAX_VOLUME = 3.0


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@dataclass
class Experiment:
    experiment_id: int
    acid_type: str
    acid_params: list[float]
    initial_ph: float
    target_ph: float


def parse_acid_params(raw: str) -> list[float]:
    value = ast.literal_eval(raw)
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def get_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"Missing expected columns: {names}")


def normalize_acid_type(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith("mono"):
        return "monoprotic"
    if lowered.startswith("di"):
        return "diprotic"
    if lowered.startswith("tri"):
        return "triprotic"
    return lowered


def load_experiments(csv_path: Path, limit: int | None = None) -> list[Experiment]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    experiments: list[Experiment] = []
    for idx, row in enumerate(rows, 1):
        acid_type = normalize_acid_type(get_field(row, "Acid_Type", "Acid type"))
        experiments.append(
            Experiment(
                experiment_id=idx,
                acid_type=acid_type,
                acid_params=parse_acid_params(get_field(row, "Acid_Params", "Acid params")),
                initial_ph=float(get_field(row, "Initial_pH", "Initial p h")),
                target_ph=float(get_field(row, "Target_pH", "Target p h")),
            )
        )
    return experiments


def summarize_times(name: str, decision_times_ms: list[float], episode_times_ms: list[float], episodes: int, total_decisions: int) -> dict[str, float | int | str]:
    if not decision_times_ms:
        raise ValueError(f"No timing samples collected for {name}")
    decision_mean = statistics.mean(decision_times_ms)
    decision_median = statistics.median(decision_times_ms)
    decision_p95 = float(np.percentile(np.array(decision_times_ms, dtype=float), 95))
    episode_mean = statistics.mean(episode_times_ms) if episode_times_ms else 0.0
    episode_median = statistics.median(episode_times_ms) if episode_times_ms else 0.0
    return {
        "controller": name,
        "episodes": episodes,
        "total_decisions": total_decisions,
        "mean_decision_ms": decision_mean,
        "median_decision_ms": decision_median,
        "p95_decision_ms": decision_p95,
        "mean_episode_controller_ms": episode_mean,
        "median_episode_controller_ms": episode_median,
        "decision_fraction_of_20s": decision_median / (STABILIZATION_TIME_S * 1000.0),
        "episode_fraction_of_20s": episode_median / (STABILIZATION_TIME_S * 1000.0),
        "times_20s_longer_than_decision": (STABILIZATION_TIME_S * 1000.0) / decision_median if decision_median > 0 else math.inf,
    }


def write_summary(summary: list[dict[str, float | int | str]], output_json: Path, output_csv: Path) -> None:
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_notes(summary: list[dict[str, float | int | str]], output_md: Path) -> None:
    lines = [
        "# Timing Comparison Benchmark",
        "",
        f"- Dataset: `{DEFAULT_CSV}`",
        f"- Physical reference delay: `{STABILIZATION_TIME_S:.0f} s` per dosing step",
        "- Metric definition:",
        "  For Bayesian, each timed controller cycle includes posterior updating after observing the current pH and then selecting the next action.",
        "  For IL/RL, each timed controller cycle includes state-vector assembly, candidate filtering, and neural-network inference.",
        "  For PID and expert-rule baselines, each timed controller cycle includes only the rule/controller computation, not the pH-solver step.",
        "",
        "| Controller | Median decision (ms) | Mean decision (ms) | P95 decision (ms) | Median episode controller time (ms) | 20 s / median decision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['controller']} | {row['median_decision_ms']:.6f} | {row['mean_decision_ms']:.6f} | "
            f"{row['p95_decision_ms']:.6f} | {row['median_episode_controller_ms']:.6f} | "
            f"{row['times_20s_longer_than_decision']:.1f} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Simple PID / expert-rule environment
# ---------------------------------------------------------------------------


def simple_get_acid_charge_factor(pH: float, pKas: list[float]) -> float:
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


def simple_charge_balance_equation(pH: float, c_A: float, c_Na: float, c_HCl: float, pKas: list[float]) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    acid_neg_charge = c_A * simple_get_acid_charge_factor(pH, pKas)
    return H + c_Na - OH - c_HCl - acid_neg_charge


def simple_solve_ph(base_vol: float, acid_vol: float, pKas: list[float]) -> float:
    total_vol_l = (SIMPLE_INITIAL_ACID_VOL + base_vol + acid_vol) / 1000.0
    c_A = (SIMPLE_INITIAL_ACID_VOL * 0.1 / 1000.0) / total_vol_l
    c_Na = (base_vol * SIMPLE_TITRANT_CONC / 1000.0) / total_vol_l
    c_HCl = (acid_vol * SIMPLE_TITRANT_CONC / 1000.0) / total_vol_l
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if simple_charge_balance_equation(mid, c_A, c_Na, c_HCl, pKas) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)


class SimpleTitrationEnv:
    def __init__(self) -> None:
        self.pKas: list[float] = []
        self.target_ph = 0.0
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = 0.0
        self.steps = 0

    def reset_state(self, pKas: list[float], target_ph: float) -> float:
        self.pKas = pKas
        self.target_ph = target_ph
        self.base_added = 0.0
        self.acid_added = 0.0
        self.current_ph = simple_solve_ph(0.0, 0.0, pKas)
        self.steps = 0
        return self.current_ph

    def step(self, volume: float) -> tuple[float, str, float, bool]:
        prev_ph = self.current_ph
        reagent = "Base" if self.current_ph < self.target_ph else "Acid"
        if reagent == "Base":
            self.base_added += volume
        else:
            self.acid_added += volume
        self.current_ph = simple_solve_ph(self.base_added, self.acid_added, self.pKas)
        self.steps += 1
        overshoot = (
            (prev_ph < self.target_ph and self.current_ph > self.target_ph)
            or (prev_ph > self.target_ph and self.current_ph < self.target_ph)
        )
        return self.current_ph, reagent, volume, overshoot


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AdaptivePIDController:
    def __init__(
        self,
        kp: float = 0.32,
        ki: float = 0.012,
        kd: float = 0.08,
        integral_limit: float = 12.0,
        output_limit: float = SIMPLE_MAX_VOLUME,
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
        return round(clamp(abs(control_signal), SIMPLE_MIN_VOLUME, self.output_limit), 3)


class ExpertRuleController:
    def __init__(
        self,
        max_volume: float = 3.0,
        min_volume: float = 0.01,
        resolution: float = 0.01,
    ) -> None:
        self.max_volume = max_volume
        self.min_volume = min_volume
        self.resolution = resolution
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
                volume = max(self.min_volume, self.last_volume * 0.5)
        if abs_error <= 0.30:
            volume = min(volume, 0.10)
        if abs_error <= 0.15:
            volume = min(volume, 0.03)
        volume = clamp(volume, self.min_volume, self.max_volume)
        units = round(volume / self.resolution)
        return round(max(self.min_volume, units * self.resolution), 3)

    def observe(self, prev_ph: float, new_ph: float, target_ph: float, reagent: str, volume: float, overshot: bool) -> None:
        if reagent == "Base":
            self.net_titrant_ml += volume
        else:
            self.net_titrant_ml -= volume
        if new_ph < target_ph:
            self.lower_bracket_ml = self.net_titrant_ml if self.lower_bracket_ml is None else max(self.lower_bracket_ml, self.net_titrant_ml)
        elif new_ph > target_ph:
            self.upper_bracket_ml = self.net_titrant_ml if self.upper_bracket_ml is None else min(self.upper_bracket_ml, self.net_titrant_ml)
        self.last_error = target_ph - new_ph
        self.last_volume = volume
        self.last_reagent = reagent
        self.last_delta_ph = abs(new_ph - prev_ph)
        self.last_overshot = overshot


# ---------------------------------------------------------------------------
# Neural policy benchmark environment
# ---------------------------------------------------------------------------


def f_monoprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa: float) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    term = 10 ** (pH - pKa)
    alpha = term / (1 + term)
    return H + c_Na - OH - c_A * alpha - c_HCl


def solve_pH_monoprotic_balance(c_A: float, c_Na: float, c_HCl: float, pKa: float) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = f_monoprotic(mid, c_A, c_Na, c_HCl, pKa)
        if abs(f_mid) < 1e-10:
            return mid
        if f_monoprotic(lo, c_A, c_Na, c_HCl, pKa) * f_mid < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def calculate_pH_monoprotic(base1_ml: float, base2_ml: float, acid1_ml: float, acid2_ml: float, pKa: float, secondary_conc: float) -> float:
    acid_vol_ml = SIMPLE_INITIAL_ACID_VOL
    n_acid = acid_vol_ml / 1000.0 * 0.1
    V_total = (acid_vol_ml + base1_ml + base2_ml + acid1_ml + acid2_ml) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_ml * 0.1 + base2_ml * secondary_conc) / 1000.0 / V_total
    c_HCl = (acid1_ml * 0.1 + acid2_ml * secondary_conc) / 1000.0 / V_total
    return round(solve_pH_monoprotic_balance(c_A, c_Na, c_HCl, pKa), 2)


def f_diprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    term1 = np.power(10, np.clip(pH - pKa1, -100, 100))
    term2 = np.power(10, np.clip(2 * pH - pKa1 - pKa2, -100, 100))
    D = 1 + term1 + term2
    alpha1 = term1 / D
    alpha2 = term2 / D
    acid_anion_charge = c_A * (alpha1 + 2 * alpha2)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH_diprotic(c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = f_diprotic(mid, c_A, c_Na, c_HCl, pKa1, pKa2)
        if abs(f_mid) < 1e-10:
            return mid
        if f_diprotic(lo, c_A, c_Na, c_HCl, pKa1, pKa2) * f_mid < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def calculate_pH_diprotic(base1_ml: float, base2_ml: float, acid1_ml: float, acid2_ml: float, pKa1: float, pKa2: float, secondary_conc: float) -> float:
    acid_vol_ml = SIMPLE_INITIAL_ACID_VOL
    n_acid = acid_vol_ml / 1000.0 * 0.1
    V_total = (acid_vol_ml + base1_ml + base2_ml + acid1_ml + acid2_ml) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_ml * 0.1 + base2_ml * secondary_conc) / 1000.0 / V_total
    c_HCl = (acid1_ml * 0.1 + acid2_ml * secondary_conc) / 1000.0 / V_total
    return round(solve_pH_diprotic(c_A, c_Na, c_HCl, pKa1, pKa2), 2)


def f_triprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float, pKa3: float) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    term1 = np.power(10, np.clip(pH - pKa1, -100, 100))
    term2 = np.power(10, np.clip(2 * pH - pKa1 - pKa2, -100, 100))
    term3 = np.power(10, np.clip(3 * pH - pKa1 - pKa2 - pKa3, -100, 100))
    D = 1 + term1 + term2 + term3
    alpha1 = term1 / D
    alpha2 = term2 / D
    alpha3 = term3 / D
    acid_anion_charge = c_A * (alpha1 + 2 * alpha2 + 3 * alpha3)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH_triprotic(c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float, pKa3: float) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = f_triprotic(mid, c_A, c_Na, c_HCl, pKa1, pKa2, pKa3)
        if abs(f_mid) < 1e-10:
            return mid
        if f_triprotic(lo, c_A, c_Na, c_HCl, pKa1, pKa2, pKa3) * f_mid < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def calculate_pH_triprotic(base1_ml: float, base2_ml: float, acid1_ml: float, acid2_ml: float, pKa1: float, pKa2: float, pKa3: float, secondary_conc: float) -> float:
    acid_vol_ml = SIMPLE_INITIAL_ACID_VOL
    n_acid = acid_vol_ml / 1000.0 * 0.1
    V_total = (acid_vol_ml + base1_ml + base2_ml + acid1_ml + acid2_ml) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_ml * 0.1 + base2_ml * secondary_conc) / 1000.0 / V_total
    c_HCl = (acid1_ml * 0.1 + acid2_ml * secondary_conc) / 1000.0 / V_total
    return round(solve_pH_triprotic(c_A, c_Na, c_HCl, pKa1, pKa2, pKa3), 2)


class NeuralEvalEnv:
    def __init__(self, secondary_conc: float) -> None:
        self.secondary_conc = secondary_conc
        self.reagents = {
            "Strong base 1": 0.1,
            "Strong base 2": secondary_conc,
            "Strong acid 1": 0.1,
            "Strong acid 2": secondary_conc,
        }
        self.min_addition_volume = 0.01
        self.addition_volumes = [round(self.min_addition_volume * i, 2) for i in range(1, 1001)]
        self.action_space = [(reagent, volume) for reagent in self.reagents.keys() for volume in self.addition_volumes]

    def reset(self, experiment: Experiment) -> np.ndarray:
        self.acid_type = experiment.acid_type
        self.acid_params = experiment.acid_params
        self.target_ph = experiment.target_ph
        self.base1_added_ml = 0.0
        self.base2_added_ml = 0.0
        self.acid1_added_ml = 0.0
        self.acid2_added_ml = 0.0
        self.last_action_volume = 0.0
        self.steps = 0
        self.oscillation_count = 0
        self.use_secondary_reagents = False
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.current_ph = experiment.initial_ph
        self.previous_ph = self.current_ph
        self.last_measured_ph = self.current_ph
        self.prev_measured_ph = self.current_ph
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        pH_delta = round(self.current_ph - self.previous_ph, 2) if self.current_ph is not None and self.previous_ph is not None else 0.0
        error = round(self.current_ph - self.target_ph, 2)
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def detect_overshoot(self, prev_ph: float, current_ph: float, reagent: str, volume: float) -> tuple[bool, float | None]:
        sign_change = (prev_ph - self.target_ph) * (current_ph - self.target_ph) < 0
        error_increased = abs(current_ph - self.target_ph) > abs(prev_ph - self.target_ph)
        if sign_change or error_increased:
            return True, max(volume / 2.0, self.min_addition_volume)
        return False, None

    def step(self, action: tuple[str, float]) -> tuple[np.ndarray, bool]:
        reagent, volume = action
        volume = float(volume)
        self.last_action_volume = volume
        self.steps += 1
        self.previous_ph = self.current_ph
        self.prev_measured_ph = self.last_measured_ph
        if reagent == "Strong base 1":
            self.base1_added_ml += volume
        elif reagent == "Strong base 2":
            self.base2_added_ml += volume
        elif reagent == "Strong acid 1":
            self.acid1_added_ml += volume
        elif reagent == "Strong acid 2":
            self.acid2_added_ml += volume

        if self.acid_type == "monoprotic":
            self.current_ph = calculate_pH_monoprotic(
                self.base1_added_ml, self.base2_added_ml, self.acid1_added_ml, self.acid2_added_ml, self.acid_params[0], self.secondary_conc
            )
        elif self.acid_type == "diprotic":
            self.current_ph = calculate_pH_diprotic(
                self.base1_added_ml, self.base2_added_ml, self.acid1_added_ml, self.acid2_added_ml, self.acid_params[0], self.acid_params[1], self.secondary_conc
            )
        else:
            self.current_ph = calculate_pH_triprotic(
                self.base1_added_ml,
                self.base2_added_ml,
                self.acid1_added_ml,
                self.acid2_added_ml,
                self.acid_params[0],
                self.acid_params[1],
                self.acid_params[2],
                self.secondary_conc,
            )
        self.last_measured_ph = self.current_ph

        if abs(volume - self.min_addition_volume) < 1e-6 and (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
            self.oscillation_count += 1
            if self.oscillation_count >= 3:
                self.use_secondary_reagents = True

        overshoot_flag, new_threshold = self.detect_overshoot(self.previous_ph, self.current_ph, reagent, volume)
        if overshoot_flag:
            self.overshoot_occurred = True
            self.overshoot_reagent = reagent
            if new_threshold is not None and (self.overshoot_threshold is None or new_threshold < self.overshoot_threshold):
                self.overshoot_threshold = new_threshold

        done = abs(self.current_ph - self.target_ph) < SIMPLE_SUCCESS_THRESHOLD or self.steps >= SIMPLE_MAX_STEPS
        return self._get_state(), done

    def select_best_action(self, state_tensor: torch.Tensor, policy_model: nn.Module) -> tuple[str, float]:
        def filter_by_global_threshold(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if self.overshoot_threshold is not None:
                filtered = [action for action in candidates if action[1] <= self.overshoot_threshold]
                if filtered:
                    return filtered
            return candidates

        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        if self.use_secondary_reagents:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents if "acid 2" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "base 2" in r.lower()]
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents if "base 2" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "acid 2" in r.lower()]
        else:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents if "acid 1" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "base 1" in r.lower()]
                self.overshoot_occurred = False
                self.overshoot_reagent = None
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents if "base 1" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "acid 1" in r.lower()]

        candidate_actions = [action for action in self.action_space if action[0] in allowed_reagent]
        candidate_actions = filter_by_global_threshold(candidate_actions)

        with torch.no_grad():
            logits = policy_model(state_tensor)
            candidate_indices = [self.addition_volumes.index(action[1]) for action in candidate_actions]
            candidate_logits = logits[0, candidate_indices]
            best_pos = candidate_logits.argmax().item()
            return candidate_actions[best_pos]


class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim: int = 5, min_volume: float = 0.01, max_volume: float = 10.0, step: float = 0.01) -> None:
        super().__init__()
        self.discrete_volumes = [round(min_volume + i * step, 2) for i in range(int((max_volume - min_volume) / step) + 1)]
        self.num_actions = len(self.discrete_volumes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Bayesian benchmark environment
# ---------------------------------------------------------------------------


BAYES_TITRATED_VOLUME = 11.0
BAYES_ANALYTE_CONC = 0.1
BAYES_MAX_STEPS = 50
BAYES_SUCCESS_THRESHOLD = 0.1
BAYES_REAGENTS = {
    "Dilute acid 1": 0.1,
    "Dilute acid 2": 0.01,
    "Dilute base 1": 0.1,
    "Dilute base 2": 0.01,
}


def calculate_acid_anion_charge(c_A: float, H: float, pKa_list: list[float]) -> float:
    n = len(pKa_list)
    K = [10 ** (-np.clip(pKa, -100, 100)) for pKa in pKa_list]
    denominator = 1.0
    cumulative_K = 1.0
    for i in range(n):
        cumulative_K *= K[i]
        denominator += cumulative_K / (H ** (i + 1))
    H_nA = c_A / denominator if denominator != 0 else 0.0
    anion_charge = 0.0
    cumulative_K = 1.0
    for k in range(1, n + 1):
        cumulative_K *= K[k - 1]
        anion_conc = H_nA * (cumulative_K / (H ** k))
        anion_charge += k * anion_conc
    return anion_charge


def charge_balance(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa_list: list[float]) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    acid_anion_charge = calculate_acid_anion_charge(c_A, H, pKa_list)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH(c_A: float, c_Na: float, c_HCl: float, pKa_list: list[float]) -> float:
    lo, hi = 0.0, 14.0
    f_lo = charge_balance(lo, c_A, c_Na, c_HCl, pKa_list)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = charge_balance(mid, c_A, c_Na, c_HCl, pKa_list)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def solve_volume_root(func, lo: float = 0.0, hi: float = 10.0, iterations: int = 80) -> float:
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return 0.0
    left, right = lo, hi
    left_value = f_lo
    for _ in range(iterations):
        mid = (left + right) / 2.0
        mid_value = func(mid)
        if abs(mid_value) < 1e-10:
            return mid
        if left_value * mid_value < 0:
            right = mid
        else:
            left = mid
            left_value = mid_value
    return (left + right) / 2.0


def calculate_acid_anion_charge_batch(c_A: float, H: np.ndarray, pKa_matrix: np.ndarray) -> np.ndarray:
    K = np.power(10.0, -np.clip(pKa_matrix, -100, 100))
    denominator = np.ones(H.shape[0], dtype=float)
    cumulative_K = np.ones(H.shape[0], dtype=float)
    for i in range(K.shape[1]):
        cumulative_K *= K[:, i]
        denominator += cumulative_K / np.power(H, i + 1)
    H_nA = c_A / denominator
    anion_charge = np.zeros(H.shape[0], dtype=float)
    cumulative_K = np.ones(H.shape[0], dtype=float)
    for i in range(K.shape[1]):
        cumulative_K *= K[:, i]
        anion_charge += (i + 1) * H_nA * (cumulative_K / np.power(H, i + 1))
    return anion_charge


def charge_balance_batch(pH: np.ndarray, c_A: float, c_Na: float, c_HCl: float, pKa_matrix: np.ndarray) -> np.ndarray:
    H = np.power(10.0, -pH)
    OH = 1e-14 / H
    acid_anion_charge = calculate_acid_anion_charge_batch(c_A, H, pKa_matrix)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH_batch(c_A: float, c_Na: float, c_HCl: float, pKa_matrix: np.ndarray) -> np.ndarray:
    n_particles = pKa_matrix.shape[0]
    lo = np.zeros(n_particles, dtype=float)
    hi = np.full(n_particles, 14.0, dtype=float)
    f_lo = charge_balance_batch(lo, c_A, c_Na, c_HCl, pKa_matrix)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = charge_balance_batch(mid, c_A, c_Na, c_HCl, pKa_matrix)
        left_mask = f_lo * f_mid < 0
        hi = np.where(left_mask, mid, hi)
        lo = np.where(left_mask, lo, mid)
        f_lo = np.where(left_mask, f_lo, f_mid)
    return (lo + hi) / 2.0


class BayesianEnv:
    def __init__(self, num_particles: int = 1000) -> None:
        self.num_particles = num_particles
        self.reagents = BAYES_REAGENTS.copy()
        self.min_addition_volume = 0.01
        self.addition_volumes = [round(self.min_addition_volume * i, 2) for i in range(1, 1000)]
        self.action_space = [(reagent, volume) for reagent in self.reagents for volume in self.addition_volumes]

        self.num_buffers = 3
        self.pKa_list = np.random.uniform(2, 6, size=self.num_buffers)
        self.ref_pKa = np.copy(self.pKa_list)
        self.pKa_std = np.full(self.num_buffers, 0.2)
        self.buffer_total_moles = np.random.uniform(1e-6, 0.5, size=self.num_buffers)
        self.buffer_total_std = np.full(self.num_buffers, 0.005)
        self.vol_ideal_factor = 0.2
        self.ph_rate_threshold = 1.0
        self.ph_rate_bonus_factor = 0.5
        self.direction_penalty_factor = 60.0

    def initialize(self, experiment: Experiment) -> None:
        self.acid_type = experiment.acid_type
        self.acid_params = experiment.acid_params
        self.true_pKas = experiment.acid_params[:]
        self.initial_ph = experiment.initial_ph
        self.current_ph = experiment.initial_ph
        self.previous_ph = experiment.initial_ph
        self.target_ph = experiment.target_ph
        self.max_steps = BAYES_MAX_STEPS
        self.steps_taken = 0
        self.done = False
        self.total_volume = BAYES_TITRATED_VOLUME
        self.previous_total_volume = BAYES_TITRATED_VOLUME
        self.acid_added_moles = 0.0
        self.base_added_moles = 0.0
        self.acid_volume = 0.0
        self.base_volume = 0.0
        self.last_acid_added = 0.0
        self.last_base_added = 0.0
        self.last_action_volume = 0.0
        self.last_measured_ph = experiment.initial_ph
        self.prev_measured_ph = experiment.initial_ph
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False

    def update_exp_ph(self, pH: float) -> None:
        self.prev_measured_ph = self.last_measured_ph if self.last_measured_ph is not None else pH
        self.current_ph = pH
        self.last_measured_ph = pH

    def get_effective_pka_array(self) -> np.ndarray:
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (self.pKa_list - self.ref_pKa)

    def get_effective_pka_matrix(self, sampled_pKa: np.ndarray) -> np.ndarray:
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (sampled_pKa - self.ref_pKa)

    def simulate_observed_ph(self) -> float:
        V_total = (BAYES_TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (BAYES_TITRATED_VOLUME / 1000.0) * BAYES_ANALYTE_CONC
        c_A = n_analyte / V_total
        c_Na = self.base_added_moles / V_total
        c_HCl = self.acid_added_moles / V_total
        return round(solve_pH(c_A, c_Na, c_HCl, self.true_pKas), 2)

    def compute_required_volume(self) -> float:
        n_analyte = (BAYES_TITRATED_VOLUME / 1000.0) * BAYES_ANALYTE_CONC
        effective_pKa = self.get_effective_pka_array().tolist()
        if self.current_ph < self.target_ph:
            reagent = "Dilute base 2" if self.use_secondary_reagents else "Dilute base 1"
            conc = self.reagents[reagent]

            def f_vol(x: float) -> float:
                add_moles = conc * (x / 1000.0)
                new_base = self.base_added_moles + add_moles
                new_total_volume = (BAYES_TITRATED_VOLUME + self.acid_volume + self.base_volume + x) / 1000.0
                c_A_new = n_analyte / new_total_volume
                c_Na_new = new_base / new_total_volume
                c_HCl_new = self.acid_added_moles / new_total_volume
                pH_new = solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
                return pH_new - self.target_ph

            return solve_volume_root(f_vol, 0.0, 10.0)

        reagent = "Dilute acid 2" if self.use_secondary_reagents else "Dilute acid 1"
        conc = self.reagents[reagent]

        def f_vol(x: float) -> float:
            add_moles = conc * (x / 1000.0)
            new_acid = self.acid_added_moles + add_moles
            new_total_volume = (BAYES_TITRATED_VOLUME + self.acid_volume + self.base_volume + x) / 1000.0
            c_A_new = n_analyte / new_total_volume
            c_Na_new = self.base_added_moles / new_total_volume
            c_HCl_new = new_acid / new_total_volume
            pH_new = solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
            return pH_new - self.target_ph

        return solve_volume_root(f_vol, 0.0, 10.0)

    def detect_overshoot(self, prev_ph: float, current_ph: float, reagent: str, last_added_moles: float) -> tuple[bool, float | None]:
        sign_change = (prev_ph - self.target_ph) * (current_ph - self.target_ph) < 0
        error_increased = abs(current_ph - self.target_ph) > abs(prev_ph - self.target_ph)
        if sign_change or error_increased:
            reagent_conc = self.reagents[reagent]
            overshoot_volume = last_added_moles * 1000.0 / reagent_conc
            return True, max(overshoot_volume / 2.0, self.min_addition_volume)
        return False, None

    def step(self, action: tuple[str, float]) -> tuple[float, bool]:
        if self.done:
            return self.current_ph, self.done
        reagent, volume = action
        volume = float(volume)
        self.last_action_volume = volume
        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        if current_for_direction > self.target_ph and "base" in reagent.lower():
            self.done = True
            return self.current_ph, self.done
        if current_for_direction < self.target_ph and "acid" in reagent.lower():
            self.done = True
            return self.current_ph, self.done
        added_moles = self.reagents[reagent] * (volume / 1000.0)
        self.previous_ph = self.current_ph
        self.previous_total_volume = self.total_volume
        self.total_volume += volume
        if "acid" in reagent.lower():
            self.acid_added_moles += added_moles
            self.acid_volume += volume
            self.last_acid_added = added_moles
        else:
            self.base_added_moles += added_moles
            self.base_volume += volume
            self.last_base_added = added_moles
        new_ph = self.simulate_observed_ph()
        self.update_exp_ph(new_ph)
        if abs(volume - self.min_addition_volume) < 1e-6 and (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
            self.oscillation_count += 1
            if self.oscillation_count >= 3:
                self.use_secondary_reagents = True
        self.steps_taken += 1
        last_added = self.last_acid_added if "acid" in reagent.lower() else self.last_base_added
        overshoot_flag, new_thresh = self.detect_overshoot(self.previous_ph, self.current_ph, reagent, last_added)
        if overshoot_flag:
            self.overshoot_occurred = True
            self.overshoot_reagent = reagent
            if new_thresh is not None and (self.overshoot_threshold is None or new_thresh < self.overshoot_threshold):
                self.overshoot_threshold = new_thresh
        if abs(self.current_ph - self.target_ph) < BAYES_SUCCESS_THRESHOLD or self.steps_taken >= self.max_steps:
            self.done = True
        return self.current_ph, self.done

    def select_best_action(self) -> tuple[str, float]:
        def filter_by_global_threshold(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if self.overshoot_threshold is not None:
                filtered = [action for action in candidates if action[1] <= self.overshoot_threshold]
                if filtered:
                    return filtered
            return candidates

        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        if self.use_secondary_reagents:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents if "acid 2" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "base 2" in r.lower()]
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents if "dilute base 2" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "dilute acid 2" in r.lower()]
        else:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if "base" in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents if "acid 1" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "base 1" in r.lower()]
                self.overshoot_occurred = False
                self.overshoot_reagent = None
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents if "dilute base 1" in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents if "dilute acid 1" in r.lower()]
        candidate_actions = [action for action in self.action_space if action[0] in allowed_reagent]
        candidate_actions = filter_by_global_threshold(candidate_actions)
        error = abs(current_for_direction - self.target_ph)
        previous = self.prev_measured_ph if self.prev_measured_ph is not None else current_for_direction
        ph_change = abs(current_for_direction - previous)
        bonus_factor = 1 + self.ph_rate_bonus_factor * (1 - min(ph_change, self.ph_rate_threshold) / self.ph_rate_threshold)
        avg_uncertainty = np.mean(self.pKa_std)
        uncertainty_factor = 1 - 0.1 * min(avg_uncertainty / 1.0, 1.0)
        buffer_mean = np.mean(self.buffer_total_moles)
        buffering_factor = np.clip(1.0 + 0.1 * (buffer_mean - 0.5), 0.95, 1.05)
        alpha = self.vol_ideal_factor * bonus_factor * uncertainty_factor * buffering_factor
        required_vol = self.compute_required_volume()
        combined_value = error + 0.1 * required_vol
        ideal_volume = self.min_addition_volume + (max(self.addition_volumes) - self.min_addition_volume) * np.tanh(alpha * combined_value)
        return min(candidate_actions, key=lambda action: abs(action[1] - ideal_volume))

    def update_posteriors(self, observed_ph: float) -> None:
        sampled_pKa = np.random.normal(self.pKa_list, self.pKa_std, size=(self.num_particles, self.num_buffers))
        sampled_total_moles = np.random.normal(self.buffer_total_moles, self.buffer_total_std, size=(self.num_particles, self.num_buffers))
        effective_pKa = self.get_effective_pka_matrix(sampled_pKa)
        V_total = (BAYES_TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (BAYES_TITRATED_VOLUME / 1000.0) * BAYES_ANALYTE_CONC
        c_A = n_analyte / V_total
        c_Na = self.base_added_moles / V_total
        c_HCl = self.acid_added_moles / V_total
        predicted_ph = solve_pH_batch(c_A, c_Na, c_HCl, effective_pKa)
        weights = np.exp(-0.5 * ((observed_ph - predicted_ph) / 0.01) ** 2)
        weights += 1e-12
        weights /= weights.sum()
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        resampled_pKa = sampled_pKa[indices]
        resampled_total_moles = sampled_total_moles[indices]
        self.pKa_list = resampled_pKa.mean(axis=0)
        self.pKa_std = resampled_pKa.std(axis=0) + 1e-3
        self.buffer_total_moles = resampled_total_moles.mean(axis=0)
        self.buffer_total_std = resampled_total_moles.std(axis=0) + 1e-3


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def warmup_model(model: nn.Module) -> None:
    dummy = torch.zeros((1, 5), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(dummy)


def benchmark_pid(experiments: Iterable[Experiment]) -> dict[str, float | int | str]:
    env = SimpleTitrationEnv()
    controller = AdaptivePIDController()
    decision_times_ms: list[float] = []
    episode_times_ms: list[float] = []
    total_decisions = 0
    experiments = list(experiments)
    for exp in experiments:
        current_ph = env.reset_state(exp.acid_params, exp.target_ph)
        controller.reset()
        episode_total = 0.0
        while True:
            t0 = time.perf_counter_ns()
            volume = controller.get_volume(current_ph, exp.target_ph)
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            decision_times_ms.append(dt_ms)
            episode_total += dt_ms
            total_decisions += 1
            current_ph, _, _, _ = env.step(volume)
            if abs(current_ph - exp.target_ph) <= SIMPLE_SUCCESS_THRESHOLD or env.steps >= SIMPLE_MAX_STEPS:
                break
        episode_times_ms.append(episode_total)
    return summarize_times("Adaptive PID", decision_times_ms, episode_times_ms, len(experiments), total_decisions)


def benchmark_expert(experiments: Iterable[Experiment]) -> dict[str, float | int | str]:
    env = SimpleTitrationEnv()
    controller = ExpertRuleController()
    decision_times_ms: list[float] = []
    episode_times_ms: list[float] = []
    total_decisions = 0
    experiments = list(experiments)
    for exp in experiments:
        current_ph = env.reset_state(exp.acid_params, exp.target_ph)
        controller.reset()
        controller.initialize_bracket(current_ph, exp.target_ph)
        episode_total = 0.0
        while True:
            t0 = time.perf_counter_ns()
            volume = controller.suggest_volume(current_ph, exp.target_ph)
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            decision_times_ms.append(dt_ms)
            episode_total += dt_ms
            total_decisions += 1
            prev_ph = current_ph
            current_ph, reagent, added_volume, overshot = env.step(volume)
            controller.observe(prev_ph, current_ph, exp.target_ph, reagent, added_volume, overshot)
            if abs(current_ph - exp.target_ph) <= SIMPLE_SUCCESS_THRESHOLD or env.steps >= SIMPLE_MAX_STEPS:
                break
        episode_times_ms.append(episode_total)
    return summarize_times("Expert rule", decision_times_ms, episode_times_ms, len(experiments), total_decisions)


def benchmark_neural(experiments: Iterable[Experiment], model_path: Path, secondary_conc: float, label: str) -> dict[str, float | int | str]:
    experiments = list(experiments)
    env = NeuralEvalEnv(secondary_conc=secondary_conc)
    model = DiscreteVolumeRegressor()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    warmup_model(model)
    decision_times_ms: list[float] = []
    episode_times_ms: list[float] = []
    total_decisions = 0
    for exp in experiments:
        state = env.reset(exp)
        episode_total = 0.0
        done = False
        while not done:
            t0 = time.perf_counter_ns()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = env.select_best_action(state_tensor, model)
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            decision_times_ms.append(dt_ms)
            episode_total += dt_ms
            total_decisions += 1
            state, done = env.step(action)
        episode_times_ms.append(episode_total)
    return summarize_times(label, decision_times_ms, episode_times_ms, len(experiments), total_decisions)


def benchmark_bayesian(experiments: Iterable[Experiment], num_particles: int = 1000) -> dict[str, float | int | str]:
    experiments = list(experiments)
    decision_times_ms: list[float] = []
    episode_times_ms: list[float] = []
    total_decisions = 0
    np.random.seed(555)
    for exp in experiments:
        env = BayesianEnv(num_particles=num_particles)
        env.initialize(exp)
        episode_total = 0.0

        t0 = time.perf_counter_ns()
        action = env.select_best_action()
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        decision_times_ms.append(dt_ms)
        episode_total += dt_ms
        total_decisions += 1

        while not env.done:
            observed_ph, done = env.step(action)
            if done:
                break
            t0 = time.perf_counter_ns()
            env.update_posteriors(observed_ph)
            action = env.select_best_action()
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            decision_times_ms.append(dt_ms)
            episode_total += dt_ms
            total_decisions += 1

        episode_times_ms.append(episode_total)
    return summarize_times(f"Bayesian ({num_particles} particles)", decision_times_ms, episode_times_ms, len(experiments), total_decisions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark controller decision latency for the pH titration manuscript revision.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV, help="Experiment summary CSV.")
    parser.add_argument("--il-weights", type=Path, default=DEFAULT_IL_WEIGHTS, help="Imitation-learning model weights.")
    parser.add_argument("--rl-weights", type=Path, default=DEFAULT_RL_WEIGHTS, help="Reinforcement-learning model weights.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on experiments.")
    parser.add_argument("--bayes-particles", type=int, default=1000, help="Particle count for Bayesian timing.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for timing outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(args.input_csv, limit=args.limit)

    summary = [
        benchmark_bayesian(experiments, num_particles=args.bayes_particles),
        benchmark_neural(experiments, args.il_weights, secondary_conc=0.1, label="Imitation learning"),
        benchmark_neural(experiments, args.rl_weights, secondary_conc=0.01, label="Reinforcement learning"),
        benchmark_pid(experiments),
        benchmark_expert(experiments),
    ]

    summary.sort(key=lambda row: float(row["median_decision_ms"]))
    write_summary(summary, args.output_dir / "timing_comparison_summary.json", args.output_dir / "timing_comparison_summary.csv")
    write_notes(summary, args.output_dir / "timing_comparison_summary.md")

    print(f"Benchmarked {len(experiments)} experiments.")
    for row in summary:
        print(
            f"{row['controller']}: median {row['median_decision_ms']:.6f} ms/decision, "
            f"mean {row['mean_decision_ms']:.6f} ms, p95 {row['p95_decision_ms']:.6f} ms"
        )
    print(f"JSON: {args.output_dir / 'timing_comparison_summary.json'}")
    print(f"CSV: {args.output_dir / 'timing_comparison_summary.csv'}")
    print(f"MD: {args.output_dir / 'timing_comparison_summary.md'}")


if __name__ == "__main__":
    main()
