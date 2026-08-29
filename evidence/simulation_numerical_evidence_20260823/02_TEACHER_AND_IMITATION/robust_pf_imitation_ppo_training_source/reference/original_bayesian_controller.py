"""
Companion export from repo_reviewcopy/main_code3.ipynb.

Workflow title: Evaluate Bayesian final
Source notebook cells: title cell 19, code cell 20
Purpose: Keep the original notebook untouched while exposing this workflow as a standalone script.

This export preserves the local logic from the source cell. Common helpers remain duplicated
across files on purpose so the exported slices stay close to the notebook semantics.
"""

# Source title cell
# Evaluate Bayesian final

import ast
import csv
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TITRATED_VOLUME = 11.0
ANALYTE_CONC = 0.1
HCL_CONC1 = 0.1
HCL_CONC2 = 0.01
NAOH_CONC1 = 0.1
NAOH_CONC2 = 0.01
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.1

REAGENTS = {
    'Dilute acid 1': HCL_CONC1,
    'Dilute acid 2': HCL_CONC2,
    'Dilute base 1': NAOH_CONC1,
    'Dilute base 2': NAOH_CONC2,
}


def parse_acid_params(raw):
    value = ast.literal_eval(raw)
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def get_field(row, *names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"Missing expected columns: {names}")


def calculate_acid_anion_charge(c_A: float, H: float, pKa_list: list) -> float:
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


def charge_balance(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa_list: list) -> float:
    H = 10 ** (-pH)
    OH = 1e-14 / H
    acid_anion_charge = calculate_acid_anion_charge(c_A, H, pKa_list)
    return H + c_Na - OH - acid_anion_charge - c_HCl


def solve_pH(c_A: float, c_Na: float, c_HCl: float, pKa_list: list) -> float:
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


def solve_volume_root(func, lo=0.0, hi=10.0, iterations=80):
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


class PHAdjustmentEnv:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.steps_taken = 0
        self.done = False
        self.total_volume = TITRATED_VOLUME
        self.previous_total_volume = TITRATED_VOLUME
        self.acid_added_moles = 0.0
        self.base_added_moles = 0.0
        self.acid_volume = 0.0
        self.base_volume = 0.0
        self.last_acid_added = 0.0
        self.last_base_added = 0.0
        self.last_action_volume = 0.0

        self.reagents = REAGENTS.copy()
        self.min_addition_volume = 0.01
        self.addition_volumes = [round(self.min_addition_volume * i, 2) for i in range(1, 1000)]
        self.action_space = [(reagent, volume) for reagent in self.reagents.keys() for volume in self.addition_volumes]

        self.num_buffers = 3
        self.pKa_list = np.random.uniform(2, 6, size=self.num_buffers)
        self.ref_pKa = np.copy(self.pKa_list)
        self.pKa_std = np.full(self.num_buffers, 0.2)
        self.buffer_total_moles = np.random.uniform(1e-6, 0.5, size=self.num_buffers)
        self.buffer_total_std = np.full(self.num_buffers, 0.005)

        self.initial_ph = None
        self.current_ph = None
        self.previous_ph = None
        self.target_ph = None
        self.max_steps = MAX_STEPS

        self.last_measured_ph = None
        self.prev_measured_ph = None

        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False

        self.vol_ideal_factor = 0.2
        self.ph_rate_threshold = 1.0
        self.ph_rate_bonus_factor = 0.5
        self.direction_penalty_factor = 60.0

        self.acid_type = None
        self.acid_params = None
        self.true_pKas = None

    def initialize(self, acid_type, acid_params, init_pH, target_pH, max_steps=MAX_STEPS):
        self.acid_type = acid_type
        self.acid_params = acid_params
        self.true_pKas = [float(x) for x in acid_params] if isinstance(acid_params, list) else [float(acid_params)]

        self.initial_ph = init_pH
        self.current_ph = init_pH
        self.previous_ph = init_pH
        self.target_ph = target_pH
        self.max_steps = max_steps

        self.steps_taken = 0
        self.done = False
        self.total_volume = TITRATED_VOLUME
        self.previous_total_volume = TITRATED_VOLUME
        self.acid_added_moles = 0.0
        self.base_added_moles = 0.0
        self.acid_volume = 0.0
        self.base_volume = 0.0
        self.last_acid_added = 0.0
        self.last_base_added = 0.0
        self.last_action_volume = 0.0

        self.last_measured_ph = init_pH
        self.prev_measured_ph = init_pH
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False

    def get_state(self):
        previous = self.prev_measured_ph if self.prev_measured_ph is not None else self.current_ph
        pH_delta = self.current_ph - previous
        error = self.current_ph - self.target_ph
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def update_exp_ph(self, pH):
        if self.last_measured_ph is not None:
            self.prev_measured_ph = self.last_measured_ph
        else:
            self.prev_measured_ph = pH
        self.current_ph = pH
        self.last_measured_ph = pH

    def get_effective_pka_array(self):
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (self.pKa_list - self.ref_pKa)

    def get_effective_pka_matrix(self, sampled_pKa):
        weight_max = 0.2
        weights = weight_max * (1 - np.tanh(self.pKa_std))
        return self.ref_pKa + weights * (sampled_pKa - self.ref_pKa)

    def simulate_observed_ph(self):
        V_total = (TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
        c_A = n_analyte / V_total
        c_Na = self.base_added_moles / V_total
        c_HCl = self.acid_added_moles / V_total
        return round(solve_pH(c_A, c_Na, c_HCl, self.true_pKas), 2)

    def compute_required_volume(self):
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
        effective_pKa = self.get_effective_pka_array().tolist()

        if self.current_ph < self.target_ph:
            reagent = 'Dilute base 2' if self.use_secondary_reagents else 'Dilute base 1'
            conc = self.reagents[reagent]

            def f_vol(x):
                add_moles = conc * (x / 1000.0)
                new_base = self.base_added_moles + add_moles
                new_total_volume = (TITRATED_VOLUME + self.acid_volume + self.base_volume + x) / 1000.0
                c_A_new = n_analyte / new_total_volume
                c_Na_new = new_base / new_total_volume
                c_HCl_new = self.acid_added_moles / new_total_volume
                pH_new = solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
                return pH_new - self.target_ph

            return solve_volume_root(f_vol, 0.0, 10.0)

        reagent = 'Dilute acid 2' if self.use_secondary_reagents else 'Dilute acid 1'
        conc = self.reagents[reagent]

        def f_vol(x):
            add_moles = conc * (x / 1000.0)
            new_acid = self.acid_added_moles + add_moles
            new_total_volume = (TITRATED_VOLUME + self.acid_volume + self.base_volume + x) / 1000.0
            c_A_new = n_analyte / new_total_volume
            c_Na_new = self.base_added_moles / new_total_volume
            c_HCl_new = new_acid / new_total_volume
            pH_new = solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
            return pH_new - self.target_ph

        return solve_volume_root(f_vol, 0.0, 10.0)

    def detect_overshoot(self, prev_ph, current_ph, reagent, last_added_moles):
        sign_change = (prev_ph - self.target_ph) * (current_ph - self.target_ph) < 0
        error_increased = abs(current_ph - self.target_ph) > abs(prev_ph - self.target_ph)
        if sign_change or error_increased:
            reagent_conc = self.reagents[reagent]
            overshoot_volume = last_added_moles * 1000.0 / reagent_conc
            return True, max(overshoot_volume / 2, self.min_addition_volume)
        return False, None

    def step(self, action, mode='Simulate'):
        if self.done:
            return self.current_ph, 0.0, self.done, {"crossed_target": False}

        reagent, volume = action
        volume = float(volume)
        reagent_lower = reagent.lower()
        self.last_action_volume = volume

        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        if current_for_direction > self.target_ph and 'base' in reagent_lower:
            self.done = True
            return self.current_ph, -100.0, self.done, {"crossed_target": False}
        if current_for_direction < self.target_ph and 'acid' in reagent_lower:
            self.done = True
            return self.current_ph, -100.0, self.done, {"crossed_target": False}

        added_moles = self.reagents[reagent] * (volume / 1000.0)
        self.previous_ph = self.current_ph
        self.previous_total_volume = self.total_volume
        self.total_volume += volume

        if 'acid' in reagent_lower:
            self.acid_added_moles += added_moles
            self.acid_volume += volume
            self.last_acid_added = added_moles
        else:
            self.base_added_moles += added_moles
            self.base_volume += volume
            self.last_base_added = added_moles

        if mode == 'Simulate':
            new_pH = self.simulate_observed_ph()
            self.update_exp_ph(new_pH)
        else:
            raise ValueError("This benchmark cell only supports mode='Simulate'.")

        if abs(volume - self.min_addition_volume) < 1e-6 and self.previous_ph is not None:
            if (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
                self.oscillation_count += 1
                if self.oscillation_count >= 3:
                    self.use_secondary_reagents = True

        self.steps_taken += 1

        crossed_target = False
        if self.previous_ph is not None:
            crossed_target = (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0
            last_added = self.last_acid_added if 'acid' in reagent_lower else self.last_base_added
            overshoot_flag, new_thresh = self.detect_overshoot(self.previous_ph, self.current_ph, reagent, last_added)
            if overshoot_flag:
                self.overshoot_occurred = True
                self.overshoot_reagent = reagent
                if new_thresh is not None:
                    if self.overshoot_threshold is None or new_thresh < self.overshoot_threshold:
                        self.overshoot_threshold = new_thresh

        current_error = abs(self.current_ph - self.target_ph)
        if current_error < SUCCESS_THRESHOLD or self.steps_taken >= self.max_steps:
            self.done = True

        reward = -current_error
        return self.current_ph, reward, self.done, {"crossed_target": crossed_target}

    def select_best_action(self):
        def filter_by_global_threshold(candidates):
            if self.overshoot_threshold is not None:
                filtered = [a for a in candidates if a[1] <= self.overshoot_threshold]
                if filtered:
                    return filtered
            return candidates

        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph

        if self.use_secondary_reagents:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if 'base' in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents.keys() if 'acid 2' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'base 2' in r.lower()]
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents.keys() if 'dilute base 2' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'dilute acid 2' in r.lower()]
        else:
            if self.overshoot_occurred and self.overshoot_reagent is not None:
                if 'base' in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents.keys() if 'acid 1' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'base 1' in r.lower()]
                self.overshoot_occurred = False
                self.overshoot_reagent = None
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents.keys() if 'dilute base 1' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'dilute acid 1' in r.lower()]

        candidate_actions = [a for a in self.action_space if a[0] in allowed_reagent]
        candidate_actions = filter_by_global_threshold(candidate_actions)

        error = abs(current_for_direction - self.target_ph)
        ph_change = abs(current_for_direction - (self.prev_measured_ph if self.prev_measured_ph is not None else current_for_direction))
        bonus_factor = 1 + self.ph_rate_bonus_factor * (1 - min(ph_change, self.ph_rate_threshold) / self.ph_rate_threshold)

        avg_uncertainty = np.mean(self.pKa_std)
        uncertainty_factor = 1 - 0.1 * min(avg_uncertainty / 1.0, 1)

        buffer_mean = np.mean(self.buffer_total_moles)
        buffering_factor = 1.0 + 0.1 * (buffer_mean - 0.5)
        buffering_factor = np.clip(buffering_factor, 0.95, 1.05)

        alpha = self.vol_ideal_factor * bonus_factor * uncertainty_factor * buffering_factor
        required_vol = self.compute_required_volume()
        combined_value = error + 0.1 * required_vol

        min_vol = self.min_addition_volume
        max_vol = max(self.addition_volumes)
        ideal_volume = min_vol + (max_vol - min_vol) * np.tanh(alpha * combined_value)

        best_action = min(candidate_actions, key=lambda a: abs(a[1] - ideal_volume))
        return best_action, self.done

    def update_posteriors(self, action, observed_ph):
        sampled_pKa = np.random.normal(self.pKa_list, self.pKa_std, size=(self.num_particles, self.num_buffers))
        sampled_total_moles = np.random.normal(self.buffer_total_moles, self.buffer_total_std, size=(self.num_particles, self.num_buffers))

        effective_pKa = self.get_effective_pka_matrix(sampled_pKa)

        V_total = (TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
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


def load_experiment_conditions(csv_path):
    experiments = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acid_type = get_field(row, "Acid_Type", "Acid type")
            acid_params = parse_acid_params(get_field(row, "Acid_Params", "Acid params"))
            initial_ph = float(get_field(row, "Initial_pH", "Initial p h"))
            target_ph = float(get_field(row, "Target_pH", "Target p h"))
            experiments.append({
                "acid_type": acid_type,
                "acid_params": acid_params,
                "initial_ph": initial_ph,
                "target_ph": target_ph,
            })
    return experiments


def main():
    np.random.seed(555)

    input_csv = Path("ph4github") / "experiment_summary.csv"
    output_file = Path("ph4github") / "data" / "bayesian_online_update.txt"
    summary_file = Path("ph4github") / "data" / "bayesian_online_update_summary.csv"

    experiments = load_experiment_conditions(input_csv)
    success_count = 0
    steps_success = []
    total_steps = 0
    total_overshoots = 0

    start = time.time()

    with open(output_file, "w", encoding="utf-8", newline="\n") as log_file, \
         open(summary_file, "w", encoding="utf-8-sig", newline="") as summary_f:

        writer = csv.writer(summary_f)
        writer.writerow(["Experiment", "Acid_Type", "Acid_Params", "Initial_pH", "Target_pH", "Final_pH", "Steps_Taken", "Success"])

        for exp_id, exp in enumerate(experiments, 1):
            env = PHAdjustmentEnv(num_particles=1000)
            env.initialize(
                acid_type=exp["acid_type"],
                acid_params=exp["acid_params"],
                init_pH=exp["initial_ph"],
                target_pH=exp["target_ph"],
                max_steps=MAX_STEPS,
            )

            log_file.write(f"==== Experiment {exp_id} Start ====\n")
            log_file.write(f"Initial state: {np.round(env.get_state(), 2)}\n")
            log_file.write(f"Acid type: {env.acid_type}, parameters: {env.acid_params}, target pH: {env.target_ph:.2f}\n")
            log_file.write("Status Action Reagent Pair:\n")

            overshoot_count = 0
            action, _ = env.select_best_action()

            while not env.done:
                current_ph, reward, done, info = env.step(action, mode='Simulate')
                env.update_posteriors(action, current_ph)
                state_after = np.round(env.get_state(), 2)

                log_file.write(
                    f"  Step {env.steps_taken}: State = {state_after}, Action = {action[1]:.4f}, Reagent = {action[0]}"
                    + (" [overshoot]" if info.get("crossed_target", False) else "")
                    + "\n"
                )

                if info.get("crossed_target", False):
                    overshoot_count += 1

                if done:
                    break

                action, _ = env.select_best_action()

            success = abs(env.current_ph - env.target_ph) <= SUCCESS_THRESHOLD

            if success:
                success_count += 1
                steps_success.append(env.steps_taken)

            total_steps += env.steps_taken
            total_overshoots += overshoot_count

            acid_params_out = exp["acid_params"] if len(exp["acid_params"]) > 1 else f"{exp['acid_params'][0]:.2f}"
            writer.writerow([
                exp_id,
                env.acid_type,
                acid_params_out,
                f"{exp['initial_ph']:.2f}",
                f"{env.target_ph:.2f}",
                f"{env.current_ph:.2f}",
                env.steps_taken,
                "Yes" if success else "No"
            ])

            log_file.write(f"Conclusion: {'success' if success else 'fail'} | Final pH: {env.current_ph:.2f} | Steps: {env.steps_taken}\n")
            log_file.write("----------------------------------------\n\n")

            if exp_id % 50 == 0:
                elapsed = time.time() - start
                print(f"Completed {exp_id}/{len(experiments)} experiments in {elapsed:.1f} s")

        success_rate = success_count / len(experiments) * 100
        avg_steps = np.mean(steps_success) if steps_success else 0.0
        std_steps = np.std(steps_success, ddof=1) if len(steps_success) > 1 else 0.0
        overshoot_rate = total_overshoots / total_steps * 100 if total_steps > 0 else 0.0

        log_file.write("============================================================\n")
        log_file.write("Summary statistics\n")
        log_file.write("------------------------------------------------------------\n")
        log_file.write(f"Total experiments: {len(experiments)}\n")
        log_file.write(f"Successful experiments: {success_count}\n")
        log_file.write(f"Success rate: {success_rate:.2f}%\n")
        log_file.write(f"Successful steps: {avg_steps:.2f} +/- {std_steps:.2f}\n")
        log_file.write(f"Total steps: {total_steps}\n")
        log_file.write(f"Total overshoots: {total_overshoots}\n")
        log_file.write(f"Overshoot rate: {overshoot_rate:.2f}%\n")
        log_file.write("============================================================\n")

    print(f"Log written to: {output_file}")
    print(f"Summary CSV written to: {summary_file}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Successful steps: {avg_steps:.2f} +/- {std_steps:.2f}")
    print(f"Overshoot rate: {overshoot_rate:.2f}%")


if __name__ == "__main__":
    main()
