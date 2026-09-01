# Source notebook: main_code3.ipynb
# Raw notebook cell index: 22
# Code-cell export index: 22
# First non-empty line: import numpy as np
import numpy as np
import math
import random
import logging
from scipy.stats import norm
from scipy.optimize import brentq
import sys
from io import StringIO

# Configuration log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global parameters (consistent with code 2)
TITRATED_VOLUME = 11.0
ANALYTE_CONC = 0.1
HCL_CONC1 = 0.1
HCL_CONC2 = 0.1
NAOH_CONC1 = 0.1
NAOH_CONC2 = 0.1
MAX_STEPS = 50

REAGENTS = {
    'Dilute acid 1': HCL_CONC1,
    'Dilute acid 2': HCL_CONC2,
    'Dilute base 1': NAOH_CONC1,
    'Dilute base 2': NAOH_CONC2,
}

# pH calculation function (directly reuse code 2)
def calculate_acid_anion_charge(c_A: float, H: float, pKa_list: list) -> float:
    n = len(pKa_list)
    K = [np.power(10, np.clip(-pKa, -100, 100)) for pKa in pKa_list]
    denominator = 1.0
    cumulative_K = 1.0
    for i in range(n):
        cumulative_K *= K[i]
        denominator += cumulative_K / np.power(H, i + 1, where=H != 0, out=np.array(np.inf))
    H_nA = c_A / denominator if denominator != 0 else 0.0
    anion_charge = 0.0
    cumulative_K = 1.0
    for k in range(1, n + 1):
        cumulative_K *= K[k - 1]
        anion_conc = H_nA * (cumulative_K / np.power(H, k, where=H != 0, out=np.array(np.inf)))
        anion_charge += k * anion_conc
    return anion_charge

class PHAdjustmentEnv:
    def __init__(self):
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
        self.reagents = REAGENTS.copy()
        self.min_addition_volume = 0.01
        self.addition_volumes = [self.min_addition_volume * i for i in range(1, 1000)]
        self.action_space = [(reagent, volume) for reagent in self.reagents.keys()
                             for volume in self.addition_volumes]
        self.epsilon = 0
        self.direction_penalty_factor = 60.0
        self.tol = 1e-4
        self.num_buffers = 3
        self.pKa_list = np.random.uniform(2, 6, size=self.num_buffers)
        self.ref_pKa = np.copy(self.pKa_list)
        self.pKa_std = np.full(self.num_buffers, 0.2)
        self.buffer_total_moles = np.random.uniform(1e-6, 0.5, size=self.num_buffers)
        self.initial_ph = None
        self.current_ph = None
        self.previous_ph = None
        self.target_ph = None
        self.max_steps = None
        self.priors = []
        for i in range(self.num_buffers):
            prior = {
                'P the': norm(loc=self.pKa_list[i], scale=0.5),
                'Total moles': norm(loc=self.buffer_total_moles[i], scale=0.005)
            }
            self.priors.append(prior)
        self.vol_ideal_factor = 0.2
        self.ph_rate_threshold = 1.0
        self.ph_rate_bonus_factor = 0.5
        self.last_measured_ph = None
        self.prev_measured_ph = None
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False
        self.acid_type = None
        self.acid_params = None

    def get_state(self):
        pH_delta = self.current_ph - self.previous_ph if self.previous_ph is not None else 0.0
        error = self.current_ph - self.target_ph
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume if hasattr(self, 'Last action volume') else 0.0], dtype=np.float32)

    def initialize(self, init_pH: float, target_pH: float, max_steps: int, initial_volume: float = TITRATED_VOLUME) -> None:
        self.acid_type = random.choice(["Monoprotic", "Diprotic", "Triprotic"])
        if self.acid_type == "Monoprotic":
            self.acid_params = random.uniform(2, 6)
        elif self.acid_type == "Diprotic":
            pKa1 = random.uniform(2, 4)
            pKa2 = random.uniform(4, 7)
            self.acid_params = [pKa1, pKa2]
        elif self.acid_type == "Triprotic":
            pKa1 = random.uniform(2, 4)
            pKa2 = random.uniform(4, 6)
            pKa3 = random.uniform(6, 8)
            self.acid_params = [pKa1, pKa2, pKa3]
        self.initial_ph = init_pH
        self.current_ph = init_pH
        self.previous_ph = init_pH
        self.target_ph = target_pH
        self.max_steps = max_steps
        self.steps_taken = 0
        self.done = False
        self.total_volume = initial_volume
        self.previous_total_volume = initial_volume
        self.acid_added_moles = 0.0
        self.base_added_moles = 0.0
        self.acid_volume = 0.0
        self.base_volume = 0.0
        self.last_measured_ph = init_pH
        self.prev_measured_ph = init_pH
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False

    def safe_pow10(self, x: float) -> float:
        return np.power(10, np.clip(x, -100, 100))

    def update_exp_ph(self, pH: float) -> None:
        if self.last_measured_ph is not None:
            self.prev_measured_ph = self.last_measured_ph
        else:
            self.prev_measured_ph = pH
        self.current_ph = pH
        self.last_measured_ph = pH

    def get_effective_pka_array(self) -> np.ndarray:
        weight_max = 0.2
        k = 1.0
        pKa_eff_array = np.zeros(self.num_buffers)
        for i in range(self.num_buffers):
            weight_i = weight_max * (1 - np.tanh(k * self.pKa_std[i]))
            pKa_eff_array[i] = self.ref_pKa[i] + weight_i * (self.pKa_list[i] - self.ref_pKa[i])
        return pKa_eff_array

    def compute_required_volume(self) -> float:
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
        effective_pKa = self.get_effective_pka_array()
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
                pH_new = self.solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
                return pH_new - self.target_ph
            try:
                x_req = brentq(f_vol, 0, 10)
            except Exception:
                x_req = 0.0
            return x_req
        else:
            reagent = 'Dilute acid 2' if self.use_secondary_reagents else 'Dilute acid 1'
            conc = self.reagents[reagent]
            def f_vol(x):
                add_moles = conc * (x / 1000.0)
                new_acid = self.acid_added_moles + add_moles
                new_total_volume = (TITRATED_VOLUME + self.acid_volume + self.base_volume + x) / 1000.0
                c_A_new = n_analyte / new_total_volume
                c_Na_new = self.base_added_moles / new_total_volume
                c_HCl_new = new_acid / new_total_volume
                pH_new = self.solve_pH(c_A_new, c_Na_new, c_HCl_new, effective_pKa)
                return pH_new - self.target_ph
            try:
                x_req = brentq(f_vol, 0, 10)
            except Exception:
                x_req = 0.0
            return x_req

    def f(self, pH: float, c_A: float, c_Na: float, c_HCl: float, pKa_list: list) -> float:
        H = 10 ** (-pH)
        Kw = 1e-14
        OH = Kw / H
        acid_anion_charge = calculate_acid_anion_charge(c_A, H, pKa_list)
        return H + c_Na - OH - acid_anion_charge - c_HCl

    def solve_pH(self, c_A: float, c_Na: float, c_HCl: float, pKa_list: list) -> float:
        lo, hi = 0.0, 14.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            f_mid = self.f(mid, c_A, c_Na, c_HCl, pKa_list)
            if abs(f_mid) < 1e-10:
                return mid
            if self.f(lo, c_A, c_Na, c_HCl, pKa_list) * f_mid < 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    def step(self, action: tuple, mode: str = 'Simulate') -> tuple:
        if self.done:
            return self.current_ph, 0, self.done, {}
        try:
            reagent, volume = action
            volume = float(volume)
            self.last_action_volume = volume
            added_moles = self.reagents[reagent] * (volume / 1000.0)
            self.previous_ph = self.current_ph
            self.previous_total_volume = self.total_volume
            self.total_volume += volume
            if 'Acid' in reagent.lower():
                self.acid_added_moles += added_moles
                self.acid_volume += volume
                self.last_acid_added = added_moles
            elif 'Base' in reagent.lower():
                self.base_added_moles += added_moles
                self.base_volume += volume
                self.last_base_added = added_moles
            current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
            if current_for_direction > self.target_ph and 'Base' in reagent.lower():
                return self.current_ph, -100, True, {}
            if current_for_direction < self.target_ph and 'Acid' in reagent.lower():
                return self.current_ph, -100, True, {}
            if mode == 'Simulate':
                new_pH = self.recalc_ph()
                self.update_exp_ph(new_pH)
            if self.previous_ph is not None and abs(volume - self.min_addition_volume) < 1e-6:
                if (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
                    self.oscillation_count += 1
                    logging.info("The pH oscillation at the minimum dripping amount was detected, the cumulative number of times:%d", self.oscillation_count)
                    if self.oscillation_count >= 3:
                        self.use_secondary_reagents = True
                        logging.info("When the continuous shaking threshold is reached, switch to secondary reagent titration.")
            self.steps_taken += 1
            if np.isnan(self.current_ph) or self.current_ph < 0 or self.current_ph > 14:
                self.done = True
                return self.current_ph, -100, self.done, {}
            error = abs(self.current_ph - self.target_ph)
            ph_change = abs(self.current_ph - (self.prev_measured_ph if self.prev_measured_ph is not None else self.current_ph))
            bonus_factor = 1 + self.ph_rate_bonus_factor * (1 - min(ph_change, self.ph_rate_threshold) / self.ph_rate_threshold)
            uncertainties = [prior['P the'].std() for prior in self.priors]
            avg_uncertainty = np.mean(uncertainties)
            max_uncertainty = 1.0
            uncertainty_factor = 1 - 0.1 * min(avg_uncertainty / max_uncertainty, 1)
            buffer_mean = np.mean(self.buffer_total_moles)
            ref_buffer = 0.5
            buffering_factor = 1.0 + 0.1 * (buffer_mean - ref_buffer)
            buffering_factor = np.clip(buffering_factor, 0.95, 1.05)
            alpha = self.vol_ideal_factor * bonus_factor * uncertainty_factor * buffering_factor
            required_vol = self.compute_required_volume()
            combined_value = error + 0.1 * required_vol
            min_vol = self.min_addition_volume
            max_vol = max(self.addition_volumes)
            ideal_volume = min_vol + (max_vol - min_vol) * np.tanh(alpha * combined_value)
            current_error = abs(self.current_ph - self.target_ph)
            error_reward = -current_error
            improvement = abs(self.previous_ph - self.target_ph) - current_error
            lambda_cost = 0.05
            action_cost = lambda_cost * ((volume - ideal_volume) ** 2)
            time_penalty = self.steps_taken * 0.1
            reward = improvement + error_reward - action_cost - time_penalty
            dynamic_direction_penalty = self.direction_penalty_factor * (0.5 if current_error > 2.0 else 1.0)
            if self.last_measured_ph is not None:
                current_for_direction = self.last_measured_ph
            if self.target_ph > current_for_direction and 'Acid' in reagent.lower():
                penalty = dynamic_direction_penalty * (self.target_ph - current_for_direction) / max(self.target_ph, 1)
                reward -= penalty
            if self.target_ph < current_for_direction and 'Base' in reagent.lower():
                penalty = dynamic_direction_penalty * (current_for_direction - self.target_ph) / max((14 - self.target_ph), 1)
                reward -= penalty
            if self.steps_taken > 0:
                if 'Acid' in reagent.lower():
                    reagent_conc = self.reagents[reagent]
                    last_added = self.last_acid_added
                elif 'Base' in reagent.lower():
                    reagent_conc = self.reagents[reagent]
                    last_added = self.last_base_added
                else:
                    reagent_conc = 1.0
                    last_added = 0.0
                overshoot_flag, new_thresh = self.detect_overshoot(self.previous_ph, self.current_ph,
                                                                   self.target_ph, reagent,
                                                                   last_added, reagent_conc,
                                                                   self.min_addition_volume)
                if overshoot_flag:
                    self.overshoot_occurred = True
                    self.overshoot_reagent = reagent
                    if new_thresh is not None:
                        if self.overshoot_threshold is None or new_thresh < self.overshoot_threshold:
                            self.overshoot_threshold = new_thresh
            if current_error < 0.1 or self.steps_taken >= self.max_steps:
                self.done = True
            return self.current_ph, reward, self.done, {}
        except Exception as e:
            logging.error("An exception occurs when executing step:%s", e)
            self.done = True
            return self.current_ph, -100, self.done, {}

    def detect_overshoot(self, prev_ph, current_ph, target_ph, reagent, last_added_moles, reagent_conc, min_addition):
        overshoot = False
        new_threshold = None
        sign_change = (prev_ph - target_ph) * (current_ph - target_ph) < 0
        error_increased = abs(current_ph - target_ph) > abs(prev_ph - target_ph)
        if sign_change or error_increased:
            overshoot = True
            overshoot_volume = last_added_moles * 1000.0 / reagent_conc
            new_threshold = max(overshoot_volume / 2, min_addition)
        return overshoot, new_threshold

    def env_copy(self) -> 'Ph adjustment env':
        env_copied = PHAdjustmentEnv()
        env_copied.total_volume = self.total_volume
        env_copied.previous_total_volume = self.previous_total_volume
        env_copied.acid_added_moles = self.acid_added_moles
        env_copied.base_added_moles = self.base_added_moles
        env_copied.acid_volume = self.acid_volume
        env_copied.base_volume = self.base_volume
        env_copied.current_ph = self.current_ph
        env_copied.previous_ph = self.previous_ph
        env_copied.target_ph = self.target_ph
        env_copied.steps_taken = self.steps_taken
        env_copied.done = self.done
        env_copied.num_buffers = self.num_buffers
        env_copied.pKa_list = np.copy(self.pKa_list)
        env_copied.buffer_total_moles = np.copy(self.buffer_total_moles)
        env_copied.priors = self.priors.copy()
        env_copied.epsilon = self.epsilon
        env_copied.direction_penalty_factor = self.direction_penalty_factor
        env_copied.tol = self.tol
        env_copied.reagents = self.reagents.copy()
        env_copied.addition_volumes = self.addition_volumes.copy()
        env_copied.action_space = self.action_space.copy()
        env_copied.max_steps = self.max_steps
        env_copied.vol_ideal_factor = self.vol_ideal_factor
        env_copied.ph_rate_threshold = self.ph_rate_threshold
        env_copied.ph_rate_bonus_factor = self.ph_rate_bonus_factor
        env_copied.last_measured_ph = self.last_measured_ph
        env_copied.prev_measured_ph = self.prev_measured_ph
        env_copied.overshoot_threshold = self.overshoot_threshold
        env_copied.oscillation_count = self.oscillation_count
        env_copied.use_secondary_reagents = self.use_secondary_reagents
        env_copied.ref_pKa = np.copy(self.ref_pKa)
        env_copied.pKa_std = np.copy(self.pKa_std)
        env_copied.acid_type = self.acid_type
        env_copied.acid_params = self.acid_params
        return env_copied

    def recalc_ph(self) -> float:
        V_total = (TITRATED_VOLUME + self.acid_volume + self.base_volume) / 1000.0
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
        c_A = n_analyte / V_total
        c_Na = self.base_added_moles / V_total
        c_HCl = self.acid_added_moles / V_total
        pKa_list = self.get_effective_pka_array().tolist()
        return self.solve_pH(c_A, c_Na, c_HCl, pKa_list)

    def select_best_action(self) -> tuple:
        def filter_by_global_threshold(candidates):
            if self.overshoot_threshold is not None:
                filtered = [a for a in candidates if a[1] <= self.overshoot_threshold]
                if filtered:
                    return filtered
            return candidates
        current_for_direction = self.last_measured_ph if self.last_measured_ph is not None else self.current_ph
        if self.use_secondary_reagents:
            if self.overshoot_occurred:
                if 'Base' in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents.keys() if 'Acid 2' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Base 2' in r.lower()]
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Dilute base 2' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Dilute acid 2' in r.lower()]
        else:
            if self.overshoot_occurred:
                if 'Base' in self.overshoot_reagent.lower():
                    allowed_reagent = [r for r in self.reagents.keys() if 'Acid 1' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Base 1' in r.lower()]
                self.overshoot_occurred = False
                self.overshoot_reagent = None
            else:
                if current_for_direction < self.target_ph:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Dilute base 1' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Dilute acid 1' in r.lower()]
        candidate_actions = [a for a in self.action_space if a[0] in allowed_reagent]
        candidate_actions = filter_by_global_threshold(candidate_actions)
        error = abs(current_for_direction - self.target_ph)
        ph_change = abs(current_for_direction - (self.prev_measured_ph if self.prev_measured_ph is not None else current_for_direction))
        bonus_factor = 1 + self.ph_rate_bonus_factor * (1 - min(ph_change, self.ph_rate_threshold) / self.ph_rate_threshold)
        uncertainties = [prior['P the'].std() for prior in self.priors]
        avg_uncertainty = np.mean(uncertainties)
        max_uncertainty = 1.0
        uncertainty_factor = 1 - 0.1 * min(avg_uncertainty / max_uncertainty, 1)
        buffer_mean = np.mean(self.buffer_total_moles)
        ref_buffer = 0.5
        buffering_factor = 1.0 + 0.1 * (buffer_mean - ref_buffer)
        buffering_factor = np.clip(buffering_factor, 0.95, 1.05)
        alpha = self.vol_ideal_factor * bonus_factor * uncertainty_factor * buffering_factor
        required_vol = self.compute_required_volume()
        combined_value = error + 0.1 * required_vol
        min_vol = self.min_addition_volume
        max_vol = max(self.addition_volumes)
        ideal_volume = min_vol + (max_vol - min_vol) * np.tanh(alpha * combined_value)
        best_action = min(candidate_actions, key=lambda a: abs(a[1] - ideal_volume))
        return best_action, self.done

    def sample_parameters(self) -> tuple:
        sampled_pKa = []
        sampled_total_moles = []
        for prior in self.priors:
            sampled_pKa.append(prior['P the'].rvs())
            sampled_total_moles.append(prior['Total moles'].rvs())
        return sampled_pKa, sampled_total_moles

    def predict_ph(self, action: tuple, sampled_pKa, sampled_total_moles) -> float:
        env_copy = self.env_copy()
        env_copy.pKa_list = np.array(sampled_pKa)
        env_copy.buffer_total_moles = np.array(sampled_total_moles)
        new_ph = env_copy.recalc_ph()
        return new_ph

    def update_posteriors(self, action: tuple, observed_ph: float) -> None:
        num_particles = 1000
        particles = []
        weights = []
        for _ in range(num_particles):
            sampled_pKa, sampled_total_moles = self.sample_parameters()
            predicted_ph = self.predict_ph(action, sampled_pKa, sampled_total_moles)
            likelihood = norm.pdf(observed_ph, loc=predicted_ph, scale=0.01)
            particles.append((sampled_pKa, sampled_total_moles))
            weights.append(likelihood)
        weights = np.array(weights) + 1e-10
        weights /= np.sum(weights)
        indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
        new_pKa = []
        new_total_moles = []
        new_pKa_std = []
        for i in range(self.num_buffers):
            pKa_samples = np.array([particles[idx][0][i] for idx in indices])
            total_moles_samples = np.array([particles[idx][1][i] for idx in indices])
            mean_pKa = np.mean(pKa_samples)
            std_pKa = np.std(pKa_samples) + 1e-3
            mean_total_moles = np.mean(total_moles_samples)
            std_total_moles = np.std(total_moles_samples) + 1e-3
            new_pKa.append((mean_pKa, std_pKa))
            new_total_moles.append((mean_total_moles, std_total_moles))
            new_pKa_std.append(std_pKa)
        for i in range(self.num_buffers):
            self.priors[i]['P the'] = norm(loc=new_pKa[i][0], scale=new_pKa[i][1])
            self.priors[i]['Total moles'] = norm(loc=new_total_moles[i][0], scale=new_total_moles[i][1])
            self.pKa_list[i] = new_pKa[i][0]
            self.buffer_total_moles[i] = new_total_moles[i][0]
            self.pKa_std[i] = new_pKa_std[i]

    def suggest_next_action(self, action: tuple, observed_ph: float) -> tuple:
        if abs(observed_ph - self.target_ph) < 0.1:
            self.done = True
            return None, True
        new_ph, reward, done, _ = self.step(action, mode='Simulate')
        self.update_posteriors(action, new_ph)
        next_action, _ = self.select_best_action()
        return next_action, done

# Helper function: Calculate initial pH (emulates the behavior of code 1)
def calculate_initial_ph(acid_type: str, acid_params, env: PHAdjustmentEnv):
    n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
    V_total = TITRATED_VOLUME / 1000.0
    c_A = n_analyte / V_total
    c_Na = 0.0
    c_HCl = 0.0
    if acid_type == "Monoprotic":
        pKa = [acid_params]
    elif acid_type == "Diprotic":
        pKa = acid_params
    elif acid_type == "Triprotic":
        pKa = acid_params
    else:
        return 7.0
    return env.solve_pH(c_A, c_Na, c_HCl, pKa)

# Custom class for outputting to console and file simultaneously
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Modified main program
def main():
    # Fixed random seed to ensure reproducibility (consistent with Code 1)
    seed = 555
    random.seed(seed)
    np.random.seed(seed)

    # Set output file
    output_file = open('Experiment output.txt', 'W', encoding='Utf 8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, output_file)

    num_experiments = 3000
    success_count = 0
    steps_success = []

    try:
        for exp in range(num_experiments):
            # Randomly generate target pH and initial pH
            target_ph = round(random.uniform(2, 11), 2)
            env = PHAdjustmentEnv()

            # Randomly select acid type and generate initial pH
            env.acid_type = random.choice(["Monoprotic", "Diprotic", "Triprotic"])
            if env.acid_type == "Monoprotic":
                env.acid_params = random.uniform(2, 6)
            elif env.acid_type == "Diprotic":
                pKa1 = random.uniform(2, 4)
                pKa2 = random.uniform(4, 7)
                env.acid_params = [pKa1, pKa2]
            elif env.acid_type == "Triprotic":
                pKa1 = random.uniform(2, 4)
                pKa2 = random.uniform(4, 6)
                pKa3 = random.uniform(6, 8)
                env.acid_params = [pKa1, pKa2, pKa3]
            initial_ph = calculate_initial_ph(env.acid_type, env.acid_params, env)

            # Initialize environment
            env.initialize(init_pH=initial_ph, target_pH=target_ph, max_steps=MAX_STEPS, initial_volume=TITRATED_VOLUME)

            print(f"==== Experiment {exp+1} Start ====")
            initial_state = env.get_state()
            print(f"Initial state: {np.round(initial_state, 2)}")
            print(f"Acid type: {env.acid_type}, parameters: {env.acid_params}, target pH: {env.target_ph}")
            print("Status Action Reagent pair:")

            trace = []
            action, _ = env.select_best_action()
            while not env.done:
                state_before = env.get_state()
                current_ph, reward, done, info = env.step(action, mode='Simulate')
                state_after = env.get_state()
                trace.append((state_after, action, action[0]))  # Record reagent name
                action, _ = env.select_best_action()

            for i, (s, a, reagent) in enumerate(trace, start=1):
                s_formatted = np.round(s, 2)
                print(f"  Step {i}: State = {s_formatted}, Action = {a[1]:.4f}, Reagent = {reagent}")
            print(f"The experiment is over, the number of shared steps is: {env.steps_taken}, final pH: {env.current_ph:.2f}\n")

            if abs(env.current_ph - env.target_ph) < 0.1:
                success_count += 1
                steps_success.append(env.steps_taken)

        success_rate = success_count / num_experiments * 100
        avg_steps = np.mean(steps_success) if steps_success else 0
        print("Total number of experiments: {}, number of successful experiments: {}, success rate: {:.2f}%, average number of steps for successful experiments: {:.2f}".format(
            num_experiments, success_count, success_rate, avg_steps))
    finally:
        # Restore original stdout and close file
        sys.stdout = original_stdout
        output_file.close()

if __name__ == 'Main':
    main()