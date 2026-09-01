# Source notebook: main_code3.ipynb
# Raw notebook cell index: 18
# Code-cell export index: 18
# First non-empty line: import torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import logging
from scipy.optimize import fsolve
import csv
import ast

# Configuration log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# fixed random seed
seed = 555
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# global constants
TITRANT_CONC1 = 0.1         # Main titrant concentration (0.1 M)
TITRANT_CONC2 = 0.1        # Secondary titrant concentration (0.01 M)
MAX_STEPS = 50              # Maximum number of steps
INITIAL_ACID_VOL = 11.0     # Initial volume of weak acid to be titrated (mL)
SUCCESS_THRESHOLD = 0.1     # Ph error threshold
MIN_ADDITION_VOLUME = 0.01  # Minimum drop volume (mL)

REAGENTS = {
    'Strong base 1': TITRANT_CONC1,
    'Strong base 2': TITRANT_CONC2,
    'Strong acid 1': TITRANT_CONC1,
    'Strong acid 2': TITRANT_CONC2,
}

# pH calculation function
def f_monoprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa: float) -> float:
    H = 10 ** (-pH)
    Kw = 1e-14
    OH = Kw / H
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

def calculate_pH_monoprotic(base1_mL: float, base2_mL: float, acid1_mL: float, acid2_mL: float, pKa: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    V_total = (acid_vol_mL + base1_mL + base2_mL + acid1_mL + acid2_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_mL * TITRANT_CONC1 + base2_mL * TITRANT_CONC2) / 1000.0 / V_total
    c_HCl = (acid1_mL * TITRANT_CONC1 + acid2_mL * TITRANT_CONC2) / 1000.0 / V_total
    return round(solve_pH_monoprotic_balance(c_A, c_Na, c_HCl, pKa), 2)

def f_diprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float) -> float:
    H = 10 ** (-pH)
    Kw = 1e-14
    OH = Kw / H
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

def calculate_pH_diprotic(base1_mL: float, base2_mL: float, acid1_mL: float, acid2_mL: float, pKa1: float, pKa2: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    V_total = (acid_vol_mL + base1_mL + base2_mL + acid1_mL + acid2_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_mL * TITRANT_CONC1 + base2_mL * TITRANT_CONC2) / 1000.0 / V_total
    c_HCl = (acid1_mL * TITRANT_CONC1 + acid2_mL * TITRANT_CONC2) / 1000.0 / V_total
    return round(solve_pH_diprotic(c_A, c_Na, c_HCl, pKa1, pKa2), 2)

def f_triprotic(pH: float, c_A: float, c_Na: float, c_HCl: float, pKa1: float, pKa2: float, pKa3: float) -> float:
    H = 10 ** (-pH)
    Kw = 1e-14
    OH = Kw / H
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

def calculate_pH_triprotic(base1_mL: float, base2_mL: float, acid1_mL: float, acid2_mL: float, pKa1: float, pKa2: float, pKa3: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    V_total = (acid_vol_mL + base1_mL + base2_mL + acid1_mL + acid2_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = (base1_mL * TITRANT_CONC1 + base2_mL * TITRANT_CONC2) / 1000.0 / V_total
    c_HCl = (acid1_mL * TITRANT_CONC1 + acid2_mL * TITRANT_CONC2) / 1000.0 / V_total
    return round(solve_pH_triprotic(c_A, c_Na, c_HCl, pKa1, pKa2, pKa3), 2)

# Environment and reward function
def calculate_reward(previous_ph, current_ph, target_ph, steps_taken, max_steps, reagent, reward_config, SUCCESS_THRESHOLD, prev_overshoot_flag=None, prev_overshoot_volume=None, last_action_volume=None):
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    remaining_ratio = (max_steps - steps_taken) / max_steps
    dense_lambda = reward_config.get("Dense lambda", 1.0)
    dense_reward = dense_lambda * (previous_error - current_error) * (1 + remaining_ratio)
    step_penalty = reward_config.get("Step penalty", -0.005)
    overshoot_weight = reward_config.get("Overshoot weight", 0.2)
    overshoot_threshold = reward_config.get("Overshoot threshold", 0.1)
    
    if (previous_ph - target_ph) * (current_ph - target_ph) < 0 and max(previous_error, current_error) > overshoot_threshold:
        overshoot_magnitude = abs(current_ph - target_ph)
        overshoot_penalty = -overshoot_weight * (1 / (1 + math.exp(- (overshoot_magnitude - overshoot_threshold))))
    else:
        overshoot_penalty = 0
        
    wrong_dir_factor = reward_config.get("Wrong dir factor", 1.0)
    wrong_dir_penalty = 0
    if (current_ph > target_ph and 'Base' in reagent.lower()) or (current_ph < target_ph and 'Acid' in reagent.lower()):
        wrong_dir_penalty = -wrong_dir_factor * abs(current_ph - target_ph)
    
    volume_penalty = 0
    volume_bonus = 0
    if prev_overshoot_flag and prev_overshoot_volume is not None and last_action_volume is not None:
        overshoot_volume_penalty = reward_config.get("Overshoot volume penalty", 0.1)
        volume_penalty = -overshoot_volume_penalty * last_action_volume
        overshoot_volume_bonus = reward_config.get("Overshoot volume bonus", 0.1)
        if last_action_volume < prev_overshoot_volume:
            volume_bonus = overshoot_volume_bonus * (prev_overshoot_volume - last_action_volume)
    
    raw_reward = dense_reward + step_penalty + overshoot_penalty + wrong_dir_penalty + volume_penalty + volume_bonus

    is_terminal = False
    if abs(current_ph - target_ph) < SUCCESS_THRESHOLD or steps_taken >= max_steps:
        is_terminal = True
        bonus_factor = 2.0 if steps_taken < max_steps * 0.5 else 1.0
        terminal_bonus = reward_config.get("Terminal bonus", 80.0) * bonus_factor
        raw_reward += terminal_bonus

    if not is_terminal:
        reward_clip_max = reward_config.get("Reward clip max", 2.0)
        reward_clip_min = reward_config.get("Reward clip min", -2.0)
        reward = max(min(raw_reward, reward_clip_max), reward_clip_min)
    else:
        reward = raw_reward

    return reward, is_terminal

class PHSimEnv:
    def __init__(self, initial_acid_vol=11.0, analyte_conc=0.1):
        self.initial_acid_vol = initial_acid_vol
        self.analyte_conc = analyte_conc
        self.n_acid = self.initial_acid_vol / 1000.0 * self.analyte_conc
        self.reagents = REAGENTS.copy()
        self.min_addition_volume = MIN_ADDITION_VOLUME
        self.addition_volumes = [self.min_addition_volume * i for i in range(1, 1001)]
        self.action_space = [(reagent, volume) for reagent in self.reagents.keys() for volume in self.addition_volumes]
        self.reward_config = {
            "Dense lambda": 1.0,
            "Step penalty": -0.005,
            "Terminal bonus": 80,
            "Overshoot weight": 0.2,
            "Overshoot threshold": 0.1,
            "Wrong dir factor": 60.0,
            "Reward clip max": 2.0,
            "Reward clip min": -2.0,
            "Overshoot volume penalty": 0.1,
            "Overshoot volume bonus": 0.1
        }
        # Do not call reset during initialization, wait for test_model to provide CSV parameters
        self.acid_type = None
        self.acid_params = None
        self.target_ph = None
        self.current_ph = None

    def reset(self, acid_type=None, acid_params=None, target_ph=None, initial_ph=None):
        if acid_type is None or acid_params is None:
            raise ValueError("acid_type and acid_params must be provided from CSV data")
        
        self.acid_type = acid_type
        self.acid_params = acid_params
        self.target_ph = float(target_ph) if target_ph is not None else 7.0
        self.base1_added_mL = 0.0
        self.base2_added_mL = 0.0
        self.acid1_added_mL = 0.0
        self.acid2_added_mL = 0.0
        self.total_volume = self.initial_acid_vol
        self.last_action_volume = 0.0
        self.last_added_moles = 0.0
        self.steps = 0
        self.prev_overshoot_flag = False
        self.prev_overshoot_volume = None
        self.oscillation_count = 0
        self.use_secondary_reagents = False
        self.overshoot_threshold = None
        self.overshoot_occurred = False
        self.overshoot_reagent = None
        
        if self.acid_type == 'Monoprotic':
            self.current_ph = initial_ph if initial_ph is not None else calculate_pH_monoprotic(0.0, 0.0, 0.0, 0.0, float(self.acid_params))
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = initial_ph if initial_ph is not None else calculate_pH_diprotic(0.0, 0.0, 0.0, 0.0, pKa1, pKa2)
        elif self.acid_type == 'Triprotic':
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = initial_ph if initial_ph is not None else calculate_pH_triprotic(0.0, 0.0, 0.0, 0.0, pKa1, pKa2, pKa3)
        else:
            raise ValueError(f"Unknown acid_type: {self.acid_type}")
        
        self.previous_ph = self.current_ph
        self.last_measured_ph = self.current_ph
        self.prev_measured_ph = self.current_ph
        return self._get_state()

    def _get_state(self):
        pH_delta = round(self.current_ph - self.previous_ph, 2) if self.current_ph is not None and self.previous_ph is not None else 0.0
        error = round(self.current_ph - self.target_ph, 2) if self.current_ph is not None and self.target_ph is not None else 0.0
        return np.array([self.current_ph or 0.0, self.target_ph or 7.0, pH_delta, error, self.last_action_volume], dtype=np.float32)

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

    def step(self, action):
        reagent, volume = action
        volume = float(volume)
        self.last_action_volume = volume
        self.last_added_moles = self.reagents[reagent] * (volume / 1000.0)
        self.steps += 1
        self.previous_ph = self.current_ph
        self.prev_measured_ph = self.last_measured_ph
        if reagent == 'Strong base 1':
            self.base1_added_mL += volume
        elif reagent == 'Strong base 2':
            self.base2_added_mL += volume
        elif reagent == 'Strong acid 1':
            self.acid1_added_mL += volume
        elif reagent == 'Strong acid 2':
            self.acid2_added_mL += volume
        self.total_volume = self.initial_acid_vol + self.base1_added_mL + self.base2_added_mL + self.acid1_added_mL + self.acid2_added_mL
        if self.acid_type == 'Monoprotic':
            self.current_ph = calculate_pH_monoprotic(
                self.base1_added_mL, self.base2_added_mL, self.acid1_added_mL, self.acid2_added_mL, float(self.acid_params)
            )
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = calculate_pH_diprotic(
                self.base1_added_mL, self.base2_added_mL, self.acid1_added_mL, self.acid2_added_mL, pKa1, pKa2
            )
        elif self.acid_type == 'Triprotic':
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(
                self.base1_added_mL, self.base2_added_mL, self.acid1_added_mL, self.acid2_added_mL, pKa1, pKa2, pKa3
            )
        self.last_measured_ph = self.current_ph

        if self.previous_ph is not None and abs(volume - self.min_addition_volume) < 1e-6:
            if (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0 and abs(self.current_ph - self.previous_ph) > 0.1:
                self.oscillation_count += 1
                logging.info(f"PH oscillation at minimum dripping volume was detected, cumulative number of times:{self.oscillation_count}")
                if self.oscillation_count >= 3:
                    self.use_secondary_reagents = True
                    logging.info("When the continuous shaking threshold is reached, switch to secondary reagent titration.")

        overshoot_flag, new_threshold = self.detect_overshoot(
            self.previous_ph, self.current_ph, self.target_ph, reagent,
            self.last_added_moles, self.reagents[reagent], self.min_addition_volume
        )
        if overshoot_flag:
            self.overshoot_occurred = True
            self.overshoot_reagent = reagent
            if new_threshold is not None:
                if self.overshoot_threshold is None or new_threshold < self.overshoot_threshold:
                    self.overshoot_threshold = new_threshold

        state = self._get_state()
        reward, done = calculate_reward(
            self.previous_ph, self.current_ph, self.target_ph, self.steps, MAX_STEPS, reagent,
            self.reward_config, SUCCESS_THRESHOLD, self.overshoot_occurred, self.overshoot_threshold, volume
        )
        return state, reward, done, {'Reagent': reagent}

    def select_best_action(self, state_tensor, policy_model):
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
                    allowed_reagent = [r for r in self.reagents.keys() if 'Base 2' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Acid 2' in r.lower()]
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
                    allowed_reagent = [r for r in self.reagents.keys() if 'Base 1' in r.lower()]
                else:
                    allowed_reagent = [r for r in self.reagents.keys() if 'Acid 1' in r.lower()]
        
        candidate_actions = [a for a in self.action_space if a[0] in allowed_reagent]
        candidate_actions = filter_by_global_threshold(candidate_actions)
        
        logging.info(f"Candidate actions: {candidate_actions[:5]}... (common{len(candidate_actions)}indivual)")
        
        with torch.no_grad():
            logits = policy_model(state_tensor)
            candidate_indices = [self.addition_volumes.index(a[1]) for a in candidate_actions]
            candidate_logits = logits[0, candidate_indices]
            best_index = candidate_indices[candidate_logits.argmax().item()]
            best_action = candidate_actions[candidate_logits.argmax().item()]
        
        logging.info(f"Select action: {best_action}")
        return best_action

class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim=5, min_volume=0.01, max_volume=10.0, step=0.01):
        super(DiscreteVolumeRegressor, self).__init__()
        self.discrete_volumes = [round(min_volume + i * step, 2) for i in range(int((max_volume - min_volume) / step) + 1)]
        self.num_actions = len(self.discrete_volumes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def sample_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        volume = self.discrete_volumes[action_index.item()]
        return torch.tensor([[volume]], dtype=torch.float32), log_prob
    
    def predict_volume(self, x):
        logits = self.forward(x)
        _, predicted_indices = torch.max(logits, dim=1)
        predicted_volume = self.discrete_volumes[predicted_indices.item()]
        return torch.tensor([[predicted_volume]], dtype=torch.float32)

def load_experiment_conditions(csv_file):
    experiments = []
    with open(csv_file, 'R', encoding='Utf 8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            acid_type = row['Acid type']
            acid_params = ast.literal_eval(row['Acid params'])
            initial_ph = float(row['Initial p h'])
            target_ph = float(row['Target p h'])
            experiments.append({
                'Acid type': acid_type,
                'Acid params': acid_params,
                'Initial ph': initial_ph,
                'Target ph': target_ph
            })
    return experiments

def test_model(policy_model, csv_file="Experiment summary.csv", output_file="Test output2 modified.txt", summary_file="Experiment summary rl.csv"):
    experiments = load_experiment_conditions(csv_file)
    num_experiments = len(experiments)
    success_count = 0
    total_steps_success = []
    
    with open(output_file, 'W', encoding='Utf 8') as f, open(summary_file, 'W', newline='', encoding='Utf 8') as summary_f:
        def log_and_print(message):
            print(message)
            f.write(message + '\n')
        
        csv_writer = csv.writer(summary_f)
        csv_writer.writerow(['Experiment', 'Acid type', 'Acid params', 'Initial p h', 'Target p h', 'Final p h', 'Steps taken', 'Success'])
        
        for i, exp in enumerate(experiments, 1):
            log_and_print(f"\n==== Experiment {i} Start ====")
            acid_type = exp['Acid type']
            acid_params = exp['Acid params']
            initial_ph = exp['Initial ph']
            target_ph = exp['Target ph']
            state = env.reset(acid_type=acid_type, acid_params=acid_params, target_ph=target_ph, initial_ph=initial_ph)
            log_and_print(f"Initial state: {state}")
            log_and_print(f"Acid type: {acid_type}, parameters: {acid_params}, target pH: {target_ph}")
            done = False
            steps = 0
            experiment_trace = []
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                log_and_print(f"Current state vector: {state}")
                action = env.select_best_action(state_tensor, policy_model)
                state, reward, done, info = env.step(action)
                experiment_trace.append((state, action[1], info.get('Reagent', '')))
                steps += 1
            log_and_print("Status Action Reagent pair:")
            for j, (s, a, reagent) in enumerate(experiment_trace):
                log_and_print(f"  Step {j+1}: State = {s}, Action = {a:.4f}, Reagent = {reagent}")
            log_and_print(f"The experiment is over, the number of shared steps is: {steps}, final pH: {env.current_ph:.2f}")
            success = abs(env.current_ph - env.target_ph) < SUCCESS_THRESHOLD
            if success:
                success_count += 1
                total_steps_success.append(steps)
            
            acid_params_str = f"{acid_params}" if isinstance(acid_params, (list, tuple)) else f"{acid_params:.2f}"
            csv_writer.writerow([i, acid_type, acid_params_str, f"{initial_ph:.2f}", f"{target_ph:.2f}",
                                f"{env.current_ph:.2f}", steps, 'Yes' if success else 'No'])
        
        success_rate = success_count / num_experiments * 100
        avg_steps_success = np.mean(total_steps_success) if total_steps_success else 0
        summary_stats = f"\nTest completed: success rate = {success_rate:.2f}%, average number of steps for successful experiments = {avg_steps_success:.2f}"
        log_and_print(summary_stats)

if __name__ == "Main":
    input_dim = 5
    learning_rate = 1e-3
    gamma = 0.99

    env = PHSimEnv(initial_acid_vol=INITIAL_ACID_VOL, analyte_conc=0.1)
    
    policy_model = DiscreteVolumeRegressor(input_dim=input_dim, min_volume=0.01, max_volume=10.0, step=0.01)
    
    try:
        policy_model.load_state_dict(torch.load("Volume regressor best big discrete new1 test.pth", map_location=torch.device('Cpu')))
        print("Loading the discrete pre-trained model was successful.")
    except Exception as e:
        print("Failed to load discrete pretrained model, using randomly initialized model.", e)
    
    policy_model.eval()
    
    test_model(policy_model, csv_file="Experiment summary.csv", output_file="The same experiment of neural network before strengthening only uses concentrated acid and alkali.txt", summary_file="The same experiment of neural network before strengthening only uses concentrated acid and alkali.csv")