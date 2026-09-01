# Source notebook: main_code3.ipynb
# Raw notebook cell index: 14
# Code-cell export index: 14
# First non-empty line: import torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from scipy.optimize import fsolve

# fixed random seed
seed = 555
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

##############################################
# global constants
##############################################
TITRANT_CONC = 0.1          # Titrant concentration (0.1 M)
MAX_STEPS = 50              # Maximum number of steps
INITIAL_ACID_VOL = 11.0     # Initial volume of weak acid to be titrated (mL)
SUCCESS_THRESHOLD = 0.1     # Ph error threshold

##############################################
# pH calculation function (mono, di, tribasic acids) -remains consistent with training
##############################################
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

def calculate_pH_monoprotic(base_added_mL: float, acid_added_mL: float, pKa: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1  
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    base_conc = TITRANT_CONC  
    n_Na = base_added_mL / 1000.0 * base_conc
    acid_added_conc = TITRANT_CONC  
    n_HCl = acid_added_mL / 1000.0 * acid_added_conc
    V_total = (acid_vol_mL + base_added_mL + acid_added_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = n_Na / V_total
    c_HCl = n_HCl / V_total
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

def calculate_pH_diprotic(base_added_mL: float, acid_added_mL: float, pKa1: float, pKa2: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    base_conc = TITRANT_CONC
    n_Na = base_added_mL / 1000.0 * base_conc
    acid_added_conc = TITRANT_CONC
    n_HCl = acid_added_mL / 1000.0 * acid_added_conc
    V_total = (acid_vol_mL + base_added_mL + acid_added_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = n_Na / V_total
    c_HCl = n_HCl / V_total
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

def calculate_pH_triprotic(base_added_mL: float, acid_added_mL: float, pKa1: float, pKa2: float, pKa3: float) -> float:
    acid_vol_mL = INITIAL_ACID_VOL
    acid_conc = 0.1
    n_acid = acid_vol_mL / 1000.0 * acid_conc
    base_conc = TITRANT_CONC
    n_Na = base_added_mL / 1000.0 * base_conc
    acid_added_conc = TITRANT_CONC
    n_HCl = acid_added_mL / 1000.0 * acid_added_conc
    V_total = (acid_vol_mL + base_added_mL + acid_added_mL) / 1000.0
    c_A = n_acid / V_total
    c_Na = n_Na / V_total
    c_HCl = n_HCl / V_total
    return round(solve_pH_triprotic(c_A, c_Na, c_HCl, pKa1, pKa2, pKa3), 2)

##############################################
# Environment and reward function
##############################################
def calculate_reward(previous_ph, current_ph, target_ph, steps_taken, max_steps, reagent, reward_config, SUCCESS_THRESHOLD,
                     prev_overshoot_flag=None, prev_overshoot_volume=None, last_action_volume=None):
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
        terminal_bonus = reward_config.get("Terminal bonus", 3.0) * bonus_factor
        raw_reward += terminal_bonus

    if not is_terminal:
        reward_clip_max = reward_config.get("Reward clip max", 4.0)
        reward_clip_min = reward_config.get("Reward clip min", -4.0)
        reward = max(min(raw_reward, reward_clip_max), reward_clip_min)
    else:
        reward = raw_reward

    return reward, is_terminal

class PHSimEnv:
    def __init__(self, initial_acid_vol=11.0, analyte_conc=0.1, titrant_conc=0.1):
        self.initial_acid_vol = initial_acid_vol
        self.analyte_conc = analyte_conc
        self.titrant_conc = titrant_conc
        self.n_acid = self.initial_acid_vol / 1000.0 * self.analyte_conc
        self.reward_config = {
            "Dense lambda": 1.0,
            "Step penalty": -0.005,
            "Terminal bonus": 80,
            "Overshoot weight": 0.2,
            "Overshoot threshold": 0.1,
            "Wrong dir factor": 1.0,
            "Reward clip max": 2.0,
            "Reward clip min": -2.0
        }
        self.monoprotic_pKa_list = np.random.uniform(2, 6, size=30)
        self.diprotic_pKa_list = []
        for _ in range(30):
            pKa1 = random.uniform(2, 4)
            pKa2 = random.uniform(4, 7)
            self.diprotic_pKa_list.append((pKa1, pKa2))
        self.triprotic_pKa_list = []
        for _ in range(30):
            pKa1 = random.uniform(2, 4)
            pKa2 = random.uniform(4, 6)
            pKa3 = random.uniform(6, 8)
            self.triprotic_pKa_list.append((pKa1, pKa2, pKa3))
        self.reset()

    def reset(self):
        self.acid_type = random.choice(['Monoprotic', 'Diprotic', 'Triprotic'])
        if self.acid_type == 'Monoprotic':
            self.acid_params = float(np.random.choice(self.monoprotic_pKa_list))
        elif self.acid_type == 'Diprotic':
            self.acid_params = random.choice(self.diprotic_pKa_list)
        elif self.acid_type == 'Triprotic':
            self.acid_params = random.choice(self.triprotic_pKa_list)
        # The original code here randomly generates the target pH, which will later be overwritten by user input.
        self.target_ph = round(random.uniform(2, 11), 2)
        self.acid_added_mL = 0.0
        self.base_added_mL = 0.0
        self.total_volume = self.initial_acid_vol
        self.last_action_volume = 0.0
        self.steps = 0
        self.prev_overshoot_flag = False
        self.prev_overshoot_volume = None
        if self.acid_type == 'Monoprotic':
            self.current_ph = calculate_pH_monoprotic(0.0, 0.0, pKa=self.acid_params)
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = calculate_pH_diprotic(0.0, 0.0, pKa1, pKa2)
        elif self.acid_type == 'Triprotic':
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(0.0, 0.0, pKa1, pKa2, pKa3)
        self.previous_ph = self.current_ph
        return self._get_state()

    def _get_state(self):
        pH_delta = round(self.current_ph - self.previous_ph, 2)
        error = round(self.current_ph - self.target_ph, 2)
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def step(self, action):
        # The step method here will update the status during automatic simulation.
        # But in manual experiments we use the user-entered pH to update the status,
        # Therefore this method is not called directly.
        pass

##############################################
# Discrete action policy model: action space [0.01, 10] mL, step size 0.01 mL, 1000 discrete actions in total
##############################################
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

##############################################
# Interactive manual titration experiments
# Description: First enter the initial pH and target pH; then the model will give suggested actions.
#        You enter the measured pH after working in the lab, and the status updates to continue giving recommendations.
##############################################
def interactive_titration_manual(env, policy_model):
    # Enter initial pH and target pH
    try:
        init_ph = float(input("Please enter initial pH value: "))
        target_ph = float(input("Please enter target pH value: "))
    except ValueError:
        print("Input format error, use environment default value.")
        init_ph = env.current_ph
        target_ph = env.target_ph

    # Reset environment and override initial pH and target pH
    state = env.reset()
    env.current_ph = init_ph
    env.previous_ph = init_ph
    env.target_ph = target_ph
    print(f"\nInitial pH: {env.current_ph:.2f}, target pH: {env.target_ph:.2f}\n")

    done = False
    while not done:
        # Print current status
        print(f"Current pH: {env.current_ph:.2f}")
        # Update state vector (with latest pH value)
        state = env._get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Based on the current status, the model gives a recommended adding volume
        with torch.no_grad():
            recommended_action, _ = policy_model.sample_action(state_tensor)
            recommended_volume = recommended_action.item()
        
        # Determine recommended reagents based on current pH and target pH (consistent with simulation environment logic)
        if env.current_ph < env.target_ph:
            recommended_reagent = "Strong base"
        else:
            recommended_reagent = "Strong acid"
        print(f"Suggestion: Join {recommended_volume:.2f} M l {recommended_reagent}")
        
        # Allow the user to choose whether to follow the recommendations directly or enter a custom volume
        user_choice = input("Is the recommended volume used? (Press enter directly to use, n enter a custom volume): ")
        if user_choice.strip().lower() == "N":
            try:
                action = float(input("Please enter the actual added volume (mL): "))
            except ValueError:
                print("Input error, use recommended value.")
                action = recommended_volume
        else:
            action = recommended_volume
        
        # Prompts the user to enter the measured pH value after working in the laboratory
        measured_ph = None
        while measured_ph is None:
            try:
                measured_ph = float(input("Please enter the pH value measured after the operation: "))
            except ValueError:
                print("Input format error, please enter a number.")
        
        # Update status: record the previous pH and update the current pH to the measured value entered by the user
        env.previous_ph = env.current_ph
        env.current_ph = measured_ph
        env.last_action_volume = action
        env.steps += 1
        
        # Update the cumulative amount of reagent added (according to the fixed logic in the simulation: if the previous pH is less than the target, add alkali, otherwise add acid)
        if env.previous_ph < env.target_ph:
            env.base_added_mL += action
            reagent_used = "Strong base"
        else:
            env.acid_added_mL += action
            reagent_used = "Strong acid"
        env.total_volume = env.initial_acid_vol + env.base_added_mL + env.acid_added_mL
        
        print(f"Action: Join {action:.2f} M l {reagent_used}, measured pH: {env.current_ph:.2f}\n")
        
        # Check whether the termination condition is met
        if abs(env.current_ph - env.target_ph) < SUCCESS_THRESHOLD:
            print("Target pH successfully achieved!")
            done = True
        elif env.steps >= MAX_STEPS:
            print("When the maximum number of steps is reached, the experiment ends.")
            done = True
                                
    print(f"The experiment is over and shared {env.steps} Step, final pH: {env.current_ph:.2f}")

##############################################
# Main program: Load the pre-trained model (if it exists) and enter interactive mode
##############################################
if __name__ == "Main":
    input_dim = 5
    learning_rate = 1e-3
    gamma = 0.99

    # Initialize environment
    env = PHSimEnv(initial_acid_vol=INITIAL_ACID_VOL, analyte_conc=0.1, titrant_conc=0.1)
    
    # Initialize the discrete action model
    policy_model = DiscreteVolumeRegressor(input_dim=input_dim, min_volume=0.01, max_volume=10.0, step=0.01)
    
    # Try to load the pre-trained model status (please keep the file name consistent)
    try:
        policy_model.load_state_dict(torch.load("Volume regressor best big discrete new1 trained 1 test.pth", map_location=torch.device('Cpu')))
        print("Loading the discrete pre-trained model was successful. \n")
    except Exception as e:
        print("Failed to load discrete pretrained model, using randomly initialized model. \n", e)
    
    policy_model.eval()
    
    # Enter interactive manual titration experiment mode
    interactive_titration_manual(env, policy_model)
