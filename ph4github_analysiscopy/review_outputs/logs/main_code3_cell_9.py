import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from scipy.optimize import fsolve
import json

##############################################
# Fixed random seeds to ensure repeatable experiments
##############################################
seed = 255
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##############################################
# global constants
##############################################
TITRANT_CONC = 0.1          # Titrant concentration (0.1 M)
MAX_STEPS = 50              # Maximum number of steps
INITIAL_ACID_VOL = 11.0     # Initial volume of weak acid to be titrated (mL)
SUCCESS_THRESHOLD = 0.1     # pH error threshold

##############################################
# pH calculation function: unit acid
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

##############################################
# pH Calculation Function: Dibasic Acid
##############################################
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

##############################################
# pH Calculation Function: Tribasic Acid
##############################################
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
# Reward calculation function (modified version)
##############################################
def calculate_reward(previous_ph, current_ph, target_ph, steps_taken, max_steps, reagent, reward_config, SUCCESS_THRESHOLD, prev_overshoot_flag, prev_overshoot_volume, last_action_volume):
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    remaining_ratio = (max_steps - steps_taken) / max_steps
    dense_lambda = reward_config.get("Dense lambda", 1.0)
    dense_reward = dense_lambda * (previous_error - current_error) * (1 + remaining_ratio)
    step_penalty = reward_config.get("Step penalty", -0.005)
    overshoot_weight = reward_config.get("Overshoot weight", 0.2)
    overshoot_threshold = reward_config.get("Overshoot threshold", 0.1)
    
    # If overshoot occurs, the overshoot penalty is calculated
    if (previous_ph - target_ph) * (current_ph - target_ph) < 0 and max(previous_error, current_error) > overshoot_threshold:
        overshoot_magnitude = abs(current_ph - target_ph)
        overshoot_penalty = -overshoot_weight * (1 / (1 + math.exp(- (overshoot_magnitude - overshoot_threshold))))
    else:
        overshoot_penalty = 0
        
    wrong_dir_factor = reward_config.get("Wrong dir factor", 1.0)
    wrong_dir_penalty = 0
    if (current_ph > target_ph and 'Base' in reagent.lower()) or (current_ph < target_ph and 'Acid' in reagent.lower()):
        wrong_dir_penalty = -wrong_dir_factor * abs(current_ph - target_ph)
    
    # If an overshoot occurs in the previous "step", a negative penalty will be applied to the current action volume,
    # If the current action volume is smaller than the last overshoot, a certain positive reward will be given
    volume_penalty = 0
    volume_bonus = 0
    if prev_overshoot_flag and prev_overshoot_volume is not None:
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

    # Cut rewards in non-terminal state
    if not is_terminal:
        reward_clip_max = reward_config.get("Reward clip max", 4.0)
        reward_clip_min = reward_config.get("Reward clip min", -4.0)
        reward = max(min(raw_reward, reward_clip_max), reward_clip_min)
    else:
        reward = raw_reward

    return reward, is_terminal

##############################################
# pH simulation environment: PHSimEnv (modified version)
##############################################
class PHSimEnv:
    def __init__(self, initial_acid_vol=11.0, analyte_conc=0.1, titrant_conc=0.1):
        self.initial_acid_vol = initial_acid_vol  # M l
        self.analyte_conc = analyte_conc          # 0.1 M
        self.titrant_conc = titrant_conc          # 0.1 M
        self.n_acid = self.initial_acid_vol / 1000.0 * self.analyte_conc
        self.reward_config = {
            "Dense lambda": -0.03,
            "Step penalty": 0,
            "Terminal bonus": 3.9, 
            "Overshoot weight": 0.2,
            "Overshoot threshold": 0.1,
            "Wrong dir factor": 1,
            "Reward clip max": 4.1,
            "Reward clip min": -4.1,
            "Overshoot volume penalty": 0.1,
            "Overshoot volume bonus": 0.1
        }
        # Randomly generate 30 sets of acid parameters for different acid types
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
        # Randomly select acid type and parameters
        self.acid_type = random.choice(['Monoprotic', 'Diprotic', 'Triprotic'])
        if self.acid_type == 'Monoprotic':
            self.acid_params = float(np.random.choice(self.monoprotic_pKa_list))
        elif self.acid_type == 'Diprotic':
            self.acid_params = random.choice(self.diprotic_pKa_list)
        else:  # Triprotic
            self.acid_params = random.choice(self.triprotic_pKa_list)
        self.target_ph = round(random.uniform(2, 11), 2)
        self.acid_added_mL = 0.0
        self.base_added_mL = 0.0
        self.total_volume = self.initial_acid_vol
        self.last_action_volume = 0.0
        self.steps = 0
        # Initialization overshoot related flags
        self.prev_overshoot_flag = False
        self.prev_overshoot_volume = None
        # Initialize current pH based on acid type
        if self.acid_type == 'Monoprotic':
            self.current_ph = calculate_pH_monoprotic(0.0, 0.0, pKa=self.acid_params)
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = calculate_pH_diprotic(0.0, 0.0, pKa1, pKa2)
        else:  # Triprotic
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(0.0, 0.0, pKa1, pKa2, pKa3)
        # Initially set the previous state pH and current pH to be the same
        self.previous_ph = self.current_ph
        return self._get_state()

    def _get_state(self):
        pH_delta = round(self.current_ph - self.previous_ph, 2)
        error = round(self.current_ph - self.target_ph, 2)
        # State vector: current pH, target pH, pH change, error, last action volume
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def step(self, action):
        volume = float(action)
        self.last_action_volume = volume
        self.steps += 1
        # Choose to add a base or acid based on the relationship between current pH and target pH
        if self.current_ph < self.target_ph:
            reagent = "Strong base"
            self.base_added_mL += volume
        else:
            reagent = "Strong acid"
            self.acid_added_mL += volume
        self.total_volume = self.initial_acid_vol + self.base_added_mL + self.acid_added_mL

        # Save current pH as previous state
        self.previous_ph = self.current_ph

        # Update the current pH (call the corresponding function based on the acid type)
        if self.acid_type == 'Monoprotic':
            self.current_ph = calculate_pH_monoprotic(self.base_added_mL, self.acid_added_mL, self.acid_params)
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = calculate_pH_diprotic(self.base_added_mL, self.acid_added_mL, pKa1, pKa2)
        else:
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(self.base_added_mL, self.acid_added_mL, pKa1, pKa2, pKa3)

        state = self._get_state()
        # Calculate rewards using new reward function
        reward, done = calculate_reward(
            previous_ph=self.previous_ph,
            current_ph=self.current_ph,
            target_ph=self.target_ph,
            steps_taken=self.steps,
            max_steps=MAX_STEPS,
            reagent=reagent,
            reward_config=self.reward_config,
            SUCCESS_THRESHOLD=SUCCESS_THRESHOLD,
            prev_overshoot_flag=self.prev_overshoot_flag,
            prev_overshoot_volume=self.prev_overshoot_volume,
            last_action_volume=self.last_action_volume
        )
        
        # Determine whether overshoot occurs in this step (ie: pH crosses the target pH from one side)
        current_overshoot = (self.previous_ph - self.target_ph) * (self.current_ph - target_ph) < 0
        if current_overshoot:
            self.prev_overshoot_flag = True
            self.prev_overshoot_volume = self.last_action_volume
        else:
            self.prev_overshoot_flag = False
            self.prev_overshoot_volume = None
        
        return state, reward, done, {'Reagent': reagent}

##############################################
# Discrete action strategy model: DiscreteVolumeRegressor
##############################################
class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim=5, min_volume=0.01, max_volume=10.0, step=0.01):
        super(DiscreteVolumeRegressor, self).__init__()
        self.discrete_volumes = [round(min_volume + i * step, 2)
                                 for i in range(int((max_volume - min_volume) / step) + 1)]
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
        if torch.isnan(logits).any():
            print("Logits contain NaN:", logits)
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        volume = self.discrete_volumes[action_index.item()]
        return torch.tensor([[volume]], dtype=torch.float32), log_prob
    
    def predict_volume(self, x):
        logits = self.forward(x)
        _, predicted_index = torch.max(logits, dim=1)
        volume = self.discrete_volumes[predicted_index.item()]
        return torch.tensor([[volume]], dtype=torch.float32)

##############################################
# Online training: updating policy model using REINFORCE algorithm
##############################################
def train_reinforce(env, policy_model, optimizer, num_episodes=500, gamma=0.99):
    best_error = float('Inf')  # tracking best error
    best_model_state = None    # Store the best model state

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob = policy_model.sample_action(state_tensor)
            action_scalar = action.item()  # for passing in environment
            next_state, reward, done, _ = env.step(action_scalar)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()
        
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate the final error of the current episode
        current_error = abs(env.current_ph - env.target_ph)
        
        # If the current error is better than the optimal error, save the model
        if current_error < best_error:
            best_error = current_error
            best_model_state = policy_model.state_dict().copy()
            torch.save(best_model_state, "Volume regressor best big discrete new1 trained 1 test.pth")
            print(f"episode {episode}, Loss: {loss.item():.4f}, Updated Best Model with Error: {best_error:.4f}, Target pH: {env.target_ph:.2f}, Final pH: {env.current_ph:.2f}")
        elif episode % 50 == 0:  # Only print every 50 steps if not optimal
            total_reward = sum(rewards)
            print(f"episode {episode}, Loss: {loss.item():.4f}, Total Reward: {total_reward:.4f}, Target pH: {env.target_ph:.2f}, Final pH: {env.current_ph:.2f}")

##############################################
# Test function: run the experiment and print the details of each step
##############################################
def test_model(policy_model, env, num_experiments=10):
    for i in range(num_experiments):
        print(f"\n==== Experiment {i+1} Start ====")
        state = env.reset()
        print(f"Initial state: {state}")
        print(f"Acid type: {env.acid_type}, parameters: {env.acid_params}, target pH: {env.target_ph}")
        done = False
        steps = 0
        experiment_trace = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy_model.sample_action(state_tensor)
            action_scalar = action.item()
            state, reward, done, info = env.step(action_scalar)
            experiment_trace.append((state, action_scalar, info.get('Reagent', '')))
            steps += 1
        for j, (s, a, reagent) in enumerate(experiment_trace):
            print(f"  Step {j+1}: State = {s}, Action = {a:.4f}, Reagent = {reagent}")
        print(f"The experiment is over, the number of shared steps is: {steps}")

##############################################
# Main program: load the pre-trained model (if available) and train and test
##############################################
if __name__ == "Main":
    input_dim = 5
    learning_rate = 1e-4
    gamma = 0.99

    env = PHSimEnv(initial_acid_vol=INITIAL_ACID_VOL, analyte_conc=0.1, titrant_conc=TITRANT_CONC)
    policy_model = DiscreteVolumeRegressor(input_dim=input_dim, min_volume=0.01, max_volume=10.0, step=0.01)
    
    pretrained_path = "Volume regressor best big discrete new1 test.pth"
    try:
        state_dict = torch.load(pretrained_path, map_location=torch.device('Cpu'))
        policy_model.load_state_dict(state_dict)
        print("Loading the pre-trained model successfully.")
    except Exception as e:
        print("Failed to load pretrained model, using randomly initialized model.", e)
    
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    train_reinforce(env, policy_model, optimizer, num_episodes=500, gamma=gamma)
    # Save the last model at the end of training (optional)
    torch.save(policy_model.state_dict(), "Volume regressor best big discrete new1 trained 1 test.pth")
    print("Model saved.")
    
    test_model(policy_model, env, num_experiments=10)