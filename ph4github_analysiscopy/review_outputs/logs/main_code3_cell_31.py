import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from scipy.optimize import fsolve
import json
import os

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
# Reward calculation function (supports ablation experiments)
##############################################
def calculate_reward(previous_ph, current_ph, target_ph, steps_taken, max_steps, reagent, reward_config, SUCCESS_THRESHOLD, prev_overshoot_flag, prev_overshoot_volume, last_action_volume, ablate_component=None):
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    remaining_ratio = (max_steps - steps_taken) / max_steps
    dense_lambda = reward_config.get("Dense lambda", 1.0)
    dense_reward = dense_lambda * (previous_error - current_error) * (1 + remaining_ratio) if ablate_component != "Dense reward" else 0
    step_penalty = reward_config.get("Step penalty", -0.005) if ablate_component != "Step penalty" else 0
    overshoot_weight = reward_config.get("Overshoot weight", 0.2)
    overshoot_threshold = reward_config.get("Overshoot threshold", 0.1)
    
    if (previous_ph - target_ph) * (current_ph - target_ph) < 0 and max(previous_error, current_error) > overshoot_threshold:
        overshoot_magnitude = abs(current_ph - target_ph)
        overshoot_penalty = -overshoot_weight * (1 / (1 + math.exp(- (overshoot_magnitude - overshoot_threshold)))) if ablate_component != "Overshoot penalty" else 0
    else:
        overshoot_penalty = 0
        
    wrong_dir_factor = reward_config.get("Wrong dir factor", 1.0)
    wrong_dir_penalty = 0
    if (current_ph > target_ph and 'Base' in reagent.lower()) or (current_ph < target_ph and 'Acid' in reagent.lower()):
        wrong_dir_penalty = -wrong_dir_factor * abs(current_ph - target_ph) if ablate_component != "Wrong dir penalty" else 0
    
    volume_penalty = 0
    volume_bonus = 0
    if prev_overshoot_flag and prev_overshoot_volume is not None:
        overshoot_volume_penalty = reward_config.get("Overshoot volume penalty", 0.1)
        volume_penalty = -overshoot_volume_penalty * last_action_volume if ablate_component != "Volume penalty" else 0
        overshoot_volume_bonus = reward_config.get("Overshoot volume bonus", 0.1)
        if last_action_volume < prev_overshoot_volume:
            volume_bonus = overshoot_volume_bonus * (prev_overshoot_volume - last_action_volume) if ablate_component != "Volume bonus" else 0
    
    raw_reward = dense_reward + step_penalty + overshoot_penalty + wrong_dir_penalty + volume_penalty + volume_bonus

    is_terminal = False
    if abs(current_ph - target_ph) < SUCCESS_THRESHOLD or steps_taken >= max_steps:
        is_terminal = True
        bonus_factor = 2.0 if steps_taken < max_steps * 0.5 else 1.0
        terminal_bonus = reward_config.get("Terminal bonus", 3.0) * bonus_factor if ablate_component != "Terminal bonus" else 0
        raw_reward += terminal_bonus

    if not is_terminal:
        reward_clip_max = reward_config.get("Reward clip max", 4.0)
        reward_clip_min = reward_config.get("Reward clip min", -4.0)
        reward = max(min(raw_reward, reward_clip_max), reward_clip_min)
    else:
        reward = raw_reward

    return reward, is_terminal

##############################################
# pH simulation environment: PHSimEnv
##############################################
class PHSimEnv:
    def __init__(self, initial_acid_vol=11.0, analyte_conc=0.1, titrant_conc=0.1):
        self.initial_acid_vol = initial_acid_vol
        self.analyte_conc = analyte_conc
        self.titrant_conc = titrant_conc
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
        self.monoprotic_pKa_list = np.random.uniform(2, 6, size=30)
        self.diprotic_pKa_list = [(random.uniform(2, 4), random.uniform(4, 7)) for _ in range(30)]
        self.triprotic_pKa_list = [(random.uniform(2, 4), random.uniform(4, 6), random.uniform(6, 8)) for _ in range(30)]
        self.reset()

    def reset(self, acid_type=None, acid_params=None, target_ph=None):
        if acid_type is None:
            self.acid_type = random.choice(['Monoprotic', 'Diprotic', 'Triprotic'])
        else:
            self.acid_type = acid_type
        
        if acid_params is None:
            if self.acid_type == 'Monoprotic':
                self.acid_params = float(np.random.choice(self.monoprotic_pKa_list))
            elif self.acid_type == 'Diprotic':
                self.acid_params = random.choice(self.diprotic_pKa_list)
            else:
                self.acid_params = random.choice(self.triprotic_pKa_list)
        else:
            self.acid_params = acid_params

        self.target_ph = round(random.uniform(2, 11), 2) if target_ph is None else target_ph
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
        else:
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(0.0, 0.0, pKa1, pKa2, pKa3)

        self.previous_ph = self.current_ph
        return self._get_state()

    def _get_state(self):
        pH_delta = round(self.current_ph - self.previous_ph, 2)
        error = round(self.current_ph - self.target_ph, 2)
        return np.array([self.current_ph, self.target_ph, pH_delta, error, self.last_action_volume], dtype=np.float32)

    def step(self, action, ablate_component=None):
        volume = float(action)
        self.last_action_volume = volume
        self.steps += 1
        if self.current_ph < self.target_ph:
            reagent = "Strong base"
            self.base_added_mL += volume
        else:
            reagent = "Strong acid"
            self.acid_added_mL += volume
        self.total_volume = self.initial_acid_vol + self.base_added_mL + self.acid_added_mL

        self.previous_ph = self.current_ph

        if self.acid_type == 'Monoprotic':
            self.current_ph = calculate_pH_monoprotic(self.base_added_mL, self.acid_added_mL, self.acid_params)
        elif self.acid_type == 'Diprotic':
            pKa1, pKa2 = self.acid_params
            self.current_ph = calculate_pH_diprotic(self.base_added_mL, self.acid_added_mL, pKa1, pKa2)
        else:
            pKa1, pKa2, pKa3 = self.acid_params
            self.current_ph = calculate_pH_triprotic(self.base_added_mL, self.acid_added_mL, pKa1, pKa2, pKa3)

        state = self._get_state()
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
            last_action_volume=self.last_action_volume,
            ablate_component=ablate_component
        )
        
        current_overshoot = (self.previous_ph - self.target_ph) * (self.current_ph - self.target_ph) < 0
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
def train_reinforce(env, policy_model, optimizer, num_episodes=200, gamma=0.99, ablate_component=None):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob = policy_model.sample_action(state_tensor)
            action_scalar = action.item()
            next_state, reward, done, _ = env.step(action_scalar, ablate_component=ablate_component)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
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
        
        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f"episode {episode}, Loss: {loss.item():.4f}, Total Reward: {total_reward:.4f}, Target pH: {env.target_ph:.2f}, Final pH: {env.current_ph:.2f}")

    model_path = f"volume regressor ablation no{ablate_component}200ep.pth" if ablate_component else "Volume regressor full 200ep.pth"
    torch.save(policy_model.state_dict(), model_path)
    print(f"The model has been saved to {model_path}")

##############################################
# Test function: run a fixed 200 experiments and count the success rate and average number of steps
##############################################
def test_model(policy_model, env, test_configs, ablate_component=None):
    success_count = 0
    success_steps = []
    
    for i, config in enumerate(test_configs):
        acid_type, acid_params, target_ph = config
        state = env.reset(acid_type=acid_type, acid_params=acid_params, target_ph=target_ph)
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy_model.sample_action(state_tensor)
            action_scalar = action.item()
            state, reward, done, info = env.step(action_scalar, ablate_component=ablate_component)
            steps += 1
        
        is_success = abs(env.current_ph - env.target_ph) < SUCCESS_THRESHOLD and steps <= MAX_STEPS
        if is_success:
            success_count += 1
            success_steps.append(steps)
    
    success_rate = success_count / len(test_configs) * 100
    avg_steps = np.mean(success_steps) if success_steps else 0.0
    
    print(f"\nTest results ({'No ' + ablate_component if ablate_component else 'Full Reward'}):")
    print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(test_configs)})")
    print(f"Average number of steps for successful experiments: {avg_steps:.2f}")
    
    return {"Success rate": success_rate, "Avg steps": avg_steps, "Success count": success_count, "Total experiments": len(test_configs)}

##############################################
# Generate fixed 200 test configurations
##############################################
def generate_test_configs(num_configs=200, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    monoprotic_pKa_list = np.random.uniform(2, 6, size=30)
    diprotic_pKa_list = [(random.uniform(2, 4), random.uniform(4, 7)) for _ in range(30)]
    triprotic_pKa_list = [(random.uniform(2, 4), random.uniform(4, 6), random.uniform(6, 8)) for _ in range(30)]
    
    configs = []
    for _ in range(num_configs):
        acid_type = random.choice(['Monoprotic', 'Diprotic', 'Triprotic'])
        if acid_type == 'Monoprotic':
            acid_params = float(np.random.choice(monoprotic_pKa_list))
        elif acid_type == 'Diprotic':
            acid_params = random.choice(diprotic_pKa_list)
        else:
            acid_params = random.choice(triprotic_pKa_list)
        target_ph = round(random.uniform(2, 11), 2)
        configs.append((acid_type, acid_params, target_ph))
    
    return configs

##############################################
# Main program: run ablation experiment and test
##############################################
if __name__ == "Main":
    input_dim = 5
    learning_rate = 1e-4
    gamma = 0.99
    num_episodes = 500
    num_test_experiments = 500

    reward_components = [
        "Dense reward",
        "Step penalty",
        "Overshoot penalty",
        "Wrong dir penalty",
        "Volume penalty",
        "Volume bonus",
        "Terminal bonus"
    ]

    env = PHSimEnv(initial_acid_vol=INITIAL_ACID_VOL, analyte_conc=0.1, titrant_conc=TITRANT_CONC)
    test_configs = generate_test_configs(num_configs=num_test_experiments, seed=123)
    
    results = {}
    
    print("\n=== Run training with full reward ===")
    policy_model = DiscreteVolumeRegressor(input_dim=input_dim, min_volume=0.01, max_volume=10.0, step=0.01)
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    train_reinforce(env, policy_model, optimizer, num_episodes=num_episodes, gamma=gamma, ablate_component=None)
    results["Full reward"] = test_model(policy_model, env, test_configs, ablate_component=None)

    for component in reward_components:
        print(f"\n=== Ablation Experiment: Removed {component} ===")
        policy_model = DiscreteVolumeRegressor(input_dim=input_dim, min_volume=0.01, max_volume=10.0, step=0.01)
        optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
        train_reinforce(env, policy_model, optimizer, num_episodes=num_episodes, gamma=gamma, ablate_component=component)
        results[f"no{component}"] = test_model(policy_model, env, test_configs, ablate_component=component)
    
    with open("Ablation test results.json", "W") as f:
        json.dump(results, f, indent=4)
    print("\nTest results have been saved to ablation_test_results.json")