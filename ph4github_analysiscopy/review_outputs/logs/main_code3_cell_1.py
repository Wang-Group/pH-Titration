import numpy as np 
import math
import logging
from scipy.stats import norm
from scipy.optimize import brentq, newton

# Configuration log (INFO level)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Global tunable parameters
# -------------------------------
TITRATED_VOLUME = 11.0    # Volume of titrant (m l)
ANALYTE_CONC = 0.1        # Concentration of acid in titrated reagent (mol/l)

# The following are two different concentrations of hydrochloric acid and two different concentrations of sodium hydroxide that can be set by the user.
HCL_CONC1 = 0.1           # Hydrochloric acid 1 concentration (mol/l)
HCL_CONC2 = 0.01          # Hydrochloric acid 2 concentration (mol/l)
NAOH_CONC1 = 0.1          # Sodium hydroxide 1 concentration (mol/l)
NAOH_CONC2 = 0.01         # Sodium hydroxide 2 concentration (mol/l)
TARGET_PH = 11           # target pH
MAX_STEPS = 50            # Maximum number of steps

# -------------------------------
# Global reagent concentration dictionary (identified with "1" or "2")
# -------------------------------
REAGENTS = {
    'Dilute acid 1': HCL_CONC1,
    'Dilute acid 2': HCL_CONC2,
    'Dilute base 1': NAOH_CONC1,
    'Dilute base 2': NAOH_CONC2,
}

# -------------------------------
# pH calculation function (based on multiple buffer pairs)
# -------------------------------

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
def f(self, pH: float, c_A: float, c_Na: float, c_HCl: float, pKa_list: list) -> float:
    H = 10 ** (-pH)
    Kw = 1e-14
    OH = Kw / H
    acid_anion_charge = self.calculate_acid_anion_charge(c_A, H, pKa_list)
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

# -------------------------------
# pH adjustment environment
# -------------------------------
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
        
        # Minimum dripping volume (m l), and construct a dripping volume list
        self.min_addition_volume = 0.01  
        self.addition_volumes = [self.min_addition_volume * i for i in range(1, 1000)]
        self.action_space = [(reagent, volume) for reagent in self.reagents.keys() 
                             for volume in self.addition_volumes]
                
        self.epsilon = 0
        self.direction_penalty_factor = 60.0
        self.tol = 1e-4

        # Set the uncertain parameters of the buffer system: initial random pKa and total number of moles
        self.num_buffers = 3
        self.pKa_list = np.random.uniform(2, 6, size=self.num_buffers)
        # Use the initial sampled pKa value as a reference
        self.ref_pKa = np.copy(self.pKa_list)
        # Used to record the updated standard deviation of each buffer pair, initially set to 0.5
        self.pKa_std = np.full(self.num_buffers, 0.2)
        self.buffer_total_moles = np.random.uniform(1e-6, 0.5, size=self.num_buffers)
        
        self.initial_ph = None
        self.current_ph = None
        self.previous_ph = None
        self.target_ph = None
        self.max_steps = None

        # Initialize the prior distribution (assuming pKa ~ N(mean, 0.5) and total_moles ~ N(mean, 0.005))
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

    def initialize(self, init_pH: float, target_pH: float, max_steps: int, initial_volume: float = TITRATED_VOLUME) -> None:
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
        """
        Calculate dynamic weights based on the current pKa_list, ref_pKa and pKa_std, construct a valid pKa array,
        The length of the array is equal to the number of buffer pairs.
        """
        weight_max = 0.2
        k = 1.0
        pKa_eff_array = np.zeros(self.num_buffers)
        for i in range(self.num_buffers):
            weight_i = weight_max * (1 - np.tanh(k * self.pKa_std[i]))
            pKa_eff_array[i] = self.ref_pKa[i] + weight_i * (self.pKa_list[i] - self.ref_pKa[i])
        return pKa_eff_array

    def compute_required_volume(self) -> float:
        """
        Calculate the theoretical drip volume required from the current pH to the target pH and solve numerically using brentq.
        Choose to add an acid or a base based on the current situation and use the updated pKa mean to calculate the pH.
        """
        n_analyte = (TITRATED_VOLUME / 1000.0) * ANALYTE_CONC
        effective_pKa = self.get_effective_pka_array()

        if self.current_ph < self.target_ph:
            # The current system is too acidic and alkali needs to be added
            if self.use_secondary_reagents:
                reagent = 'Dilute base 2'
            else:
                reagent = 'Dilute base 1'
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

            try:
                x_req = brentq(f_vol, 0, 10)
            except Exception:
                x_req = 0.0
            return x_req
        else:
            # The current system is too alkaline and acid needs to be added.
            if self.use_secondary_reagents:
                reagent = 'Dilute acid 2'
            else:
                reagent = 'Dilute acid 1'
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

            try:
                x_req = brentq(f_vol, 0, 10)
            except Exception:
                x_req = 0.0
            return x_req

    def step(self, action: tuple, mode: str = 'Simulate') -> tuple:
        """
        mode parameter:
          -'simulate': automatically calculate the current pH (call recalc_ph calculation),
          -'manual': Prompts the user for pH value (interactive).
        """
        if self.done:
            return self.current_ph, 0, self.done, {}
        try:
            reagent, volume = action
            volume = float(volume)
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

            # Update measured pH based on mode selection
            if mode == 'Simulate':
                new_pH = self.recalc_ph()
                self.update_exp_ph(new_pH)
            elif mode == 'Manual':
                while True:
                    user_input = input("Please enter the currently measured pH value: ")
                    try:
                        manual_ph = float(user_input)
                        break
                    except ValueError:
                        print("Incorrect input format, please enter a number (e.g. 7.0).")
                self.update_exp_ph(manual_ph)

            # Detection of pH oscillations at minimum dosage
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

            # ---Modified ideal volume calculation part ---
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
            # --------------------------------------------

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
        """
        Use the parameters obtained from sampling to update the environment copy and calculate the pH.
        This reflects the impact of parameter changes on pH prediction.
        """
        env_copy = self.env_copy()
        env_copy.pKa_list = np.array(sampled_pKa)
        env_copy.buffer_total_moles = np.array(sampled_total_moles)
        new_ph = env_copy.recalc_ph()
        return new_ph

    def update_posteriors(self, action: tuple, observed_ph: float) -> None:
        """
        Bayesian update process (based on particle filtering):
          ① Sampling: Sample num_particles particles from the current prior;
          ② Prediction: For each particle, predict the pH after operation based on action;
          ③ Evaluation: Calculate the likelihood of each particle;
          ④ Resampling: obtain a new particle set based on likelihood resampling;
          ⑤ Statistical update: Use the resampled particles to update the prior distribution of the buffer pair.
        """
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
        new_ph, reward, done, _ = self.step(action, mode='Manual')
        self.update_posteriors(action, new_ph)
        next_action, _ = self.select_best_action()
        return next_action, done

# -------------------------------
# main program
# -------------------------------
def main():
    initial_volume = TITRATED_VOLUME

    REAGENTS['Dilute acid 1'] = HCL_CONC1
    REAGENTS['Dilute acid 2'] = HCL_CONC2
    REAGENTS['Dilute base 1'] = NAOH_CONC1
    REAGENTS['Dilute base 2'] = NAOH_CONC2

    # Manually set initial pH
    initial_ph = 6.9
    logging.info("Initial pH = %.2f", initial_ph)
    
    env = PHAdjustmentEnv()
    env.initialize(init_pH=initial_ph, target_pH=TARGET_PH, max_steps=MAX_STEPS, initial_volume=initial_volume)
    
    measured_ph = env.current_ph
    action, done = env.select_best_action()
    
    while not done:
        if abs(measured_ph - env.target_ph) < 0.1:
            break
        overshoot_msg = ""
        if env.overshoot_threshold is not None:
            overshoot_msg = "(Overshoot limit: the maximum dripping volume is {:.2f} M l）".format(env.overshoot_threshold)
        print("Current pH = {:.2f}, recommended operation: add {} {}".format(measured_ph, action, overshoot_msg))
        action, done = env.suggest_next_action(action, measured_ph)
        measured_ph = env.current_ph

    print("The experiment ends when the target pH is reached or the maximum number of steps is exceeded.")
    print("Total amount of added acid:{:.2f} mL, total alkali added amount:{:.2f} mL, total number of steps:{}, final pH = {:.2f}"
          .format(env.acid_volume, env.base_volume, env.steps_taken, measured_ph))

if __name__ == 'Main':
    main()
