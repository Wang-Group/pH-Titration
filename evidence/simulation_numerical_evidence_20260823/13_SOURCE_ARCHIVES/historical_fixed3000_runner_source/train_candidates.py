from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from benchmark_core import MAX_STEPS, MIN_VOLUME_ML, NeuralVolumePolicy, PolicyEnvironment, seed_everything
from challenge_common import (
    CategoricalActorCritic,
    DeterministicActor,
    GaussianActor,
    QNetwork,
    SCENARIOS,
    categorical_action,
    continuous_action,
    deterministic_action,
    feature_dim,
    initialize_from_imitation,
    make_features,
    make_task_pool,
    residual_action,
    reward_value,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    algorithm: str
    state_mode: str
    residual: bool
    robust: bool
    risk_aware: bool
    reward_profile: str = "standard"


CANDIDATES = {
    "ppo_nominal": Candidate("ppo_nominal", "ppo", "basic", False, False, False),
    "ppo_robust": Candidate("ppo_robust", "ppo", "basic", False, True, True),
    "a2c_robust": Candidate("a2c_robust", "a2c", "basic", False, True, True),
    "ppo_history_robust": Candidate("ppo_history_robust", "ppo", "history", False, True, True),
    "sac_history_robust": Candidate("sac_history_robust", "sac", "history", False, True, True),
    "ppo_residual_robust": Candidate("ppo_residual_robust", "ppo", "history", True, True, True),
    "ppo_filtered_robust": Candidate("ppo_filtered_robust", "ppo", "filtered", False, True, True),
    "ppo_conservative_robust": Candidate("ppo_conservative_robust", "ppo", "filtered", False, True, True, "conservative"),
    "td3_filtered_robust": Candidate("td3_filtered_robust", "td3", "filtered", False, True, True, "conservative"),
}


@dataclass
class Transition:
    state: np.ndarray
    action: int
    action_limit: int
    reward: float
    old_log_prob: float
    value: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.index
        self.states[i] = state
        self.actions[i, 0] = action
        self.rewards[i, 0] = reward
        self.next_states[i] = next_state
        self.dones[i, 0] = float(done)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], dtype=torch.float32, device=device)
            for array in (self.states, self.actions, self.rewards, self.next_states, self.dones)
        )


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    output = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        output.append(running)
    output.reverse()
    return torch.tensor(output, dtype=torch.float32)


def rollout_categorical(
    model: CategoricalActorCritic,
    candidate: Candidate,
    task,
    scenario,
    imitation: NeuralVolumePolicy,
    device: torch.device,
    rng_seed: int,
    stochastic: bool,
) -> tuple[list[Transition], dict]:
    env = PolicyEnvironment(task, scenario, np.random.default_rng(rng_seed))
    history: deque = deque(maxlen=3)
    transitions: list[Transition] = []
    while not env.done:
        raw_before = env.state().copy()
        maximum = env.maximum_requested_volume()
        base_volume = imitation.select_volume(raw_before, maximum) if candidate.residual else None
        features = make_features(env, candidate.state_mode, base_volume=base_volume, history=history)
        if candidate.residual:
            action, volume, log_prob, value = residual_action(model, features, float(base_volume), maximum, device, stochastic)
            action_limit = 9
        else:
            action, volume, log_prob, value = categorical_action(model, features, maximum, device, stochastic)
            action_limit = max(1, min(1000, int(math.floor(maximum / 0.01 + 1e-9))))
        previous_error = abs(env.true_ph - env.target_ph)
        overshoots_before = env.overshoots
        env.step(volume)
        current_error = abs(env.true_ph - env.target_ph)
        reward = reward_value(
            previous_error,
            current_error,
            volume,
            env.overshoots > overshoots_before,
            env.done,
            current_error <= 0.1,
            abs(env.measured_ph - env.target_ph) <= 0.1,
            env.steps,
            candidate.risk_aware,
            candidate.reward_profile,
        )
        transitions.append(Transition(features, action, action_limit, reward, log_prob, value, env.done))
        history.append(raw_before)
    return transitions, {
        "success": abs(env.true_ph - env.target_ph) <= 0.1,
        "steps": env.steps,
        "overshoots": env.overshoots,
        "volume": env.acid_added_ml + env.base_added_ml,
        "final_error": abs(env.true_ph - env.target_ph),
        "false_stop": abs(env.measured_ph - env.target_ph) <= 0.1 and abs(env.true_ph - env.target_ph) > 0.1,
    }


def ppo_update(model, optimizer, batch: list[Transition], gamma: float, device: torch.device, epochs: int) -> None:
    episode_parts: list[list[Transition]] = []
    current: list[Transition] = []
    for item in batch:
        current.append(item)
        if item.done:
            episode_parts.append(current)
            current = []
    if current:
        episode_parts.append(current)
    returns = torch.cat([discounted_returns([x.reward for x in episode], gamma) for episode in episode_parts]).to(device)
    old_values = torch.tensor([x.value for x in batch], dtype=torch.float32, device=device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = np.arange(len(batch))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 256):
            selected = indices[start:start + 256]
            actor_loss = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)
            entropy = torch.tensor(0.0, device=device)
            for index in selected:
                item = batch[int(index)]
                state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.tensor([item.action], device=device)
                logits = model.actor(state)[:, : item.action_limit]
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_prob = distribution.log_prob(action).squeeze()
                ratio = torch.exp(new_log_prob - item.old_log_prob)
                advantage = advantages[int(index)]
                clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
                actor_loss = actor_loss - torch.minimum(ratio * advantage, clipped)
                critic_loss = critic_loss + (model.critic(state).squeeze() - returns[int(index)]).pow(2)
                entropy = entropy + distribution.entropy().squeeze()
            count = max(1, len(selected))
            loss = actor_loss / count + 0.5 * critic_loss / count - 0.01 * entropy / count
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def a2c_update(model, optimizer, trajectory: list[Transition], gamma: float, device: torch.device) -> None:
    returns = discounted_returns([x.reward for x in trajectory], gamma).to(device)
    losses = []
    for item, target in zip(trajectory, returns):
        state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([item.action], device=device)
        distribution = torch.distributions.Categorical(logits=model.actor(state)[:, : item.action_limit])
        value = model.critic(state).squeeze()
        advantage = target - value.detach()
        losses.append(-distribution.log_prob(action).squeeze() * advantage + 0.5 * (value - target).pow(2) - 0.01 * distribution.entropy().squeeze())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def sample_sac(actor: GaussianActor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = actor(states)
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = normal.rsample()
    squashed = torch.tanh(raw)
    fraction = (squashed + 1.0) / 2.0
    log_prob = normal.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + 1e-6)
    return fraction, log_prob.sum(dim=-1, keepdim=True)


def evaluate_categorical(model, candidate, imitation, device, seed: int, tasks_per_scenario: int) -> dict:
    names = ["nominal"] if not candidate.robust else ["nominal", "analyte_high", "noise_005", "partial_response", "tetraprotic"]
    metrics = []
    model.eval()
    with torch.no_grad():
        for offset, name in enumerate(names):
            scenario = SCENARIOS[name]
            tasks = make_task_pool(seed + 5000 + offset, tasks_per_scenario, False) if name == "nominal" else [
                (task, scenario) for task in __import__("benchmark_core").generate_tasks(seed + 5000 + offset, tasks_per_scenario, scenario)
            ]
            for task, used_scenario in tasks:
                _, result = rollout_categorical(model, candidate, task, used_scenario, imitation, device, seed * 1_000_003 + task.task_id, False)
                metrics.append(result)
    model.train()
    success = [x for x in metrics if x["success"]]
    return {
        "success_rate_percent": 100.0 * len(success) / max(1, len(metrics)),
        "successful_steps_mean": float(np.mean([x["steps"] for x in success])) if success else math.nan,
        "overshoot_rate_percent": 100.0 * sum(x["overshoots"] for x in metrics) / max(1, sum(x["steps"] for x in metrics)),
        "total_volume_mean": float(np.mean([x["volume"] for x in metrics])),
        "final_error_mean": float(np.mean([x["final_error"] for x in metrics])),
        "false_stop_percent": 100.0 * sum(x["false_stop"] for x in metrics) / max(1, len(metrics)),
    }


def evaluate_sac(actor, candidate, device, seed: int, tasks_per_scenario: int) -> dict:
    names = ["nominal", "analyte_high", "noise_005", "partial_response", "tetraprotic"]
    metrics = []
    actor.eval()
    with torch.no_grad():
        for offset, name in enumerate(names):
            scenario = SCENARIOS[name]
            tasks = __import__("benchmark_core").generate_tasks(seed + 7000 + offset, tasks_per_scenario, scenario)
            for task in tasks:
                env = PolicyEnvironment(task, scenario, np.random.default_rng(seed * 1_000_003 + task.task_id))
                history: deque = deque(maxlen=3)
                while not env.done:
                    raw = env.state().copy()
                    features = make_features(env, candidate.state_mode, history=history)
                    volume, _ = continuous_action(actor, features, env.maximum_requested_volume(), device, False)
                    env.step(volume)
                    history.append(raw)
                metrics.append({
                    "success": abs(env.true_ph - env.target_ph) <= 0.1,
                    "steps": env.steps,
                    "overshoots": env.overshoots,
                    "volume": env.acid_added_ml + env.base_added_ml,
                    "final_error": abs(env.true_ph - env.target_ph),
                    "false_stop": abs(env.measured_ph - env.target_ph) <= 0.1 and abs(env.true_ph - env.target_ph) > 0.1,
                })
    actor.train()
    success = [x for x in metrics if x["success"]]
    return {
        "success_rate_percent": 100.0 * len(success) / max(1, len(metrics)),
        "successful_steps_mean": float(np.mean([x["steps"] for x in success])) if success else math.nan,
        "overshoot_rate_percent": 100.0 * sum(x["overshoots"] for x in metrics) / max(1, sum(x["steps"] for x in metrics)),
        "total_volume_mean": float(np.mean([x["volume"] for x in metrics])),
        "final_error_mean": float(np.mean([x["final_error"] for x in metrics])),
        "false_stop_percent": 100.0 * sum(x["false_stop"] for x in metrics) / max(1, len(metrics)),
    }


def evaluate_td3(actor, candidate, device, seed: int, tasks_per_scenario: int) -> dict:
    names = ["nominal", "analyte_high", "noise_005", "partial_response", "tetraprotic"]
    metrics = []
    actor.eval()
    with torch.no_grad():
        for offset, name in enumerate(names):
            scenario = SCENARIOS[name]
            tasks = __import__("benchmark_core").generate_tasks(seed + 9000 + offset, tasks_per_scenario, scenario)
            for task in tasks:
                env = PolicyEnvironment(task, scenario, np.random.default_rng(seed * 1_000_003 + task.task_id))
                history: deque = deque(maxlen=3)
                while not env.done:
                    raw = env.state().copy()
                    features = make_features(env, candidate.state_mode, history=history)
                    volume, _ = deterministic_action(actor, features, env.maximum_requested_volume(), device)
                    env.step(volume)
                    history.append(raw)
                metrics.append({
                    "success": abs(env.true_ph - env.target_ph) <= 0.1,
                    "steps": env.steps,
                    "overshoots": env.overshoots,
                    "volume": env.acid_added_ml + env.base_added_ml,
                    "final_error": abs(env.true_ph - env.target_ph),
                    "false_stop": abs(env.measured_ph - env.target_ph) <= 0.1 and abs(env.true_ph - env.target_ph) > 0.1,
                })
    actor.train()
    success = [x for x in metrics if x["success"]]
    return {
        "success_rate_percent": 100.0 * len(success) / max(1, len(metrics)),
        "successful_steps_mean": float(np.mean([x["steps"] for x in success])) if success else math.nan,
        "overshoot_rate_percent": 100.0 * sum(x["overshoots"] for x in metrics) / max(1, sum(x["steps"] for x in metrics)),
        "total_volume_mean": float(np.mean([x["volume"] for x in metrics])),
        "final_error_mean": float(np.mean([x["final_error"] for x in metrics])),
        "false_stop_percent": 100.0 * sum(x["false_stop"] for x in metrics) / max(1, len(metrics)),
    }


def train_categorical(candidate, seed, args, device, imitation):
    seed_everything(seed)
    input_dim = feature_dim(candidate.state_mode, candidate.residual)
    action_dim = 9 if candidate.residual else 1000
    model = CategoricalActorCritic(input_dim, action_dim).to(device)
    if not candidate.residual:
        initialize_from_imitation(model, args.imitation_weights, device)
    lr = args.learning_rate if candidate.algorithm == "ppo" else args.a2c_learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pool = make_task_pool(seed + 100_000, args.training_pool_size, candidate.robust)
    interactions = 0
    episode = 0
    next_eval = args.eval_interval
    ppo_batch: list[Transition] = []
    curves = [{"environment_steps": 0, **evaluate_categorical(model, candidate, imitation, device, seed, args.eval_tasks)}]
    while interactions < args.train_steps:
        task, scenario = pool[episode % len(pool)]
        trajectory, _ = rollout_categorical(model, candidate, task, scenario, imitation, device, seed * 10_000_019 + episode, True)
        interactions += len(trajectory)
        episode += 1
        if not trajectory:
            continue
        if candidate.algorithm == "a2c":
            a2c_update(model, optimizer, trajectory, args.gamma, device)
        else:
            ppo_batch.extend(trajectory)
            if len(ppo_batch) >= args.ppo_batch_steps:
                ppo_update(model, optimizer, ppo_batch, args.gamma, device, args.ppo_epochs)
                ppo_batch = []
        if interactions >= next_eval or interactions >= args.train_steps:
            point = {"environment_steps": interactions, **evaluate_categorical(model, candidate, imitation, device, seed, args.eval_tasks)}
            curves.append(point)
            print(f"{candidate.name}/seed{seed}: {interactions} steps, validation success={point['success_rate_percent']:.2f}%")
            next_eval += args.eval_interval
    if ppo_batch:
        ppo_update(model, optimizer, ppo_batch, args.gamma, device, args.ppo_epochs)
    return model, curves


def train_sac(candidate, seed, args, device):
    seed_everything(seed)
    state_dim = feature_dim(candidate.state_mode, False)
    actor = GaussianActor(state_dim).to(device)
    q1 = QNetwork(state_dim).to(device)
    q2 = QNetwork(state_dim).to(device)
    target1 = QNetwork(state_dim).to(device)
    target2 = QNetwork(state_dim).to(device)
    target1.load_state_dict(q1.state_dict())
    target2.load_state_dict(q2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=args.sac_learning_rate)
    q1_opt = optim.Adam(q1.parameters(), lr=args.sac_learning_rate)
    q2_opt = optim.Adam(q2.parameters(), lr=args.sac_learning_rate)
    replay = ReplayBuffer(args.replay_size, state_dim)
    pool = make_task_pool(seed + 100_000, args.training_pool_size, True)
    interactions = 0
    episode = 0
    next_eval = args.eval_interval
    curves = [{"environment_steps": 0, **evaluate_sac(actor, candidate, device, seed, args.eval_tasks)}]
    while interactions < args.train_steps:
        task, scenario = pool[episode % len(pool)]
        env = PolicyEnvironment(task, scenario, np.random.default_rng(seed * 10_000_019 + episode))
        history: deque = deque(maxlen=3)
        while not env.done and interactions < args.train_steps:
            raw = env.state().copy()
            state = make_features(env, candidate.state_mode, history=history)
            maximum = env.maximum_requested_volume()
            if interactions < args.sac_warmup:
                fraction = float(np.random.random())
                volume = MIN_VOLUME_ML + (maximum - MIN_VOLUME_ML) * fraction
            else:
                volume, _ = continuous_action(actor, state, maximum, device, True)
                fraction = (volume - MIN_VOLUME_ML) / max(1e-9, maximum - MIN_VOLUME_ML)
            previous_error = abs(env.true_ph - env.target_ph)
            overshoots_before = env.overshoots
            env.step(volume)
            history.append(raw)
            next_state = make_features(env, candidate.state_mode, history=history)
            current_error = abs(env.true_ph - env.target_ph)
            reward = reward_value(previous_error, current_error, volume, env.overshoots > overshoots_before, env.done, current_error <= 0.1, abs(env.measured_ph - env.target_ph) <= 0.1, env.steps, True, candidate.reward_profile)
            replay.add(state, fraction, reward, next_state, env.done)
            interactions += 1
            if replay.size >= args.sac_batch_size:
                states, actions, rewards, next_states, dones = replay.sample(args.sac_batch_size, device)
                with torch.no_grad():
                    next_actions, next_log_prob = sample_sac(actor, next_states)
                    target_q = torch.minimum(target1(next_states, next_actions), target2(next_states, next_actions)) - args.sac_alpha * next_log_prob
                    target = rewards + args.gamma * (1.0 - dones) * target_q
                q1_loss = F.mse_loss(q1(states, actions), target)
                q2_loss = F.mse_loss(q2(states, actions), target)
                q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
                q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()
                new_actions, log_prob = sample_sac(actor, states)
                actor_loss = (args.sac_alpha * log_prob - torch.minimum(q1(states, new_actions), q2(states, new_actions))).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                with torch.no_grad():
                    for target_param, param in zip(target1.parameters(), q1.parameters()):
                        target_param.mul_(1.0 - args.sac_tau).add_(args.sac_tau * param)
                    for target_param, param in zip(target2.parameters(), q2.parameters()):
                        target_param.mul_(1.0 - args.sac_tau).add_(args.sac_tau * param)
            if interactions >= next_eval or interactions >= args.train_steps:
                point = {"environment_steps": interactions, **evaluate_sac(actor, candidate, device, seed, args.eval_tasks)}
                curves.append(point)
                print(f"{candidate.name}/seed{seed}: {interactions} steps, validation success={point['success_rate_percent']:.2f}%")
                next_eval += args.eval_interval
        episode += 1
    return actor, curves


def train_td3(candidate, seed, args, device):
    seed_everything(seed)
    state_dim = feature_dim(candidate.state_mode, False)
    actor = DeterministicActor(state_dim).to(device)
    target_actor = DeterministicActor(state_dim).to(device)
    q1 = QNetwork(state_dim).to(device)
    q2 = QNetwork(state_dim).to(device)
    target1 = QNetwork(state_dim).to(device)
    target2 = QNetwork(state_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target1.load_state_dict(q1.state_dict())
    target2.load_state_dict(q2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=args.td3_learning_rate)
    q1_opt = optim.Adam(q1.parameters(), lr=args.td3_learning_rate)
    q2_opt = optim.Adam(q2.parameters(), lr=args.td3_learning_rate)
    replay = ReplayBuffer(args.replay_size, state_dim)
    pool = make_task_pool(seed + 100_000, args.training_pool_size, True)
    interactions = 0
    episode = 0
    updates = 0
    next_eval = args.eval_interval
    curves = [{"environment_steps": 0, **evaluate_td3(actor, candidate, device, seed, args.eval_tasks)}]
    while interactions < args.train_steps:
        task, scenario = pool[episode % len(pool)]
        env = PolicyEnvironment(task, scenario, np.random.default_rng(seed * 10_000_019 + episode))
        history: deque = deque(maxlen=3)
        while not env.done and interactions < args.train_steps:
            raw = env.state().copy()
            state = make_features(env, candidate.state_mode, history=history)
            maximum = env.maximum_requested_volume()
            if interactions < args.sac_warmup:
                fraction = float(np.random.random())
                volume = MIN_VOLUME_ML + (maximum - MIN_VOLUME_ML) * fraction
            else:
                volume, fraction = deterministic_action(actor, state, maximum, device, args.td3_exploration_noise)
            previous_error = abs(env.true_ph - env.target_ph)
            overshoots_before = env.overshoots
            env.step(volume)
            history.append(raw)
            next_state = make_features(env, candidate.state_mode, history=history)
            current_error = abs(env.true_ph - env.target_ph)
            reward = reward_value(previous_error, current_error, volume, env.overshoots > overshoots_before, env.done, current_error <= 0.1, abs(env.measured_ph - env.target_ph) <= 0.1, env.steps, True, candidate.reward_profile)
            replay.add(state, fraction, reward, next_state, env.done)
            interactions += 1
            if replay.size >= args.sac_batch_size:
                states, actions, rewards, next_states, dones = replay.sample(args.sac_batch_size, device)
                with torch.no_grad():
                    noise = (torch.randn_like(actions) * args.td3_target_noise).clamp(-args.td3_noise_clip, args.td3_noise_clip)
                    next_actions = (target_actor(next_states) + noise).clamp(0.0, 1.0)
                    target_q = torch.minimum(target1(next_states, next_actions), target2(next_states, next_actions))
                    target = rewards + args.gamma * (1.0 - dones) * target_q
                q1_loss = F.mse_loss(q1(states, actions), target)
                q2_loss = F.mse_loss(q2(states, actions), target)
                q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
                q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()
                updates += 1
                if updates % args.td3_policy_delay == 0:
                    actor_loss = -q1(states, actor(states)).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                    with torch.no_grad():
                        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                            target_param.mul_(1.0 - args.sac_tau).add_(args.sac_tau * param)
                        for target_param, param in zip(target1.parameters(), q1.parameters()):
                            target_param.mul_(1.0 - args.sac_tau).add_(args.sac_tau * param)
                        for target_param, param in zip(target2.parameters(), q2.parameters()):
                            target_param.mul_(1.0 - args.sac_tau).add_(args.sac_tau * param)
            if interactions >= next_eval or interactions >= args.train_steps:
                point = {"environment_steps": interactions, **evaluate_td3(actor, candidate, device, seed, args.eval_tasks)}
                curves.append(point)
                print(f"{candidate.name}/seed{seed}: {interactions} steps, validation success={point['success_rate_percent']:.2f}%")
                next_eval += args.eval_interval
        episode += 1
    return actor, curves


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train predeclared RL candidates for the Bayesian challenge.")
    parser.add_argument("--imitation-weights", type=Path, default=base / "models" / "imitation.pth")
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--train-steps", type=int, default=60000)
    parser.add_argument("--training-pool-size", type=int, default=8000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-tasks", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--a2c-learning-rate", type=float, default=5e-5)
    parser.add_argument("--ppo-batch-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--sac-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sac-alpha", type=float, default=0.15)
    parser.add_argument("--sac-tau", type=float, default=0.005)
    parser.add_argument("--sac-warmup", type=int, default=1000)
    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--td3-learning-rate", type=float, default=3e-4)
    parser.add_argument("--td3-exploration-noise", type=float, default=0.10)
    parser.add_argument("--td3-target-noise", type=float, default=0.20)
    parser.add_argument("--td3-noise-clip", type=float, default=0.50)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=base / "results_challenge" / "training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    imitation = NeuralVolumePolicy(args.imitation_weights.resolve(), str(device))
    curve_rows: list[dict] = read_csv(args.output_dir / "learning_curves.csv") if args.resume else []
    final_rows: list[dict] = read_csv(args.output_dir / "validation_results.csv") if args.resume else []
    for seed in args.seeds:
        for name in args.candidates:
            candidate = CANDIDATES[name]
            model_path = model_dir / f"{name}_seed{seed}.pth"
            if args.resume and model_path.exists() and any(int(float(row["seed"])) == seed and row["candidate"] == name for row in final_rows):
                print(f"SKIP {name}/seed{seed}: completed model and validation row exist.")
                continue
            if candidate.algorithm == "sac":
                model, curves = train_sac(candidate, seed, args, device)
                payload = {
                    "metadata": candidate.__dict__ | {"seed": seed, "input_dim": feature_dim(candidate.state_mode)},
                    "actor_state_dict": model.state_dict(),
                }
            elif candidate.algorithm == "td3":
                model, curves = train_td3(candidate, seed, args, device)
                payload = {
                    "metadata": candidate.__dict__ | {"seed": seed, "input_dim": feature_dim(candidate.state_mode)},
                    "actor_state_dict": model.state_dict(),
                }
            else:
                model, curves = train_categorical(candidate, seed, args, device, imitation)
                payload = {
                    "metadata": candidate.__dict__ | {"seed": seed, "input_dim": feature_dim(candidate.state_mode, candidate.residual), "action_dim": 9 if candidate.residual else 1000},
                    "model_state_dict": model.state_dict(),
                }
            torch.save(payload, model_path)
            for point in curves:
                curve_rows.append({"seed": seed, "candidate": name, **point})
            final_rows.append({"seed": seed, "candidate": name, **curves[-1]})
            write_csv(args.output_dir / "learning_curves.csv", curve_rows)
            write_csv(args.output_dir / "validation_results.csv", final_rows)
    settings = vars(args).copy()
    settings["imitation_weights"] = str(args.imitation_weights)
    settings["output_dir"] = str(args.output_dir)
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
