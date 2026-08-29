from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from benchmark_core import (
    DiscreteVolumeRegressor,
    PolicyEnvironment,
    StressScenario,
    Task,
    _unwrap_state_dict,
    generate_tasks,
    portable_settings,
    seed_everything,
)


@dataclass(frozen=True)
class RewardVariant:
    name: str
    dense_lambda: float = 0.03
    step_penalty: float = 0.0
    terminal_bonus: float = 3.9
    overshoot_weight: float = 0.2
    volume_penalty: float = 0.1
    volume_bonus: float = 0.1
    reward_clip: float = 4.1


REWARD_VARIANTS = {
    "full": RewardVariant("full"),
    "step_penalty_minus_0p005": RewardVariant("step_penalty_minus_0p005", step_penalty=-0.005),
    "no_dense": RewardVariant("no_dense", dense_lambda=0.0),
    "no_overshoot": RewardVariant("no_overshoot", overshoot_weight=0.0),
    "no_volume_shaping": RewardVariant("no_volume_shaping", volume_penalty=0.0, volume_bonus=0.0),
    "no_terminal": RewardVariant("no_terminal", terminal_bonus=0.0),
    "simple_progress": RewardVariant(
        "simple_progress",
        dense_lambda=1.0,
        step_penalty=-0.005,
        terminal_bonus=3.9,
        overshoot_weight=0.0,
        volume_penalty=0.0,
        volume_bonus=0.0,
    ),
}


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = DiscreteVolumeRegressor()
        self.critic = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def load_imitation(self, path: Path, device: torch.device) -> None:
        payload = torch.load(path, map_location=device)
        self.actor.load_state_dict(_unwrap_state_dict(payload), strict=True)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    action_limit: int
    reward: float
    old_log_prob: float
    value: float
    done: bool


def action_distribution(model: ActorCritic, state_tensor: torch.Tensor, action_limit: int) -> Categorical:
    logits = model.actor(state_tensor)[:, :action_limit]
    return Categorical(logits=logits)


def calculate_reward(
    previous_ph: float,
    current_ph: float,
    target_ph: float,
    steps: int,
    current_overshoot: bool,
    previous_overshoot: bool,
    previous_overshoot_volume: float | None,
    action_volume: float,
    done: bool,
    variant: RewardVariant,
) -> float:
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    remaining_ratio = (50 - steps) / 50.0
    dense = variant.dense_lambda * (previous_error - current_error) * (1.0 + remaining_ratio)
    overshoot = 0.0
    if current_overshoot and max(previous_error, current_error) > 0.1:
        magnitude = abs(current_ph - target_ph)
        overshoot = -variant.overshoot_weight / (1.0 + math.exp(-(magnitude - 0.1)))
    volume_term = 0.0
    if previous_overshoot and previous_overshoot_volume is not None:
        volume_term -= variant.volume_penalty * action_volume
        if action_volume < previous_overshoot_volume:
            volume_term += variant.volume_bonus * (previous_overshoot_volume - action_volume)
    raw = dense + variant.step_penalty + overshoot + volume_term
    if done and current_error <= 0.1:
        raw += variant.terminal_bonus * (2.0 if steps < 25 else 1.0)
    if not done:
        raw = float(np.clip(raw, -variant.reward_clip, variant.reward_clip))
    return raw


def rollout_episode(
    model: ActorCritic,
    task: Task,
    variant: RewardVariant,
    device: torch.device,
    rng_seed: int,
    stochastic: bool,
) -> tuple[list[Transition], dict]:
    env = PolicyEnvironment(task, StressScenario("nominal"), np.random.default_rng(rng_seed))
    transitions: list[Transition] = []
    previous_overshoot = False
    previous_overshoot_volume: float | None = None
    while not env.done:
        state = env.state()
        action_limit = max(1, min(1000, int(math.floor(env.maximum_requested_volume() / 0.01 + 1e-9))))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        distribution = action_distribution(model, state_tensor, action_limit)
        if stochastic:
            action_tensor = distribution.sample()
        else:
            action_tensor = torch.argmax(distribution.logits, dim=1)
        action_index = int(action_tensor.item())
        log_prob = float(distribution.log_prob(action_tensor).item())
        value = float(model.critic(state_tensor).squeeze(-1).item())
        action_volume = (action_index + 1) * 0.01
        previous_ph = env.measured_ph
        overshoots_before = env.overshoots
        env.step(action_volume)
        current_overshoot = env.overshoots > overshoots_before
        reward = calculate_reward(
            previous_ph,
            env.measured_ph,
            env.target_ph,
            env.steps,
            current_overshoot,
            previous_overshoot,
            previous_overshoot_volume,
            action_volume,
            env.done,
            variant,
        )
        transitions.append(
            Transition(
                state=state,
                action=action_index,
                action_limit=action_limit,
                reward=reward,
                old_log_prob=log_prob,
                value=value,
                done=env.done,
            )
        )
        previous_overshoot = current_overshoot
        previous_overshoot_volume = action_volume if current_overshoot else None
    metrics = {
        "success": abs(env.true_ph - env.target_ph) <= 0.1,
        "steps": env.steps,
        "overshoots": env.overshoots,
        "final_abs_error": abs(env.true_ph - env.target_ph),
    }
    return transitions, metrics


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    values = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        values.append(running)
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


def reinforce_update(model: ActorCritic, optimizer, trajectory: list[Transition], gamma: float, device: torch.device) -> None:
    if not trajectory:
        return
    returns = discounted_returns([item.reward for item in trajectory], gamma).to(device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    loss = torch.tensor(0.0, device=device)
    for item, value in zip(trajectory, returns):
        state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([item.action], device=device)
        loss = loss - action_distribution(model, state, item.action_limit).log_prob(action).squeeze(0) * value
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
    optimizer.step()


def a2c_update(model: ActorCritic, optimizer, trajectory: list[Transition], gamma: float, device: torch.device) -> None:
    if not trajectory:
        return
    returns = discounted_returns([item.reward for item in trajectory], gamma).to(device)
    actor_loss = torch.tensor(0.0, device=device)
    critic_loss = torch.tensor(0.0, device=device)
    entropy = torch.tensor(0.0, device=device)
    for item, target in zip(trajectory, returns):
        state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([item.action], device=device)
        distribution = action_distribution(model, state, item.action_limit)
        value = model.critic(state).squeeze()
        advantage = target - value.detach()
        actor_loss = actor_loss - distribution.log_prob(action).squeeze() * advantage
        critic_loss = critic_loss + (value - target).pow(2)
        entropy = entropy + distribution.entropy().squeeze()
    count = max(1, len(trajectory))
    loss = actor_loss / count + 0.5 * critic_loss / count - 0.01 * entropy / count
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def ppo_update(
    model: ActorCritic,
    optimizer,
    batch: list[Transition],
    gamma: float,
    device: torch.device,
    epochs: int,
    clip_ratio: float,
) -> None:
    if not batch:
        return
    returns_parts: list[torch.Tensor] = []
    current_rewards: list[float] = []
    for item in batch:
        current_rewards.append(item.reward)
        if item.done:
            returns_parts.append(discounted_returns(current_rewards, gamma))
            current_rewards = []
    if current_rewards:
        returns_parts.append(discounted_returns(current_rewards, gamma))
    returns = torch.cat(returns_parts).to(device)
    old_values = torch.tensor([item.value for item in batch], dtype=torch.float32, device=device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

    indices = np.arange(len(batch))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            actor_loss = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)
            entropy = torch.tensor(0.0, device=device)
            for index in selected:
                item = batch[int(index)]
                state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.tensor([item.action], device=device)
                distribution = action_distribution(model, state, item.action_limit)
                new_log_prob = distribution.log_prob(action).squeeze()
                ratio = torch.exp(new_log_prob - item.old_log_prob)
                advantage = advantages[int(index)]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                actor_loss = actor_loss - torch.minimum(ratio * advantage, clipped)
                value = model.critic(state).squeeze()
                critic_loss = critic_loss + (value - returns[int(index)]).pow(2)
                entropy = entropy + distribution.entropy().squeeze()
            count = max(1, len(selected))
            loss = actor_loss / count + 0.5 * critic_loss / count - 0.01 * entropy / count
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model: ActorCritic, tasks: list[Task], device: torch.device, seed: int) -> dict[str, float]:
    success = 0
    successful_steps: list[int] = []
    total_steps = 0
    overshoots = 0
    final_errors = []
    model.eval()
    with torch.no_grad():
        for task in tasks:
            _, metrics = rollout_episode(model, task, REWARD_VARIANTS["full"], device, seed * 1_000_003 + task.task_id, False)
            success += int(metrics["success"])
            if metrics["success"]:
                successful_steps.append(int(metrics["steps"]))
            total_steps += int(metrics["steps"])
            overshoots += int(metrics["overshoots"])
            final_errors.append(float(metrics["final_abs_error"]))
    model.train()
    return {
        "success_rate_percent": 100.0 * success / len(tasks),
        "successful_steps_mean": float(np.mean(successful_steps)) if successful_steps else math.nan,
        "overshoot_rate_percent": 100.0 * overshoots / total_steps if total_steps else 0.0,
        "final_abs_error_mean": float(np.mean(final_errors)),
    }


def train_one(
    algorithm: str,
    variant: RewardVariant,
    imitation_weights: Path,
    train_tasks: list[Task],
    eval_tasks: list[Task],
    train_steps: int,
    eval_interval: int,
    learning_rate: float,
    gamma: float,
    device: torch.device,
    seed: int,
    ppo_batch_steps: int,
    ppo_epochs: int,
) -> tuple[ActorCritic, list[dict]]:
    seed_everything(seed)
    model = ActorCritic().to(device)
    model.load_imitation(imitation_weights, device)
    parameters = model.actor.parameters() if algorithm == "reinforce" else model.parameters()
    optimizer = optim.Adam(parameters, lr=learning_rate)
    learning_curve = [{"environment_steps": 0, **evaluate(model, eval_tasks, device, seed)}]
    interactions = 0
    episode_index = 0
    next_evaluation = eval_interval
    ppo_batch: list[Transition] = []

    while interactions < train_steps:
        task = train_tasks[episode_index % len(train_tasks)]
        trajectory, _ = rollout_episode(
            model,
            task,
            variant,
            device,
            seed * 10_000_019 + episode_index,
            True,
        )
        interactions += len(trajectory)
        episode_index += 1
        if algorithm == "reinforce":
            reinforce_update(model, optimizer, trajectory, gamma, device)
        elif algorithm == "a2c":
            a2c_update(model, optimizer, trajectory, gamma, device)
        elif algorithm == "ppo":
            ppo_batch.extend(trajectory)
            if len(ppo_batch) >= ppo_batch_steps:
                ppo_update(model, optimizer, ppo_batch, gamma, device, ppo_epochs, 0.2)
                ppo_batch = []
        else:
            raise ValueError(algorithm)

        if interactions >= next_evaluation or interactions >= train_steps:
            learning_curve.append({"environment_steps": interactions, **evaluate(model, eval_tasks, device, seed)})
            next_evaluation += eval_interval
            print(
                f"{algorithm}/{variant.name}/seed{seed}: {interactions} steps, "
                f"success={learning_curve[-1]['success_rate_percent']:.2f}%"
            )

    if algorithm == "ppo" and ppo_batch:
        ppo_update(model, optimizer, ppo_batch, gamma, device, ppo_epochs, 0.2)
        learning_curve.append({"environment_steps": interactions, **evaluate(model, eval_tasks, device, seed)})
    return model, learning_curve


def run_study_job(job: tuple) -> tuple[int, str, str, list[dict]]:
    (
        seed,
        algorithm,
        variant_name,
        imitation_weights,
        train_steps,
        training_pool_size,
        eval_tasks_count,
        eval_interval,
        learning_rate,
        gamma,
        device_name,
        ppo_batch_steps,
        ppo_epochs,
        model_path,
    ) = job
    torch.set_num_threads(1)
    device = torch.device(device_name)
    train_tasks = generate_tasks(seed + 100_000, training_pool_size, StressScenario("nominal"))
    eval_tasks = generate_tasks(seed + 200_000, eval_tasks_count, StressScenario("nominal"))
    model, curve = train_one(
        algorithm,
        REWARD_VARIANTS[variant_name],
        Path(imitation_weights),
        train_tasks,
        eval_tasks,
        train_steps,
        eval_interval,
        learning_rate,
        gamma,
        device,
        seed,
        ppo_batch_steps,
        ppo_epochs,
    )
    torch.save(model.actor.state_dict(), Path(model_path))
    return seed, algorithm, variant_name, curve


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Equal-budget RL algorithm comparison and reward ablation.")
    parser.add_argument("mode", choices=["algorithms", "rewards"])
    parser.add_argument("--imitation-weights", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", choices=["reinforce", "a2c", "ppo"], default=["reinforce", "a2c", "ppo"])
    parser.add_argument("--reward-algorithm", choices=["reinforce", "a2c", "ppo"], default="reinforce")
    parser.add_argument("--reward-variants", nargs="+", choices=sorted(REWARD_VARIANTS), default=list(REWARD_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--training-pool-size", type=int, default=5000)
    parser.add_argument("--eval-tasks", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo-batch-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=base / "results" / "rl_algorithm_reward")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    studies = (
        [(algorithm, "full") for algorithm in args.algorithms]
        if args.mode == "algorithms"
        else [(args.reward_algorithm, variant) for variant in args.reward_variants]
    )
    curve_rows: list[dict] = []
    final_rows: list[dict] = []

    if args.workers > 1:
        if device.type != "cpu":
            raise SystemExit("--workers greater than 1 is supported only with --device cpu.")
        jobs = [
            (
                seed,
                algorithm,
                variant_name,
                str(args.imitation_weights.resolve()),
                args.train_steps,
                args.training_pool_size,
                args.eval_tasks,
                args.eval_interval,
                args.learning_rate,
                args.gamma,
                args.device,
                args.ppo_batch_steps,
                args.ppo_epochs,
                str(model_dir / f"{algorithm}_{variant_name}_seed{seed}.pth"),
            )
            for seed in args.seeds
            for algorithm, variant_name in studies
        ]
        worker_count = min(args.workers, len(jobs))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            completed = executor.map(run_study_job, jobs)
            for seed, algorithm, variant_name, curve in completed:
                for point in curve:
                    curve_rows.append({"seed": seed, "algorithm": algorithm, "reward_variant": variant_name, **point})
                final_rows.append({"seed": seed, "algorithm": algorithm, "reward_variant": variant_name, **curve[-1]})
    else:
        for seed in args.seeds:
            train_tasks = generate_tasks(seed + 100_000, args.training_pool_size, StressScenario("nominal"))
            eval_tasks = generate_tasks(seed + 200_000, args.eval_tasks, StressScenario("nominal"))
            for algorithm, variant_name in studies:
                variant = REWARD_VARIANTS[variant_name]
                model, curve = train_one(
                    algorithm,
                    variant,
                    args.imitation_weights.resolve(),
                    train_tasks,
                    eval_tasks,
                    args.train_steps,
                    args.eval_interval,
                    args.learning_rate,
                    args.gamma,
                    device,
                    seed,
                    args.ppo_batch_steps,
                    args.ppo_epochs,
                )
                for point in curve:
                    curve_rows.append({"seed": seed, "algorithm": algorithm, "reward_variant": variant_name, **point})
                final_rows.append({"seed": seed, "algorithm": algorithm, "reward_variant": variant_name, **curve[-1]})
                torch.save(model.actor.state_dict(), model_dir / f"{algorithm}_{variant_name}_seed{seed}.pth")

    write_csv(args.output_dir / "learning_curves.csv", curve_rows)
    write_csv(args.output_dir / "final_results.csv", final_rows)
    (args.output_dir / "settings.json").write_text(
        json.dumps(portable_settings(vars(args), base), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
