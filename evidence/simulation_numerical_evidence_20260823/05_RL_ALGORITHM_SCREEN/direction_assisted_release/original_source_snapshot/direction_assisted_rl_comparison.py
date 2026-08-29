from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
    seed_everything,
)


PROTOCOL_ID = "direction_assisted_volume_only_v1"
ACTION_COUNT = 1000
RANDOM_ACTOR_SEED_OFFSET = 9_000_000


@dataclass(frozen=True)
class RewardConfig:
    # Reducing absolute pH error must give a positive dense reward.
    dense_lambda: float = 0.03
    step_penalty: float = 0.0
    terminal_bonus: float = 3.9
    overshoot_weight: float = 0.2
    volume_penalty: float = 0.1
    volume_bonus: float = 0.1
    reward_clip: float = 4.1


@dataclass
class Transition:
    state: np.ndarray
    action: int
    action_limit: int
    reward: float
    old_log_prob: float
    value: float
    done: bool


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


def load_imitation_actor(model: ActorCritic, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    model.actor.load_state_dict(_unwrap_state_dict(payload), strict=True)


def reset_random_actor(model: ActorCritic, seed: int, device: torch.device) -> None:
    """Reset actor weights without advancing the later training RNG stream."""
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for module in model.actor.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def action_distribution(model: ActorCritic, state: torch.Tensor, action_limit: int) -> Categorical:
    if action_limit != ACTION_COUNT:
        raise ValueError(f"Direction-assisted protocol requires {ACTION_COUNT} volume actions")
    return Categorical(logits=model.actor(state))


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
    config: RewardConfig,
) -> float:
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    remaining_ratio = (50 - steps) / 50.0
    dense = config.dense_lambda * (previous_error - current_error) * (1.0 + remaining_ratio)
    overshoot = 0.0
    if current_overshoot and max(previous_error, current_error) > 0.1:
        magnitude = abs(current_ph - target_ph)
        overshoot = -config.overshoot_weight / (1.0 + math.exp(-(magnitude - 0.1)))
    volume_term = 0.0
    if previous_overshoot and previous_overshoot_volume is not None:
        volume_term -= config.volume_penalty * action_volume
        if action_volume < previous_overshoot_volume:
            volume_term += config.volume_bonus * (previous_overshoot_volume - action_volume)
    raw = dense + config.step_penalty + overshoot + volume_term
    if done and current_error <= 0.1:
        raw += config.terminal_bonus * (2.0 if steps < 25 else 1.0)
    return float(raw if done else np.clip(raw, -config.reward_clip, config.reward_clip))


def rollout_episode(
    model: ActorCritic,
    task: Task,
    device: torch.device,
    rng_seed: int,
    stochastic: bool,
    reward_config: RewardConfig,
) -> tuple[list[Transition], dict[str, Any]]:
    env = PolicyEnvironment(task, StressScenario("nominal"), np.random.default_rng(rng_seed))
    transitions: list[Transition] = []
    previous_overshoot = False
    previous_overshoot_volume: float | None = None
    while not env.done:
        state = env.state()
        action_limit = ACTION_COUNT
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        distribution = action_distribution(model, state_tensor, action_limit)
        action_tensor = distribution.sample() if stochastic else torch.argmax(distribution.logits, dim=1)
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
            reward_config,
        )
        transitions.append(Transition(state, action_index, action_limit, reward, log_prob, value, env.done))
        previous_overshoot = current_overshoot
        previous_overshoot_volume = action_volume if current_overshoot else None
    true_error = abs(env.true_ph - env.target_ph)
    measured_error = abs(env.measured_ph - env.target_ph)
    metrics = {
        "true_success": int(true_error <= 0.10),
        "strict_success": int(true_error <= 0.05),
        "severe_failure": int(true_error > 0.50),
        "measured_success": int(measured_error <= 0.10),
        "false_stop": int(measured_error <= 0.10 and true_error > 0.10),
        "steps": int(env.steps),
        "overshoots": int(env.overshoots),
        "final_abs_error": float(true_error),
        "final_measured_abs_error": float(measured_error),
        "total_volume_ml": float(env.acid_added_ml + env.base_added_ml),
        "acid_added_ml": float(env.acid_added_ml),
        "base_added_ml": float(env.base_added_ml),
        "final_true_ph": float(env.true_ph),
        "final_measured_ph": float(env.measured_ph),
    }
    return transitions, metrics


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        values.append(running)
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


def reinforce_update(model: ActorCritic, optimizer: optim.Optimizer, trajectory: list[Transition], gamma: float, device: torch.device) -> None:
    returns = discounted_returns([x.reward for x in trajectory], gamma).to(device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    loss = torch.tensor(0.0, device=device)
    for item, target in zip(trajectory, returns):
        state = torch.as_tensor(item.state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([item.action], device=device)
        loss = loss - action_distribution(model, state, item.action_limit).log_prob(action).squeeze(0) * target
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
    optimizer.step()


def a2c_update(model: ActorCritic, optimizer: optim.Optimizer, trajectory: list[Transition], gamma: float, device: torch.device) -> None:
    returns = discounted_returns([x.reward for x in trajectory], gamma).to(device)
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
    optimizer: optim.Optimizer,
    batch: list[Transition],
    gamma: float,
    device: torch.device,
    epochs: int,
    clip_ratio: float,
) -> None:
    return_parts: list[torch.Tensor] = []
    current: list[float] = []
    for item in batch:
        current.append(item.reward)
        if item.done:
            return_parts.append(discounted_returns(current, gamma))
            current = []
    if current:
        return_parts.append(discounted_returns(current, gamma))
    returns = torch.cat(return_parts).to(device)
    old_values = torch.tensor([x.value for x in batch], dtype=torch.float32, device=device)
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


def evaluate_model(model: ActorCritic, tasks: list[Task], device: torch.device, seed: int, config: RewardConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for task in tasks:
            _, metrics = rollout_episode(model, task, device, seed * 1_000_003 + task.task_id, False, config)
            row = {
                "task_seed": task.seed,
                "task_id": task.task_id,
                "acid_type": task.acid_type,
                "pka_values": json.dumps(task.pka_values),
                "initial_ph": task.initial_ph,
                "target_ph": task.target_ph,
                **metrics,
            }
            rows.append(row)
    model.train()
    return rows


def summarize_task_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    success = values("true_success")
    successful_steps = values("steps")[success > 0.5]
    return {
        "tasks": len(rows),
        "success_rate_percent": float(100 * np.mean(success)),
        "strict_success_rate_percent": float(100 * np.mean(values("strict_success"))),
        "severe_failure_rate_percent": float(100 * np.mean(values("severe_failure"))),
        "false_stop_rate_percent": float(100 * np.mean(values("false_stop"))),
        "successful_steps_mean": float(np.mean(successful_steps)) if len(successful_steps) else math.nan,
        "successful_steps_sd_task": float(np.std(successful_steps, ddof=1)) if len(successful_steps) > 1 else 0.0,
        "steps_mean": float(np.mean(values("steps"))),
        "overshoots_mean": float(np.mean(values("overshoots"))),
        "total_volume_mean_ml": float(np.mean(values("total_volume_ml"))),
        "final_abs_error_mean": float(np.mean(values("final_abs_error"))),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def task_manifest_rows(tasks: list[Task]) -> list[dict[str, Any]]:
    return [
        {
            "seed": task.seed,
            "task_id": task.task_id,
            "acid_type": task.acid_type,
            "pka_values": json.dumps(task.pka_values),
            "initial_ph": task.initial_ph,
            "target_ph": task.target_ph,
            "initial_volume_ml": task.initial_volume_ml,
            "analyte_conc_m": task.analyte_conc_m,
        }
        for task in tasks
    ]


def train_one(
    algorithm: str,
    initialization: str,
    imitation_weights: Path,
    train_tasks: list[Task],
    eval_tasks: list[Task],
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_path = run_dir / "learning_curve.csv"
    task_path = run_dir / "task_results.csv"
    done_path = run_dir / "COMPLETE.json"
    run_config_path = run_dir / "run_config.json"
    expected_run_config = {
        "protocol_id": PROTOCOL_ID,
        "algorithm": algorithm,
        "initialization": initialization,
        "seed": seed,
        "random_actor_seed": seed + RANDOM_ACTOR_SEED_OFFSET,
        "train_task_seed": seed + 100_000,
        "evaluation_task_seed": seed + 200_000,
        "training_pool_size": args.training_pool_size,
        "evaluation_tasks": args.eval_tasks,
        "train_steps": args.train_steps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "reward_config": asdict(RewardConfig()),
        "state": [
            "measured_ph",
            "target_ph",
            "measured_ph_change",
            "measured_ph_minus_target_ph",
            "last_requested_volume_ml",
        ],
        "policy_action": "1000 volume classes from 0.01 to 10.00 mL",
        "external_direction_rule": "base if measured_ph < target_ph else acid",
        "titrant_concentration_m": 0.1,
        "overshoot_action_mask": False,
        "automatic_titrant_switching": False,
    }
    if args.resume and done_path.exists() and curve_path.exists() and task_path.exists() and run_config_path.exists():
        try:
            stored_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored_config = {}
        comparable = {key: stored_config.get(key) for key in expected_run_config}
        if comparable == expected_run_config:
            return read_csv(curve_path), read_csv(task_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device(args.device)
    model = ActorCritic().to(device)
    if initialization == "imitation":
        load_imitation_actor(model, imitation_weights, device)
    elif initialization == "random":
        reset_random_actor(model, seed + RANDOM_ACTOR_SEED_OFFSET, device)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    initial_actor_sha256 = state_dict_sha256(model.actor.state_dict())
    expected_run_config["initial_actor_sha256"] = initial_actor_sha256
    run_config_path.write_text(json.dumps(expected_run_config, indent=2), encoding="utf-8")
    optimizer = optim.Adam(model.actor.parameters() if algorithm == "reinforce" else model.parameters(), lr=args.learning_rate)
    config = RewardConfig()
    curve: list[dict[str, Any]] = []
    curve.append({"environment_steps": 0, **summarize_task_rows(evaluate_model(model, eval_tasks, device, seed + 700_000, config))})
    interactions = 0
    episode_index = 0
    next_eval = args.eval_interval
    ppo_batch: list[Transition] = []
    start_time = time.perf_counter()
    while interactions < args.train_steps:
        task = train_tasks[episode_index % len(train_tasks)]
        trajectory, _ = rollout_episode(
            model,
            task,
            device,
            seed * 10_000_019 + episode_index,
            True,
            config,
        )
        interactions += len(trajectory)
        episode_index += 1
        if algorithm == "reinforce":
            reinforce_update(model, optimizer, trajectory, args.gamma, device)
        elif algorithm == "a2c":
            a2c_update(model, optimizer, trajectory, args.gamma, device)
        elif algorithm == "ppo":
            ppo_batch.extend(trajectory)
            if len(ppo_batch) >= args.ppo_batch_steps:
                ppo_update(model, optimizer, ppo_batch, args.gamma, device, args.ppo_epochs, 0.2)
                ppo_batch = []
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if interactions >= next_eval or interactions >= args.train_steps:
            metrics = summarize_task_rows(evaluate_model(model, eval_tasks, device, seed + 700_000, config))
            curve.append({"environment_steps": interactions, **metrics})
            write_csv(curve_path, curve)
            print(f"{algorithm}/{initialization}/seed{seed}: {interactions} interactions, success={metrics['success_rate_percent']:.2f}%", flush=True)
            next_eval += args.eval_interval
    if algorithm == "ppo" and ppo_batch:
        ppo_update(model, optimizer, ppo_batch, args.gamma, device, args.ppo_epochs, 0.2)
    final_task_rows = evaluate_model(model, eval_tasks, device, seed + 700_000, config)
    final_summary = summarize_task_rows(final_task_rows)
    if curve and int(curve[-1]["environment_steps"]) == interactions:
        curve[-1] = {"environment_steps": interactions, **final_summary}
    else:
        curve.append({"environment_steps": interactions, **final_summary})
    write_csv(curve_path, curve)
    write_csv(task_path, final_task_rows)
    torch.save(
        {"model_state_dict": model.state_dict(), "algorithm": algorithm, "initialization": initialization, "seed": seed},
        run_dir / "final_model.pth",
    )
    done_path.write_text(json.dumps({"elapsed_seconds": time.perf_counter() - start_time, "seed": seed}, indent=2), encoding="utf-8")
    return curve, final_task_rows


def exact_mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    from scipy.stats import binomtest

    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    return a_only, b_only, p


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    if observed == 0:
        return 1.0
    counts = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        total += 1
        if abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12:
            counts += 1
    return counts / total


def paired_seed_bootstrap(values: np.ndarray, seed: int = 12345, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_all(task_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in task_rows:
        key = (str(row["algorithm"]), str(row["initialization"]), int(row["seed"]))
        groups.setdefault(key, []).append(row)
    seed_summary: list[dict[str, Any]] = []
    for (algorithm, initialization, seed), rows in sorted(groups.items()):
        seed_summary.append({"algorithm": algorithm, "initialization": initialization, "seed": seed, **summarize_task_rows(rows)})
    write_csv(output_dir / "seed_summary.csv", seed_summary)

    aggregate: list[dict[str, Any]] = []
    metrics = ["success_rate_percent", "strict_success_rate_percent", "severe_failure_rate_percent", "false_stop_rate_percent", "successful_steps_mean", "steps_mean", "overshoots_mean", "total_volume_mean_ml", "final_abs_error_mean"]
    for algorithm in sorted({x["algorithm"] for x in seed_summary}):
        for initialization in ("imitation", "random"):
            rows = [x for x in seed_summary if x["algorithm"] == algorithm and x["initialization"] == initialization]
            if not rows:
                continue
            output = {"algorithm": algorithm, "initialization": initialization, "training_seeds": len(rows)}
            for metric in metrics:
                values = np.asarray([float(x[metric]) for x in rows if np.isfinite(float(x[metric]))])
                output[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
                output[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregate.append(output)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)

    tests: list[dict[str, Any]] = []
    raw_p: list[float] = []
    pending: list[dict[str, Any]] = []
    for algorithm in sorted({x["algorithm"] for x in seed_summary}):
        imitation = {(int(x["seed"]), int(x["task_id"])): int(x["true_success"]) for x in task_rows if x["algorithm"] == algorithm and x["initialization"] == "imitation"}
        random_init = {(int(x["seed"]), int(x["task_id"])): int(x["true_success"]) for x in task_rows if x["algorithm"] == algorithm and x["initialization"] == "random"}
        keys = sorted(set(imitation) & set(random_init))
        a = [imitation[k] for k in keys]
        b = [random_init[k] for k in keys]
        a_only, b_only, p = exact_mcnemar(a, b)
        seed_diffs = []
        for seed in sorted({int(x["seed"]) for x in task_rows}):
            a_seed = [x for x in task_rows if x["algorithm"] == algorithm and x["initialization"] == "imitation" and int(x["seed"]) == seed]
            b_seed = [x for x in task_rows if x["algorithm"] == algorithm and x["initialization"] == "random" and int(x["seed"]) == seed]
            if a_seed and b_seed:
                seed_diffs.append(100 * (np.mean([int(x["true_success"]) for x in b_seed]) - np.mean([int(x["true_success"]) for x in a_seed])))
        diffs = np.asarray(seed_diffs, dtype=float)
        ci_low, ci_high = paired_seed_bootstrap(diffs) if len(diffs) else (math.nan, math.nan)
        row = {
            "algorithm": algorithm,
            "paired_tasks": len(keys),
            "imitation_success_only": a_only,
            "random_success_only": b_only,
            "task_level_mcnemar_p": p,
            "seed_success_difference_random_minus_imitation_pp_mean": float(np.mean(diffs)) if len(diffs) else math.nan,
            "seed_success_difference_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
            "seed_difference_bootstrap_95ci_low": ci_low,
            "seed_difference_bootstrap_95ci_high": ci_high,
            "seed_sign_flip_p": sign_flip_p(diffs) if len(diffs) else math.nan,
        }
        pending.append(row)
        raw_p.append(p)
    adjusted = holm_adjust(raw_p) if raw_p else []
    for row, adj in zip(pending, adjusted):
        row["task_level_mcnemar_holm_p"] = adj
        tests.append(row)
    write_csv(output_dir / "paired_initialization_tests.csv", tests)

    algorithm_pending: list[dict[str, Any]] = []
    algorithm_raw_p: list[float] = []
    algorithms = sorted({str(x["algorithm"]) for x in task_rows})
    seeds = sorted({int(x["seed"]) for x in task_rows})
    for initialization in ("imitation", "random"):
        for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
            outcomes_a = {
                (int(x["seed"]), int(x["task_id"])): int(x["true_success"])
                for x in task_rows
                if x["algorithm"] == algorithm_a and x["initialization"] == initialization
            }
            outcomes_b = {
                (int(x["seed"]), int(x["task_id"])): int(x["true_success"])
                for x in task_rows
                if x["algorithm"] == algorithm_b and x["initialization"] == initialization
            }
            keys = sorted(set(outcomes_a) & set(outcomes_b))
            a = [outcomes_a[key] for key in keys]
            b = [outcomes_b[key] for key in keys]
            a_only, b_only, p = exact_mcnemar(a, b)
            seed_diffs: list[float] = []
            for seed in seeds:
                seed_keys = [key for key in keys if key[0] == seed]
                if seed_keys:
                    seed_diffs.append(
                        100.0
                        * (
                            np.mean([outcomes_b[key] for key in seed_keys])
                            - np.mean([outcomes_a[key] for key in seed_keys])
                        )
                    )
            diffs = np.asarray(seed_diffs, dtype=float)
            ci_low, ci_high = paired_seed_bootstrap(diffs) if len(diffs) else (math.nan, math.nan)
            algorithm_pending.append(
                {
                    "initialization": initialization,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "paired_tasks": len(keys),
                    "algorithm_a_success_only": a_only,
                    "algorithm_b_success_only": b_only,
                    "task_level_mcnemar_p": p,
                    "seed_success_difference_b_minus_a_pp_mean": float(np.mean(diffs)) if len(diffs) else math.nan,
                    "seed_success_difference_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                    "seed_difference_bootstrap_95ci_low": ci_low,
                    "seed_difference_bootstrap_95ci_high": ci_high,
                    "seed_sign_flip_p": sign_flip_p(diffs) if len(diffs) else math.nan,
                }
            )
            algorithm_raw_p.append(p)
    algorithm_adjusted = holm_adjust(algorithm_raw_p) if algorithm_raw_p else []
    for row, adjusted_p in zip(algorithm_pending, algorithm_adjusted):
        row["task_level_mcnemar_holm_p"] = adjusted_p
    write_csv(output_dir / "paired_algorithm_tests.csv", algorithm_pending)


def plot_results(output_dir: Path, curve_rows: list[dict[str, Any]], seed_summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "PLOTS_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    plt.rcParams["font.family"] = "Arial"
    algorithms = sorted({str(x["algorithm"]) for x in curve_rows})
    colors = {"imitation": "#1f77b4", "random": "#d62728"}
    algorithm_colors = {"a2c": "#2ca02c", "ppo": "#1f77b4", "reinforce": "#9467bd"}
    fig, axes = plt.subplots(1, len(algorithms), figsize=(11.4, 3.8), dpi=180, sharey=True)
    if len(algorithms) == 1:
        axes = [axes]
    for ax, algorithm in zip(axes, algorithms):
        for initialization in ("imitation", "random"):
            selected = [x for x in curve_rows if x["algorithm"] == algorithm and x["initialization"] == initialization]
            if not selected:
                continue
            by_seed: dict[int, list[dict[str, Any]]] = {}
            for row in selected:
                by_seed.setdefault(int(row["seed"]), []).append(row)
            checkpoint_values: dict[int, list[tuple[float, float]]] = {}
            for rows in by_seed.values():
                for checkpoint, row in enumerate(sorted(rows, key=lambda x: int(x["environment_steps"]))):
                    checkpoint_values.setdefault(checkpoint, []).append(
                        (float(row["environment_steps"]), float(row["success_rate_percent"]))
                    )
            checkpoints = sorted(checkpoint_values)
            steps = np.asarray([np.mean([x[0] for x in checkpoint_values[i]]) for i in checkpoints])
            means = np.asarray([np.mean([x[1] for x in checkpoint_values[i]]) for i in checkpoints])
            sds = np.asarray([
                np.std([x[1] for x in checkpoint_values[i]], ddof=1)
                if len(checkpoint_values[i]) > 1 else 0.0
                for i in checkpoints
            ])
            color = colors[initialization]
            ax.plot(steps, means, label=initialization, color=color, marker="o", linewidth=2, markersize=3.5)
            ax.fill_between(steps, np.maximum(0.0, means - sds), means + sds, color=color, alpha=0.15)
        ax.set_title(algorithm.upper())
        ax.set_xlabel("Environment interactions")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out success rate (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "learning_curves.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
    for index, algorithm in enumerate(algorithms):
        rows = [x for x in seed_summary if x["algorithm"] == algorithm]
        seeds = sorted({int(x["seed"]) for x in rows})
        for seed in seeds:
            imitation = [float(x["success_rate_percent"]) for x in rows if int(x["seed"]) == seed and x["initialization"] == "imitation"]
            random_init = [float(x["success_rate_percent"]) for x in rows if int(x["seed"]) == seed and x["initialization"] == "random"]
            if imitation and random_init:
                ax.plot([index - 0.14, index + 0.14], [imitation[0], random_init[0]], color="#999999", alpha=0.6, linewidth=1)
                ax.scatter(index - 0.14, imitation[0], color=colors["imitation"], s=28, zorder=3)
                ax.scatter(index + 0.14, random_init[0], color=colors["random"], s=28, zorder=3)
    ax.scatter([], [], color=colors["imitation"], label="Imitation initialization")
    ax.scatter([], [], color=colors["random"], label="Random initialization")
    ax.set_xticks(np.arange(len(algorithms)), [x.upper() for x in algorithms])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "seed_success_rates.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "seed_success_rates.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), dpi=180)
    metric_specs = [("success_rate_percent", "Success (%)"), ("steps_mean", "Mean steps"), ("total_volume_mean_ml", "Total volume (mL)")]
    for ax, (metric, label) in zip(axes, metric_specs):
        for initialization, color in colors.items():
            values = [float(x[metric]) for x in seed_summary if x["initialization"] == initialization]
            positions = np.arange(len(algorithms)) + (-0.18 if initialization == "imitation" else 0.18)
            grouped = []
            for algorithm in algorithms:
                grouped.append([float(x[metric]) for x in seed_summary if x["algorithm"] == algorithm and x["initialization"] == initialization])
            means = [np.mean(x) if x else np.nan for x in grouped]
            sds = [np.std(x, ddof=1) if len(x) > 1 else 0 for x in grouped]
            ax.bar(positions, means, width=0.34, yerr=sds, capsize=3, color=color, label=initialization)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(algorithms)), [x.upper() for x in algorithms])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "aggregate_metrics.svg", bbox_inches="tight")
    plt.close(fig)


def build_report(output_dir: Path, args: argparse.Namespace) -> None:
    aggregate = read_csv(output_dir / "aggregate_summary.csv")
    tests = read_csv(output_dir / "paired_initialization_tests.csv")
    algorithm_tests = read_csv(output_dir / "paired_algorithm_tests.csv")
    lines = [
        "# Direction-assisted volume-policy RL comparison",
        "",
        "This report is generated from the saved task-level outputs.",
        "",
        "## Protocol",
        "",
        "The neural actor receives current pH, target pH, measured pH change, current-minus-target error, and the last requested volume. It selects only one of 1,000 dosing volumes from 0.01 to 10.00 mL.",
        "A common external rule selects base when measured pH is below the target and acid otherwise. The titrant concentration is fixed at 0.1 M. No overshoot-based action masking, automatic dose reduction, or dilute-titrant switching is used.",
        "This allocation matches the deployed 1,000-output policy: the algorithms are compared as volume policies inside the same direction-assisted controller.",
        "",
        f"Algorithms: {', '.join(args.algorithms)}; initialization modes: imitation and random; seeds: {args.seeds}.",
        f"Each run used {args.train_steps:,} environment interactions, a {args.training_pool_size:,}-task training pool, and {args.eval_tasks:,} held-out nominal test tasks.",
        "For each seed, both initialization modes used the identical generated training and test tasks, reward, optimizer settings, and evaluation action rule.",
        "",
        "## Aggregate results (mean +/- SD across training seeds)",
        "",
        "| Algorithm | Initialization | Success (%) | Strict +/-0.05 (%) | Severe failure (%) | Steps | Volume (mL) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(f"| {row['algorithm'].upper()} | {row['initialization']} | {float(row['success_rate_percent_mean']):.2f} +/- {float(row['success_rate_percent_sd']):.2f} | {float(row['strict_success_rate_percent_mean']):.2f} +/- {float(row['strict_success_rate_percent_sd']):.2f} | {float(row['severe_failure_rate_percent_mean']):.2f} +/- {float(row['severe_failure_rate_percent_sd']):.2f} | {float(row['steps_mean_mean']):.2f} +/- {float(row['steps_mean_sd']):.2f} | {float(row['total_volume_mean_ml_mean']):.2f} +/- {float(row['total_volume_mean_ml_sd']):.2f} |")
    lines.extend(["", "## Initialization-paired tests", "", f"The task-level McNemar test is conditional on the frozen model from each seed. The seed-level difference and sign-flip test treat the {len(args.seeds)} training run(s) as the independent units.", "", "| Algorithm | Random - imitation success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |", "|---|---:|---|---:|---:|"])
    for row in tests:
        lines.append(f"| {row['algorithm'].upper()} | {float(row['seed_success_difference_random_minus_imitation_pp_mean']):.2f} | [{float(row['seed_difference_bootstrap_95ci_low']):.2f}, {float(row['seed_difference_bootstrap_95ci_high']):.2f}] | {float(row['task_level_mcnemar_holm_p']):.4g} | {float(row['seed_sign_flip_p']):.4g} |")
    lines.extend([
        "",
        "## Algorithm-paired tests",
        "",
        f"Differences are reported as algorithm B minus algorithm A in success-rate percentage points. Task-level tests are conditional on the frozen trained models; seed-level results use the {len(args.seeds)} independent training run(s).",
        "",
        "| Initialization | Algorithm A | Algorithm B | B - A success (percentage points) | 95% bootstrap CI | Task-level Holm-adjusted McNemar p | Seed sign-flip p |",
        "|---|---|---|---:|---|---:|---:|",
    ])
    for row in algorithm_tests:
        lines.append(
            f"| {row['initialization']} | {row['algorithm_a'].upper()} | {row['algorithm_b'].upper()} | "
            f"{float(row['seed_success_difference_b_minus_a_pp_mean']):.2f} | "
            f"[{float(row['seed_difference_bootstrap_95ci_low']):.2f}, {float(row['seed_difference_bootstrap_95ci_high']):.2f}] | "
            f"{float(row['task_level_mcnemar_holm_p']):.4g} | {float(row['seed_sign_flip_p']):.4g} |"
        )
    lines.extend(["", "## Interpretation guardrail", "", "The random-initialization condition is an independent training control, not a claim that random initialization is universally inferior. Conclusions should be based on paired seed-level differences, uncertainty, learning curves, and the prespecified protocol. These results evaluate volume-policy learning with a shared external direction rule; they are not policy-only direction-and-volume control results.", ""])
    (output_dir / "RESULT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def validate_results(
    output_dir: Path,
    args: argparse.Namespace,
    curve_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> None:
    errors: list[str] = []
    expected_conditions = len(args.algorithms) * 2 * len(args.seeds)
    expected_task_rows = expected_conditions * args.eval_tasks
    expected_checkpoints = 1 + int(math.ceil(args.train_steps / args.eval_interval))
    maximum_expected_curve_rows = expected_conditions * expected_checkpoints
    conditions = {
        (str(row["algorithm"]), str(row["initialization"]), int(row["seed"]))
        for row in task_rows
    }
    if len(conditions) != expected_conditions:
        errors.append(f"Expected {expected_conditions} trained conditions, found {len(conditions)}")
    if len(task_rows) != expected_task_rows:
        errors.append(f"Expected {expected_task_rows} task rows, found {len(task_rows)}")
    curve_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in curve_rows:
        key = (str(row["algorithm"]), str(row["initialization"]), int(row["seed"]))
        curve_groups.setdefault(key, []).append(row)
    for condition in conditions:
        rows = sorted(curve_groups.get(condition, []), key=lambda x: int(x["environment_steps"]))
        if len(rows) < 2:
            errors.append(f"Missing zero/final learning-curve checkpoints for {condition}")
            continue
        if int(rows[0]["environment_steps"]) != 0:
            errors.append(f"First learning-curve checkpoint is not zero for {condition}")
        if int(rows[-1]["environment_steps"]) < args.train_steps:
            errors.append(f"Final learning-curve checkpoint is below the training budget for {condition}")
        if len(rows) > expected_checkpoints:
            errors.append(f"Too many learning-curve checkpoints for {condition}")

    run_configs: list[dict[str, Any]] = []
    for algorithm, initialization, seed in sorted(conditions):
        run_dir = output_dir / "runs" / f"{algorithm}_{initialization}_seed{seed}"
        config_path = run_dir / "run_config.json"
        model_path = run_dir / "final_model.pth"
        if not config_path.exists():
            errors.append(f"Missing run config: {config_path}")
            continue
        if not model_path.exists():
            errors.append(f"Missing final model: {model_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_configs.append(config)
        if config.get("protocol_id") != PROTOCOL_ID:
            errors.append(f"Wrong protocol id in {config_path}")
        if config.get("external_direction_rule") != "base if measured_ph < target_ph else acid":
            errors.append(f"Wrong direction rule in {config_path}")
        if config.get("overshoot_action_mask") is not False:
            errors.append(f"Overshoot masking was not disabled in {config_path}")
        if config.get("automatic_titrant_switching") is not False:
            errors.append(f"Automatic titrant switching was not disabled in {config_path}")

    for seed in args.seeds:
        random_hashes = {
            str(config.get("initial_actor_sha256"))
            for config in run_configs
            if config.get("initialization") == "random" and int(config.get("seed")) == seed
        }
        if len(random_hashes) != 1:
            errors.append(f"Random actor initialization is not paired across algorithms for seed {seed}")
    imitation_hashes = {
        str(config.get("initial_actor_sha256"))
        for config in run_configs
        if config.get("initialization") == "imitation"
    }
    if len(imitation_hashes) != 1:
        errors.append("Imitation actor initialization differs across runs")

    validation = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_id": PROTOCOL_ID,
        "condition_count": len(conditions),
        "task_row_count": len(task_rows),
        "learning_curve_row_count": len(curve_rows),
        "expected_task_rows": expected_task_rows,
        "maximum_expected_learning_curve_rows": maximum_expected_curve_rows,
        "external_direction_rule": "base if measured_ph < target_ph else acid",
        "policy_action": "1000 volume classes from 0.01 to 10.00 mL",
        "overshoot_action_mask": False,
        "automatic_titrant_switching": False,
    }
    (output_dir / "RESULT_VALIDATION.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if errors:
        raise RuntimeError("Result validation failed: " + "; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Direction-assisted volume-policy RL algorithm and initialization comparison.")
    parser.add_argument("--algorithms", nargs="+", choices=["reinforce", "a2c", "ppo"], default=["reinforce", "a2c", "ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--training-pool-size", type=int, default=5000)
    parser.add_argument("--eval-tasks", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo-batch-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--imitation-weights", type=Path, default=base / "models" / "imitation.pth")
    parser.add_argument("--output-dir", type=Path, default=base / "results_full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.resume = not args.no_resume
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.imitation_weights.exists():
        raise FileNotFoundError(f"Imitation weights not found: {args.imitation_weights}")
    device = torch.device(args.device)
    all_curve_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    manifest_dir = args.output_dir / "task_manifests"
    for seed in args.seeds:
        train_tasks = generate_tasks(seed + 100_000, args.training_pool_size, StressScenario("nominal"))
        eval_tasks = generate_tasks(seed + 200_000, args.eval_tasks, StressScenario("nominal"))
        write_csv(manifest_dir / f"train_tasks_seed{seed}.csv", task_manifest_rows(train_tasks))
        write_csv(manifest_dir / f"eval_tasks_seed{seed}.csv", task_manifest_rows(eval_tasks))
        for algorithm in args.algorithms:
            for initialization in ("imitation", "random"):
                run_name = f"{algorithm}_{initialization}_seed{seed}"
                run_dir = args.output_dir / "runs" / run_name
                curve, task_rows = train_one(
                    algorithm, initialization, args.imitation_weights.resolve(), train_tasks, eval_tasks, args, seed, run_dir
                )
                all_curve_rows.extend({"algorithm": algorithm, "initialization": initialization, "seed": seed, **row} for row in curve)
                all_task_rows.extend({"algorithm": algorithm, "initialization": initialization, "seed": seed, **row} for row in task_rows)
    write_csv(args.output_dir / "learning_curves.csv", all_curve_rows)
    write_csv(args.output_dir / "task_results.csv", all_task_rows)
    summarize_all(all_task_rows, all_curve_rows, args.output_dir)
    validate_results(args.output_dir, args, all_curve_rows, all_task_rows)
    plot_results(args.output_dir, all_curve_rows, read_csv(args.output_dir / "seed_summary.csv"))
    build_report(args.output_dir, args)
    settings = vars(args).copy()
    settings["protocol_id"] = PROTOCOL_ID
    settings["policy_action"] = "1000 volume classes from 0.01 to 10.00 mL"
    settings["external_direction_rule"] = "base if measured_ph < target_ph else acid"
    settings["titrant_concentration_m"] = 0.1
    settings["overshoot_action_mask"] = False
    settings["automatic_titrant_switching"] = False
    settings["random_actor_seed_offset"] = RANDOM_ACTOR_SEED_OFFSET
    settings["imitation_weights"] = str(args.imitation_weights.resolve())
    settings["output_dir"] = str(args.output_dir.resolve())
    settings["torch_version"] = torch.__version__
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "RUN_COMPLETE.txt").write_text(f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
