from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .control_environment import ControlEnvironment, sample_training_domain
    from .models import ActorCritic, StateNormalizer, checkpoint_sha256, load_actor_checkpoint
    from .policy_evaluation import evaluate_actor, summarize_rows
    from .protocol import training_checkpoint_protocol_metadata
    from .task_distribution import generate_tasks, save_tasks
except ImportError:  # pragma: no cover - direct script compatibility
    from control_environment import ControlEnvironment, sample_training_domain
    from models import ActorCritic, StateNormalizer, checkpoint_sha256, load_actor_checkpoint
    from policy_evaluation import evaluate_actor, summarize_rows
    from protocol import training_checkpoint_protocol_metadata
    from task_distribution import generate_tasks, save_tasks


@dataclass
class PPOBatch:
    states: list[np.ndarray]
    actions: list[int]
    old_log_probs: list[float]
    returns: list[float]
    advantages: list[float]


REWARD_VARIANTS = (
    "full",
    "no_progress",
    "no_step_cost",
    "no_volume_cost",
    "no_overshoot_penalty",
    "no_terminal_success",
    "no_terminal_failure",
)


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reward_value(
    previous_ph,
    current_ph,
    target_ph,
    volume_ml,
    crossed,
    done,
    success,
    strict,
    reward_variant="full",
    step_cost=0.005,
):
    if reward_variant not in REWARD_VARIANTS:
        raise ValueError(f"Unknown reward variant: {reward_variant}")
    if not math.isfinite(step_cost) or step_cost < 0.0:
        raise ValueError(f"step_cost must be a finite non-negative number: {step_cost}")
    previous_error = abs(previous_ph - target_ph)
    current_error = abs(current_ph - target_ph)
    progress = float(np.clip(previous_error - current_error, -2.0, 2.0))
    reward = 0.0 if reward_variant == "no_progress" else 0.50 * progress
    if reward_variant != "no_step_cost":
        reward -= step_cost
    if reward_variant != "no_volume_cost":
        reward -= 0.002 * (volume_ml / 10.0)
    if (
        reward_variant != "no_overshoot_penalty"
        and crossed
        and current_error > 0.10
    ):
        reward -= 0.10 * min(1.0, current_error)
    if done:
        if success:
            if reward_variant != "no_terminal_success":
                reward += 4.0 + (0.5 if strict else 0.0)
        elif reward_variant != "no_terminal_failure":
            reward -= 1.0 + min(2.0, current_error)
    return float(reward)


def rollout_episode(
    model,
    normalizer,
    task,
    device,
    rng,
    gamma,
    gae_lambda,
    reward_variant="full",
    step_cost=0.005,
):
    env = ControlEnvironment(task, rng, sample_training_domain(rng))
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    while not env.done:
        state = env.state()
        normalized = normalizer.transform_numpy(state)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model.actor(tensor)
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            log_prob = float(distribution.log_prob(action_tensor).item())
            value = float(model.critic(tensor).squeeze().item())
        previous_ph = env.measured_ph
        info = env.step((action + 1) * 0.01)
        success = abs(env.true_ph - env.target_ph) < 0.10
        strict = abs(env.true_ph - env.target_ph) < 0.05
        reward = reward_value(
            previous_ph,
            env.measured_ph,
            env.target_ph,
            (action + 1) * 0.01,
            bool(info["crossed_target"]),
            env.done,
            success,
            strict,
            reward_variant,
            step_cost,
        )
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return states, actions, log_probs, returns.tolist(), advantages.tolist(), env.metrics()


def ppo_update(model, optimizer, batch: PPOBatch, normalizer, device, epochs, minibatch_size, clip_ratio, entropy_coef):
    states = torch.as_tensor(
        normalizer.transform_numpy(np.stack(batch.states)),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch.old_log_probs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    indices = np.arange(len(actions))
    losses = []
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            selected = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=device)
            logits = model.actor(states[selected])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[selected])
            ratio = torch.exp(new_log_probs - old_log_probs[selected])
            unclipped = ratio * advantages[selected]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[selected]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            values = model.critic(states[selected]).squeeze(-1)
            critic_loss = F.mse_loss(values, returns[selected])
            entropy = distribution.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((float(actor_loss.item()), float(critic_loss.item()), float(entropy.item())))
    return {
        "actor_loss": float(np.mean([item[0] for item in losses])),
        "critic_loss": float(np.mean([item[1] for item in losses])),
        "entropy": float(np.mean([item[2] for item in losses])),
    }


def save_ppo_checkpoint(path, model, normalizer, metadata):
    metadata = dict(metadata)
    metadata.setdefault("protocol", training_checkpoint_protocol_metadata())
    validation = metadata.get("validation")
    if isinstance(validation, dict):
        # Keep internal strict-threshold diagnostics in result tables, but do
        # not embed them in released policy metadata.
        metadata["validation"] = {
            key: value
            for key, value in validation.items()
            if key != "strict_success_rate_percent"
        }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "actor_state_dict": model.actor.state_dict(),
            "state_mean": normalizer.mean,
            "state_std": normalizer.std,
            "metadata": metadata,
        },
        path,
    )


def train_seed(seed, imitation_path, args, output_dir):
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "COMPLETE.json"
    if args.resume and complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    imitation_actor, normalizer, imitation_metadata = load_actor_checkpoint(
        imitation_path,
        device,
        expected_file_sha256=getattr(args, "imitation_checkpoint_sha256", None),
    )
    model = ActorCritic().to(device)
    model.actor.load_state_dict(imitation_actor.state_dict(), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_tasks = generate_tasks(seed + 610_000, args.training_pool_size, "ppo_train")
    validation_tasks = generate_tasks(seed + 710_000, args.validation_tasks, "ppo_validation")
    save_tasks(run_dir / "training_tasks.jsonl", train_tasks)
    save_tasks(run_dir / "validation_tasks.jsonl", validation_tasks)
    initial_rows = evaluate_actor(model.actor, normalizer, validation_tasks, device, seed_offset=seed * 101)
    initial_summary = summarize_rows(initial_rows)
    curve = [{"environment_steps": 0, **initial_summary}]
    best_key = (
        float(initial_summary["success_rate_percent"]),
        -float(initial_summary["final_abs_error_mean"]),
        -float(initial_summary["steps_mean"]),
    )
    best_path = run_dir / "best_ppo.pth"
    save_ppo_checkpoint(
        best_path,
        model,
        normalizer,
        {
            "seed": seed,
            "environment_steps": 0,
            "validation": initial_summary,
            "source": "imitation_start",
            "reward_variant": args.reward_variant,
            "step_cost": args.step_cost,
        },
    )

    rng = np.random.default_rng(seed + 810_000)
    interactions = 0
    next_evaluation = args.eval_interval
    batch = PPOBatch([], [], [], [], [])
    update_metrics = {"actor_loss": math.nan, "critic_loss": math.nan, "entropy": math.nan}
    start_time = time.perf_counter()
    while interactions < args.train_interactions:
        task = train_tasks[int(rng.integers(0, len(train_tasks)))]
        episode = rollout_episode(
            model,
            normalizer,
            task,
            device,
            rng,
            args.gamma,
            args.gae_lambda,
            args.reward_variant,
            args.step_cost,
        )
        states, actions, log_probs, returns, advantages, _ = episode
        batch.states.extend(states)
        batch.actions.extend(actions)
        batch.old_log_probs.extend(log_probs)
        batch.returns.extend(returns)
        batch.advantages.extend(advantages)
        interactions += len(states)
        if len(batch.states) >= args.ppo_batch_steps:
            update_metrics = ppo_update(
                model,
                optimizer,
                batch,
                normalizer,
                device,
                args.ppo_epochs,
                args.minibatch_size,
                args.clip_ratio,
                args.entropy_coefficient,
            )
            batch = PPOBatch([], [], [], [], [])
        if interactions >= next_evaluation or interactions >= args.train_interactions:
            validation_rows = evaluate_actor(
                model.actor,
                normalizer,
                validation_tasks,
                device,
                seed_offset=seed * 10_000_019 + interactions,
            )
            summary = summarize_rows(validation_rows)
            curve.append({"environment_steps": interactions, **summary, **update_metrics})
            write_csv(run_dir / "learning_curve.csv", curve)
            key = (
                float(summary["success_rate_percent"]),
                -float(summary["final_abs_error_mean"]),
                -float(summary["steps_mean"]),
            )
            if key > best_key:
                best_key = key
                save_ppo_checkpoint(
                    best_path,
                    model,
                    normalizer,
                    {
                        "seed": seed,
                        "environment_steps": interactions,
                        "validation": summary,
                        "source": "ppo",
                        "reward_variant": args.reward_variant,
                        "step_cost": args.step_cost,
                    },
                )
            print(
                f"PPO seed {seed}: {interactions} interactions, "
                f"validation success={summary['success_rate_percent']:.2f}%",
                flush=True,
            )
            next_evaluation += args.eval_interval

    if batch.states:
        update_metrics = ppo_update(
            model,
            optimizer,
            batch,
            normalizer,
            device,
            args.ppo_epochs,
            args.minibatch_size,
            args.clip_ratio,
            args.entropy_coefficient,
        )
        validation_rows = evaluate_actor(
            model.actor,
            normalizer,
            validation_tasks,
            device,
            seed_offset=seed * 10_000_019 + interactions + 1,
        )
        summary = summarize_rows(validation_rows)
        final_row = {"environment_steps": interactions, **summary, **update_metrics, "post_final_batch": 1}
        if curve and int(curve[-1]["environment_steps"]) == interactions:
            curve[-1] = final_row
        else:
            curve.append(final_row)
        write_csv(run_dir / "learning_curve.csv", curve)
        key = (
            float(summary["success_rate_percent"]),
            -float(summary["final_abs_error_mean"]),
            -float(summary["steps_mean"]),
        )
        if key > best_key:
            best_key = key
            save_ppo_checkpoint(
                best_path,
                model,
                normalizer,
                {
                    "seed": seed,
                    "environment_steps": interactions,
                    "validation": summary,
                    "source": "ppo_final_batch",
                    "reward_variant": args.reward_variant,
                    "step_cost": args.step_cost,
                },
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    payload = {
        "training_seed": seed,
        "reward_variant": args.reward_variant,
        "step_cost": args.step_cost,
        "elapsed_seconds": time.perf_counter() - start_time,
        "train_interactions": interactions,
        "best_checkpoint": str(best_path),
        "best_checkpoint_source": checkpoint["metadata"]["source"],
        "best_environment_steps": checkpoint["metadata"]["environment_steps"],
        "best_validation": checkpoint["metadata"]["validation"],
        "actor_sha256": checkpoint_sha256(checkpoint["actor_state_dict"]),
        "imitation_metadata": imitation_metadata,
    }
    complete.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="PPO refinement from the new PF-distilled imitation policy")
    parser.add_argument("--imitation-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--imitation-checkpoint-sha256",
        default=None,
        help="Optional SHA-256 digest checked before loading the imitation checkpoint.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 555])
    parser.add_argument("--train-interactions", type=int, default=100000)
    parser.add_argument("--training-pool-size", type=int, default=5000)
    parser.add_argument(
        "--validation-tasks",
        type=int,
        default=500,
        help="Validation tasks used for checkpoint selection (publication workflow: 500).",
    )
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--ppo-batch-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--reward-variant", choices=REWARD_VARIANTS, default="full")
    parser.add_argument("--step-cost", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        train_seed(seed, args.imitation_checkpoint, args, args.output_dir)
        for seed in args.seeds
    ]
    (args.output_dir / "PPO_COMPLETE.json").write_text(
        json.dumps({"seeds": args.seeds, "runs": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
