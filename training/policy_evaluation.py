from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

try:
    from .control_environment import ControlEnvironment, DomainRandomization
    from .models import StateNormalizer, VolumeActor
except ImportError:  # pragma: no cover - direct script compatibility
    from control_environment import ControlEnvironment, DomainRandomization
    from models import StateNormalizer, VolumeActor


def actor_volume(
    actor: VolumeActor,
    normalizer: StateNormalizer,
    state: np.ndarray,
    device: torch.device,
    stochastic: bool = False,
) -> float:
    normalized = normalizer.transform_numpy(state)
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = actor(tensor)
        if stochastic:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            action = int(torch.argmax(logits, dim=1).item())
    return (action + 1) * 0.01


def rollout_actor(
    actor: VolumeActor,
    normalizer: StateNormalizer,
    task,
    device: torch.device,
    seed: int,
    domain: DomainRandomization | None = None,
) -> dict:
    env = ControlEnvironment(task, np.random.default_rng(seed), domain)
    actor.eval()
    while not env.done:
        env.step(actor_volume(actor, normalizer, env.state(), device, stochastic=False))
    return {
        "task_seed": task.seed,
        "task_id": task.task_id,
        "acid_type": task.acid_type,
        "difficulty": task.difficulty,
        "direction": task.direction,
        "pka_family": task.pka_family,
        "true_pair_count": len(task.pka_values),
        "true_concentration_m": task.analyte_conc_m,
        "initial_volume_ml": task.initial_volume_ml,
        "initial_ph": task.initial_ph,
        "target_ph": task.target_ph,
        **env.metrics(),
    }


def evaluate_actor(actor, normalizer, tasks, device, seed_offset=0, domain=None):
    return [
        rollout_actor(
            actor,
            normalizer,
            task,
            device,
            seed_offset + task.seed * 1_000_003 + task.task_id,
            domain,
        )
        for task in tasks
    ]


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    success = np.asarray([int(row["true_success"]) for row in rows], dtype=float)
    successful_steps = np.asarray(
        [int(row["steps"]) for row in rows if int(row["true_success"])],
        dtype=float,
    )
    return {
        "tasks": len(rows),
        "success_rate_percent": 100.0 * float(np.mean(success)),
        "strict_success_rate_percent": 100.0 * float(np.mean([int(row["strict_success"]) for row in rows])),
        "severe_failure_rate_percent": 100.0 * float(np.mean([int(row["severe_failure"]) for row in rows])),
        "false_stop_rate_percent": 100.0 * float(np.mean([int(row["false_stop"]) for row in rows])),
        "successful_steps_mean": float(np.mean(successful_steps)) if len(successful_steps) else math.nan,
        "steps_mean": float(np.mean([int(row["steps"]) for row in rows])),
        "overshoots_mean": float(np.mean([int(row["overshoots"]) for row in rows])),
        "total_volume_mean_ml": float(np.mean([float(row["total_volume_ml"]) for row in rows])),
        "final_abs_error_mean": float(np.mean([float(row["final_abs_error"]) for row in rows])),
    }
