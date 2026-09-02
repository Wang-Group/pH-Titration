from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset

from models import ACTION_VOLUMES_ML, StateNormalizer, VolumeActor, checkpoint_sha256, save_actor_checkpoint
from policy_evaluation import evaluate_actor, summarize_rows
from task_distribution import load_tasks


class DirectionBalancedSampler(Sampler[int]):
    """Yield equal acid/base state counts while retaining every majority state."""

    def __init__(self, direction_codes: np.ndarray, seed: int):
        self.groups = {
            code: np.flatnonzero(direction_codes == code).astype(np.int64)
            for code in (0, 1)
        }
        if any(len(indices) == 0 for indices in self.groups.values()):
            raise ValueError("Imitation data must contain both acid- and base-direction states")
        self.samples_per_direction = max(len(indices) for indices in self.groups.values())
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return 2 * self.samples_per_direction

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        selected = []
        for code in (0, 1):
            indices = self.groups[code]
            if len(indices) == self.samples_per_direction:
                values = rng.permutation(indices)
            else:
                values = rng.choice(indices, size=self.samples_per_direction, replace=True)
            selected.extend(int(value) for value in values)
        rng.shuffle(selected)
        self.epoch += 1
        return iter(selected)


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification_metrics(actor, normalizer, states, labels, device, batch_size=2048):
    actor.eval()
    volumes = torch.as_tensor(ACTION_VOLUMES_ML, dtype=torch.float32, device=device)
    predicted = []
    expected = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.as_tensor(
                normalizer.transform_numpy(states[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            logits = actor(batch)
            predicted.append(torch.argmax(logits, dim=1).cpu().numpy())
            expected.append((torch.softmax(logits, dim=1) @ volumes).cpu().numpy())
    predicted = np.concatenate(predicted)
    expected = np.concatenate(expected)
    target_volume = (labels + 1) * 0.01
    class_distance = np.abs(predicted - labels)
    return {
        "top1_class_accuracy_percent": 100.0 * float(np.mean(predicted == labels)),
        "within_0p05ml_percent": 100.0 * float(np.mean(class_distance <= 5)),
        "within_0p10ml_percent": 100.0 * float(np.mean(class_distance <= 10)),
        "argmax_volume_mae_ml": float(np.mean(class_distance) * 0.01),
        "expected_volume_mae_ml": float(np.mean(np.abs(expected - target_volume))),
    }


def train_seed(
    seed,
    train_data,
    val_data,
    validation_tasks,
    args,
    normalizer,
    output_dir,
):
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "COMPLETE.json"
    if args.resume and complete_path.exists():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    actor = VolumeActor().to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    train_states = normalizer.transform_numpy(train_data["states"])
    dataset = TensorDataset(
        torch.as_tensor(train_states, dtype=torch.float32),
        torch.as_tensor(train_data["labels"], dtype=torch.long),
        torch.as_tensor(train_data["weights"], dtype=torch.float32),
    )
    direction_codes = (train_data["states"][:, 0] < train_data["states"][:, 1]).astype(np.int8)
    sampler = DirectionBalancedSampler(direction_codes, seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    action_volumes = torch.as_tensor(ACTION_VOLUMES_ML, dtype=torch.float32, device=device)
    history = []
    best_key = None
    best_path = run_dir / "best_imitation.pth"
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        actor.train()
        loss_sum = 0.0
        count = 0
        for states, labels, weights in loader:
            states = states.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = actor(states)
            ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=0.01)
            probabilities = torch.softmax(logits, dim=1)
            expected_volume = probabilities @ action_volumes
            target_volume = (labels.float() + 1.0) * 0.01
            distance = F.smooth_l1_loss(
                expected_volume / 10.0,
                target_volume / 10.0,
                reduction="none",
                beta=0.02,
            )
            loss = torch.mean(weights * (ce + args.volume_loss_weight * distance))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(states)
            count += len(states)

        class_metrics = classification_metrics(
            actor,
            normalizer,
            val_data["states"],
            val_data["labels"],
            device,
        )
        closed_rows = evaluate_actor(
            actor,
            normalizer,
            validation_tasks[: args.closed_loop_validation_tasks],
            device,
            seed_offset=seed * 10_000_019,
        )
        closed = summarize_rows(closed_rows)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(1, count),
            **class_metrics,
            **{f"closed_loop_{key}": value for key, value in closed.items()},
        }
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        key = (
            float(closed["success_rate_percent"]),
            -float(closed["final_abs_error_mean"]),
            -float(class_metrics["argmax_volume_mae_ml"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            patience_counter = 0
            save_actor_checkpoint(
                best_path,
                actor,
                normalizer,
                {
                    "training_seed": seed,
                    "selected_epoch": epoch,
                    "selection_metrics": row,
                    "teacher": "pf_pka_conc_variable_k",
                },
            )
        else:
            patience_counter += 1
        print(
            f"imitation seed {seed} epoch {epoch}: "
            f"success={closed['success_rate_percent']:.2f}% "
            f"volume_mae={class_metrics['argmax_volume_mae_ml']:.3f} mL",
            flush=True,
        )
        if patience_counter >= args.early_stopping_patience:
            break
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    selected = checkpoint["metadata"]["selection_metrics"]
    result = {
        "seed": seed,
        "checkpoint": str(best_path),
        "actor_sha256": checkpoint_sha256(checkpoint["actor_state_dict"]),
        "raw_acid_direction_states": int(np.sum(direction_codes == 0)),
        "raw_base_direction_states": int(np.sum(direction_codes == 1)),
        "samples_per_direction_per_epoch": int(sampler.samples_per_direction),
        "direction_sampling": "exact 1:1 acid/base state sampling in every epoch",
        **selected,
    }
    complete_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a new imitation policy from robust PF actions")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--volume-loss-weight", type=float, default=0.20)
    parser.add_argument("--closed-loop-validation-tasks", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "imitation_best.pth"
    complete_path = args.output_dir / "IMITATION_COMPLETE.json"
    if args.resume and selected_path.exists() and complete_path.exists():
        print(f"Imitation training already complete: {complete_path}")
        return

    train_data = np.load(args.data_dir / "train_teacher_dataset.npz")
    val_data = np.load(args.data_dir / "validation_teacher_dataset.npz")
    validation_tasks = load_tasks(args.data_dir / "validation_tasks.jsonl")
    mean = np.mean(train_data["states"], axis=0).astype(np.float32)
    std = np.std(train_data["states"], axis=0).astype(np.float32)
    std = np.maximum(std, 1e-3)
    normalizer = StateNormalizer(mean, std)
    candidates = [
        train_seed(seed, train_data, val_data, validation_tasks, args, normalizer, args.output_dir)
        for seed in args.seeds
    ]
    candidates.sort(
        key=lambda row: (
            float(row["closed_loop_success_rate_percent"]),
            -float(row["closed_loop_final_abs_error_mean"]),
            -float(row["argmax_volume_mae_ml"]),
        ),
        reverse=True,
    )
    winner = candidates[0]
    shutil.copy2(winner["checkpoint"], selected_path)
    write_csv(args.output_dir / "imitation_seed_comparison.csv", candidates)
    payload = {
        "selected_seed": winner["seed"],
        "selected_checkpoint": str(selected_path),
        "selection_rule": "validation closed-loop success, then final error, then teacher-volume MAE",
        "candidates": candidates,
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
    }
    complete_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
