from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BLOCK = (
    ROOT
    / "evidence"
    / "simulation_numerical_evidence_20260823"
    / "03_PPO_TRAINING_STABILITY"
)
SEEDS = (101, 202, 303, 404, 555)
EXPECTED_ACTOR_SHA256 = {
    101: "1cf2ed57408d34b78d8858285f48d66e584ade26ef952ef721352ea3198c3f4c",
    202: "a79394a7474d80803c798514003abe063f5498fb5dbf9f6f41337277b4f2bb18",
    303: "8c2ccdbc7879d1d54b151eeccb76ed3d354e58ace23e8052f6619d57369214fb",
    404: "e96c221227d911fa8ceef5a18085dd3ac1fd2fd8b0b5fbf37e9726371479684d",
    555: "d3e3fccfb9970e6ac3803c5b83bfba54d3ca5f77a7b39e354de096a7ce3d0a2b",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_sha256(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["actor_state_dict"]
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    config = json.loads((BLOCK / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    if config.get("study_id") != "ppo_training_stability":
        raise SystemExit("PPO stability RUN_CONFIG has the wrong study identity")
    if config.get("ppo_validation_tasks_per_seed") != 500:
        raise SystemExit("PPO stability validation size must be 500 tasks per seed")
    if count_jsonl(BLOCK / "common_locked_test_tasks.jsonl") != 1000:
        raise SystemExit("Common PPO stability test manifest must contain 1,000 tasks")

    manifest_keys = []
    with (BLOCK / "common_locked_test_tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest_keys.append((int(row["seed"]), int(row["task_id"])))
    if len(set(manifest_keys)) != 1000:
        raise SystemExit("Common PPO stability test manifest has duplicate task keys")

    success_rates: list[float] = []
    checkpoint_rows = []
    for seed in SEEDS:
        run = BLOCK / "runs" / f"seed_{seed}"
        if count_jsonl(run / "training_tasks.jsonl") != 5000:
            raise SystemExit(f"Seed {seed} training manifest must contain 5,000 tasks")
        if count_jsonl(run / "validation_tasks.jsonl") != 500:
            raise SystemExit(f"Seed {seed} validation manifest must contain 500 tasks")
        rows = read_csv(run / "locked_test_results.csv")
        if len(rows) != 1000:
            raise SystemExit(f"Seed {seed} locked outcomes must contain 1,000 rows")
        keys = [(int(row["task_seed"]), int(row["task_id"])) for row in rows]
        if keys != manifest_keys:
            raise SystemExit(f"Seed {seed} locked outcomes do not match the common manifest")
        success_rate = 100.0 * sum(int(row["true_success"]) for row in rows) / len(rows)
        success_rates.append(success_rate)

        checkpoint = BLOCK / "checkpoints" / f"ppo_seed_{seed}.pth"
        observed_actor_hash = actor_sha256(checkpoint)
        if observed_actor_hash != EXPECTED_ACTOR_SHA256[seed]:
            raise SystemExit(f"Seed {seed} actor hash mismatch")
        checkpoint_rows.append(
            {
                "seed": seed,
                "file_sha256": file_sha256(checkpoint),
                "actor_sha256": observed_actor_hash,
            }
        )

    mean = statistics.mean(success_rates)
    sample_sd = statistics.stdev(success_rates)
    if abs(mean - 89.54) > 1e-12 or abs(sample_sd - 2.2266566866043833) > 1e-12:
        raise SystemExit(f"PPO stability summary mismatch: {mean=}, {sample_sd=}")

    aggregate = read_csv(BLOCK / "evaluation" / "aggregate_summary.csv")
    nominal = next(
        row for row in aggregate if row["suite"] == "nominal_locked" and row["method"] == "ppo"
    )
    if abs(float(nominal["success_rate_percent_mean"]) - mean) > 1e-12:
        raise SystemExit("Released PPO stability aggregate mean is inconsistent")
    if abs(float(nominal["success_rate_percent_sd"]) - sample_sd) > 1e-12:
        raise SystemExit("Released PPO stability aggregate sample SD is inconsistent")

    report = {
        "status": "PASS",
        "protocol_family": "pH-control",
        "protocol_version": "2026.08",
        "training_profile": "training_environment_strict",
        "seeds": list(SEEDS),
        "common_locked_test_tasks": len(manifest_keys),
        "success_rate_percent_by_seed": dict(zip(map(str, SEEDS), success_rates)),
        "success_rate_percent_mean": mean,
        "success_rate_percent_sample_sd": sample_sd,
        "checkpoints": checkpoint_rows,
    }
    (BLOCK / "PPO_STABILITY_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
