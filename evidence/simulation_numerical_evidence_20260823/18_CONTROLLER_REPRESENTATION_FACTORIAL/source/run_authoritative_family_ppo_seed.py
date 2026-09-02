from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path


FAMILY_DOMAINS = {
    "F2": "sequential_k123",
    "F3": "sequential_k123",
    "F4": "independent_j123",
    "F5": "sequential_k123",
    "F6": "independent_j123",
}


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("independent_j123_ppo_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILY_DOMAINS, required=True)
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--independent-generator", type=Path, required=True)
    parser.add_argument("--imitation-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--interactions", type=int, default=100000)
    parser.add_argument("--training-pool-size", type=int, default=5000)
    parser.add_argument("--validation-tasks", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.formal_source.resolve(), args.staging_root.resolve()):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_full_pf_factorial as runner
    runner.configure(args.formal_source, args.staging_root)
    import policy_family_environment as environment
    import train_ppo as training

    if FAMILY_DOMAINS[args.family] == "independent_j123":
        generator = load_generator(args.independent_generator)

        def generate_tasks(seed: int, count: int, split: str):
            tasks = []
            for task in generator.generate_seed(seed, count):
                row = asdict(task)
                row["split"] = split
                tasks.append(environment.task_object(row))
            return tasks

        training.generate_tasks = generate_tasks
        training.load_tasks = environment.load_tasks
        training.save_tasks = environment.save_tasks
        training.ControlEnvironment = environment.GenericControlEnvironment
        training.evaluate_actor = environment.evaluate_actor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old = sys.argv
    sys.argv = [
        "train_ppo.py",
        "--imitation-checkpoint", str(args.imitation_checkpoint),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--seeds", str(args.seed),
        "--train-interactions", str(args.interactions),
        "--training-pool-size", str(args.training_pool_size),
        "--validation-tasks", str(args.validation_tasks),
        "--eval-interval", "10000",
        "--ppo-batch-steps", "2048",
        "--ppo-epochs", "4",
        "--minibatch-size", "256",
        "--device", "cpu",
        *( ["--resume"] if args.resume else [] ),
    ]
    try:
        training.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
