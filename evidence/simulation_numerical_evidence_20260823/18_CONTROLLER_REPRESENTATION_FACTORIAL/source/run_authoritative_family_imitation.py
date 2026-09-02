from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-tasks", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for path in (args.formal_source.resolve(), args.staging_root.resolve()):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_full_pf_factorial as runner
    runner.configure(args.formal_source, args.staging_root)
    import policy_family_environment as environment
    import train_imitation as training

    training.load_tasks = environment.load_tasks
    training.evaluate_actor = environment.evaluate_actor
    old = sys.argv
    sys.argv = [
        "train_imitation.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--epochs", str(args.epochs),
        "--batch-size", "512",
        "--closed-loop-validation-tasks", str(args.validation_tasks),
        "--early-stopping-patience", str(args.patience),
        "--device", "cpu",
        *( ["--resume"] if args.resume else [] ),
    ]
    try:
        training.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
