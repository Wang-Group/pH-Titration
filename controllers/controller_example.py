from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .new_pf_controller import RobustPFController
    from .new_rl_controller import PPOVolumeController
except ImportError:  # pragma: no cover - direct script compatibility
    from new_pf_controller import RobustPFController
    from new_rl_controller import PPOVolumeController


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ppo_seed_303.pth"


def interactive_loop(controller) -> None:
    while True:
        action = controller.recommend()
        print(json.dumps(action.as_dict(), ensure_ascii=False, indent=2))
        if action.stop:
            break
        measured = float(input("执行动作后输入新的测量 pH: ").strip())
        actual_text = input(
            f"输入实际加液体积 mL，直接回车表示 {action.volume_ml:.2f}: "
        ).strip()
        actual = action.volume_ml if not actual_text else float(actual_text)
        status = controller.observe(measured, actual, action.reagent)
        print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive PF/PPO controller example")
    parser.add_argument("controller", choices=("pf", "ppo"))
    parser.add_argument("--initial-ph", type=float, required=True)
    parser.add_argument("--target-ph", type=float, required=True)
    parser.add_argument("--initial-volume-ml", type=float, default=11.0)
    parser.add_argument("--initial-base-moles", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.controller == "pf":
        controller = RobustPFController(seed=args.seed)
        controller.reset(
            args.initial_ph,
            args.target_ph,
            args.initial_volume_ml,
            args.initial_base_moles,
        )
    else:
        controller = PPOVolumeController(args.checkpoint, device=args.device)
        controller.reset(args.initial_ph, args.target_ph)
    interactive_loop(controller)


if __name__ == "__main__":
    main()
