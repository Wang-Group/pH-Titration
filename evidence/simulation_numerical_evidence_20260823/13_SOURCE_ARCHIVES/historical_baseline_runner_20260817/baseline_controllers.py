from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from control_environment import ControlEnvironment, DomainRandomization


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.32
    ki: float = 0.012
    kd: float = 0.08
    integral_limit: float = 12.0
    overshoot_decay: float = 0.10
    minimum_output_ml: float = 0.01
    maximum_output_ml: float = 3.00

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PIDConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: float(value) for key, value in payload.items() if key in allowed})


PRESPECIFIED_PID = PIDConfig()


class FixedGainPIDController:
    def __init__(self, config: PIDConfig = PRESPECIFIED_PID) -> None:
        self.config = config

    def reset(self, current_ph: float, target_ph: float) -> None:
        del current_ph
        self.target_ph = float(target_ph)
        self.integral = 0.0
        self.previous_error: float | None = None

    def recommend(self, current_ph: float) -> float:
        error = self.target_ph - float(current_ph)
        if self.previous_error is not None and error * self.previous_error < 0.0:
            self.integral *= self.config.overshoot_decay
        self.integral = float(
            np.clip(
                self.integral + error,
                -self.config.integral_limit,
                self.config.integral_limit,
            )
        )
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error
        signed_output = (
            self.config.kp * error
            + self.config.ki * self.integral
            + self.config.kd * derivative
        )
        return float(
            np.clip(
                abs(signed_output),
                self.config.minimum_output_ml,
                self.config.maximum_output_ml,
            )
        )

    def observe(
        self,
        measured_ph: float,
        added_base: bool,
        actual_volume_ml: float,
        crossed: bool,
    ) -> None:
        del measured_ph, added_base, actual_volume_ml, crossed

    def parameters(self) -> dict[str, float]:
        return asdict(self.config)


class SimpleRuleController:
    """The disclosed error-bucket, half-after-crossing, bracket-refinement rule."""

    def reset(self, current_ph: float, target_ph: float) -> None:
        self.target_ph = float(target_ph)
        self.last_volume_ml: float | None = None
        self.last_overshot = False
        self.net_titrant_ml = 0.0
        self.lower_bracket = 0.0 if current_ph <= target_ph else None
        self.upper_bracket = 0.0 if current_ph >= target_ph else None

    @staticmethod
    def bucket(error: float) -> float:
        if error > 6.0:
            return 3.00
        if error > 4.0:
            return 2.50
        if error > 2.0:
            return 2.00
        if error > 1.0:
            return 1.00
        if error > 0.5:
            return 0.50
        if error > 0.2:
            return 0.20
        return 0.05

    def recommend(self, current_ph: float) -> float:
        error = abs(self.target_ph - float(current_ph))
        has_bracket = self.lower_bracket is not None and self.upper_bracket is not None
        if has_bracket:
            target_net = (self.lower_bracket + self.upper_bracket) / 2.0
            volume = max(0.01, abs(target_net - self.net_titrant_ml))
        else:
            volume = self.bucket(error)
            if self.last_overshot and self.last_volume_ml is not None:
                volume = max(0.01, self.last_volume_ml * 0.5)
        if error <= 0.30:
            volume = min(volume, 0.10)
        if error <= 0.15:
            volume = min(volume, 0.03)
        return float(np.clip(volume, 0.01, 10.0))

    def observe(
        self,
        measured_ph: float,
        added_base: bool,
        actual_volume_ml: float,
        crossed: bool,
    ) -> None:
        self.net_titrant_ml += actual_volume_ml if added_base else -actual_volume_ml
        if measured_ph < self.target_ph:
            self.lower_bracket = (
                self.net_titrant_ml
                if self.lower_bracket is None
                else max(self.lower_bracket, self.net_titrant_ml)
            )
        elif measured_ph > self.target_ph:
            self.upper_bracket = (
                self.net_titrant_ml
                if self.upper_bracket is None
                else min(self.upper_bracket, self.net_titrant_ml)
            )
        self.last_volume_ml = float(actual_volume_ml)
        self.last_overshot = bool(crossed)


def deterministic_task_rng(task) -> np.random.Generator:
    seed = int(task.seed * 1_000_003 + task.task_id) % (2**32 - 1)
    return np.random.default_rng(seed)


def rollout_baseline(task, controller) -> dict[str, Any]:
    env = ControlEnvironment(task, deterministic_task_rng(task), DomainRandomization())
    controller.reset(env.measured_ph, env.target_ph)
    while not env.done:
        before_ph = float(env.measured_ph)
        added_base = before_ph < env.target_ph
        volume_ml = controller.recommend(before_ph)
        info = env.step(volume_ml)
        controller.observe(
            float(env.measured_ph),
            added_base,
            float(info["actual_volume_ml"]),
            bool(info["crossed_target"]),
        )
    return env.metrics()
