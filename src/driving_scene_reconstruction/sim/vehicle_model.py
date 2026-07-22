"""Dependency-free ego vehicle motion model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .control import HumanControl
from .state import EgoState


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class SimpleVehicleModel:
    """A deterministic kinematic bicycle model for simulator smoke tests.

    The model treats normalized steering as a wheel angle, uses a linear
    throttle/brake acceleration approximation, and ignores tire slip, reverse
    gear, suspension, and collision response. It is an interface baseline, not
    a high-fidelity dynamics model.
    """

    wheelbase: float = 2.8
    max_steer_angle: float = math.radians(30.0)
    max_acceleration: float = 3.0
    max_braking: float = 7.0
    linear_drag: float = 0.1
    max_speed: float | None = None

    def __post_init__(self) -> None:
        parameters = {
            "wheelbase": self.wheelbase,
            "max_steer_angle": self.max_steer_angle,
            "max_acceleration": self.max_acceleration,
            "max_braking": self.max_braking,
            "linear_drag": self.linear_drag,
        }
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("vehicle model parameters must be finite")
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.max_steer_angle <= 0.0:
            raise ValueError("max_steer_angle must be positive")
        if self.max_acceleration < 0.0:
            raise ValueError("max_acceleration cannot be negative")
        if self.max_braking < 0.0:
            raise ValueError("max_braking cannot be negative")
        if self.linear_drag < 0.0:
            raise ValueError("linear_drag cannot be negative")
        if self.max_speed is not None:
            if not math.isfinite(self.max_speed) or self.max_speed <= 0.0:
                raise ValueError("max_speed must be finite and positive when set")

    def step(
        self,
        state: EgoState,
        control: HumanControl,
        dt: float,
    ) -> EgoState:
        """Advance ``state`` by ``dt`` seconds and return a new state."""

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        state_values = (state.x, state.y, state.yaw, state.speed, state.time)
        if not all(math.isfinite(value) for value in state_values):
            raise ValueError("ego state values must be finite")

        control = control.clamped()
        acceleration = (
            control.throttle * self.max_acceleration
            - control.brake * self.max_braking
            - self.linear_drag * state.speed
        )
        next_speed = max(0.0, state.speed + acceleration * dt)
        if self.max_speed is not None:
            next_speed = min(next_speed, self.max_speed)
        travel_speed = 0.5 * (state.speed + next_speed)

        steering_angle = control.steer * self.max_steer_angle
        yaw_rate = travel_speed * math.tan(steering_angle) / self.wheelbase
        midpoint_yaw = state.yaw + 0.5 * yaw_rate * dt

        return EgoState(
            x=state.x + travel_speed * math.cos(midpoint_yaw) * dt,
            y=state.y + travel_speed * math.sin(midpoint_yaw) * dt,
            yaw=_wrap_angle(state.yaw + yaw_rate * dt),
            speed=next_speed,
            time=state.time + dt,
        )
