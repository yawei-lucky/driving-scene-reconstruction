"""Human-control integration for a logged trajectory plus nearby offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .control import HumanControl
from .scene_coordinates import NearbyPoseLimits
from .state import EgoState


@dataclass(frozen=True)
class LoggedEgoOffsetController:
    """Advance log time while integrating a bounded human-controlled offset.

    The logged trajectory supplies the main road motion. ``EgoState.speed`` is
    only the additional forward-offset speed; it is not the source log's
    physical speed. Steering changes the offset yaw even at zero relative
    speed so the first human input is immediately visible.
    """

    logged_duration: float
    limits: NearbyPoseLimits = field(
        default_factory=lambda: NearbyPoseLimits(
            max_abs_forward_meters=0.5,
            max_abs_left_meters=0.25,
            max_abs_yaw_radians=math.radians(2.0),
        )
    )
    max_relative_speed: float = 0.5
    relative_acceleration: float = 0.5
    relative_braking: float = 1.0
    max_yaw_rate_radians: float = math.radians(4.0)

    def __post_init__(self) -> None:
        values = (
            self.logged_duration,
            self.max_relative_speed,
            self.relative_acceleration,
            self.relative_braking,
            self.max_yaw_rate_radians,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("controller parameters must be finite and non-negative")

    @staticmethod
    def reset() -> EgoState:
        return EgoState()

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def step(
        self,
        state: EgoState,
        control: HumanControl,
        dt: float,
    ) -> EgoState:
        values = (state.x, state.y, state.yaw, state.speed, state.time, dt)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("state and dt must contain only finite values")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if state.time < 0.0 or state.time > self.logged_duration:
            raise ValueError("state time is outside the logged trajectory")
        control = control.clamped()

        next_time = min(self.logged_duration, state.time + dt)
        if state.time >= self.logged_duration:
            return EgoState(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                speed=0.0,
                time=self.logged_duration,
            )
        acceleration = (
            control.throttle * self.relative_acceleration
            - control.brake * self.relative_braking
        )
        speed = max(
            0.0,
            min(
                self.max_relative_speed,
                state.speed + acceleration * dt,
            ),
        )
        yaw = self._clamp(
            state.yaw + control.steer * self.max_yaw_rate_radians * dt,
            self.limits.max_abs_yaw_radians,
        )
        x = self._clamp(
            state.x + math.cos(yaw) * speed * dt,
            self.limits.max_abs_forward_meters,
        )
        y = self._clamp(
            state.y + math.sin(yaw) * speed * dt,
            self.limits.max_abs_left_meters,
        )
        result = EgoState(x=x, y=y, yaw=yaw, speed=speed, time=next_time)
        self.limits.validate(result)
        return result
