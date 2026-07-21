"""Human-control integration for a logged trajectory plus nearby offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .control import HumanControl
from .scene_coordinates import NearbyPoseLimits
from .state import EgoState


@dataclass(frozen=True)
class LoggedMovementProfile:
    """Named control envelope for logged-time reconstruction playback."""

    name: str
    limits: NearbyPoseLimits
    max_relative_speed: float
    relative_acceleration: float
    relative_braking: float
    max_yaw_rate_radians: float
    probe_forward_meters: float
    probe_left_meters: float
    probe_yaw_radians: float

    def __post_init__(self) -> None:
        values = (
            self.max_relative_speed,
            self.relative_acceleration,
            self.relative_braking,
            self.max_yaw_rate_radians,
            self.probe_forward_meters,
            self.probe_left_meters,
            self.probe_yaw_radians,
        )
        if not self.name:
            raise ValueError("movement profile name cannot be empty")
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "movement profile values must be finite and non-negative"
            )
        self.limits.validate(
            EgoState(
                x=self.probe_forward_meters,
                y=self.probe_left_meters,
                yaw=self.probe_yaw_radians,
            )
        )


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

    @classmethod
    def from_profile(
        cls,
        logged_duration: float,
        profile: LoggedMovementProfile,
    ) -> "LoggedEgoOffsetController":
        return cls(
            logged_duration=logged_duration,
            limits=profile.limits,
            max_relative_speed=profile.max_relative_speed,
            relative_acceleration=profile.relative_acceleration,
            relative_braking=profile.relative_braking,
            max_yaw_rate_radians=profile.max_yaw_rate_radians,
        )

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


H3_SAFE_LOGGED_MOVEMENT_PROFILE = LoggedMovementProfile(
    name="safe",
    limits=NearbyPoseLimits(
        max_abs_forward_meters=0.5,
        max_abs_left_meters=0.25,
        max_abs_yaw_radians=math.radians(2.0),
    ),
    max_relative_speed=0.5,
    relative_acceleration=0.5,
    relative_braking=1.0,
    max_yaw_rate_radians=math.radians(4.0),
    probe_forward_meters=0.25,
    probe_left_meters=0.1,
    probe_yaw_radians=math.radians(1.0),
)
H3_VISIBLE_LOGGED_MOVEMENT_PROFILE = LoggedMovementProfile(
    name="visible",
    limits=NearbyPoseLimits(
        max_abs_forward_meters=2.0,
        max_abs_left_meters=0.75,
        max_abs_yaw_radians=math.radians(8.0),
    ),
    max_relative_speed=2.0,
    relative_acceleration=4.0,
    relative_braking=6.0,
    max_yaw_rate_radians=math.radians(24.0),
    probe_forward_meters=1.25,
    probe_left_meters=0.6,
    probe_yaw_radians=math.radians(7.0),
)
LOGGED_MOVEMENT_PROFILES = {
    profile.name: profile
    for profile in (
        H3_SAFE_LOGGED_MOVEMENT_PROFILE,
        H3_VISIBLE_LOGGED_MOVEMENT_PROFILE,
    )
}


def logged_movement_profile(name: str) -> LoggedMovementProfile:
    """Return a named H3 logged-time movement profile."""

    try:
        return LOGGED_MOVEMENT_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(LOGGED_MOVEMENT_PROFILES))
        raise ValueError(
            f"unknown logged movement profile {name!r}; available: {available}"
        ) from error
