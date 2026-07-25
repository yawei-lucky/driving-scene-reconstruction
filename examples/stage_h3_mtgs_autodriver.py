#!/usr/bin/env python3
"""Deterministic route follower for the MTGS no-browser driving loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    EgoState,
    HumanControl,
    LoggedCenterlineCorridor,
    SimpleVehicleModel,
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _cosine_interpolate(start: float, end: float, fraction: float) -> float:
    eased = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return start + (end - start) * eased


def humanized_lateral_target(
    progress_meters: float,
    *,
    spawn_progress_meters: float,
    route_end_meters: float,
    amplitude_meters: float,
) -> float:
    """Return a smooth centre/left/centre/right/centre target."""

    if route_end_meters <= spawn_progress_meters:
        raise ValueError("route end must lie after spawn")
    if not math.isfinite(amplitude_meters) or amplitude_meters <= 0.0:
        raise ValueError("lateral amplitude must be finite and positive")
    fraction = _clamp(
        (progress_meters - spawn_progress_meters)
        / (route_end_meters - spawn_progress_meters),
        0.0,
        1.0,
    )
    knots = (
        (0.00, 0.0),
        (0.12, 0.0),
        (0.28, amplitude_meters),
        (0.42, 0.0),
        (0.58, -amplitude_meters),
        (0.74, 0.0),
        (1.00, 0.0),
    )
    for (left_x, left_y), (right_x, right_y) in zip(knots, knots[1:]):
        if fraction <= right_x:
            local = (fraction - left_x) / (right_x - left_x)
            return _cosine_interpolate(left_y, right_y, local)
    return 0.0


@dataclass(frozen=True)
class MtgsAutodriverConfig:
    """Small set of controls that genuinely vary for the first closed loop."""

    cruise_speed_mps: float = 12.0
    lateral_amplitude_meters: float = 3.0
    lookahead_meters: float = 6.0
    speed_gain: float = 1.2
    comfortable_braking_mps2: float = 6.0
    endpoint_margin_meters: float = 0.5
    spawn_progress_meters: float = 5.0
    max_steer_rate_per_second: float = 1.8

    def __post_init__(self) -> None:
        values = (
            self.cruise_speed_mps,
            self.lateral_amplitude_meters,
            self.lookahead_meters,
            self.speed_gain,
            self.comfortable_braking_mps2,
            self.endpoint_margin_meters,
            self.spawn_progress_meters,
            self.max_steer_rate_per_second,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("autodriver parameters must be finite and positive")


@dataclass(frozen=True)
class MtgsAutodriverDecision:
    """One simulated-human control request and its inspectable targets."""

    control: HumanControl
    target_speed_mps: float
    target_lateral_meters: float
    lookahead_progress_meters: float
    remaining_support_meters: float


class MtgsAutodriver:
    """Pure-pursuit steering plus bounded speed control."""

    def __init__(
        self,
        vehicle_model: SimpleVehicleModel,
        config: MtgsAutodriverConfig | None = None,
    ) -> None:
        self.vehicle_model = vehicle_model
        self.config = config or MtgsAutodriverConfig()
        self._previous_steer = 0.0

    def reset(self) -> None:
        self._previous_steer = 0.0

    def decide(
        self,
        state: EgoState,
        corridor: LoggedCenterlineCorridor,
        dt: float,
    ) -> MtgsAutodriverDecision:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        measurement = corridor.measure(state)
        route_end = corridor.length - self.config.endpoint_margin_meters
        remaining = max(0.0, route_end - measurement.progress)
        stopping_speed = math.sqrt(
            2.0 * self.config.comfortable_braking_mps2 * remaining
        )
        target_speed = min(self.config.cruise_speed_mps, stopping_speed)

        lookahead_progress = min(
            route_end,
            measurement.progress + self.config.lookahead_meters,
        )
        target_lateral = humanized_lateral_target(
            lookahead_progress,
            spawn_progress_meters=self.config.spawn_progress_meters,
            route_end_meters=route_end,
            amplitude_meters=self.config.lateral_amplitude_meters,
        )
        centre_target = corridor.pose_at_progress(lookahead_progress)
        target_x = (
            centre_target.x
            - math.sin(centre_target.yaw) * target_lateral
        )
        target_y = (
            centre_target.y
            + math.cos(centre_target.yaw) * target_lateral
        )
        dx = target_x - state.x
        dy = target_y - state.y
        target_distance = max(1e-3, math.hypot(dx, dy))
        alpha = _wrap_angle(math.atan2(dy, dx) - state.yaw)
        steering_angle = math.atan2(
            2.0 * self.vehicle_model.wheelbase * math.sin(alpha),
            target_distance,
        )
        requested_steer = _clamp(
            steering_angle / self.vehicle_model.max_steer_angle,
            -1.0,
            1.0,
        )
        max_steer_change = self.config.max_steer_rate_per_second * dt
        steer = _clamp(
            requested_steer,
            self._previous_steer - max_steer_change,
            self._previous_steer + max_steer_change,
        )
        self._previous_steer = steer

        requested_acceleration = self.config.speed_gain * (
            target_speed - state.speed
        )
        if (
            target_speed < self.config.cruise_speed_mps
            and state.speed > target_speed
        ):
            requested_acceleration = min(
                requested_acceleration,
                -self.config.comfortable_braking_mps2,
            )
        drag_acceleration = self.vehicle_model.linear_drag * state.speed
        if requested_acceleration >= 0.0:
            throttle = _clamp(
                (requested_acceleration + drag_acceleration)
                / self.vehicle_model.max_acceleration,
                0.0,
                1.0,
            )
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(
                (-requested_acceleration - drag_acceleration)
                / self.vehicle_model.max_braking,
                0.0,
                1.0,
            )
        return MtgsAutodriverDecision(
            control=HumanControl(
                steer=steer,
                throttle=throttle,
                brake=brake,
            ),
            target_speed_mps=target_speed,
            target_lateral_meters=target_lateral,
            lookahead_progress_meters=lookahead_progress,
            remaining_support_meters=remaining,
        )
