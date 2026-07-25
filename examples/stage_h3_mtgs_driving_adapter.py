#!/usr/bin/env python3
"""Minimal control/support adapter for the released MTGS route."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CorridorMeasurement,
    EgoState,
    HumanControl,
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    SimpleVehicleModel,
    WorldDrivingController,
)


MTGS_CORRIDOR_HALF_WIDTH_METERS = 5.0
MTGS_MAX_HEADING_ERROR_DEGREES = 30.0
MTGS_MAX_SPEED_MPS = 15.0
MTGS_SPAWN_PROGRESS_METERS = 5.0
MTGS_ENDPOINT_MARGIN_METERS = 0.5


@dataclass(frozen=True)
class MtgsRenderQuery:
    """Route-relative values needed to construct one MTGS camera pose."""

    progress_meters: float
    lateral_meters: float
    heading_error_radians: float
    support_margin_meters: float


@dataclass(frozen=True)
class MtgsDrivingUpdate:
    """One bounded control update plus its render/evidence query."""

    state: EgoState
    render_query: MtgsRenderQuery
    boundary_hit: bool = False
    boundary_reason: str | None = None
    endpoint_reached: bool = False


@dataclass(frozen=True)
class MtgsDrivingAdapter:
    """Free vehicle dynamics inside the reviewed MTGS +/-5 m route tube."""

    controller: WorldDrivingController
    endpoint_margin_meters: float = MTGS_ENDPOINT_MARGIN_METERS

    def reset(self) -> EgoState:
        return self.controller.reset()

    def render_query(self, state: EgoState) -> MtgsRenderQuery:
        measurement = self.controller.corridor_measurement(state)
        assert measurement is not None
        return MtgsRenderQuery(
            progress_meters=measurement.progress,
            lateral_meters=measurement.lateral_offset,
            heading_error_radians=measurement.heading_error,
            support_margin_meters=(
                self.controller.corridor.half_width - measurement.distance
            ),
        )

    def step(
        self,
        state: EgoState,
        control: HumanControl,
        dt: float,
    ) -> MtgsDrivingUpdate:
        update = self.controller.step(state, control, dt)
        query = self.render_query(update.state)
        endpoint_reached = (
            query.progress_meters
            >= self.controller.corridor.length - self.endpoint_margin_meters
        )
        if endpoint_reached and update.state.speed > 0.0:
            stopped = EgoState(
                x=update.state.x,
                y=update.state.y,
                yaw=update.state.yaw,
                speed=0.0,
                time=update.state.time,
            )
            return MtgsDrivingUpdate(
                state=stopped,
                render_query=self.render_query(stopped),
                boundary_hit=update.boundary_hit,
                boundary_reason=update.boundary_reason,
                endpoint_reached=True,
            )
        return MtgsDrivingUpdate(
            state=update.state,
            render_query=query,
            boundary_hit=update.boundary_hit,
            boundary_reason=update.boundary_reason,
            endpoint_reached=endpoint_reached,
        )


def camera_route_centerline(
    route_poses: Any,
    route_distances: list[float],
) -> LoggedCenterlineCorridor:
    """Convert observed MTGS front-camera poses into a support centreline."""

    if len(route_poses) != len(route_distances) or len(route_poses) < 2:
        raise ValueError("route poses/distances must have matching length >= 2")
    samples = []
    for index, pose in enumerate(route_poses):
        if index + 1 < len(route_poses):
            next_pose = route_poses[index + 1]
            dx = float(next_pose[0, 3] - pose[0, 3])
            dy = float(next_pose[1, 3] - pose[1, 3])
        else:
            previous = route_poses[index - 1]
            dx = float(pose[0, 3] - previous[0, 3])
            dy = float(pose[1, 3] - previous[1, 3])
        if math.hypot(dx, dy) <= 1e-6:
            raise ValueError("MTGS camera route contains a zero-length segment")
        samples.append(
            LoggedCenterlineSample(
                logical_frame=index,
                log_time=float(route_distances[index]),
                x=float(pose[0, 3]),
                y=float(pose[1, 3]),
                yaw=math.atan2(dy, dx),
            )
        )
    return LoggedCenterlineCorridor(
        samples=tuple(samples),
        half_width=MTGS_CORRIDOR_HALF_WIDTH_METERS,
        max_heading_error=math.radians(MTGS_MAX_HEADING_ERROR_DEGREES),
    )


def make_mtgs_driving_adapter(
    route_poses: Any,
    route_distances: list[float],
    *,
    spawn_progress_meters: float = MTGS_SPAWN_PROGRESS_METERS,
) -> MtgsDrivingAdapter:
    """Build the small control boundary used before any GUI integration."""

    corridor = camera_route_centerline(route_poses, route_distances)
    spawn = corridor.pose_at_progress(spawn_progress_meters)
    controller = WorldDrivingController(
        corridor=corridor,
        spawn_state=spawn,
        vehicle_model=SimpleVehicleModel(
            wheelbase=2.8,
            max_steer_angle=math.radians(25.0),
            max_acceleration=4.0,
            max_braking=8.0,
            linear_drag=0.05,
            max_speed=MTGS_MAX_SPEED_MPS,
        ),
    )
    return MtgsDrivingAdapter(controller=controller)


def measurement_to_render_slope(measurement: CorridorMeasurement) -> float:
    """Convert a route-relative heading into the pose sampler's lateral slope."""

    return math.tan(measurement.heading_error)
