#!/usr/bin/env python3
"""Minimal route-constrained TbV browser with an auditable evidence outlet."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    BranchedRouteDrivingAdapter,
    EgoState,
    HumanControl,
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    RouteDrivingEvidenceRecorder,
    SimpleVehicleModel,
    SupportedRoute,
)
from stage_h3_tbv_world_pose_probe import (  # noqa: E402
    CAMERAS,
    RIGHT_TRAVERSAL,
    STRAIGHT_TRAVERSAL,
    LocalWorldPose,
    RouteSample,
    TbVWorldRenderer,
    pose_at_progress,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/tbv_h3/"
    "tbv_branch_pair_splatad_static_8000/splatad/"
    "2026-07-22_resume_2k_to_8k/config.yml"
)
DEFAULT_EVIDENCE = Path(
    "/home/yawei/stage3_external/artifacts/"
    "tbv_branch_pair_driving_adapter/tbv_driving_evidence.json"
)
COMMON_START_METERS = -20.0
BRANCH_ANCHOR_METERS = 0.0
STRAIGHT_END_METERS = 40.0
RIGHT_END_METERS = 30.0
CORRIDOR_HALF_WIDTH_METERS = 1.0
CORRIDOR_HEADING_LIMIT_DEGREES = 30.0
SELECTION_WINDOW_METERS = 0.5
FRONT_CAMERAS = (
    "ring_front_left",
    "ring_front_center",
    "ring_front_right",
)
OVERHEAD_EXTRA_CAMERAS = tuple(
    name for name in CAMERAS if name not in FRONT_CAMERAS
)
COCKPIT_WIDTH = 1600
COCKPIT_VIEW_HEIGHT = 620
COCKPIT_STATUS_HEIGHT = 48
COCKPIT_HORIZONTAL_FOV_DEGREES = 150.0
COCKPIT_TOP_ANGLE_DEGREES = 18.0
COCKPIT_BOTTOM_ANGLE_DEGREES = -24.0
BEV_SIZE = 284
DEFAULT_MAX_SPEED_MPS = 4.0
DEFAULT_OUTPUT_SCALE = 0.75
DEFAULT_OVERHEAD_EXTRA_SCALE = 0.375
DEFAULT_OVERHEAD_UPDATE_EVERY = 1
OVERHEAD_FORWARD_METERS = 16.0
OVERHEAD_REAR_METERS = 5.0
OVERHEAD_SIDE_METERS = 10.5
OVERHEAD_GROUND_Z_METERS = -1.43
OVERHEAD_BOWL_FLAT_RADIUS_METERS = 4.0
OVERHEAD_BOWL_RISE_PER_METER_SQUARED = 0.015
OVERHEAD_BOWL_MAX_RISE_METERS = 1.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-scale", type=float, default=DEFAULT_OUTPUT_SCALE
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--expected-checkpoint-step", type=int, default=7999)
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS)
    parser.add_argument(
        "--overhead-extra-scale",
        type=float,
        default=DEFAULT_OVERHEAD_EXTRA_SCALE,
    )
    parser.add_argument(
        "--overhead-update-every",
        type=int,
        default=DEFAULT_OVERHEAD_UPDATE_EVERY,
    )
    return parser.parse_args()


def control_for_keys(keys: set[str]) -> HumanControl:
    braking = "s" in keys
    return HumanControl(
        steer=float("a" in keys) - float("d" in keys),
        throttle=float("w" in keys and not braking),
        brake=float(braking),
    )


def route_segment(
    route: tuple[RouteSample, ...], start: float, end: float
) -> tuple[LocalWorldPose, ...]:
    if not start < end:
        raise ValueError("route segment start must precede end")
    poses = [pose_at_progress(route, start)]
    poses.extend(
        LocalWorldPose(sample.x, sample.y, sample.z, sample.yaw)
        for sample in route
        if start < sample.progress < end
    )
    poses.append(pose_at_progress(route, end))
    deduplicated: list[LocalWorldPose] = []
    for pose in poses:
        if deduplicated and math.hypot(
            pose.x - deduplicated[-1].x, pose.y - deduplicated[-1].y
        ) <= 1e-6:
            continue
        deduplicated.append(pose)
    if len(deduplicated) < 2:
        raise RuntimeError("route segment contains fewer than two distinct poses")
    return tuple(deduplicated)


def supported_route(
    *,
    name: str,
    renderer_profile: str,
    route: tuple[RouteSample, ...],
    start: float,
    end: float,
) -> SupportedRoute:
    poses = route_segment(route, start, end)
    samples = tuple(
        LoggedCenterlineSample(
            logical_frame=index,
            log_time=float(index),
            x=pose.x,
            y=pose.y,
            yaw=pose.yaw,
        )
        for index, pose in enumerate(poses)
    )
    return SupportedRoute(
        name=name,
        renderer_profile=renderer_profile,
        corridor=LoggedCenterlineCorridor(
            samples,
            half_width=CORRIDOR_HALF_WIDTH_METERS,
            max_heading_error=math.radians(CORRIDOR_HEADING_LIMIT_DEGREES),
        ),
        start_progress_from_anchor=start,
    )


def make_adapter(
    renderer: TbVWorldRenderer,
    *,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> BranchedRouteDrivingAdapter:
    if not math.isfinite(max_speed_mps) or max_speed_mps <= 0.0:
        raise ValueError("maximum speed must be finite and positive")
    right_route = renderer.routes[RIGHT_TRAVERSAL]
    straight_route = renderer.routes[STRAIGHT_TRAVERSAL]
    common = supported_route(
        name="common",
        renderer_profile=RIGHT_TRAVERSAL,
        route=right_route,
        start=COMMON_START_METERS,
        end=BRANCH_ANCHOR_METERS,
    )
    straight = supported_route(
        name="straight",
        renderer_profile=STRAIGHT_TRAVERSAL,
        route=straight_route,
        start=COMMON_START_METERS,
        end=STRAIGHT_END_METERS,
    )
    right = supported_route(
        name="right",
        renderer_profile=RIGHT_TRAVERSAL,
        route=right_route,
        start=COMMON_START_METERS,
        end=RIGHT_END_METERS,
    )
    spawn = pose_at_progress(right_route, COMMON_START_METERS)
    return BranchedRouteDrivingAdapter(
        common_route=common,
        branches={"straight": straight, "right": right},
        spawn_state=EgoState(x=spawn.x, y=spawn.y, yaw=spawn.yaw),
        selection_window_meters=SELECTION_WINDOW_METERS,
        vehicle_model=SimpleVehicleModel(
            max_steer_angle=math.radians(15.0),
            max_acceleration=2.0,
            max_braking=5.0,
            max_speed=max_speed_mps,
        ),
    )


def route_height(
    renderer: TbVWorldRenderer,
    renderer_profile: str,
    progress_from_anchor: float,
) -> float:
    route = renderer.routes[renderer_profile]
    progress = min(route[-1].progress, max(route[0].progress, progress_from_anchor))
    return pose_at_progress(route, progress).z


@dataclass(frozen=True)
class CameraProjection:
    """One calibrated pinhole camera expressed in the rig-local frame."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    local_to_camera: tuple[tuple[float, float, float], ...]
    camera_center_local: tuple[float, float, float]


def _tensor_scalar(value: Any) -> float:
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def camera_projections(
    renderer: TbVWorldRenderer,
    camera_names: Iterable[str],
    output_scales: Mapping[str, float] | None = None,
) -> dict[str, tuple[CameraProjection, ...]]:
    """Extract scaled camera calibration without changing the renderer."""

    import numpy as np

    if not renderer.is_loaded:
        raise RuntimeError("renderer must be loaded before extracting calibration")
    requested = tuple(camera_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("camera names must be non-empty and unique")
    if any(name not in CAMERAS for name in requested):
        raise ValueError("unknown camera requested")
    unknown_scales = tuple(
        name for name in (output_scales or {}) if name not in requested
    )
    if unknown_scales:
        raise ValueError(
            f"output scales provided for unrequested cameras {unknown_scales}"
        )
    scales = {
        name: float((output_scales or {}).get(name, renderer.output_scale))
        for name in requested
    }
    if not all(
        math.isfinite(scale) and 0.0 < scale <= 1.0
        for scale in scales.values()
    ):
        raise ValueError("camera output scales must be finite and in (0, 1]")
    profiles: dict[str, tuple[CameraProjection, ...]] = {}
    for traversal in (RIGHT_TRAVERSAL, STRAIGHT_TRAVERSAL):
        front_pose = renderer.source_poses[traversal]["ring_front_center"]
        forward_xy = renderer._horizontal_forward(front_pose)
        local_to_world = np.asarray(
            (
                (forward_xy[0], -forward_xy[1], 0.0),
                (forward_xy[1], forward_xy[0], 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        rig_origin = np.asarray(
            renderer.source_origins[traversal], dtype=np.float64
        )
        projections = []
        for camera_name in requested:
            camera = deepcopy(renderer.source_cameras[traversal][camera_name])
            output_scale = scales[camera_name]
            if output_scale != 1.0:
                camera.rescale_output_resolution(output_scale)
            camera_to_world = (
                renderer.source_poses[traversal][camera_name]
                .detach()
                .cpu()
                .numpy()
            )
            local_to_camera = camera_to_world[:, :3].T @ local_to_world
            camera_center_local = (
                local_to_world.T @ (camera_to_world[:, 3] - rig_origin)
            )
            projections.append(
                CameraProjection(
                    name=camera_name,
                    width=int(round(_tensor_scalar(camera.width))),
                    height=int(round(_tensor_scalar(camera.height))),
                    fx=_tensor_scalar(camera.fx),
                    fy=_tensor_scalar(camera.fy),
                    cx=_tensor_scalar(camera.cx),
                    cy=_tensor_scalar(camera.cy),
                    local_to_camera=tuple(
                        tuple(float(value) for value in row)
                        for row in local_to_camera
                    ),
                    camera_center_local=tuple(
                        float(value) for value in camera_center_local
                    ),
                )
            )
        profiles[traversal] = tuple(projections)
    return profiles


def front_camera_projections(
    renderer: TbVWorldRenderer,
) -> dict[str, tuple[CameraProjection, ...]]:
    return camera_projections(renderer, FRONT_CAMERAS)


class CylindricalCockpitComposer:
    """Calibrated three-camera cylindrical projection for the driving view."""

    def __init__(
        self,
        profiles: dict[str, tuple[CameraProjection, ...]],
        *,
        width: int = COCKPIT_WIDTH,
        height: int = COCKPIT_VIEW_HEIGHT,
        horizontal_fov_degrees: float = COCKPIT_HORIZONTAL_FOV_DEGREES,
        top_angle_degrees: float = COCKPIT_TOP_ANGLE_DEGREES,
        bottom_angle_degrees: float = COCKPIT_BOTTOM_ANGLE_DEGREES,
    ) -> None:
        import cv2
        import numpy as np

        if width <= 0 or height <= 0:
            raise ValueError("cockpit dimensions must be positive")
        if not 0.0 < horizontal_fov_degrees < 180.0:
            raise ValueError("horizontal field of view must be in (0, 180)")
        if not -89.0 < bottom_angle_degrees < top_angle_degrees < 89.0:
            raise ValueError("invalid cockpit vertical angle range")
        self.cv2 = cv2
        self.np = np
        self.width = width
        self.height = height
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self.top_angle_degrees = top_angle_degrees
        self.bottom_angle_degrees = bottom_angle_degrees
        self.profiles = profiles
        self.maps: dict[
            str, dict[str, tuple[int, int, Any, Any, Any]]
        ] = {}
        self.coverage_fraction: dict[str, float] = {}
        for profile, projections in profiles.items():
            maps, coverage = self._build_maps(projections)
            if coverage < 0.90:
                raise RuntimeError(
                    f"front cylindrical coverage is only {coverage:.1%} for {profile}"
                )
            self.maps[profile] = maps
            self.coverage_fraction[profile] = coverage

    @classmethod
    def from_renderer(
        cls, renderer: TbVWorldRenderer
    ) -> "CylindricalCockpitComposer":
        return cls(front_camera_projections(renderer))

    def _build_maps(
        self, projections: tuple[CameraProjection, ...]
    ) -> tuple[dict[str, tuple[int, int, Any, Any, Any]], float]:
        np = self.np
        half_fov = math.radians(self.horizontal_fov_degrees) / 2.0
        theta = np.linspace(
            half_fov, -half_fov, self.width, dtype=np.float32
        )[None, :]
        vertical = np.linspace(
            math.radians(self.top_angle_degrees),
            math.radians(self.bottom_angle_degrees),
            self.height,
            dtype=np.float32,
        )[:, None]
        horizontal_x = np.broadcast_to(np.cos(theta), (self.height, self.width))
        horizontal_y = np.broadcast_to(np.sin(theta), (self.height, self.width))
        vertical_z = np.broadcast_to(np.tan(vertical), (self.height, self.width))
        directions = np.stack(
            (horizontal_x, horizontal_y, vertical_z), axis=0
        ).reshape(3, -1)
        direction_norm = np.linalg.norm(directions, axis=0)

        maps: dict[str, tuple[int, int, Any, Any, Any]] = {}
        total_weight = np.zeros((self.height, self.width), dtype=np.float32)
        for projection in projections:
            local_to_camera = np.asarray(
                projection.local_to_camera, dtype=np.float32
            )
            camera_rays = local_to_camera @ directions
            depth = -camera_rays[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                map_x = (
                    projection.fx * camera_rays[0] / depth + projection.cx
                ).reshape(self.height, self.width)
                map_y = (
                    projection.cy - projection.fy * camera_rays[1] / depth
                ).reshape(self.height, self.width)
            valid = (
                (depth.reshape(self.height, self.width) > 1e-6)
                & (map_x >= 0.0)
                & (map_x <= projection.width - 1.0)
                & (map_y >= 0.0)
                & (map_y <= projection.height - 1.0)
            )
            edge_distance = np.minimum.reduce(
                (
                    map_x,
                    projection.width - 1.0 - map_x,
                    map_y,
                    projection.height - 1.0 - map_y,
                )
            )
            feather_pixels = max(
                8.0, min(projection.width, projection.height) * 0.08
            )
            feather = np.clip(
                edge_distance / feather_pixels, 0.0, 1.0
            ).astype(np.float32)
            incidence = np.clip(
                depth / direction_norm, 0.0, 1.0
            ).reshape(self.height, self.width)
            weight = (
                np.power(incidence, 8.0).astype(np.float32)
                * feather
                * valid.astype(np.float32)
            )
            map_x = np.where(valid, map_x, -1.0).astype(np.float32)
            map_y = np.where(valid, map_y, -1.0).astype(np.float32)
            covered_columns = np.flatnonzero(np.any(weight > 1e-6, axis=0))
            if not len(covered_columns):
                raise RuntimeError(
                    f"{projection.name} contributes no cylindrical pixels"
                )
            left = int(covered_columns[0])
            right = int(covered_columns[-1]) + 1
            maps[projection.name] = (
                left,
                right,
                map_x[:, left:right].copy(),
                map_y[:, left:right].copy(),
                weight[:, left:right].copy(),
            )
            total_weight += weight
        coverage = float((total_weight > 1e-6).mean())
        return maps, coverage

    def compose(self, profile: str, frames: dict[str, Any]) -> Any:
        if profile not in self.maps:
            raise KeyError(f"no cylindrical calibration for profile {profile!r}")
        np = self.np
        accumulator = np.zeros(
            (self.height, self.width, 3), dtype=np.float32
        )
        total_weight = np.zeros((self.height, self.width), dtype=np.float32)
        projections = {item.name: item for item in self.profiles[profile]}
        for camera_name in FRONT_CAMERAS:
            frame = frames[camera_name]
            projection = projections[camera_name]
            if tuple(frame.shape[:2]) != (projection.height, projection.width):
                raise RuntimeError(
                    f"{camera_name} rendered {tuple(frame.shape[:2])}, expected "
                    f"{(projection.height, projection.width)}"
                )
            left, right, map_x, map_y, weight = self.maps[profile][camera_name]
            warped = self.cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=self.cv2.INTER_LINEAR,
                borderMode=self.cv2.BORDER_CONSTANT,
            )
            accumulator[:, left:right] += (
                warped.astype(np.float32) * weight[:, :, None]
            )
            total_weight[:, left:right] += weight
        panorama = np.zeros_like(accumulator, dtype=np.uint8)
        covered = total_weight > 1e-6
        panorama[covered] = np.clip(
            accumulator[covered] / total_weight[covered, None],
            0.0,
            255.0,
        ).astype(np.uint8)
        return panorama


class BowlOverheadComposer:
    """Classic AVM-style calibrated projection onto one fixed virtual bowl."""

    def __init__(
        self,
        profiles: dict[str, tuple[CameraProjection, ...]],
        *,
        size: int = BEV_SIZE,
        forward_meters: float = OVERHEAD_FORWARD_METERS,
        rear_meters: float = OVERHEAD_REAR_METERS,
        side_meters: float = OVERHEAD_SIDE_METERS,
        ground_z_meters: float = OVERHEAD_GROUND_Z_METERS,
        flat_radius_meters: float = OVERHEAD_BOWL_FLAT_RADIUS_METERS,
        rise_per_meter_squared: float = (
            OVERHEAD_BOWL_RISE_PER_METER_SQUARED
        ),
        maximum_rise_meters: float = OVERHEAD_BOWL_MAX_RISE_METERS,
    ) -> None:
        import cv2
        import numpy as np

        dimensions = (
            forward_meters,
            rear_meters,
            side_meters,
            flat_radius_meters,
            rise_per_meter_squared,
            maximum_rise_meters,
        )
        if size <= 0 or not all(
            math.isfinite(value) and value > 0.0 for value in dimensions
        ):
            raise ValueError("overhead dimensions must be finite and positive")
        if not math.isfinite(ground_z_meters) or ground_z_meters >= 0.0:
            raise ValueError("overhead ground height must be finite and negative")
        self.cv2 = cv2
        self.np = np
        self.size = size
        self.forward_meters = forward_meters
        self.rear_meters = rear_meters
        self.side_meters = side_meters
        self.ground_z_meters = ground_z_meters
        self.flat_radius_meters = flat_radius_meters
        self.rise_per_meter_squared = rise_per_meter_squared
        self.maximum_rise_meters = maximum_rise_meters
        self.profiles = profiles
        self.forward_grid, self.left_grid, points = self._surface_points()
        self.maps: dict[str, dict[str, tuple[Any, Any, Any]]] = {}
        self.coverage_fraction: dict[str, float] = {}
        for profile, projections in profiles.items():
            maps, coverage = self._build_maps(projections, points)
            self.maps[profile] = maps
            self.coverage_fraction[profile] = coverage

    def _surface_points(self) -> tuple[Any, Any, Any]:
        np = self.np
        forward = np.linspace(
            self.forward_meters,
            -self.rear_meters,
            self.size,
            dtype=np.float32,
        )[:, None]
        left = np.linspace(
            self.side_meters,
            -self.side_meters,
            self.size,
            dtype=np.float32,
        )[None, :]
        forward_grid = np.broadcast_to(forward, (self.size, self.size))
        left_grid = np.broadcast_to(left, (self.size, self.size))
        radius = np.hypot(forward_grid, left_grid)
        outside = np.maximum(radius - self.flat_radius_meters, 0.0)
        rise = np.minimum(
            self.maximum_rise_meters,
            self.rise_per_meter_squared * outside * outside,
        )
        height = self.ground_z_meters + rise
        points = np.stack(
            (forward_grid, left_grid, height), axis=0
        ).reshape(3, -1)
        return forward_grid, left_grid, points

    def _build_maps(
        self,
        projections: tuple[CameraProjection, ...],
        points: Any,
    ) -> tuple[dict[str, tuple[Any, Any, Any]], float]:
        np = self.np
        maps: dict[str, tuple[Any, Any, Any]] = {}
        total_weight = np.zeros((self.size, self.size), dtype=np.float32)
        for projection in projections:
            rotation = np.asarray(
                projection.local_to_camera, dtype=np.float32
            )
            centre = np.asarray(
                projection.camera_center_local, dtype=np.float32
            )[:, None]
            camera_points = rotation @ (points - centre)
            depth = -camera_points[2]
            distance = np.linalg.norm(camera_points, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                map_x = (
                    projection.fx * camera_points[0] / depth + projection.cx
                ).reshape(self.size, self.size)
                map_y = (
                    projection.cy - projection.fy * camera_points[1] / depth
                ).reshape(self.size, self.size)
            valid = (
                (depth.reshape(self.size, self.size) > 1e-6)
                & (map_x >= 0.0)
                & (map_x <= projection.width - 1.0)
                & (map_y >= 0.0)
                & (map_y <= projection.height - 1.0)
            )
            edge_distance = np.minimum.reduce(
                (
                    map_x,
                    projection.width - 1.0 - map_x,
                    map_y,
                    projection.height - 1.0 - map_y,
                )
            )
            feather_pixels = max(
                6.0, min(projection.width, projection.height) * 0.08
            )
            feather = np.clip(
                edge_distance / feather_pixels, 0.0, 1.0
            ).astype(np.float32)
            incidence = np.clip(
                depth / np.maximum(distance, 1e-6), 0.0, 1.0
            ).reshape(self.size, self.size)
            weight = (
                np.power(incidence, 4.0).astype(np.float32)
                * feather
                * valid.astype(np.float32)
            )
            maps[projection.name] = (
                np.where(valid, map_x, -1.0).astype(np.float32),
                np.where(valid, map_y, -1.0).astype(np.float32),
                weight,
            )
            total_weight += weight
        return maps, float((total_weight > 1e-6).mean())

    def compose(self, profile: str, frames: Mapping[str, Any]) -> Any:
        if profile not in self.maps:
            raise KeyError(f"no overhead calibration for profile {profile!r}")
        np = self.np
        accumulator = np.zeros(
            (self.size, self.size, 3), dtype=np.float32
        )
        total_weight = np.zeros((self.size, self.size), dtype=np.float32)
        projections = {item.name: item for item in self.profiles[profile]}
        for camera_name, (map_x, map_y, weight) in self.maps[profile].items():
            if camera_name not in frames:
                raise KeyError(f"overhead frame missing {camera_name}")
            frame = frames[camera_name]
            projection = projections[camera_name]
            if tuple(frame.shape[:2]) != (projection.height, projection.width):
                raise RuntimeError(
                    f"{camera_name} rendered {tuple(frame.shape[:2])}, expected "
                    f"{(projection.height, projection.width)}"
                )
            warped = self.cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=self.cv2.INTER_LINEAR,
                borderMode=self.cv2.BORDER_CONSTANT,
            )
            accumulator += warped.astype(np.float32) * weight[:, :, None]
            total_weight += weight
        overhead = np.zeros_like(accumulator, dtype=np.uint8)
        covered = total_weight > 1e-6
        overhead[covered] = np.clip(
            accumulator[covered] / total_weight[covered, None],
            0.0,
            255.0,
        ).astype(np.uint8)
        return overhead

    def reproject(
        self,
        image: Any,
        source_state: EgoState,
        target_state: EgoState,
    ) -> Any:
        """Motion-compensate the last ground projection into the current pose."""

        if tuple(image.shape[:2]) != (self.size, self.size):
            raise ValueError("overhead image has the wrong dimensions")
        np = self.np
        target_cosine, target_sine = (
            math.cos(target_state.yaw),
            math.sin(target_state.yaw),
        )
        world_x = (
            target_state.x
            + target_cosine * self.forward_grid
            - target_sine * self.left_grid
        )
        world_y = (
            target_state.y
            + target_sine * self.forward_grid
            + target_cosine * self.left_grid
        )
        dx, dy = world_x - source_state.x, world_y - source_state.y
        source_cosine, source_sine = (
            math.cos(source_state.yaw),
            math.sin(source_state.yaw),
        )
        source_forward = source_cosine * dx + source_sine * dy
        source_left = -source_sine * dx + source_cosine * dy
        map_x = (
            (self.side_meters - source_left)
            / (2.0 * self.side_meters)
            * (self.size - 1)
        ).astype(np.float32)
        map_y = (
            (self.forward_meters - source_forward)
            / (self.forward_meters + self.rear_meters)
            * (self.size - 1)
        ).astype(np.float32)
        return self.cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=self.cv2.INTER_LINEAR,
            borderMode=self.cv2.BORDER_CONSTANT,
        )

    def screen(self, forward: float, left: float) -> tuple[int, int]:
        return (
            int(
                round(
                    (self.side_meters - left)
                    / (2.0 * self.side_meters)
                    * (self.size - 1)
                )
            ),
            int(
                round(
                    (self.forward_meters - forward)
                    / (self.forward_meters + self.rear_meters)
                    * (self.size - 1)
                )
            ),
        )


@dataclass(frozen=True)
class OverheadSnapshot:
    image: Any
    state: EgoState
    profile: str
    sequence: int
    source_render_seconds: float
    compose_seconds: float
    coverage_fraction: float


def make_trajectory_bev(
    image_module: Any,
    draw_module: Any,
    adapter: BranchedRouteDrivingAdapter,
    state: EgoState,
    support: dict[str, object],
    *,
    size: int = BEV_SIZE,
    overhead: Any | None = None,
    overhead_composer: BowlOverheadComposer | None = None,
) -> Any:
    """Draw trusted route support over an optional visual-only AVM projection."""

    if (overhead is None) != (overhead_composer is None):
        raise ValueError("overhead image and composer must be provided together")
    if overhead_composer is not None and size != overhead_composer.size:
        raise ValueError("overhead composer size does not match inset size")
    if overhead is None:
        canvas = image_module.new("RGB", (size, size), (12, 18, 24))
        scale = 5.5
        ego_x, ego_y = size // 2, size - 48
    else:
        canvas = image_module.fromarray(overhead).convert("RGB")
        assert overhead_composer is not None
        scale = min(
            (size - 1)
            / (overhead_composer.forward_meters + overhead_composer.rear_meters),
            (size - 1) / (2.0 * overhead_composer.side_meters),
        )
        ego_x, ego_y = overhead_composer.screen(0.0, 0.0)
    draw = draw_module.Draw(canvas)
    grid_distances = (
        (5.0, 10.0, 15.0)
        if overhead is not None
        else (10.0, 20.0, 30.0, 40.0)
    )
    for delta_meters in grid_distances:
        _, grid_y = (
            overhead_composer.screen(delta_meters, 0.0)
            if overhead_composer is not None
            else (ego_x, int(round(ego_y - delta_meters * scale)))
        )
        if 0 <= grid_y < size:
            draw.line((0, grid_y, size, grid_y), fill=(38, 48, 54), width=1)
    draw.line((ego_x, 28, ego_x, size), fill=(29, 39, 48), width=1)

    cosine, sine = math.cos(state.yaw), math.sin(state.yaw)

    def screen(x: float, y: float) -> tuple[int, int]:
        dx, dy = x - state.x, y - state.y
        forward = dx * cosine + dy * sine
        left = -dx * sine + dy * cosine
        if overhead_composer is not None:
            return overhead_composer.screen(forward, left)
        return (
            int(round(ego_x - left * scale)),
            int(round(ego_y - forward * scale)),
        )

    def points(route: SupportedRoute) -> list[tuple[int, int]]:
        return [screen(sample.x, sample.y) for sample in route.corridor.samples]

    selected = adapter.selected_branch
    branch_colours = {
        "straight": (74, 171, 255),
        "right": (255, 142, 73),
    }
    for name, route in adapter.branches.items():
        colour = branch_colours.get(name, (135, 150, 160))
        if selected is not None and name != selected:
            colour = tuple(channel // 3 for channel in colour)
        route_points = points(route)
        if len(route_points) >= 2:
            draw.line(route_points, fill=(38, 51, 61), width=13, joint="curve")
            draw.line(route_points, fill=colour, width=3, joint="curve")
            endpoint = route_points[-1]
            draw.ellipse(
                (
                    endpoint[0] - 8,
                    endpoint[1] - 8,
                    endpoint[0] + 8,
                    endpoint[1] + 8,
                ),
                fill=colour,
            )
            draw.text(
                (endpoint[0] - 3, endpoint[1] - 6),
                "1" if name == "straight" else "2",
                fill=(8, 12, 16),
            )

    active_points = points(adapter.active_route)
    if len(active_points) >= 2:
        support_width = max(
            5,
            int(
                round(
                    2.0
                    * adapter.active_route.corridor.half_width
                    * scale
                )
            ),
        )
        draw.line(
            active_points,
            fill=(77, 91, 99),
            width=support_width,
            joint="curve",
        )
        draw.line(active_points, fill=(238, 238, 224), width=2, joint="curve")

    margin = float(support["distance_margin_meters"])
    ego_colour = (255, 78, 69) if margin < 0.2 else (100, 240, 150)
    if overhead_composer is None:
        draw.polygon(
            (
                (ego_x, ego_y - 12),
                (ego_x - 8, ego_y + 9),
                (ego_x, ego_y + 5),
                (ego_x + 8, ego_y + 9),
            ),
            fill=ego_colour,
            outline=(245, 245, 245),
        )
    else:
        vehicle = tuple(
            overhead_composer.screen(forward, left)
            for forward, left in (
                (2.4, 0.80),
                (2.4, -0.80),
                (1.7, -1.05),
                (-2.4, -1.05),
                (-2.4, 1.05),
                (1.7, 1.05),
            )
        )
        draw.polygon(
            vehicle,
            fill=(24, 29, 34),
            outline=(224, 232, 235),
        )
        draw.line(
            (
                overhead_composer.screen(0.0, 0.0),
                overhead_composer.screen(1.7, 0.0),
            ),
            fill=ego_colour,
            width=3,
        )
    draw.rectangle((0, 0, size - 1, size - 1), outline=(112, 126, 136), width=2)
    draw.rectangle((1, 1, size - 2, 44), fill=(8, 12, 16))
    if overhead is None:
        title = "TRAJECTORY SUPPORT  +/-1 m"
        subtitle = "not overhead RGB"
    else:
        title = "OVERHEAD  ·  VISUAL ONLY"
        subtitle = "black = no camera coverage"
    draw.text((10, 8), title, fill=(240, 240, 225))
    draw.text((10, 25), subtitle, fill=(151, 170, 181))
    draw.rectangle((1, size - 30, size - 2, size - 2), fill=(8, 12, 16))
    draw.text(
        (10, size - 23),
        (
            f"offset {float(support['lateral_offset_meters']):+.2f} m  "
            f"margin {margin:.2f} m"
        ),
        fill=ego_colour,
    )
    return canvas


def make_cockpit_frame(
    image_module: Any,
    draw_module: Any,
    composer: CylindricalCockpitComposer,
    frames: dict[str, Any],
    adapter: BranchedRouteDrivingAdapter,
    state: EgoState,
    support: dict[str, object],
    *,
    overhead: Any | None = None,
    overhead_composer: BowlOverheadComposer | None = None,
) -> Any:
    profile = str(support["renderer_profile"])
    panorama = image_module.fromarray(composer.compose(profile, frames)).convert("RGB")
    canvas = image_module.new(
        "RGB",
        (composer.width, composer.height + COCKPIT_STATUS_HEIGHT),
        (4, 6, 8),
    )
    canvas.paste(panorama, (0, 0))
    inset = make_trajectory_bev(
        image_module,
        draw_module,
        adapter,
        state,
        support,
        overhead=overhead,
        overhead_composer=overhead_composer,
    )
    inset_x, inset_y = composer.width - inset.width - 14, 14
    canvas.paste(inset, (inset_x, inset_y))
    draw = draw_module.Draw(canvas)
    draw.rectangle((8, 8, 344, 37), fill=(5, 9, 12))
    draw.text(
        (18, 16),
        (
            f"FORWARD SURROUND  "
            f"{composer.horizontal_fov_degrees:.0f} DEG"
        ),
        fill=(220, 235, 240),
    )
    status = (
        f"TbV evidence-only | {support['phase']} | "
        f"branch={support['selected_branch'] or '-'} | "
        f"progress={float(support['progress_from_anchor_meters']):+.1f}m | "
        f"offset={float(support['lateral_offset_meters']):+.2f}m | "
        f"speed={state.speed:.2f}m/s"
    )
    draw.text(
        (12, composer.height + 15), status, fill=(255, 216, 77)
    )
    return canvas


def make_diagnostic_mosaic(
    image_module: Any,
    draw_module: Any,
    frames: dict[str, Any],
    state: EgoState,
    support: dict[str, object],
) -> Any:
    order = (
        "ring_front_left",
        "ring_front_center",
        "ring_front_right",
        "ring_side_left",
        "ring_side_right",
        "ring_rear_left",
        "ring_rear_right",
    )
    canvas_width = 1560
    top_height = 520
    bottom_height = 300
    status_height = 50
    canvas = image_module.new(
        "RGB", (canvas_width, top_height + bottom_height + status_height), "black"
    )
    boxes = (
        (0, 0, 520, top_height),
        (520, 0, 520, top_height),
        (1040, 0, 520, top_height),
        (0, top_height, 390, bottom_height),
        (1170, top_height, 390, bottom_height),
        (390, top_height, 390, bottom_height),
        (780, top_height, 390, bottom_height),
    )
    for name, (x, y, width, height) in zip(order, boxes):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail((width, height), image_module.Resampling.LANCZOS)
        draw_module.Draw(tile).text(
            (7, 6), name, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0)
        )
        canvas.paste(
            tile,
            (x + (width - tile.width) // 2, y + (height - tile.height) // 2),
        )
    status = (
        f"DIAGNOSTIC ONLY | {support['phase']} | "
        f"branch={support['selected_branch'] or '-'} | "
        f"progress={float(support['progress_from_anchor_meters']):+.1f}m | "
        f"offset={float(support['lateral_offset_meters']):+.2f}m | "
        f"speed={state.speed:.2f}m/s"
    )
    draw_module.Draw(canvas).text(
        (10, top_height + bottom_height + 14), status, fill=(255, 216, 77)
    )
    return canvas


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TbV 前向环视驾驶舱</title>
  <style>
    body { margin:0; background:#0d0d0d; color:#eee; font:15px sans-serif; text-align:center; }
    h1 { font-size:18px; margin:5px; }
    #view { display:block; width:calc(100vw - 8px); max-height:calc(100vh - 128px); object-fit:contain; margin:auto; border:1px solid #444; }
    #status { min-height:22px; color:#ffd84d; margin:4px; }
    button { margin:2px; padding:7px 12px; font-size:14px; }
    #branches { color:#ffb27a; }
    #branches[hidden] { display:none; }
    a { color:#8ed8ff; }
  </style>
</head>
<body>
  <h1>TbV 前向环视 · 约 150° · 俯视辅助 · 轨迹支持 ±1m</h1>
  <img id="view" src="/frame.jpg" alt="three calibrated front cameras projected into one cylindrical driving view">
  <div id="status">W 油门 · S 刹车 · A/D 转向 · 到锚点后选择分支 · R 重置</div>
  <div id="branches" hidden>
    已到共享锚点：<button data-branch="straight">1 / 直行</button>
    <button data-branch="right">2 / 右转</button>
  </div>
  <div>
    <button id="reset">R 重置</button>
    <a href="/diagnostic" target="_blank">原相机图</a> ·
    <a href="/evidence.json" target="_blank">可信证据 JSON</a>
  </div>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    const branches = document.getElementById("branches");
    const held = new Set();
    const aliases = new Map([["arrowup","w"],["arrowdown","s"],["arrowleft","a"],["arrowright","d"]]);
    const driving = new Set(["w","a","s","d"]);
    const tickPeriodMs = __TICK_PERIOD_MS__;
    let speed = 0, inFlight = false, timer = null, imageSequence = 0, generation = 0;
    let blocked = false, selectionRequired = false, pendingInputAt = null;

    function keyName(value) { const key=value.toLowerCase(); return aliases.get(key)||key; }
    function setHeld(key, active) {
      const changed = held.has(key) !== active;
      if (active) held.add(key); else held.delete(key);
      if (changed) { pendingInputAt=performance.now(); requestTick(); }
    }
    function requestTick() {
      if (inFlight || blocked || selectionRequired) return;
      if (timer !== null) { clearTimeout(timer); timer=null; }
      tick(generation);
    }
    function schedule(token, started) {
      if (blocked || selectionRequired || token !== generation) return;
      if (!held.has("w") && speed <= 1e-6) return;
      timer=setTimeout(()=>tick(token), Math.max(0,tickPeriodMs-(performance.now()-started)));
    }
    async function loadFrame() {
      await new Promise((resolve,reject)=>{
        view.onload=resolve; view.onerror=reject; view.src="/frame.jpg?n="+(++imageSequence);
      });
    }
    function stateText(result, requestMs, inputMs) {
      const route=result.route_support;
      let text=`${route.phase} · ${result.selected_branch||"未选路"} · 进度 ${route.progress_from_anchor_meters.toFixed(1)}m · `+
        `横向 ${route.lateral_offset_meters.toFixed(2)}m · 速度 ${result.speed_mps.toFixed(2)}m/s · 请求→画面 ${requestMs.toFixed(0)}ms`;
      if (inputMs !== null) text+=` · 输入→画面 ${inputMs.toFixed(0)}ms`;
      if (result.boundary_hit) text+=` · 已停止：${result.boundary_reason}`;
      return text;
    }
    async function saveBrowserTiming(result, requestMs, inputMs) {
      const response=await fetch("/evidence-sample", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
        sequence:result.sequence, client_unix_ms:Date.now(), browser_request_to_image_ms:requestMs,
        browser_input_to_image_ms:inputMs
      })});
      if (!response.ok) throw new Error((await response.json()).error||"证据写入失败");
    }
    async function tick(token) {
      if (token!==generation || inFlight || blocked || selectionRequired) return;
      inFlight=true; if (timer!==null) { clearTimeout(timer); timer=null; }
      const started=performance.now(), inputStarted=pendingInputAt;
      try {
        const keys=[...held].sort().join("");
        const response=await fetch("/tick?keys="+encodeURIComponent(keys),{method:"POST"});
        const result=await response.json(); if (!response.ok) throw new Error(result.error||"控制失败");
        await loadFrame(); const loaded=performance.now();
        const requestMs=loaded-started, inputMs=inputStarted===null?null:loaded-inputStarted;
        await saveBrowserTiming(result,requestMs,inputMs);
        if (pendingInputAt===inputStarted) pendingInputAt=null;
        speed=result.speed_mps; blocked=result.boundary_hit; selectionRequired=result.selection_required;
        branches.hidden=!selectionRequired; status.textContent=stateText(result,requestMs,inputMs);
      } catch(error) { status.textContent=error.toString(); }
      finally { inFlight=false; }
      schedule(token,started);
    }
    async function chooseBranch(branch) {
      if (inFlight) return; inFlight=true; const started=performance.now();
      try {
        const response=await fetch("/branch?name="+branch,{method:"POST"});
        const result=await response.json(); if (!response.ok) throw new Error(result.error||"选路失败");
        await loadFrame(); const loaded=performance.now();
        const timingResponse=await fetch("/evidence-route-timing", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
          event_index:result.route_event_index, client_unix_ms:Date.now(), browser_selection_to_image_ms:loaded-started
        })});
        if (!timingResponse.ok) throw new Error((await timingResponse.json()).error||"选路证据写入失败");
        speed=result.speed_mps; selectionRequired=false; blocked=false; branches.hidden=true;
        status.textContent=`已选择 ${branch} · 切换渲染 ${result.server_route_selection_to_jpeg_ms.toFixed(0)}ms`;
      } catch(error) { status.textContent=error.toString(); }
      finally { inFlight=false; }
      requestTick();
    }
    async function reset() {
      held.clear(); ++generation; blocked=false; selectionRequired=false; branches.hidden=true;
      if (timer!==null) { clearTimeout(timer); timer=null; }
      const response=await fetch("/reset",{method:"POST"}); const result=await response.json();
      if (!response.ok) { status.textContent=result.error||"重置失败"; return; }
      await loadFrame(); speed=0; pendingInputAt=null; status.textContent="已重置到公共入口 -20m";
    }
    document.addEventListener("keydown",event=>{
      const key=keyName(event.key);
      if (driving.has(key)) { event.preventDefault(); setHeld(key,true); }
      if (key==="r"&&!event.repeat) { event.preventDefault(); reset(); }
      if (selectionRequired&&key==="1") chooseBranch("straight");
      if (selectionRequired&&key==="2") chooseBranch("right");
    });
    document.addEventListener("keyup",event=>{ const key=keyName(event.key); if(driving.has(key)){event.preventDefault();setHeld(key,false);} });
    window.addEventListener("blur",()=>{ if(held.size){held.clear();pendingInputAt=performance.now();requestTick();} });
    document.querySelectorAll("button[data-branch]").forEach(button=>button.addEventListener("click",()=>chooseBranch(button.dataset.branch)));
    document.getElementById("reset").addEventListener("click",reset);
  </script>
</body>
</html>
"""


DIAGNOSTIC_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TbV 原相机图</title>
  <style>
    body { margin:0; background:#0d0d0d; color:#eee; font:15px sans-serif; text-align:center; }
    h1 { font-size:18px; margin:7px 4px 2px; }
    p { color:#b9c3c9; margin:3px 4px 7px; }
    #view { display:block; width:calc(100vw - 8px); max-height:calc(100vh - 75px); object-fit:contain; margin:auto; border:1px solid #444; }
    a { color:#8ed8ff; }
  </style>
</head>
<body>
  <h1>原相机图</h1>
  <p>
    七相机会在打开或手动刷新时按需渲染，不占用正常驾驶帧预算；
    仅用于检查覆盖、变形和重影，不作为真人驾驶视图。
    <a href="/">返回驾驶舱</a>
  </p>
  <button id="refresh">刷新当前状态</button>
  <img id="view" src="/diagnostic.jpg" alt="seven reconstructed camera diagnostics">
  <script>
    const view=document.getElementById("view");
    let refreshSequence=0;
    document.getElementById("refresh").addEventListener("click",()=>{
      view.src="/diagnostic.jpg?n="+(++refreshSequence);
    });
  </script>
</body>
</html>
"""


def render_web_page(dt: float) -> str:
    return WEB_PAGE.replace("__TICK_PERIOD_MS__", f"{dt * 1000.0:.6f}")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0 or length > 64_000:
        raise ValueError("JSON request body is empty or too large")
    value = json.loads(handler.rfile.read(length))
    if not isinstance(value, dict):
        raise ValueError("JSON request body must be an object")
    return value


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if not 0.0 < args.output_scale <= 1.0:
        raise ValueError("--output-scale must be in (0, 1]")
    if not math.isfinite(args.max_speed_mps) or args.max_speed_mps <= 0.0:
        raise ValueError("--max-speed-mps must be finite and positive")
    if (
        not math.isfinite(args.overhead_extra_scale)
        or not 0.0 < args.overhead_extra_scale <= 1.0
    ):
        raise ValueError("--overhead-extra-scale must be in (0, 1]")
    if args.overhead_update_every < 1:
        raise ValueError("--overhead-update-every must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    renderer = TbVWorldRenderer(args.config, args.output_scale)
    adapter: BranchedRouteDrivingAdapter
    composer: CylindricalCockpitComposer
    overhead_composer: BowlOverheadComposer
    evidence: RouteDrivingEvidenceRecorder
    renderer_lock = threading.Lock()
    overhead_lock = threading.Lock()
    overhead_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="tbv-overhead"
    )
    overhead_output_scales = {
        name: args.overhead_extra_scale for name in OVERHEAD_EXTRA_CAMERAS
    }
    runtime: dict[str, Any] = {
        "state": EgoState(), "jpeg": b"", "render": {},
        "diagnostic_jpeg": b"", "sequence": -1,
        "overhead_snapshot": None,
        "overhead_future": None,
        "overhead_error": None,
        "overhead_skipped_updates": 0,
        "boundary_hit": False, "boundary_reason": None, "selection_required": False,
    }

    def current_overhead_snapshot() -> OverheadSnapshot | None:
        with overhead_lock:
            return runtime["overhead_snapshot"]

    def build_overhead_snapshot(
        state: EgoState,
        profile: str,
        pose: LocalWorldPose,
        front_frames: Mapping[str, Any],
        sequence: int,
    ) -> OverheadSnapshot:
        with renderer_lock:
            observation = renderer.render(
                profile,
                pose,
                OVERHEAD_EXTRA_CAMERAS,
                overhead_output_scales,
            )
        frames = {**front_frames, **dict(observation["frames"])}
        compose_started = time.perf_counter()
        image = overhead_composer.compose(profile, frames)
        compose_seconds = time.perf_counter() - compose_started
        return OverheadSnapshot(
            image=image,
            state=state,
            profile=profile,
            sequence=sequence,
            source_render_seconds=float(observation["render_seconds"]),
            compose_seconds=compose_seconds,
            coverage_fraction=overhead_composer.coverage_fraction[profile],
        )

    def publish_overhead_snapshot(snapshot: OverheadSnapshot) -> None:
        with overhead_lock:
            current = runtime["overhead_snapshot"]
            if current is None or snapshot.sequence >= current.sequence:
                runtime["overhead_snapshot"] = snapshot
            runtime["overhead_error"] = None

    def refresh_overhead_sync(
        state: EgoState,
        profile: str,
        pose: LocalWorldPose,
        front_frames: Mapping[str, Any],
        sequence: int,
    ) -> None:
        publish_overhead_snapshot(
            build_overhead_snapshot(
                state, profile, pose, front_frames, sequence
            )
        )

    def schedule_overhead_update(
        state: EgoState,
        profile: str,
        pose: LocalWorldPose,
        front_frames: Mapping[str, Any],
        sequence: int,
    ) -> None:
        if sequence % args.overhead_update_every:
            return
        with overhead_lock:
            future: Future[Any] | None = runtime["overhead_future"]
            if future is not None and not future.done():
                runtime["overhead_skipped_updates"] += 1
                return

            def update() -> None:
                try:
                    snapshot = build_overhead_snapshot(
                        state, profile, pose, front_frames, sequence
                    )
                    publish_overhead_snapshot(snapshot)
                except Exception as error:
                    with overhead_lock:
                        runtime["overhead_error"] = str(error)

            runtime["overhead_future"] = overhead_executor.submit(update)

    def overhead_for_state(
        state: EgoState,
        profile: str,
    ) -> tuple[Any | None, dict[str, object]]:
        snapshot = current_overhead_snapshot()
        if snapshot is None or snapshot.profile != profile:
            return None, {
                "overhead_available": False,
                "overhead_profile_match": False,
            }
        distance = math.hypot(
            state.x - snapshot.state.x,
            state.y - snapshot.state.y,
        )
        yaw_difference = abs(
            math.degrees(
                math.atan2(
                    math.sin(state.yaw - snapshot.state.yaw),
                    math.cos(state.yaw - snapshot.state.yaw),
                )
            )
        )
        image = overhead_composer.reproject(
            snapshot.image, snapshot.state, state
        )
        return image, {
            "overhead_available": True,
            "overhead_profile_match": True,
            "overhead_source_sequence": snapshot.sequence,
            "overhead_motion_compensation_distance_meters": distance,
            "overhead_motion_compensation_yaw_degrees": yaw_difference,
            "overhead_source_render_seconds": snapshot.source_render_seconds,
            "overhead_compose_seconds": snapshot.compose_seconds,
            "overhead_coverage_fraction": snapshot.coverage_fraction,
        }

    def payload(extra: dict[str, object] | None = None) -> dict[str, object]:
        state: EgoState = runtime["state"]
        support = adapter.support(state)
        overhead_snapshot = current_overhead_snapshot()
        overhead_matches = (
            overhead_snapshot is not None
            and overhead_snapshot.profile == support.renderer_profile
        )
        if overhead_matches:
            assert overhead_snapshot is not None
            overhead_distance = math.hypot(
                state.x - overhead_snapshot.state.x,
                state.y - overhead_snapshot.state.y,
            )
        else:
            overhead_distance = None
        value: dict[str, object] = {
            "backend": "tbv_route_driving_adapter_v0",
            "sequence": runtime["sequence"],
            "simulation_time_seconds": state.time,
            "x_meters": state.x,
            "y_meters": state.y,
            "yaw_degrees": math.degrees(state.yaw),
            "speed_mps": state.speed,
            "maximum_speed_mps": args.max_speed_mps,
            "selected_branch": adapter.selected_branch,
            "selection_required": runtime["selection_required"],
            "boundary_hit": runtime["boundary_hit"],
            "boundary_reason": runtime["boundary_reason"],
            "route_support": support.as_dict(),
            "branch_options": (
                {
                    name: adapter.branch_support(name, state).as_dict()
                    for name in sorted(adapter.branches)
                }
                if support.selection_required
                else None
            ),
            "renderer_profile": support.renderer_profile,
            "renderer_ms": float(runtime["render"].get("render_seconds", 0.0)) * 1000.0,
            "presentation_ms": (
                float(runtime["render"].get("presentation_seconds", 0.0))
                * 1000.0
            ),
            "frozen_scene_time_seconds": renderer.model_scene_time,
            "display_mode": "forward_surround_with_visual_overhead",
            "display_names": {
                "primary": "forward_surround",
                "auxiliary": "overhead",
                "diagnostic": "original_camera_views",
            },
            "driving_camera_count": len(FRONT_CAMERAS),
            "diagnostic_camera_count": len(CAMERAS),
            "front_horizontal_fov_degrees": (
                composer.horizontal_fov_degrees
            ),
            "front_projection_coverage_fraction": float(
                runtime["render"].get("front_projection_coverage_fraction", 0.0)
            ),
            "overhead_mode": "calibrated_virtual_bowl_visual_only",
            "overhead_extra_camera_names": list(OVERHEAD_EXTRA_CAMERAS),
            "overhead_extra_output_scale": args.overhead_extra_scale,
            "overhead_update_every": args.overhead_update_every,
            "overhead_available": overhead_matches,
            "overhead_coverage_fraction": (
                overhead_snapshot.coverage_fraction
                if overhead_matches
                else None
            ),
            "overhead_source_sequence": (
                overhead_snapshot.sequence if overhead_matches else None
            ),
            "overhead_source_render_ms": (
                overhead_snapshot.source_render_seconds * 1000.0
                if overhead_matches
                else None
            ),
            "overhead_compose_ms": (
                overhead_snapshot.compose_seconds * 1000.0
                if overhead_matches
                else None
            ),
            "overhead_motion_compensation_distance_meters": overhead_distance,
            "overhead_background_error": runtime["overhead_error"],
            "overhead_skipped_updates": runtime["overhead_skipped_updates"],
            "overhead_url": "/overhead.jpg",
            "bev_source": (
                "visual_only_camera_bowl_with_logged_trajectory_support"
            ),
            "diagnostic_url": "/diagnostic",
            "evidence_url": "/evidence.json",
            "certified_drivable_corridor": False,
        }
        if extra:
            value.update(extra)
        return value

    def render_state(
        state: EgoState,
    ) -> tuple[bytes, dict[str, object]]:
        from PIL import Image, ImageDraw

        support = adapter.support(state)
        z = route_height(
            renderer, support.renderer_profile, support.progress_from_anchor_meters
        )
        pose = LocalWorldPose(state.x, state.y, z, state.yaw)
        with renderer_lock:
            observation = renderer.render(
                support.renderer_profile,
                pose,
                FRONT_CAMERAS,
            )
        frames = dict(observation["frames"])
        sequence = runtime["sequence"] + 1
        snapshot = current_overhead_snapshot()
        requires_sync = (
            snapshot is None
            or snapshot.profile != support.renderer_profile
            or math.hypot(
                state.x - snapshot.state.x,
                state.y - snapshot.state.y,
            )
            > 3.0
            or abs(
                math.degrees(
                    math.atan2(
                        math.sin(state.yaw - snapshot.state.yaw),
                        math.cos(state.yaw - snapshot.state.yaw),
                    )
                )
            )
            > 20.0
        )
        if requires_sync:
            refresh_overhead_sync(
                state,
                support.renderer_profile,
                pose,
                frames,
                sequence,
            )
        else:
            schedule_overhead_update(
                state,
                support.renderer_profile,
                pose,
                frames,
                sequence,
            )
        overhead, overhead_metadata = overhead_for_state(
            state, support.renderer_profile
        )
        presentation_started = time.perf_counter()
        cockpit = make_cockpit_frame(
            Image,
            ImageDraw,
            composer,
            frames,
            adapter,
            state,
            support.as_dict(),
            overhead=overhead,
            overhead_composer=(
                overhead_composer if overhead is not None else None
            ),
        )
        buffer = io.BytesIO()
        cockpit.save(buffer, format="JPEG", quality=88)
        presentation_seconds = time.perf_counter() - presentation_started
        return buffer.getvalue(), {
            "render_seconds": float(observation["render_seconds"]),
            "presentation_seconds": presentation_seconds,
            "scene_time_seconds": float(observation["scene_time_seconds"]),
            "renderer_profile": support.renderer_profile,
            "camera_count": len(frames),
            "front_projection_coverage_fraction": (
                composer.coverage_fraction[support.renderer_profile]
            ),
            **overhead_metadata,
        }

    def commit(
        state: EgoState,
        jpeg: bytes,
        render: dict[str, object],
        *,
        boundary_hit: bool = False,
        boundary_reason: str | None = None,
        selection_required: bool = False,
    ) -> None:
        runtime.update(
            state=state,
            jpeg=jpeg,
            render=render,
            diagnostic_jpeg=b"",
            boundary_hit=boundary_hit,
            boundary_reason=boundary_reason,
            selection_required=selection_required,
        )

    def diagnostic_jpeg() -> bytes:
        cached = runtime["diagnostic_jpeg"]
        if cached:
            return cached
        from PIL import Image, ImageDraw

        state: EgoState = runtime["state"]
        support = adapter.support(state).as_dict()
        z = route_height(
            renderer,
            str(support["renderer_profile"]),
            float(support["progress_from_anchor_meters"]),
        )
        with renderer_lock:
            observation = renderer.render(
                str(support["renderer_profile"]),
                LocalWorldPose(state.x, state.y, z, state.yaw),
                CAMERAS,
            )
        mosaic = make_diagnostic_mosaic(
            Image, ImageDraw, dict(observation["frames"]), state, support
        )
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=88)
        runtime["diagnostic_jpeg"] = buffer.getvalue()
        return runtime["diagnostic_jpeg"]

    def overhead_jpeg() -> bytes:
        from PIL import Image, ImageDraw

        state: EgoState = runtime["state"]
        support = adapter.support(state).as_dict()
        overhead, _ = overhead_for_state(
            state, str(support["renderer_profile"])
        )
        inset = make_trajectory_bev(
            Image,
            ImageDraw,
            adapter,
            state,
            support,
            overhead=overhead,
            overhead_composer=(
                overhead_composer if overhead is not None else None
            ),
        )
        buffer = io.BytesIO()
        inset.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, value: object) -> None:
            self.send_bytes(status, "application/json", json.dumps(value).encode())

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", render_web_page(args.dt).encode())
            elif path == "/diagnostic":
                self.send_bytes(
                    200,
                    "text/html; charset=utf-8",
                    DIAGNOSTIC_PAGE.encode(),
                )
            elif path == "/frame.jpg":
                self.send_bytes(200, "image/jpeg", runtime["jpeg"])
            elif path == "/diagnostic.jpg":
                self.send_bytes(200, "image/jpeg", diagnostic_jpeg())
            elif path == "/overhead.jpg":
                self.send_bytes(200, "image/jpeg", overhead_jpeg())
            elif path == "/state.json":
                self.send_json(200, payload())
            elif path == "/evidence.json":
                self.send_bytes(200, "application/json", evidence.report_bytes())
            else:
                self.send_bytes(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/tick":
                    if runtime["boundary_hit"]:
                        raise ValueError("reconstruction boundary reached; reset is required")
                    if runtime["selection_required"]:
                        raise ValueError("branch selection is required before driving")
                    keys = set(parse_qs(parsed.query).get("keys", [""])[0])
                    if not keys <= set("wasd"):
                        raise ValueError("keys must contain only W/S/A/D")
                    state: EgoState = runtime["state"]
                    update = adapter.step(state, control_for_keys(keys), args.dt)
                    jpeg, render = render_state(update.state)
                    runtime["sequence"] += 1
                    commit(
                        update.state,
                        jpeg,
                        render,
                        boundary_hit=update.boundary_hit,
                        boundary_reason=update.boundary_reason,
                        selection_required=update.selection_required,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    response = payload({"server_control_to_jpeg_ms": elapsed_ms})
                    support = adapter.support(update.state)
                    evidence.record_server_sample(
                        {
                            "sequence": runtime["sequence"],
                            "control_keys": "".join(sorted(keys)),
                            "simulation_time_seconds": update.state.time,
                            "x_meters": update.state.x,
                            "y_meters": update.state.y,
                            "yaw_degrees": math.degrees(update.state.yaw),
                            "speed_mps": update.state.speed,
                            "route_support": support.as_dict(),
                            "renderer_profile": support.renderer_profile,
                            "frozen_scene_time_seconds": renderer.model_scene_time,
                            "camera_count": render["camera_count"],
                            "all_camera_frames_finite": True,
                            "renderer_ms": float(render["render_seconds"]) * 1000.0,
                            "server_control_to_jpeg_ms": elapsed_ms,
                            "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "boundary_hit": update.boundary_hit,
                            "boundary_reason": update.boundary_reason,
                        }
                    )
                    if update.selection_required and not any(
                        event["event"] == "branch_selection_required"
                        for event in evidence.route_events[-1:]
                    ):
                        evidence.record_route_event(
                            "branch_selection_required",
                            {
                                "simulation_time_seconds": update.state.time,
                                "route_support": support.as_dict(),
                                "branch_options": {
                                    name: adapter.branch_support(name, update.state).as_dict()
                                    for name in sorted(adapter.branches)
                                },
                            },
                        )
                    self.send_json(200, response)
                    return
                if parsed.path == "/branch":
                    branch = parse_qs(parsed.query).get("name", [""])[0]
                    state = runtime["state"]
                    before_hash = hashlib.sha256(runtime["jpeg"]).hexdigest()
                    before_profile = adapter.support(state).renderer_profile
                    support = adapter.select_branch(branch, state)
                    jpeg, render = render_state(state)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    commit(state, jpeg, render)
                    event = evidence.record_route_event(
                        "branch_selected",
                        {
                            "branch": branch,
                            "simulation_time_seconds": state.time,
                            "route_support": support.as_dict(),
                            "renderer_profile_before": before_profile,
                            "renderer_profile_after": support.renderer_profile,
                            "frame_sha256_before": before_hash,
                            "frame_sha256_after": hashlib.sha256(jpeg).hexdigest(),
                            "server_route_selection_to_jpeg_ms": elapsed_ms,
                        },
                    )
                    self.send_json(
                        200,
                        payload(
                            {
                                "server_route_selection_to_jpeg_ms": elapsed_ms,
                                "route_event_index": event["event_index"],
                            }
                        ),
                    )
                    return
                if parsed.path == "/reset":
                    state = adapter.reset()
                    jpeg, render = render_state(state)
                    commit(state, jpeg, render)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    evidence.record_reset(
                        {
                            "simulation_time_seconds": state.time,
                            "route_support": adapter.support(state).as_dict(),
                            "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "server_reset_to_jpeg_ms": elapsed_ms,
                        }
                    )
                    self.send_json(200, payload({"server_reset_to_jpeg_ms": elapsed_ms}))
                    return
                if parsed.path == "/evidence-sample":
                    value = read_json_body(self)
                    evidence.record_browser_timing(
                        sequence=int(value.get("sequence", -1)),
                        browser_request_to_image_ms=value.get("browser_request_to_image_ms"),
                        browser_input_to_image_ms=value.get("browser_input_to_image_ms"),
                        client_unix_ms=value.get("client_unix_ms"),
                    )
                    self.send_json(200, evidence.summary())
                    return
                if parsed.path == "/evidence-route-timing":
                    value = read_json_body(self)
                    evidence.record_route_browser_timing(
                        event_index=int(value.get("event_index", -1)),
                        browser_selection_to_image_ms=value.get(
                            "browser_selection_to_image_ms"
                        ),
                        client_unix_ms=value.get("client_unix_ms"),
                    )
                    self.send_json(200, evidence.summary())
                    return
                self.send_bytes(404, "text/plain", b"not found")
            except Exception as error:
                self.send_json(400, {"error": str(error)})

        def log_message(self, format: str, *values: object) -> None:
            print(f"web: {format % values}")

    # Reserve the port before loading the 1.65 GB checkpoint. Unknown services
    # are never killed automatically.
    server = HTTPServer((args.host, args.port), Handler)
    try:
        renderer.load()
        if renderer.checkpoint_step != args.expected_checkpoint_step:
            raise RuntimeError(
                f"expected checkpoint step {args.expected_checkpoint_step}, "
                f"got {renderer.checkpoint_step}"
            )
        adapter = make_adapter(renderer, max_speed_mps=args.max_speed_mps)
        composer = CylindricalCockpitComposer.from_renderer(renderer)
        overhead_composer = BowlOverheadComposer(
            camera_projections(
                renderer,
                CAMERAS,
                overhead_output_scales,
            )
        )
        assert renderer.checkpoint_path is not None
        evidence = RouteDrivingEvidenceRecorder(
            scene="tbv_miami_shared_entrance_straight_right",
            config_path=args.config,
            checkpoint_path=renderer.checkpoint_path,
            checkpoint_step=renderer.checkpoint_step,
            output_scale=args.output_scale,
            dt_seconds=args.dt,
            camera_names=FRONT_CAMERAS,
            route_contract={
                "spawn_progress_from_anchor_meters": COMMON_START_METERS,
                "branch_anchor_progress_meters": BRANCH_ANCHOR_METERS,
                "straight_end_progress_meters": STRAIGHT_END_METERS,
                "right_end_progress_meters": RIGHT_END_METERS,
                "corridor_half_width_meters": CORRIDOR_HALF_WIDTH_METERS,
                "maximum_heading_error_degrees": CORRIDOR_HEADING_LIMIT_DEGREES,
                "selection_window_meters": SELECTION_WINDOW_METERS,
                "maximum_speed_mps": args.max_speed_mps,
                "boundary_policy": "fail_closed_keep_last_valid_pose_and_stop",
                "driving_display": {
                    "primary_name": "forward_surround",
                    "primary_mode": "calibrated_three_camera_cylindrical_front",
                    "camera_names": list(FRONT_CAMERAS),
                    "horizontal_fov_degrees": (
                        composer.horizontal_fov_degrees
                    ),
                    "top_angle_degrees": composer.top_angle_degrees,
                    "bottom_angle_degrees": composer.bottom_angle_degrees,
                    "auxiliary_name": "overhead",
                    "overhead": {
                        "mode": "calibrated_virtual_bowl_visual_only",
                        "camera_names": list(CAMERAS),
                        "extra_camera_names": list(OVERHEAD_EXTRA_CAMERAS),
                        "extra_camera_output_scale": args.overhead_extra_scale,
                        "background_update_every": args.overhead_update_every,
                        "size_pixels": overhead_composer.size,
                        "forward_meters": overhead_composer.forward_meters,
                        "rear_meters": overhead_composer.rear_meters,
                        "side_meters": overhead_composer.side_meters,
                        "ground_z_meters": overhead_composer.ground_z_meters,
                        "flat_radius_meters": (
                            overhead_composer.flat_radius_meters
                        ),
                        "maximum_rise_meters": (
                            overhead_composer.maximum_rise_meters
                        ),
                        "unknown_pixel_policy": "black_no_completion",
                        "vehicle_blind_zone_policy": "opaque_vehicle_mask",
                        "motion_compensation": "ground_plane_se2",
                        "route_overlay": "logged_trajectory_support_corridor",
                        "free_space_claim": False,
                    },
                    "diagnostic_name": "original_camera_views",
                    "diagnostic_camera_names": list(CAMERAS),
                    "diagnostic_render_policy": "manual_on_demand",
                    "seven_camera_diagnostic_url": "/diagnostic",
                },
            },
            limitations=(
                "Evidence-only static route-following pilot; not a certified simulator.",
                "No ground truth exists for counterfactual lateral poses.",
                "Vehicles are baked into static geometry and cannot respond.",
                "Branch selection may switch traversal-specific appearance/sensor profiles.",
                "The driving panorama is a calibrated three-camera projection; overlap "
                "parallax and seams are presentation artifacts, not geometry evidence.",
                "The overhead inset is a virtual-bowl camera projection for visual "
                "comfort only; it is not free-space, collision, or geometry evidence.",
                "Black overhead pixels are intentionally uncovered and never completed.",
                "The latency-critical forward path renders three front cameras; four "
                "reduced-scale side/rear cameras update the overhead in the background.",
                "The original seven-camera views remain a separate manual diagnostic.",
                "Browser timing excludes monitor scan-out.",
            ),
            output_path=args.evidence_output,
        )
        initial = adapter.reset()
        jpeg, render = render_state(initial)
        commit(initial, jpeg, render)
        evidence.record_reset(
            {
                "simulation_time_seconds": initial.time,
                "route_support": adapter.support(initial).as_dict(),
                "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                "server_reset_to_jpeg_ms": None,
                "reason": "server_start",
            }
        )
        print(f"TbV route adapter: http://{args.host}:{args.port}")
        print(f"original camera views: http://{args.host}:{args.port}/diagnostic")
        print(f"evidence: {args.evidence_output.expanduser().resolve()}")
        print("controls: W/S/A/D, R reset, select 1=straight or 2=right at anchor")
        print("scope: static evidence-only route following; +/-1m fail-closed support")
        print("warning: trusted localhost/tunnel only; one shared driving state")
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping TbV route driving adapter")
    finally:
        overhead_executor.shutdown(wait=True, cancel_futures=True)
        server.server_close()


if __name__ == "__main__":
    main()
