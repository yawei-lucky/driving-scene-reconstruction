#!/usr/bin/env python3
"""Probe the TbV two-traversal checkpoint from shared world poses.

This is intentionally an experiment-local renderer.  It keeps the joint
SplatAD model loaded once, preserves traversal-specific camera sensor IDs, and
moves either calibrated seven-camera rig through one common local world frame.
The source scene time is fixed; only the requested rig pose changes.
"""

from __future__ import annotations

import argparse
import bisect
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/tbv_h3/"
    "tbv_branch_pair_splatad_pilot_2000/splatad/"
    "2026-07-22_train90_seed750k/config.yml"
)
RIGHT_TRAVERSAL = "OCaNX1bQSmlP3jEQH80C0TZYzZhKLV81__Spring_2020"
STRAIGHT_TRAVERSAL = "QMnNKZiFaxnuGQmxpGkZFdM2EE7uWqDQ__Spring_2020"
CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)


@dataclass(frozen=True)
class RouteSample:
    """One synchronized seven-camera rig centre on a recorded traversal."""

    time: float
    progress: float
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class LocalWorldPose:
    """Metric pose in the shared branch-local world frame."""

    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class AnchorMatch:
    right_index: int
    straight_index: int
    distance: float
    heading_difference_radians: float
    shared_match_count: int
    shared_right_span: float


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def angle_difference(first: float, second: float) -> float:
    return abs(wrap_angle(first - second))


def csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return result


def offset_pose(pose: LocalWorldPose, left_meters: float) -> LocalWorldPose:
    """Offset a pose along its own horizontal left axis."""

    if not math.isfinite(left_meters):
        raise ValueError("left offset must be finite")
    return replace(
        pose,
        x=pose.x - math.sin(pose.yaw) * left_meters,
        y=pose.y + math.cos(pose.yaw) * left_meters,
    )


def select_branch_anchor(
    right_route: tuple[RouteSample, ...],
    straight_route: tuple[RouteSample, ...],
    *,
    distance_limit: float = 3.0,
    heading_limit_radians: float = math.radians(20.0),
    retreat_meters: float = 2.0,
) -> AnchorMatch:
    """Select a matched point just before the two recorded routes diverge."""

    if not right_route or not straight_route:
        raise ValueError("both routes must contain samples")
    if distance_limit <= 0.0 or retreat_meters < 0.0:
        raise ValueError("distance limit must be positive and retreat non-negative")

    matches: list[tuple[int, int, float, float]] = []
    for right_index, right in enumerate(right_route):
        straight_index, closest = min(
            enumerate(straight_route),
            key=lambda item: math.hypot(
                right.x - item[1].x,
                right.y - item[1].y,
            ),
        )
        distance = math.hypot(right.x - closest.x, right.y - closest.y)
        heading_difference = angle_difference(
            right.yaw, straight_route[straight_index].yaw
        )
        if distance <= distance_limit and heading_difference <= heading_limit_radians:
            matches.append(
                (right_index, straight_index, distance, heading_difference)
            )
    if len(matches) < 3:
        raise RuntimeError(
            f"only {len(matches)} shared route samples satisfy the branch anchor gate"
        )

    shared_progresses = [right_route[index].progress for index, _, _, _ in matches]
    terminal = max(matches, key=lambda item: right_route[item[0]].progress)
    target_progress = right_route[terminal[0]].progress - retreat_meters
    right_index = min(
        (item[0] for item in matches),
        key=lambda index: abs(right_route[index].progress - target_progress),
    )
    right = right_route[right_index]
    straight_index, closest = min(
        enumerate(straight_route),
        key=lambda item: math.hypot(right.x - item[1].x, right.y - item[1].y),
    )
    distance = math.hypot(right.x - closest.x, right.y - closest.y)
    heading_difference = angle_difference(right.yaw, straight_route[straight_index].yaw)
    if distance > distance_limit or heading_difference > heading_limit_radians:
        raise RuntimeError("retreated anchor no longer satisfies the shared-route gate")
    return AnchorMatch(
        right_index=right_index,
        straight_index=straight_index,
        distance=distance,
        heading_difference_radians=heading_difference,
        shared_match_count=len(matches),
        shared_right_span=max(shared_progresses) - min(shared_progresses),
    )


def pose_at_progress(
    route: tuple[RouteSample, ...], progress: float
) -> LocalWorldPose:
    """Interpolate one local-world route pose at signed branch progress."""

    if not route:
        raise ValueError("route cannot be empty")
    progresses = tuple(sample.progress for sample in route)
    if progress < progresses[0] - 1e-6 or progress > progresses[-1] + 1e-6:
        raise ValueError(
            f"progress {progress:.3f}m is outside "
            f"[{progresses[0]:.3f}, {progresses[-1]:.3f}]m"
        )
    insertion = bisect.bisect_left(progresses, progress)
    if insertion == 0:
        sample = route[0]
        return LocalWorldPose(sample.x, sample.y, sample.z, sample.yaw)
    if insertion == len(route):
        sample = route[-1]
        return LocalWorldPose(sample.x, sample.y, sample.z, sample.yaw)
    left = route[insertion - 1]
    right = route[insertion]
    span = right.progress - left.progress
    fraction = 0.0 if span <= 1e-9 else (progress - left.progress) / span
    yaw_delta = wrap_angle(right.yaw - left.yaw)
    return LocalWorldPose(
        x=left.x * (1.0 - fraction) + right.x * fraction,
        y=left.y * (1.0 - fraction) + right.y * fraction,
        z=left.z * (1.0 - fraction) + right.z * fraction,
        yaw=wrap_angle(left.yaw + yaw_delta * fraction),
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def mean_absolute_pixel_difference(first: Any, second: Any) -> float:
    import numpy as np

    return float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean())


def near_black_fraction(frame: Any) -> float:
    import numpy as np

    return float(np.all(frame < 5, axis=2).mean())


class TbVWorldRenderer:
    """Small shared-world renderer for the two fixed TbV traversals."""

    def __init__(self, config_path: Path, output_scale: float) -> None:
        self.config_path = config_path.expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not 0.0 < output_scale <= 1.0:
            raise ValueError("output scale must be in (0, 1]")
        self.output_scale = output_scale
        self.pipeline: Any | None = None
        self.torch: Any | None = None
        self.records: dict[str, dict[str, list[tuple[float, Any, Path]]]] = {}
        self.routes: dict[str, tuple[RouteSample, ...]] = {}
        self.source_cameras: dict[str, dict[str, Any]] = {}
        self.source_poses: dict[str, dict[str, Any]] = {}
        self.source_origins: dict[str, tuple[float, float, float]] = {}
        self.source_yaws: dict[str, float] = {}
        self.common_origin: tuple[float, float, float] | None = None
        self.common_yaw: float | None = None
        self.model_scene_time: float | None = None
        self.anchor_match: AnchorMatch | None = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_step: int | None = None
        self.dataparser_scale: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @staticmethod
    def _horizontal_forward(pose: Any) -> tuple[float, float]:
        x = -float(pose[0, 2])
        y = -float(pose[1, 2])
        length = math.hypot(x, y)
        if length <= 1e-9:
            raise RuntimeError("front camera has no horizontal forward direction")
        return x / length, y / length

    @staticmethod
    def _interpolate_pose(torch_module: Any, left: Any, right: Any, fraction: float) -> Any:
        translation = left[:, 3] * (1.0 - fraction) + right[:, 3] * fraction
        blended = left[:, :3] * (1.0 - fraction) + right[:, :3] * fraction
        u, _, vh = torch_module.linalg.svd(blended)
        rotation = u @ vh
        if float(torch_module.linalg.det(rotation)) < 0.0:
            u = u.clone()
            u[:, -1] *= -1.0
            rotation = u @ vh
        return torch_module.cat((rotation, translation[:, None]), dim=1)

    def _pose_at_time(self, traversal: str, camera_name: str, query: float) -> Any:
        assert self.torch is not None
        records = self.records[traversal][camera_name]
        times = tuple(item[0] for item in records)
        insertion = bisect.bisect_left(times, query)
        if insertion == 0:
            return records[0][1].camera_to_worlds[0].clone()
        if insertion == len(records):
            return records[-1][1].camera_to_worlds[0].clone()
        left_time, left_camera, _ = records[insertion - 1]
        right_time, right_camera, _ = records[insertion]
        fraction = (query - left_time) / (right_time - left_time)
        return self._interpolate_pose(
            self.torch,
            left_camera.camera_to_worlds[0],
            right_camera.camera_to_worlds[0],
            fraction,
        )

    def _rig_poses_at_time(self, traversal: str, query: float) -> dict[str, Any]:
        return {
            name: self._pose_at_time(traversal, name, query) for name in CAMERAS
        }

    def _common_time_range(self, traversal: str) -> tuple[float, float]:
        start = max(self.records[traversal][name][0][0] for name in CAMERAS)
        end = min(self.records[traversal][name][-1][0] for name in CAMERAS)
        if start >= end:
            raise RuntimeError(f"empty seven-camera time interval for {traversal}")
        return start, end

    def _build_route(self, traversal: str) -> tuple[RouteSample, ...]:
        start, end = self._common_time_range(traversal)
        front_records = self.records[traversal]["ring_front_center"]
        samples: list[RouteSample] = []
        progress = 0.0
        previous: tuple[float, float] | None = None
        for query, _, _ in front_records:
            if query < start or query > end:
                continue
            poses = self._rig_poses_at_time(traversal, query)
            centres = [pose[:, 3].detach().cpu().tolist() for pose in poses.values()]
            centre = tuple(sum(item[axis] for item in centres) / len(centres) for axis in range(3))
            if previous is not None:
                progress += math.hypot(centre[0] - previous[0], centre[1] - previous[1])
            forward = self._horizontal_forward(poses["ring_front_center"])
            samples.append(
                RouteSample(
                    time=query,
                    progress=progress,
                    x=centre[0],
                    y=centre[1],
                    z=centre[2],
                    yaw=math.atan2(forward[1], forward[0]),
                )
            )
            previous = centre[0], centre[1]
        if len(samples) < 20:
            raise RuntimeError(f"too few synchronized route samples for {traversal}")
        return tuple(samples)

    @staticmethod
    def _zero_source_motion_metadata(camera: Any) -> None:
        if camera.metadata is None:
            return
        for name in (
            "velocities",
            "linear_velocities_local",
            "angular_velocities_local",
            "rolling_shutter_time",
            "time_to_center_pixel",
        ):
            value = camera.metadata.get(name)
            if value is None:
                continue
            if hasattr(value, "clone") and hasattr(value, "zero_"):
                camera.metadata[name] = value.clone().zero_()
            else:
                camera.metadata[name] = 0.0

    def load(self) -> None:
        if self.is_loaded:
            return
        try:
            import torch
            from nerfstudio.scripts.render import streamline_ad_config
            from nerfstudio.utils.eval_utils import eval_setup
        except ImportError as error:
            raise RuntimeError("the pinned Stage H3 environment is required") from error

        config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
            self.config_path,
            test_mode="test",
            update_config_callback=streamline_ad_config,
        )
        pipeline.model.eval()
        datamanager = pipeline.datamanager
        outputs = datamanager.train_dataparser_outputs
        scale = float(outputs.dataparser_scale)
        if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"expected metre-scale TbV poses, got scale {scale}")
        mapping = {
            int(index): str(name)
            for index, name in outputs.metadata["sensor_idx_to_name"].items()
        }
        exact_by_prefix = {
            RIGHT_TRAVERSAL[:4]: RIGHT_TRAVERSAL,
            STRAIGHT_TRAVERSAL[:4]: STRAIGHT_TRAVERSAL,
        }
        records: dict[str, dict[str, list[tuple[float, Any, Path]]]] = {
            traversal: {name: [] for name in CAMERAS}
            for traversal in (RIGHT_TRAVERSAL, STRAIGHT_TRAVERSAL)
        }
        for dataset, cache in (
            (datamanager.train_dataset, datamanager.cached_train),
            (datamanager.eval_dataset, datamanager.cached_eval),
        ):
            for index, raw_filename in enumerate(dataset.image_filenames):
                _ = cache[index]
                filename = Path(raw_filename)
                camera = deepcopy(dataset.cameras[index : index + 1]).to(
                    datamanager.device
                )
                sensor_index = int(camera.metadata["sensor_idxs"].item())
                source_name = mapping[sensor_index]
                prefix, camera_name = source_name.split("/", 1)
                if prefix not in exact_by_prefix or camera_name not in CAMERAS:
                    continue
                traversal = exact_by_prefix[prefix]
                if traversal not in filename.parts:
                    raise RuntimeError(
                        f"sensor mapping {source_name!r} disagrees with {filename}"
                    )
                records[traversal][camera_name].append(
                    (float(camera.times.item()), camera, filename)
                )
        for traversal, cameras in records.items():
            for camera_name, items in cameras.items():
                items.sort(key=lambda item: item[0])
                if len(items) != 100:
                    raise RuntimeError(
                        f"expected 100 {traversal}/{camera_name} images, got {len(items)}"
                    )
                if any(right[0] <= left[0] for left, right in zip(items, items[1:])):
                    raise RuntimeError(
                        f"non-increasing camera times for {traversal}/{camera_name}"
                    )

        self.torch = torch
        self.pipeline = pipeline
        self.records = records
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_step = int(checkpoint_step)
        self.dataparser_scale = scale
        raw_routes = {
            RIGHT_TRAVERSAL: self._build_route(RIGHT_TRAVERSAL),
            STRAIGHT_TRAVERSAL: self._build_route(STRAIGHT_TRAVERSAL),
        }
        match = select_branch_anchor(
            raw_routes[RIGHT_TRAVERSAL], raw_routes[STRAIGHT_TRAVERSAL]
        )
        right_anchor = raw_routes[RIGHT_TRAVERSAL][match.right_index]
        straight_anchor = raw_routes[STRAIGHT_TRAVERSAL][match.straight_index]
        origin = (
            (right_anchor.x + straight_anchor.x) / 2.0,
            (right_anchor.y + straight_anchor.y) / 2.0,
            (right_anchor.z + straight_anchor.z) / 2.0,
        )
        heading_x = math.cos(right_anchor.yaw) + math.cos(straight_anchor.yaw)
        heading_y = math.sin(right_anchor.yaw) + math.sin(straight_anchor.yaw)
        common_yaw = math.atan2(heading_y, heading_x)
        forward = math.cos(common_yaw), math.sin(common_yaw)
        left = -forward[1], forward[0]

        local_routes: dict[str, tuple[RouteSample, ...]] = {}
        anchor_indices = {
            RIGHT_TRAVERSAL: match.right_index,
            STRAIGHT_TRAVERSAL: match.straight_index,
        }
        for traversal, route in raw_routes.items():
            anchor_progress = route[anchor_indices[traversal]].progress
            localized = []
            for sample in route:
                dx, dy = sample.x - origin[0], sample.y - origin[1]
                localized.append(
                    RouteSample(
                        time=sample.time,
                        progress=sample.progress - anchor_progress,
                        x=dx * forward[0] + dy * forward[1],
                        y=dx * left[0] + dy * left[1],
                        z=sample.z - origin[2],
                        yaw=wrap_angle(sample.yaw - common_yaw),
                    )
                )
            local_routes[traversal] = tuple(localized)

        source_cameras: dict[str, dict[str, Any]] = {}
        source_poses: dict[str, dict[str, Any]] = {}
        source_origins: dict[str, tuple[float, float, float]] = {}
        source_yaws: dict[str, float] = {}
        anchor_times = {
            RIGHT_TRAVERSAL: right_anchor.time,
            STRAIGHT_TRAVERSAL: straight_anchor.time,
        }
        for traversal, anchor_time in anchor_times.items():
            poses = self._rig_poses_at_time(traversal, anchor_time)
            source_poses[traversal] = poses
            centres = [pose[:, 3].detach().cpu().tolist() for pose in poses.values()]
            source_origins[traversal] = tuple(
                sum(item[axis] for item in centres) / len(centres) for axis in range(3)
            )
            source_forward = self._horizontal_forward(poses["ring_front_center"])
            source_yaws[traversal] = math.atan2(source_forward[1], source_forward[0])
            source_cameras[traversal] = {}
            for camera_name in CAMERAS:
                nearest = min(
                    records[traversal][camera_name],
                    key=lambda item: abs(item[0] - anchor_time),
                )
                source_cameras[traversal][camera_name] = nearest[1]

        model = pipeline.model
        if not hasattr(model.config, "compensate_rs_camera"):
            raise RuntimeError("SplatAD model has no camera RS compensation switch")
        model.config.compensate_rs_camera = False
        if hasattr(model, "rs_editing"):
            for name in (
                "rs_time",
                "lin_vel_x",
                "lin_vel_y",
                "lin_vel_z",
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
            ):
                model.rs_editing[name] = 0.0

        self.routes = local_routes
        self.source_cameras = source_cameras
        self.source_poses = source_poses
        self.source_origins = source_origins
        self.source_yaws = source_yaws
        self.common_origin = origin
        self.common_yaw = common_yaw
        self.model_scene_time = (right_anchor.time + straight_anchor.time) / 2.0
        self.anchor_match = match

    def _transform_pose(self, traversal: str, camera_name: str, pose: LocalWorldPose) -> Any:
        assert self.torch is not None
        assert self.common_origin is not None
        assert self.common_yaw is not None
        source_pose = self.source_poses[traversal][camera_name]
        source_origin = self.source_origins[traversal]
        delta_yaw = self.common_yaw + pose.yaw - self.source_yaws[traversal]
        cosine, sine = math.cos(delta_yaw), math.sin(delta_yaw)
        rotation_z = self.torch.tensor(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=source_pose.dtype,
            device=source_pose.device,
        )
        rotation = rotation_z @ source_pose[:, :3]
        forward = math.cos(self.common_yaw), math.sin(self.common_yaw)
        left = -forward[1], forward[0]
        desired = self.torch.tensor(
            (
                self.common_origin[0] + pose.x * forward[0] + pose.y * left[0],
                self.common_origin[1] + pose.x * forward[1] + pose.y * left[1],
                self.common_origin[2] + pose.z,
            ),
            dtype=source_pose.dtype,
            device=source_pose.device,
        )
        source_origin_tensor = self.torch.tensor(
            source_origin, dtype=source_pose.dtype, device=source_pose.device
        )
        relative = source_pose[:, 3] - source_origin_tensor
        centre = desired + rotation_z @ relative
        return self.torch.cat((rotation, centre[:, None]), dim=1)

    def render(self, traversal: str, pose: LocalWorldPose) -> dict[str, Any]:
        if not self.is_loaded:
            self.load()
        if traversal not in (RIGHT_TRAVERSAL, STRAIGHT_TRAVERSAL):
            raise KeyError(f"unknown traversal {traversal!r}")
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.z, pose.yaw)):
            raise ValueError("world pose must be finite")
        assert self.pipeline is not None
        assert self.torch is not None
        assert self.model_scene_time is not None
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        frames: dict[str, Any] = {}
        with torch.no_grad():
            for camera_name in CAMERAS:
                camera = deepcopy(self.source_cameras[traversal][camera_name])
                camera.camera_to_worlds = self._transform_pose(
                    traversal, camera_name, pose
                )[None, ...]
                camera.times = torch.full_like(camera.times, self.model_scene_time)
                self._zero_source_motion_metadata(camera)
                if self.output_scale != 1.0:
                    camera.rescale_output_resolution(self.output_scale)
                outputs = self.pipeline.model.get_outputs_for_camera(camera)
                if "rgb" not in outputs:
                    raise RuntimeError(f"SplatAD returned no RGB for {camera_name}")
                rgb = outputs["rgb"]
                if not bool(torch.isfinite(rgb).all()):
                    raise RuntimeError(f"non-finite RGB for {camera_name}")
                frames[camera_name] = (
                    rgb.detach()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "frames": frames,
            "render_seconds": elapsed,
            "traversal": traversal,
            "scene_time_seconds": self.model_scene_time,
            "pose": pose,
        }


def labelled_grid(
    image_module: Any,
    draw_module: Any,
    rows: list[list[tuple[str, Any]]],
) -> Any:
    cell_width, cell_height = 520, 360
    columns = max(len(row) for row in rows)
    canvas = image_module.new(
        "RGB", (cell_width * columns, cell_height * len(rows)), "black"
    )
    for row_index, row in enumerate(rows):
        for column, (label, frame) in enumerate(row):
            tile = image_module.fromarray(frame).convert("RGB")
            tile.thumbnail((cell_width, cell_height), image_module.Resampling.LANCZOS)
            draw_module.Draw(tile).text(
                (8, 7),
                label,
                fill=(255, 216, 77),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            x = column * cell_width + (cell_width - tile.width) // 2
            y = row_index * cell_height + (cell_height - tile.height) // 2
            canvas.paste(tile, (x, y))
    return canvas


def seven_camera_mosaic(
    image_module: Any,
    draw_module: Any,
    frames: dict[str, Any],
    label: str,
) -> Any:
    cell_width, cell_height = 420, 310
    canvas = image_module.new("RGB", (cell_width * 4, cell_height * 2 + 34), "black")
    for index, camera_name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[camera_name]).convert("RGB")
        tile.thumbnail((cell_width, cell_height), image_module.Resampling.LANCZOS)
        draw_module.Draw(tile).text(
            (7, 6),
            camera_name,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        x = (index % 4) * cell_width + (cell_width - tile.width) // 2
        y = (index // 4) * cell_height + (cell_height - tile.height) // 2
        canvas.paste(tile, (x, y))
    draw_module.Draw(canvas).text((8, cell_height * 2 + 8), label, fill=(255, 216, 77))
    return canvas


def draw_routes(
    image_module: Any,
    draw_module: Any,
    routes: dict[str, tuple[RouteSample, ...]],
    selected: dict[str, tuple[float, ...]],
) -> Any:
    width, height, margin = 1100, 760, 70
    canvas = image_module.new("RGB", (width, height), (18, 18, 18))
    draw = draw_module.Draw(canvas)
    all_samples = [sample for route in routes.values() for sample in route]
    xs = [sample.x for sample in all_samples]
    ys = [sample.y for sample in all_samples]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span, y_span = max(1.0, x_max - x_min), max(1.0, y_max - y_min)

    def point(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - x_min) / x_span * (width - 2 * margin))
        py = height - margin - int((y - y_min) / y_span * (height - 2 * margin))
        return px, py

    colours = {RIGHT_TRAVERSAL: (255, 120, 70), STRAIGHT_TRAVERSAL: (80, 170, 255)}
    labels = {RIGHT_TRAVERSAL: "OCa right turn", STRAIGHT_TRAVERSAL: "QMn straight"}
    for traversal, route in routes.items():
        draw.line(
            [point(sample.x, sample.y) for sample in route],
            fill=colours[traversal],
            width=5,
        )
        for progress in selected[traversal]:
            pose = pose_at_progress(route, progress)
            px, py = point(pose.x, pose.y)
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(255, 235, 80))
        draw.text(
            (margin, 24 + 28 * list(routes).index(traversal)),
            labels[traversal],
            fill=colours[traversal],
        )
    origin = point(0.0, 0.0)
    draw.ellipse(
        (origin[0] - 9, origin[1] - 9, origin[0] + 9, origin[1] + 9),
        outline=(255, 255, 255),
        width=3,
    )
    draw.text(
        (margin, height - 36),
        "yellow=render stations, white ring=shared branch anchor",
        fill=(220, 220, 220),
    )
    return canvas


def ensure_stations_fit(
    route: tuple[RouteSample, ...], stations: Iterable[float], label: str
) -> None:
    low, high = route[0].progress, route[-1].progress
    requested = tuple(stations)
    if min(requested) < low - 1e-6 or max(requested) > high + 1e-6:
        raise ValueError(
            f"{label} stations [{min(requested):.1f}, {max(requested):.1f}]m "
            f"exceed route coverage [{low:.1f}, {high:.1f}]m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--expected-checkpoint-step", type=int, default=1999)
    parser.add_argument("--common-stations", type=csv_floats, default=csv_floats("-20,-10,-2"))
    parser.add_argument("--straight-stations", type=csv_floats, default=csv_floats("5,20,40"))
    parser.add_argument("--right-stations", type=csv_floats, default=csv_floats("5,15,30"))
    parser.add_argument("--lateral-offsets", type=csv_floats, default=csv_floats("-1,0,1"))
    args = parser.parse_args()

    import numpy as np
    from PIL import Image, ImageDraw

    if 0.0 not in args.lateral_offsets:
        raise ValueError("lateral offsets must include 0 for the centre reference")
    if not any(offset != 0.0 for offset in args.lateral_offsets):
        raise ValueError("lateral offsets must include at least one counterfactual")
    expected_observation_count = (
        len(args.common_stations) * 2
        + len(args.straight_stations)
        + len(args.right_stations)
    ) * len(args.lateral_offsets)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = TbVWorldRenderer(args.config, args.output_scale)
    renderer.load()
    if renderer.checkpoint_step != args.expected_checkpoint_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_checkpoint_step}, "
            f"got {renderer.checkpoint_step}"
        )
    right_route = renderer.routes[RIGHT_TRAVERSAL]
    straight_route = renderer.routes[STRAIGHT_TRAVERSAL]
    ensure_stations_fit(right_route, args.common_stations, "right common")
    ensure_stations_fit(straight_route, args.common_stations, "straight common")
    ensure_stations_fit(right_route, args.right_stations, "right branch")
    ensure_stations_fit(straight_route, args.straight_stations, "straight branch")

    render_times: list[float] = []
    records: list[dict[str, Any]] = []
    all_valid = True
    all_near_black_ok = True
    lateral_differences: list[float] = []

    def render_section(
        section: str,
        traversal: str,
        stations: tuple[float, ...],
    ) -> list[list[tuple[str, Any]]]:
        nonlocal all_valid, all_near_black_ok
        rows: list[list[tuple[str, Any]]] = []
        for station in stations:
            centre_pose = pose_at_progress(renderer.routes[traversal], station)
            observations: dict[float, dict[str, Any]] = {}
            row: list[tuple[str, Any]] = []
            for lateral in args.lateral_offsets:
                pose = offset_pose(centre_pose, lateral)
                observation = renderer.render(traversal, pose)
                observations[lateral] = observation
                frames = observation["frames"]
                valid = all(
                    name in frames
                    and frames[name].dtype == np.uint8
                    and frames[name].ndim == 3
                    and frames[name].shape[2] == 3
                    and bool(np.isfinite(frames[name]).all())
                    for name in CAMERAS
                )
                black_fraction = near_black_fraction(frames["ring_front_center"])
                black_ok = black_fraction < 0.25
                all_valid &= valid
                all_near_black_ok &= black_ok
                render_times.append(float(observation["render_seconds"]))
                short = traversal[:4]
                label = f"{short} {station:+.0f}m left={lateral:+.0f}m"
                row.append((label, frames["ring_front_center"]))
                front_dir = output_dir / "front" / section
                front_dir.mkdir(parents=True, exist_ok=True)
                front_path = front_dir / f"{short}_{station:+05.1f}_left_{lateral:+.1f}.jpg"
                Image.fromarray(frames["ring_front_center"]).save(front_path, quality=95)
                mosaic_path: Path | None = None
                if lateral == 0.0:
                    mosaic_dir = output_dir / "seven_camera"
                    mosaic_dir.mkdir(parents=True, exist_ok=True)
                    mosaic_path = mosaic_dir / f"{section}_{short}_{station:+05.1f}.jpg"
                    seven_camera_mosaic(Image, ImageDraw, frames, label).save(
                        mosaic_path, quality=94
                    )
                records.append(
                    {
                        "section": section,
                        "traversal": traversal,
                        "route_role": "right_turn" if traversal == RIGHT_TRAVERSAL else "straight",
                        "progress_from_branch_anchor_meters": station,
                        "left_from_recorded_route_meters": lateral,
                        "world_pose": {
                            "x_meters": pose.x,
                            "y_meters": pose.y,
                            "z_meters": pose.z,
                            "yaw_degrees": math.degrees(pose.yaw),
                        },
                        "all_seven_frames_valid": valid,
                        "front_near_black_fraction_diagnostic": black_fraction,
                        "render_seconds": observation["render_seconds"],
                        "front_artifact": str(front_path),
                        "seven_camera_artifact": str(mosaic_path) if mosaic_path else None,
                    }
                )
            centre_front = observations[0.0]["frames"]["ring_front_center"]
            for lateral, observation in observations.items():
                if lateral != 0.0:
                    lateral_differences.append(
                        mean_absolute_pixel_difference(
                            centre_front, observation["frames"]["ring_front_center"]
                        )
                    )
            rows.append(row)
        return rows

    # Warm the seven heads once. The warm observation is not part of the gate count.
    renderer.render(RIGHT_TRAVERSAL, pose_at_progress(right_route, 0.0))
    common_rows: list[list[tuple[str, Any]]] = []
    for station in args.common_stations:
        combined: list[tuple[str, Any]] = []
        for traversal in (RIGHT_TRAVERSAL, STRAIGHT_TRAVERSAL):
            combined.extend(render_section("common", traversal, (station,))[0])
        common_rows.append(combined)
    straight_rows = render_section("straight", STRAIGHT_TRAVERSAL, args.straight_stations)
    right_rows = render_section("right", RIGHT_TRAVERSAL, args.right_stations)

    common_sheet = output_dir / "common_approach_front.jpg"
    straight_sheet = output_dir / "straight_branch_front.jpg"
    right_sheet = output_dir / "right_branch_front.jpg"
    labelled_grid(Image, ImageDraw, common_rows).save(common_sheet, quality=95)
    labelled_grid(Image, ImageDraw, straight_rows).save(straight_sheet, quality=95)
    labelled_grid(Image, ImageDraw, right_rows).save(right_sheet, quality=95)
    route_plot = output_dir / "tbv_branch_world_routes.jpg"
    draw_routes(
        Image,
        ImageDraw,
        renderer.routes,
        {
            RIGHT_TRAVERSAL: tuple(args.common_stations) + tuple(args.right_stations),
            STRAIGHT_TRAVERSAL: tuple(args.common_stations) + tuple(args.straight_stations),
        },
    ).save(route_plot, quality=95)

    assert renderer.anchor_match is not None
    technical_gates = {
        "expected_checkpoint_step": renderer.checkpoint_step
        == args.expected_checkpoint_step,
        "metre_scale_world_poses": renderer.dataparser_scale == 1.0,
        "shared_anchor_distance_under_3m": renderer.anchor_match.distance <= 3.0,
        "shared_anchor_heading_under_20deg": (
            renderer.anchor_match.heading_difference_radians
            <= math.radians(20.0)
        ),
        "shared_route_span_at_least_30m": renderer.anchor_match.shared_right_span >= 30.0,
        "all_requested_observations_have_seven_finite_frames": (
            len(records) == expected_observation_count and all_valid
        ),
        "front_views_not_mostly_black": all_near_black_ok,
        "all_lateral_offsets_change_front_pixels": (
            bool(lateral_differences) and min(lateral_differences) > 1.0
        ),
        "single_frozen_scene_time": renderer.model_scene_time is not None,
    }
    report = {
        "automated_probe_status": "pass" if all(technical_gates.values()) else "fail",
        "visual_corridor_status": "requires_human_review",
        "certified_drivable_corridor": False,
        "scope": "TbV shared-world coverage probe; no GT exists for lateral offsets",
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(renderer.checkpoint_path),
        "checkpoint_step": renderer.checkpoint_step,
        "expected_checkpoint_step": args.expected_checkpoint_step,
        "output_scale": args.output_scale,
        "camera_count": len(CAMERAS),
        "observation_count": len(records),
        "expected_observation_count": expected_observation_count,
        "camera_render_count": len(records) * len(CAMERAS),
        "route_roles": {
            "right_turn": RIGHT_TRAVERSAL,
            "straight": STRAIGHT_TRAVERSAL,
        },
        "anchor": {
            "right_index": renderer.anchor_match.right_index,
            "straight_index": renderer.anchor_match.straight_index,
            "cross_route_distance_meters": renderer.anchor_match.distance,
            "heading_difference_degrees": math.degrees(
                renderer.anchor_match.heading_difference_radians
            ),
            "shared_match_count": renderer.anchor_match.shared_match_count,
            "shared_right_span_meters": renderer.anchor_match.shared_right_span,
            "common_origin_model_coordinates": renderer.common_origin,
            "common_heading_degrees": math.degrees(renderer.common_yaw or 0.0),
            "frozen_model_scene_time_seconds": renderer.model_scene_time,
        },
        "route_coverage_from_anchor_meters": {
            "right_turn": [right_route[0].progress, right_route[-1].progress],
            "straight": [straight_route[0].progress, straight_route[-1].progress],
        },
        "stations": {
            "common_meters": list(args.common_stations),
            "straight_meters": list(args.straight_stations),
            "right_meters": list(args.right_stations),
            "lateral_offsets_meters": list(args.lateral_offsets),
        },
        "technical_gates": technical_gates,
        "lateral_front_mean_abs_pixel_difference": distribution(lateral_differences),
        "render_seconds_per_seven_camera_observation": distribution(render_times),
        "records": records,
        "artifacts": {
            "route_plot": str(route_plot),
            "common_approach_front": str(common_sheet),
            "straight_branch_front": str(straight_sheet),
            "right_branch_front": str(right_sheet),
        },
        "limitations": [
            (
                "The checkpoint remains an experimental reconstruction until "
                "this visual gate is reviewed."
            ),
            "No ground truth exists for the +/-1m counterfactual poses.",
            "The model has no actor decomposition; vehicles may be baked into static geometry.",
            "Near-black and pixel-difference gates test plumbing, not road usability.",
            "Human visual review decides whether longer training is justified.",
        ],
    }
    report_path = output_dir / "tbv_world_pose_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["automated_probe_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
