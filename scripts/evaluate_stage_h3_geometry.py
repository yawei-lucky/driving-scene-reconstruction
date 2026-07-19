#!/usr/bin/env python3
"""Evaluate H3 SplatAD geometry without retraining.

This script intentionally uses the pinned neurad-studio model and datamanager
objects instead of reimplementing their camera, rolling-shutter, actor, or
LiDAR raster conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from nerfstudio.scripts.render import streamline_ad_config
from nerfstudio.utils.eval_utils import eval_setup


CAMERA_NAMES = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "back",
    4: "left",
    5: "right",
}
CAMERA_ORDER = (1, 0, 2, 4, 3, 5)
POSES = (
    ("center", 0.0, 0.0, 0.0),
    ("forward_p0.50m", 0.50, 0.0, 0.0),
    ("forward_n0.50m", -0.50, 0.0, 0.0),
    ("left_p0.25m", 0.0, 0.25, 0.0),
    ("left_n0.25m", 0.0, -0.25, 0.0),
    ("yaw_p2.0deg", 0.0, 0.0, 2.0),
    ("yaw_n2.0deg", 0.0, 0.0, -2.0),
)


def parse_frames(value: str) -> tuple[str, ...]:
    frames = tuple(item.strip().zfill(2) for item in value.split(",") if item.strip())
    if not frames:
        raise argparse.ArgumentTypeError("at least one frame is required")
    return frames


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": float(np.mean(values)),
    }


def tensor_distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def image_tile(path: Path, size: tuple[int, int], label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(17)
    bounds = draw.textbbox((0, 0), label, font=label_font)
    draw.rectangle((5, 5, bounds[2] + 15, bounds[3] + 13), fill="black")
    draw.text((10, 8), label, fill=(255, 230, 60), font=label_font)
    return canvas


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path, quality=95)


def yaw_matrix(degrees: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def transform_camera(
    camera: Any,
    rig_center: torch.Tensor,
    forward: torch.Tensor,
    left: torch.Tensor,
    forward_metres: float,
    left_metres: float,
    yaw_degrees: float,
) -> Any:
    result = deepcopy(camera)
    pose = result.camera_to_worlds[0].clone()
    rotation = yaw_matrix(yaw_degrees, pose.device)
    translated_center = (
        rig_center + forward * forward_metres + left * left_metres
    )
    pose[:3, :3] = rotation @ pose[:3, :3]
    pose[:3, 3] = translated_center + rotation @ (pose[:3, 3] - rig_center)
    result.camera_to_worlds[0] = pose
    return result


def build_camera_records(datamanager: Any) -> dict[str, dict[int, Any]]:
    records: dict[str, dict[int, Any]] = {}
    filenames = datamanager.eval_dataset.image_filenames
    camera_batches = datamanager.fixed_indices_eval_dataloader
    if len(filenames) != len(camera_batches):
        raise RuntimeError("eval filename/camera count mismatch")
    for index, ((camera, _), filename) in enumerate(zip(camera_batches, filenames)):
        frame = Path(filename).stem
        sensor_index = int(camera.metadata["sensor_idxs"].item())
        if int(camera.metadata["cam_idx"]) != index:
            raise RuntimeError("unexpected eval camera index")
        records.setdefault(frame, {})[sensor_index] = camera
    return records


def rig_centers(camera_records: dict[str, dict[int, Any]]) -> dict[str, torch.Tensor]:
    centers = {}
    for frame, cameras in camera_records.items():
        if set(cameras) != set(CAMERA_NAMES):
            raise RuntimeError(f"frame {frame} does not contain all six cameras")
        centers[frame] = torch.stack(
            [camera.camera_to_worlds[0, :3, 3] for camera in cameras.values()]
        ).mean(dim=0)
    return centers


def ego_basis(
    frame: str,
    centers: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = sorted(centers, key=int)
    index = ordered.index(frame)
    before = centers[ordered[max(0, index - 1)]]
    after = centers[ordered[min(len(ordered) - 1, index + 1)]]
    forward = after - before
    forward[2] = 0.0
    forward = forward / torch.linalg.norm(forward)
    up = torch.tensor([0.0, 0.0, 1.0], device=forward.device)
    left = torch.linalg.cross(up, forward)
    left = left / torch.linalg.norm(left)
    return forward, left


def nearby_pose_gate(
    pipeline: Any,
    frames: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    model = pipeline.model
    datamanager = pipeline.datamanager
    camera_records = build_camera_records(datamanager)
    missing = [frame for frame in frames if frame not in camera_records]
    if missing:
        raise KeyError(f"frames are not in the fixed eval split: {missing}")
    centers = rig_centers(camera_records)

    # Warm each sensor once so timings exclude first-call setup.
    with torch.no_grad():
        for sensor_index in CAMERA_ORDER:
            model.get_outputs_for_camera(camera_records[frames[0]][sensor_index])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_view: list[dict[str, Any]] = []
    rig_latency_ms: list[float] = []
    finite_failures = 0
    for frame in frames:
        forward, left = ego_basis(frame, centers)
        for pose_name, forward_m, left_m, yaw_deg in POSES:
            rig_render_latency = 0.0
            for sensor_index in CAMERA_ORDER:
                camera = transform_camera(
                    camera_records[frame][sensor_index],
                    centers[frame],
                    forward,
                    left,
                    forward_m,
                    left_m,
                    yaw_deg,
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0
                rig_render_latency += latency_ms

                rgb = outputs["rgb"]
                depth = outputs["depth"]
                accumulation = outputs["accumulation"]
                finite = bool(
                    torch.isfinite(rgb).all()
                    and torch.isfinite(depth).all()
                    and torch.isfinite(accumulation).all()
                )
                if not finite:
                    finite_failures += 1

                height, width = accumulation.shape[:2]
                lower_center_roi = accumulation[
                    int(height * 0.55) :,
                    int(width * 0.20) : int(width * 0.80),
                ]
                record = {
                    "frame": frame,
                    "pose": pose_name,
                    "sensor_index": sensor_index,
                    "camera": CAMERA_NAMES[sensor_index],
                    "time_seconds": float(camera.times.item()),
                    "latency_ms": latency_ms,
                    "finite": finite,
                    "all_accumulation_below_0_5_fraction": float(
                        (accumulation < 0.5).float().mean()
                    ),
                    "lower_center_roi_accumulation_below_0_5_fraction": float(
                        (lower_center_roi < 0.5).float().mean()
                    ),
                    "lower_center_roi_accumulation_below_0_1_fraction": float(
                        (lower_center_roi < 0.1).float().mean()
                    ),
                    "depth_median_metres": float(torch.median(depth)),
                }
                per_view.append(record)
                save_rgb(
                    rgb,
                    output_dir
                    / "nearby_rgb"
                    / frame
                    / pose_name
                    / f"{CAMERA_NAMES[sensor_index]}.jpg",
                )
            rig_latency_ms.append(rig_render_latency)

    latency_values = [record["latency_ms"] for record in per_view]
    lower_center_holes = [
        record["lower_center_roi_accumulation_below_0_5_fraction"]
        for record in per_view
    ]
    result = {
        "frames": list(frames),
        "poses": [pose[0] for pose in POSES],
        "views": len(per_view),
        "finite_failures": finite_failures,
        "single_camera_latency_ms": distribution(latency_values),
        "six_camera_rig_latency_ms": distribution(rig_latency_ms),
        "rig_pivot_note": (
            "The yaw pivot is the mean of the six per-camera logged poses. "
            "Because PandaSet cameras are asynchronous, this is an approximate "
            "ego center; the same rigid transform is still applied to all six "
            "cameras."
        ),
        "lower_center_roi_definition": {
            "height": "bottom 45%",
            "width": "central 60%",
            "note": (
                "image-space opacity-hole diagnostic only; this is not a "
                "semantic road mask and does not prove correct depth"
            ),
        },
        "lower_center_roi_accumulation_below_0_5_fraction": distribution(
            lower_center_holes
        ),
        "lower_center_roi_views_over_1_percent_below_0_5": sum(
            value > 0.01 for value in lower_center_holes
        ),
        "gpu_peak_allocated_gib_during_nearby_render": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
        "per_view": per_view,
    }
    return result


def lidar_gate(pipeline: Any) -> dict[str, Any]:
    model = pipeline.model
    dataloader = pipeline.datamanager.fixed_indices_eval_lidar_dataloader
    # Warm the exact held-out raster path once.
    with torch.no_grad():
        model.get_lidar_outputs(dataloader[0][0])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_frame: list[dict[str, Any]] = []
    all_abs: list[torch.Tensor] = []
    all_rel: list[torch.Tensor] = []
    all_median_abs: list[torch.Tensor] = []
    all_median_rel: list[torch.Tensor] = []
    all_static_abs: list[torch.Tensor] = []
    all_static_rel: list[torch.Tensor] = []
    all_static_median_abs: list[torch.Tensor] = []
    all_static_median_rel: list[torch.Tensor] = []
    latency_values: list[float] = []
    finite_failures = 0
    dynamic_rays_excluded = 0
    camera_records = build_camera_records(pipeline.datamanager)
    front_times = {
        frame: float(cameras[0].times.item())
        for frame, cameras in camera_records.items()
    }
    for index, (lidar, batch) in enumerate(dataloader):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.get_lidar_outputs(lidar)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        latency_values.append(latency_ms)

        metrics, _ = model.get_image_metrics_and_images(outputs, batch)
        pred, gt = model.filter_lidar_pred_and_gt(
            outputs,
            batch,
            output_point_cloud=False,
        )
        depth = gt["depth"]
        valid_range = (depth >= 2.0) & (depth <= 50.0)
        abs_error = torch.abs(pred["depth"][valid_range] - depth[valid_range])
        rel_error = abs_error / depth[valid_range]
        median_abs_error = torch.abs(
            pred["median_depth"][valid_range] - depth[valid_range]
        )
        median_rel_error = median_abs_error / depth[valid_range]
        valid_return_indices = batch[
            "raster_pts_valid_depth_and_did_return"
        ]
        raster_returns = batch["raster_pts"].reshape(-1, 5)[
            valid_return_indices
        ]
        angles = torch.deg2rad(raster_returns[:, :2])
        directions = torch.stack(
            [
                torch.cos(angles[:, 0]) * torch.cos(angles[:, 1]),
                torch.sin(angles[:, 0]) * torch.cos(angles[:, 1]),
                torch.sin(angles[:, 1]),
            ],
            dim=-1,
        )
        points_lidar = directions * raster_returns[:, 2:3]
        lidar_pose = lidar.lidar_to_worlds[0]
        points_world = (
            points_lidar @ lidar_pose[:3, :3].transpose(0, 1)
            + lidar_pose[:3, 3]
        )
        lidar_time = float(lidar.times.item())
        dynamic_mask = torch.zeros(
            len(points_world),
            dtype=torch.bool,
            device=points_world.device,
        )
        trajectories = pipeline.datamanager.eval_lidar_dataset.metadata[
            "trajectories"
        ]
        for trajectory in trajectories:
            timestamps = trajectory["timestamps"].to(points_world.device)
            time_index = torch.argmin(torch.abs(timestamps - lidar_time))
            box_pose = trajectory["poses"][time_index].to(points_world.device)
            box_size = trajectory["dims"].to(points_world.device) * 1.15
            world2box = torch.linalg.inv(box_pose)
            points_box = (
                points_world @ world2box[:3, :3].transpose(0, 1)
                + world2box[:3, 3]
            )
            dynamic_mask |= torch.all(
                torch.abs(points_box) < box_size / 2,
                dim=-1,
            )
        static_range = valid_range & ~dynamic_mask
        dynamic_rays_excluded += int((valid_range & dynamic_mask).sum())
        static_abs_error = torch.abs(
            pred["depth"][static_range] - depth[static_range]
        )
        static_rel_error = static_abs_error / depth[static_range]
        static_median_abs_error = torch.abs(
            pred["median_depth"][static_range] - depth[static_range]
        )
        static_median_rel_error = (
            static_median_abs_error / depth[static_range]
        )
        finite = bool(
            torch.isfinite(abs_error).all()
            and torch.isfinite(rel_error).all()
            and torch.isfinite(median_abs_error).all()
            and torch.isfinite(median_rel_error).all()
            and torch.isfinite(static_abs_error).all()
            and torch.isfinite(static_rel_error).all()
            and torch.isfinite(static_median_abs_error).all()
            and torch.isfinite(static_median_rel_error).all()
        )
        finite_failures += int(not finite)
        all_abs.append(abs_error.detach().cpu())
        all_rel.append(rel_error.detach().cpu())
        all_median_abs.append(median_abs_error.detach().cpu())
        all_median_rel.append(median_rel_error.detach().cpu())
        all_static_abs.append(static_abs_error.detach().cpu())
        all_static_rel.append(static_rel_error.detach().cpu())
        all_static_median_abs.append(
            static_median_abs_error.detach().cpu()
        )
        all_static_median_rel.append(
            static_median_rel_error.detach().cpu()
        )
        source_frame = min(
            front_times,
            key=lambda frame: abs(front_times[frame] - lidar_time),
        )
        per_frame.append(
            {
                "eval_lidar_index": index,
                "source_frame": source_frame,
                "time_seconds": lidar_time,
                "latency_ms": latency_ms,
                "finite": finite,
                "native_metrics": {
                    key: float(value) for key, value in metrics.items()
                },
                "returned_depth_2_to_50m": {
                    "expected_depth": {
                        "absolute_error_metres": tensor_distribution(abs_error),
                        "relative_absolute_error": tensor_distribution(rel_error),
                    },
                    "median_depth": {
                        "absolute_error_metres": tensor_distribution(
                            median_abs_error
                        ),
                        "relative_absolute_error": tensor_distribution(
                            median_rel_error
                        ),
                    },
                    "cuboid_excluded_static_expected_depth": {
                        "absolute_error_metres": tensor_distribution(
                            static_abs_error
                        ),
                        "relative_absolute_error": tensor_distribution(
                            static_rel_error
                        ),
                    },
                },
            }
        )

    absolute = torch.cat(all_abs)
    relative = torch.cat(all_rel)
    median_absolute = torch.cat(all_median_abs)
    median_relative = torch.cat(all_median_rel)
    static_absolute = torch.cat(all_static_abs)
    static_relative = torch.cat(all_static_rel)
    static_median_absolute = torch.cat(all_static_median_abs)
    static_median_relative = torch.cat(all_static_median_rel)
    native_metric_names = sorted(per_frame[0]["native_metrics"])
    return {
        "frames": len(per_frame),
        "finite_failures": finite_failures,
        "scope": (
            "all valid returned held-out Pandar64 rays at 2-50m; "
            "not yet filtered to static semantics/cuboids"
        ),
        "expected_depth": {
            "absolute_error_metres": tensor_distribution(absolute),
            "relative_absolute_error": tensor_distribution(relative),
        },
        "median_depth": {
            "absolute_error_metres": tensor_distribution(median_absolute),
            "relative_absolute_error": tensor_distribution(median_relative),
        },
        "cuboid_excluded_static": {
            "scope": (
                "2-50m returns outside annotated moving-actor cuboids at the "
                "nearest interpolated central lidar time; cuboids use 15% "
                "size padding. This does not yet use point semantics or "
                "per-ray actor-time correction."
            ),
            "dynamic_rays_excluded": dynamic_rays_excluded,
            "expected_depth": {
                "absolute_error_metres": tensor_distribution(static_absolute),
                "relative_absolute_error": tensor_distribution(static_relative),
            },
            "median_depth": {
                "absolute_error_metres": tensor_distribution(
                    static_median_absolute
                ),
                "relative_absolute_error": tensor_distribution(
                    static_median_relative
                ),
            },
        },
        "native_metric_means": {
            name: float(
                np.mean(
                    [frame["native_metrics"][name] for frame in per_frame]
                )
            )
            for name in native_metric_names
        },
        "late_sequence_note": (
            "Inspect per-frame native Chamfer metrics separately; aggregate "
            "depth percentiles can hide late-sequence degradation."
        ),
        "latency_ms": distribution(latency_values),
        "gpu_peak_allocated_gib_during_lidar_render": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
        "per_frame": per_frame,
    }


def actor_gate(pipeline: Any) -> dict[str, Any]:
    model = pipeline.model
    gaussian_ids = model.id.detach().flatten().cpu()
    actor_count = int(model.dynamic_actors.actor_positions.shape[1])
    counts = {
        str(actor_id): int((gaussian_ids == actor_id).sum())
        for actor_id in range(actor_count)
    }
    background_count = int((gaussian_ids == actor_count).sum())
    present = model.dynamic_actors.actor_present_at_time.detach().cpu()
    configured_minimum = int(model.config.min_points_per_actor)
    return {
        "total_gaussians": int(gaussian_ids.numel()),
        "actor_count": actor_count,
        "actor_gaussians": counts,
        "actor_gaussians_total": sum(counts.values()),
        "actor_gaussians_fraction": sum(counts.values()) / gaussian_ids.numel(),
        "background_id": actor_count,
        "background_gaussians": background_count,
        "configured_min_points_per_actor": configured_minimum,
        "actors_below_configured_minimum": [
            int(actor_id)
            for actor_id, count in counts.items()
            if count < configured_minimum
        ],
        "actor_present_timestamps": {
            str(actor_id): int(present[:, actor_id].sum())
            for actor_id in range(actor_count)
        },
        "actor_present_timestamps_scope": (
            "72 train-split timeline samples in the 0.9 temporal split, not "
            "all 80 raw PandaSet frames"
        ),
        "checkpoint_evidence_note": (
            "Low actor Gaussian counts diagnose the checkpoint representation; "
            "they do not by themselves identify why an actor collapsed."
        ),
    }


def contact_sheets(
    frames: tuple[str, ...],
    output_dir: Path,
) -> list[str]:
    paths = []
    cell_size = (320, 180)
    for frame in frames:
        sheet = Image.new(
            "RGB",
            (cell_size[0] * len(CAMERA_ORDER), cell_size[1] * len(POSES)),
            "black",
        )
        for row, (pose_name, _, _, _) in enumerate(POSES):
            for column, sensor_index in enumerate(CAMERA_ORDER):
                image_path = (
                    output_dir
                    / "nearby_rgb"
                    / frame
                    / pose_name
                    / f"{CAMERA_NAMES[sensor_index]}.jpg"
                )
                cell = image_tile(
                    image_path,
                    cell_size,
                    f"{pose_name}  {CAMERA_NAMES[sensor_index]}",
                )
                sheet.paste(
                    cell,
                    (column * cell_size[0], row * cell_size[1]),
                )
        path = output_dir / f"scene_040_frame_{frame}_nearby_pose_grid.jpg"
        sheet.save(path, quality=95)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=parse_frames, default=("19", "39", "59"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    _, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        test_mode="test",
        update_config_callback=streamline_ad_config,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    report = {
        "config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "load_seconds": load_seconds,
        "gpu_allocated_gib_after_load": torch.cuda.memory_allocated() / 2**30,
        "gpu_reserved_gib_after_load": torch.cuda.memory_reserved() / 2**30,
        "nearby_pose": nearby_pose_gate(pipeline, args.frames, args.output_dir),
        "heldout_lidar": lidar_gate(pipeline),
        "actors": actor_gate(pipeline),
    }
    report["nearby_pose"]["contact_sheets"] = contact_sheets(
        args.frames,
        args.output_dir,
    )
    lidar_expected = report["heldout_lidar"]["cuboid_excluded_static"][
        "expected_depth"
    ]
    nearby_latency = report["nearby_pose"]["single_camera_latency_ms"]
    rig_latency = report["nearby_pose"]["six_camera_rig_latency_ms"]
    report["gate_summary"] = {
        "nearby_126_views_finite": (
            report["nearby_pose"]["views"] == 126
            and report["nearby_pose"]["finite_failures"] == 0
        ),
        "single_camera_latency_p95_at_most_100ms": (
            nearby_latency["p95"] <= 100.0
        ),
        "sequential_six_camera_rig_p95_at_most_100ms": (
            rig_latency["p95"] <= 100.0
        ),
        "cuboid_excluded_static_lidar_provisional": {
            "thresholds": {
                "absolute_p50_metres_at_most": 0.5,
                "absolute_p90_metres_at_most": 2.0,
                "relative_p90_at_most": 0.15,
            },
            "passed": (
                lidar_expected["absolute_error_metres"]["p50"] <= 0.5
                and lidar_expected["absolute_error_metres"]["p90"] <= 2.0
                and lidar_expected["relative_absolute_error"]["p90"] <= 0.15
                and report["heldout_lidar"]["finite_failures"] == 0
            ),
            "scope_note": (
                "Moving-actor cuboids are excluded, but point semantics and "
                "per-ray actor-time filtering are still required for a final "
                "class-resolved geometry gate."
            ),
        },
        "overall": "not_yet_stable_drivable",
        "open_gates": [
            "manual nearby-pose image quality",
            "static-semantic LiDAR depth",
            "six-camera 10Hz latency",
            "dynamic-object representation",
            "cross-camera seams and exact-path calibration overlay",
        ],
    }
    report_path = args.output_dir / "scene_040_geometry_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE: {report_path}")


if __name__ == "__main__":
    main()
