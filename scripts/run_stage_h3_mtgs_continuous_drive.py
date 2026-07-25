#!/usr/bin/env python3
"""Render a fixed-time, high-speed MTGS lane-change smoke video."""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


SOURCE_COMMIT = "7ab67a3e386e5a4830017819324922a7bb9f26f7"
CAMERA_NAME = "CAM_F0"
TRAVEL_ID = 3
CORRIDOR_HALF_WIDTH_METERS = 5.0
DEFAULT_SPEED_MPS = 12.0
DEFAULT_DURATION_SECONDS = 6.0
DEFAULT_FPS = 20
DEFAULT_LATERAL_AMPLITUDE_METERS = 4.0
DEFAULT_START_PROGRESS_METERS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-block-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed-mps", type=float, default=DEFAULT_SPEED_MPS)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--lateral-amplitude-meters",
        type=float,
        default=DEFAULT_LATERAL_AMPLITUDE_METERS,
    )
    parser.add_argument(
        "--start-progress-meters",
        type=float,
        default=DEFAULT_START_PROGRESS_METERS,
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def lane_change_profile(
    route_fraction: float,
    *,
    amplitude_meters: float,
    distance_meters: float,
) -> tuple[float, float]:
    """Return a four-leg lane-change offset with zero-heading endpoints."""

    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError("route_fraction must be within [0, 1]")
    if not math.isfinite(amplitude_meters) or amplitude_meters <= 0.0:
        raise ValueError("amplitude_meters must be finite and positive")
    if not math.isfinite(distance_meters) or distance_meters <= 0.0:
        raise ValueError("distance_meters must be finite and positive")
    targets = (
        0.0,
        amplitude_meters,
        0.0,
        -amplitude_meters,
        0.0,
    )
    scaled = route_fraction * 4.0
    segment = min(int(scaled), 3)
    segment_fraction = scaled - segment
    start = targets[segment]
    end = targets[segment + 1]
    delta = end - start
    eased = 0.5 * (1.0 - math.cos(math.pi * segment_fraction))
    lateral = start + delta * eased
    segment_distance = distance_meters / 4.0
    lateral_slope = (
        delta
        * 0.5
        * math.pi
        * math.sin(math.pi * segment_fraction)
        / segment_distance
    )
    return lateral, lateral_slope


def cumulative_route_distances(route_poses: object, np: object) -> list[float]:
    positions = route_poses[:, :3, 3]
    segment_lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    distances = [0.0]
    for length in segment_lengths.tolist():
        distances.append(distances[-1] + float(length))
    return distances


def normalize(vector: object, np: object) -> object:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError("cannot normalize a degenerate route basis")
    return vector / norm


def sample_route_pose(
    route_poses: object,
    route_distances: list[float],
    progress_meters: float,
    lateral_meters: float,
    lateral_slope: float,
    np: object,
) -> tuple[object, float]:
    """Interpolate an observed front-camera pose and apply a lane-change pose."""

    if progress_meters < 0.0 or progress_meters > route_distances[-1]:
        raise ValueError("progress lies outside the observed route")
    upper = min(
        max(1, bisect.bisect_right(route_distances, progress_meters)),
        len(route_distances) - 1,
    )
    lower = upper - 1
    span = route_distances[upper] - route_distances[lower]
    ratio = 0.0 if span <= 1e-9 else (
        progress_meters - route_distances[lower]
    ) / span
    lower_pose = route_poses[lower]
    upper_pose = route_poses[upper]
    center = (
        lower_pose[:3, 3] * (1.0 - ratio)
        + upper_pose[:3, 3] * ratio
    )
    observed_right = normalize(
        lower_pose[:3, 0] * (1.0 - ratio)
        + upper_pose[:3, 0] * ratio,
        np,
    )
    observed_up = normalize(
        lower_pose[:3, 1] * (1.0 - ratio)
        + upper_pose[:3, 1] * ratio,
        np,
    )
    observed_forward = normalize(
        -(
            lower_pose[:3, 2] * (1.0 - ratio)
            + upper_pose[:3, 2] * ratio
        ),
        np,
    )

    # Re-orthogonalize the observed basis before adding the path tangent.
    observed_right = normalize(np.cross(observed_forward, observed_up), np)
    observed_up = normalize(np.cross(observed_right, observed_forward), np)
    drive_forward = normalize(
        observed_forward + lateral_slope * observed_right,
        np,
    )
    drive_right = normalize(
        observed_right - lateral_slope * observed_forward,
        np,
    )
    drive_up = normalize(np.cross(drive_right, drive_forward), np)

    pose = np.zeros((3, 4), dtype=np.float32)
    pose[:3, 0] = drive_right
    pose[:3, 1] = drive_up
    pose[:3, 2] = -drive_forward
    pose[:3, 3] = center + lateral_meters * observed_right
    return pose, math.degrees(math.atan(lateral_slope))


def configure_for_inference(config: object, args: argparse.Namespace) -> None:
    config.load_dir = args.checkpoint.parent.resolve()
    dataparser = config.pipeline.datamanager.dataparser
    dataparser.road_block_config = str(args.road_block_config.resolve())
    dataparser.eval_2hz = True
    datamanager = config.pipeline.datamanager
    datamanager.eval_cache_strategy = "on_demand"
    datamanager.load_mask = False
    datamanager.load_custom_masks = ()
    datamanager.load_instance_masks = False
    datamanager.load_semantic_masks_from = False
    datamanager.load_lidar_depth = False
    datamanager.load_pseudo_depth = False
    model = config.pipeline.model
    model.output_depth_during_training = False
    model.predict_normals = False
    model.color_corrected_metrics = False
    model.lpips_metric = False
    model.dinov2_metric = False


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.road_block_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    numeric = (
        args.speed_mps,
        args.duration_seconds,
        args.lateral_amplitude_meters,
        args.start_progress_meters,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("drive parameters must be finite")
    if args.speed_mps <= 0.0 or args.duration_seconds <= 0.0:
        raise ValueError("speed and duration must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 < args.lateral_amplitude_meters < CORRIDOR_HALF_WIDTH_METERS:
        raise ValueError("lateral amplitude must remain inside the probe corridor")
    if args.start_progress_meters < 0.0:
        raise ValueError("start progress must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "NERFSTUDIO_DATAPARSER_CONFIGS",
        "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser",
    )
    os.environ.setdefault(
        "NERFSTUDIO_METHOD_CONFIGS",
        "mtgs=mtgs.config.MTGS:method",
    )

    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import yaml
    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible in the MTGS environment")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    setup_started = time.perf_counter()

    config = yaml.load(args.config.read_text(), Loader=yaml.Loader)
    configure_for_inference(config, args)
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    pipeline.eval()
    checkpoint_path, checkpoint_step = eval_load_checkpoint(config, pipeline)
    setup_seconds = time.perf_counter() - setup_started

    outputs = pipeline.datamanager.eval_dataparser_outputs
    front_indices = [
        index
        for index, (travel_id, image_path) in enumerate(
            zip(outputs.travel_ids, outputs.image_filenames)
        )
        if travel_id == TRAVEL_ID and image_path.parent.name == CAMERA_NAME
    ]
    if len(front_indices) < 2:
        raise RuntimeError(
            f"expected at least two {CAMERA_NAME} poses for travel {TRAVEL_ID}"
        )
    route_poses = (
        outputs.cameras.camera_to_worlds[front_indices]
        .detach()
        .cpu()
        .numpy()
    )
    route_distances = cumulative_route_distances(route_poses, np)
    drive_distance = args.speed_mps * args.duration_seconds
    end_progress = args.start_progress_meters + drive_distance
    if end_progress > route_distances[-1]:
        raise RuntimeError(
            f"requested end {end_progress:.2f} m exceeds observed "
            f"{route_distances[-1]:.2f} m route"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from stage_h3_mtgs_driving_adapter import make_mtgs_driving_adapter

    driving_adapter = make_mtgs_driving_adapter(
        route_poses,
        route_distances,
        spawn_progress_meters=args.start_progress_meters,
    )
    adapter_spawn = driving_adapter.reset()
    adapter_spawn_query = driving_adapter.render_query(adapter_spawn)

    _, base_camera = pipeline.datamanager.cached_eval(front_indices[0])
    base_camera = base_camera.to(device)
    fixed_time = (
        float(base_camera.times.item())
        if base_camera.times is not None
        else None
    )
    fixed_metadata = {
        key: (
            value.tolist()
            if hasattr(value, "tolist")
            else value
        )
        for key, value in (base_camera.metadata or {}).items()
        if key in {"cam_token", "frame_token", "travel_id"}
    }

    frame_count = round(args.duration_seconds * args.fps) + 1
    if frame_count < 2:
        raise ValueError("drive must contain at least two frames")
    frame_records = []
    render_seconds = []
    saved_frames = []
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for frame_index in range(frame_count):
            elapsed = frame_index / args.fps
            progress = args.start_progress_meters + args.speed_mps * elapsed
            fraction = (progress - args.start_progress_meters) / drive_distance
            lateral, lateral_slope = lane_change_profile(
                fraction,
                amplitude_meters=args.lateral_amplitude_meters,
                distance_meters=drive_distance,
            )
            pose, lane_change_heading_degrees = sample_route_pose(
                route_poses,
                route_distances,
                progress,
                lateral,
                lateral_slope,
                np,
            )
            base_camera.camera_to_worlds = torch.from_numpy(pose)[None].to(
                device
            )
            torch.cuda.synchronize()
            render_started = time.perf_counter()
            rendered = pipeline.model.get_outputs_for_camera(
                camera=base_camera
            )
            torch.cuda.synchronize()
            render_duration = time.perf_counter() - render_started
            rgb = rendered["rgb"].detach().float().cpu().numpy()
            finite = bool(np.isfinite(rgb).all())
            image = Image.fromarray(
                np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            )
            draw = ImageDraw.Draw(image)
            probe_margin = (
                CORRIDOR_HALF_WIDTH_METERS - abs(lateral)
            )
            draw.rectangle((8, 8, 590, 72), fill=(4, 8, 12))
            draw.text(
                (18, 16),
                (
                    f"MTGS FIXED-TIME DRIVE | {args.speed_mps:.1f} m/s | "
                    f"progress {progress:.1f} m"
                ),
                fill=(235, 245, 250),
            )
            draw.text(
                (18, 42),
                (
                    f"lateral {lateral:+.2f} m | heading "
                    f"{lane_change_heading_degrees:+.1f} deg | "
                    f"probe margin {probe_margin:.2f} m"
                ),
                fill=(72, 224, 142) if probe_margin >= 1.0 else (255, 92, 72),
            )
            frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
            image.save(frame_path, format="JPEG", quality=94)
            frame_hash = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            render_seconds.append(render_duration)
            saved_frames.append(frame_path)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "elapsed_seconds": elapsed,
                    "progress_meters": progress,
                    "speed_mps": args.speed_mps,
                    "lateral_meters": lateral,
                    "lane_change_heading_degrees": (
                        lane_change_heading_degrees
                    ),
                    "probe_margin_meters": probe_margin,
                    "finite": finite,
                    "render_seconds": render_duration,
                    "frame_sha256": frame_hash,
                }
            )
    render_wall_seconds = time.perf_counter() - wall_started

    video_path = output_dir / "mtgs_fixed_time_12mps_lane_change.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )

    contact_indices = sorted(
        {
            round(index * (frame_count - 1) / 11)
            for index in range(12)
        }
    )
    contact_images = [Image.open(saved_frames[index]) for index in contact_indices]
    tile_width, tile_height = contact_images[0].size
    contact_sheet = Image.new(
        "RGB",
        (tile_width * 4, tile_height * 3),
        color=(0, 0, 0),
    )
    for contact_index, image in enumerate(contact_images):
        x = (contact_index % 4) * tile_width
        y = (contact_index // 4) * tile_height
        contact_sheet.paste(image, (x, y))
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    contact_sheet.save(contact_sheet_path, format="JPEG", quality=90)

    p50 = statistics.median(render_seconds)
    p95 = percentile(render_seconds, 0.95)
    max_abs_lateral = max(
        abs(record["lateral_meters"]) for record in frame_records
    )
    min_probe_margin = min(
        record["probe_margin_meters"] for record in frame_records
    )
    all_finite = all(record["finite"] for record in frame_records)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    numerical_pass = (
        all_finite
        and min_probe_margin >= 1.0 - 1e-6
        and p95 <= 0.05
        and end_progress <= route_distances[-1]
    )
    report = {
        "format": "driving_scene_reconstruction.mtgs_continuous_drive.v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source_commit": SOURCE_COMMIT,
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "road_block_config": str(args.road_block_config.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "setup_seconds": setup_seconds,
        "fixed_scene": {
            "camera_name": CAMERA_NAME,
            "travel_id": TRAVEL_ID,
            "normalized_time": fixed_time,
            "camera_metadata": fixed_metadata,
            "time_advanced": False,
        },
        "observed_route": {
            "front_pose_count": len(front_indices),
            "route_length_meters": route_distances[-1],
            "start_progress_meters": args.start_progress_meters,
            "end_progress_meters": end_progress,
        },
        "control_adapter": {
            "constructed_from_actual_route": True,
            "spawn_progress_meters": adapter_spawn_query.progress_meters,
            "spawn_support_margin_meters": (
                adapter_spawn_query.support_margin_meters
            ),
            "corridor_half_width_meters": (
                driving_adapter.controller.corridor.half_width
            ),
            "max_speed_mps": (
                driving_adapter.controller.vehicle_model.max_speed
            ),
            "fail_closed_boundary": True,
            "endpoint_stop": True,
        },
        "drive": {
            "speed_mps": args.speed_mps,
            "simulation_duration_seconds": args.duration_seconds,
            "encoded_duration_seconds": frame_count / args.fps,
            "fps": args.fps,
            "frame_count": frame_count,
            "distance_meters": drive_distance,
            "requested_lateral_amplitude_meters": (
                args.lateral_amplitude_meters
            ),
            "max_abs_lateral_meters": max_abs_lateral,
            "minimum_probe_margin_meters": min_probe_margin,
            "all_finite": all_finite,
            "render_wall_seconds": render_wall_seconds,
            "render_seconds_p50": p50,
            "render_seconds_p95": p95,
            "render_fps_p50": 1.0 / p50,
            "render_fps_p95": 1.0 / p95,
        },
        "peak_memory": {
            "reserved_bytes": peak_reserved,
            "reserved_gib": peak_reserved / 1024**3,
        },
        "artifacts": {
            "video_path": str(video_path),
            "contact_sheet_path": str(contact_sheet_path),
            "frames_dir": str(frames_dir),
        },
        "frames": frame_records,
        "summary": {
            "numerical_verdict": "pass" if numerical_pass else "fail",
            "visual_verdict": "pending_manual_review",
            "training_performed": False,
            "keyboard_control_performed": False,
            "dynamic_time_advanced": False,
        },
    }
    report_path = output_dir / "mtgs_continuous_drive.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"video: {video_path}")
    print(f"contact sheet: {contact_sheet_path}")
    print(f"report: {report_path}")
    print(f"numerical verdict: {report['summary']['numerical_verdict']}")


if __name__ == "__main__":
    main()
