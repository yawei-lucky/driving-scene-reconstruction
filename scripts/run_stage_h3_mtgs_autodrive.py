#!/usr/bin/env python3
"""Run a simulated-human MTGS vehicle/render closed loop without a browser."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from run_stage_h3_mtgs_continuous_drive import (  # noqa: E402
    CAMERA_NAME,
    SOURCE_COMMIT,
    TRAVEL_ID,
    configure_for_inference,
    cumulative_route_distances,
    normalize,
    percentile,
    sample_route_pose,
)
from stage_h3_mtgs_autodriver import (  # noqa: E402
    MtgsAutodriver,
    MtgsAutodriverConfig,
)
from stage_h3_mtgs_driving_adapter import (  # noqa: E402
    make_mtgs_driving_adapter,
)


DEFAULT_FPS = 20
DEFAULT_MAX_DURATION_SECONDS = 15.0
DEFAULT_CRUISE_SPEED_MPS = 12.0
DEFAULT_LATERAL_AMPLITUDE_METERS = 3.0
DEFAULT_START_PROGRESS_METERS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-block-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    parser.add_argument(
        "--cruise-speed-mps",
        type=float,
        default=DEFAULT_CRUISE_SPEED_MPS,
    )
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


def camera_right_to_route_left_alignment(
    route_poses: object,
    route_distances: list[float],
    progress_meters: float,
    np: object,
) -> float:
    """Measure the sign/scale from camera-right to route-left."""

    upper = min(
        max(1, bisect.bisect_right(route_distances, progress_meters)),
        len(route_distances) - 1,
    )
    lower = upper - 1
    span = route_distances[upper] - route_distances[lower]
    ratio = 0.0 if span <= 1e-9 else (
        progress_meters - route_distances[lower]
    ) / span
    pose = route_poses[lower]
    next_pose = route_poses[upper]
    camera_right = normalize(
        pose[:3, 0] * (1.0 - ratio) + next_pose[:3, 0] * ratio,
        np,
    )
    tangent = next_pose[:2, 3] - pose[:2, 3]
    tangent = tangent / np.linalg.norm(tangent)
    route_left = np.asarray([-tangent[1], tangent[0]])
    alignment = float(np.dot(camera_right[:2], route_left))
    if not math.isfinite(alignment) or abs(alignment) < 0.5:
        raise RuntimeError(
            "camera-right axis is not reliably aligned with route lateral"
        )
    return alignment


def simulate_closed_loop(
    driving_adapter: object,
    *,
    fps: int,
    max_duration_seconds: float,
    cruise_speed_mps: float,
    lateral_amplitude_meters: float,
    start_progress_meters: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the vehicle/controller loop before spending time on rendering."""

    dt = 1.0 / fps
    driver = MtgsAutodriver(
        driving_adapter.controller.vehicle_model,
        MtgsAutodriverConfig(
            cruise_speed_mps=cruise_speed_mps,
            lateral_amplitude_meters=lateral_amplitude_meters,
            spawn_progress_meters=start_progress_meters,
            endpoint_margin_meters=driving_adapter.endpoint_margin_meters,
        ),
    )
    state = driving_adapter.reset()
    samples: list[dict[str, object]] = []
    boundary_hits = 0
    endpoint_reached = False
    maximum_steps = round(max_duration_seconds * fps)
    for step_index in range(maximum_steps):
        query = driving_adapter.render_query(state)
        decision = driver.decide(
            state,
            driving_adapter.controller.corridor,
            dt,
        )
        update = driving_adapter.step(state, decision.control, dt)
        samples.append(
            {
                "frame_index": step_index,
                "elapsed_seconds": state.time,
                "state": state,
                "query": query,
                "decision": decision,
                "boundary_hit_after_control": update.boundary_hit,
                "boundary_reason_after_control": update.boundary_reason,
                "endpoint_reached_after_control": update.endpoint_reached,
            }
        )
        boundary_hits += int(update.boundary_hit)
        state = update.state
        if update.endpoint_reached:
            endpoint_reached = True
            final_query = driving_adapter.render_query(state)
            final_decision = driver.decide(
                state,
                driving_adapter.controller.corridor,
                dt,
            )
            samples.append(
                {
                    "frame_index": step_index + 1,
                    "elapsed_seconds": state.time,
                    "state": state,
                    "query": final_query,
                    "decision": final_decision,
                    "boundary_hit_after_control": False,
                    "boundary_reason_after_control": None,
                    "endpoint_reached_after_control": True,
                }
            )
            break
    summary = {
        "endpoint_reached": endpoint_reached,
        "boundary_hits": boundary_hits,
        "maximum_speed_mps": max(
            float(item["state"].speed) for item in samples
        ),
        "maximum_left_meters": max(
            float(item["query"].lateral_meters) for item in samples
        ),
        "maximum_right_meters": min(
            float(item["query"].lateral_meters) for item in samples
        ),
        "minimum_support_margin_meters": min(
            float(item["query"].support_margin_meters) for item in samples
        ),
        "maximum_abs_heading_error_degrees": max(
            abs(math.degrees(float(item["query"].heading_error_radians)))
            for item in samples
        ),
        "maximum_abs_steer": max(
            abs(float(item["decision"].control.steer)) for item in samples
        ),
        "maximum_brake": max(
            float(item["decision"].control.brake) for item in samples
        ),
    }
    steers = [
        float(item["decision"].control.steer) for item in samples
    ]
    summary["maximum_abs_steer_step"] = max(
        abs(right - left) for left, right in zip(steers, steers[1:])
    )
    summary["numerical_control_pass"] = (
        endpoint_reached
        and boundary_hits == 0
        and summary["maximum_speed_mps"] >= 10.0
        and summary["maximum_left_meters"] >= 2.0
        and summary["maximum_right_meters"] <= -2.0
        and summary["minimum_support_margin_meters"] >= 1.0
        and summary["maximum_brake"] > 0.0
    )
    return samples, summary


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.road_block_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    numeric = (
        args.max_duration_seconds,
        args.cruise_speed_mps,
        args.lateral_amplitude_meters,
        args.start_progress_meters,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in numeric):
        raise ValueError("drive parameters must be finite and positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.lateral_amplitude_meters >= 5.0:
        raise ValueError("lateral target must remain inside the probe corridor")

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
    driving_adapter = make_mtgs_driving_adapter(
        route_poses,
        route_distances,
        spawn_progress_meters=args.start_progress_meters,
    )
    samples, control_summary = simulate_closed_loop(
        driving_adapter,
        fps=args.fps,
        max_duration_seconds=args.max_duration_seconds,
        cruise_speed_mps=args.cruise_speed_mps,
        lateral_amplitude_meters=args.lateral_amplitude_meters,
        start_progress_meters=args.start_progress_meters,
    )
    if not control_summary["numerical_control_pass"]:
        raise RuntimeError(f"closed-loop control gate failed: {control_summary}")

    _, base_camera = pipeline.datamanager.cached_eval(front_indices[0])
    base_camera = base_camera.to(device)
    fixed_time = (
        float(base_camera.times.item())
        if base_camera.times is not None
        else None
    )
    corridor_length = driving_adapter.controller.corridor.length
    route_scale = route_distances[-1] / corridor_length
    render_seconds: list[float] = []
    frame_records = []
    saved_frames = []
    alignments = []
    image_body_mads = []
    previous_image_body = None
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for item in samples:
            state = item["state"]
            query = item["query"]
            decision = item["decision"]
            render_progress = min(
                route_distances[-1],
                float(query.progress_meters) * route_scale,
            )
            alignment = camera_right_to_route_left_alignment(
                route_poses,
                route_distances,
                render_progress,
                np,
            )
            camera_lateral = float(query.lateral_meters) / alignment
            camera_slope = math.tan(
                float(query.heading_error_radians)
            ) / alignment
            pose, _ = sample_route_pose(
                route_poses,
                route_distances,
                render_progress,
                camera_lateral,
                camera_slope,
                np,
            )
            base_camera.camera_to_worlds = torch.from_numpy(pose)[None].to(
                device
            )
            torch.cuda.synchronize()
            render_started = time.perf_counter()
            rendered = pipeline.model.get_outputs_for_camera(camera=base_camera)
            torch.cuda.synchronize()
            render_duration = time.perf_counter() - render_started
            rgb = rendered["rgb"].detach().float().cpu().numpy()
            finite = bool(np.isfinite(rgb).all())
            image_array = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            image_body = (
                0.299 * image_array[105:, :, 0]
                + 0.587 * image_array[105:, :, 1]
                + 0.114 * image_array[105:, :, 2]
            ).astype(np.float32)
            if previous_image_body is not None:
                image_body_mads.append(
                    float(np.mean(np.abs(image_body - previous_image_body)))
                )
            previous_image_body = image_body
            image = Image.fromarray(image_array)
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, 670, 98), fill=(4, 8, 12))
            draw.text(
                (18, 16),
                (
                    f"MTGS VEHICLE CLOSED LOOP | speed {state.speed:.1f} / "
                    f"{decision.target_speed_mps:.1f} m/s | "
                    f"progress {query.progress_meters:.1f} m"
                ),
                fill=(235, 245, 250),
            )
            draw.text(
                (18, 42),
                (
                    f"actual lateral {query.lateral_meters:+.2f} m | "
                    f"driver target {decision.target_lateral_meters:+.2f} m | "
                    f"heading {math.degrees(query.heading_error_radians):+.1f} deg"
                ),
                fill=(72, 224, 142),
            )
            draw.text(
                (18, 68),
                (
                    f"steer {decision.control.steer:+.2f} | "
                    f"throttle {decision.control.throttle:.2f} | "
                    f"brake {decision.control.brake:.2f} | "
                    f"support margin {query.support_margin_meters:.2f} m"
                ),
                fill=(235, 210, 90),
            )
            frame_index = int(item["frame_index"])
            frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
            image.save(frame_path, format="JPEG", quality=94)
            frame_hash = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            render_seconds.append(render_duration)
            saved_frames.append(frame_path)
            alignments.append(alignment)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "elapsed_seconds": state.time,
                    "world_state": {
                        "x": state.x,
                        "y": state.y,
                        "yaw_radians": state.yaw,
                        "speed_mps": state.speed,
                    },
                    "route": {
                        "progress_meters": query.progress_meters,
                        "lateral_meters": query.lateral_meters,
                        "heading_error_radians": query.heading_error_radians,
                        "support_margin_meters": query.support_margin_meters,
                    },
                    "driver_target": {
                        "speed_mps": decision.target_speed_mps,
                        "lateral_meters": decision.target_lateral_meters,
                        "lookahead_progress_meters": (
                            decision.lookahead_progress_meters
                        ),
                    },
                    "control": {
                        "steer": decision.control.steer,
                        "throttle": decision.control.throttle,
                        "brake": decision.control.brake,
                    },
                    "camera_right_to_route_left_alignment": alignment,
                    "finite": finite,
                    "render_seconds": render_duration,
                    "frame_sha256": frame_hash,
                    "endpoint_reached_after_control": (
                        item["endpoint_reached_after_control"]
                    ),
                }
            )
    render_wall_seconds = time.perf_counter() - wall_started

    video_path = output_dir / "mtgs_autodrive_12mps_closed_loop.mp4"
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
            round(index * (len(saved_frames) - 1) / 11)
            for index in range(12)
        }
    )
    contact_images = [
        Image.open(saved_frames[index]) for index in contact_indices
    ]
    tile_width, tile_height = contact_images[0].size
    contact_sheet = Image.new(
        "RGB",
        (tile_width * 4, tile_height * 3),
        color=(0, 0, 0),
    )
    for contact_index, image in enumerate(contact_images):
        contact_sheet.paste(
            image,
            (
                (contact_index % 4) * tile_width,
                (contact_index // 4) * tile_height,
            ),
        )
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    contact_sheet.save(contact_sheet_path, format="JPEG", quality=90)

    p50 = statistics.median(render_seconds)
    p95 = percentile(render_seconds, 0.95)
    all_finite = all(record["finite"] for record in frame_records)
    numerical_pass = (
        bool(control_summary["numerical_control_pass"])
        and all_finite
        and p95 <= 0.05
    )
    report = {
        "format": "driving_scene_reconstruction.mtgs_autodrive.v0",
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
            "time_advanced": False,
        },
        "observed_route": {
            "front_pose_count": len(front_indices),
            "camera_route_length_meters": route_distances[-1],
            "control_corridor_length_meters": corridor_length,
        },
        "simulation": {
            "fps": args.fps,
            "dt_seconds": 1.0 / args.fps,
            "frame_count": len(samples),
            "encoded_duration_seconds": len(samples) / args.fps,
            "cruise_speed_mps": args.cruise_speed_mps,
            "requested_lateral_amplitude_meters": (
                args.lateral_amplitude_meters
            ),
            **control_summary,
        },
        "render": {
            "all_finite": all_finite,
            "render_wall_seconds": render_wall_seconds,
            "render_seconds_p50": p50,
            "render_seconds_p95": p95,
            "render_fps_p50": 1.0 / p50,
            "render_fps_p95": 1.0 / p95,
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 1024**3
            ),
            "camera_right_alignment_min": min(alignments),
            "camera_right_alignment_max": max(alignments),
            "consecutive_image_body_mad_p50": statistics.median(
                image_body_mads
            ),
            "consecutive_image_body_mad_p95": percentile(
                image_body_mads,
                0.95,
            ),
            "consecutive_image_body_mad_max": max(image_body_mads),
            "consecutive_image_body_mad_max_transition": (
                image_body_mads.index(max(image_body_mads))
            ),
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
            "vehicle_dynamics_used": True,
            "simulated_driver_used": True,
            "direct_camera_path_used": False,
            "training_performed": False,
            "dynamic_time_advanced": False,
        },
    }
    report_path = output_dir / "mtgs_autodrive.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"video: {video_path}")
    print(f"contact sheet: {contact_sheet_path}")
    print(f"report: {report_path}")
    print(f"control summary: {control_summary}")
    print(f"numerical verdict: {report['summary']['numerical_verdict']}")


if __name__ == "__main__":
    main()
