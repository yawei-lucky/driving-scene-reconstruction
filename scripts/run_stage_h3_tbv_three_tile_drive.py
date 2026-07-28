#!/usr/bin/env python3
"""Render a simulated-human 260 m drive through three adjacent TbV tiles."""

from __future__ import annotations

import argparse
import bisect
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
EXAMPLES_DIR = REPO_ROOT / "examples"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze_stage_h3_tbv_trajectories import (  # noqa: E402
    DEFAULT_CACHE,
    load_manifest,
    load_track,
)
from audit_stage_h3_tbv_long_route_tiles import (  # noqa: E402
    cumulative_distances,
)
from driving_scene_reconstruction.sim import (  # noqa: E402
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    SceneTile,
    SceneTileRoute,
    SimpleVehicleModel,
)
from probe_stage_h3_tbv_tile_continuous import (  # noqa: E402
    _encode_video,
    _interpolated_camera,
    _render_rgb,
    distribution,
    streamline_tbv_probe_config,
)
from stage_h3_mtgs_autodriver import (  # noqa: E402
    MtgsAutodriver,
    MtgsAutodriverConfig,
)


REFERENCE_LOG = "V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020"
CAMERA_NAME = "ring_front_center"
ROUTE_START_METERS = 160.0
ROUTE_END_METERS = 420.0
TILE_RANGES = {
    "tile_2": (160.0, 260.0),
    "tile_3": (240.0, 340.0),
    "tile_4": (320.0, 420.0),
}


def _interpolate(values: Sequence[float], progress: Sequence[float], target: float) -> float:
    if not progress[0] <= target <= progress[-1]:
        raise ValueError("target progress lies outside the source route")
    upper = min(max(1, bisect.bisect_right(progress, target)), len(progress) - 1)
    lower = upper - 1
    span = progress[upper] - progress[lower]
    fraction = 0.0 if span <= 1e-9 else (target - progress[lower]) / span
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def build_corridor(track: Any) -> tuple[LoggedCenterlineCorridor, list[float]]:
    progress = cumulative_distances(track.xy_metres)
    route_progress = [ROUTE_START_METERS]
    route_progress.extend(
        value for value in progress if ROUTE_START_METERS < value < ROUTE_END_METERS
    )
    route_progress.append(ROUTE_END_METERS)
    xs = [point[0] for point in track.xy_metres]
    ys = [point[1] for point in track.xy_metres]
    origin_x = _interpolate(xs, progress, ROUTE_START_METERS)
    origin_y = _interpolate(ys, progress, ROUTE_START_METERS)
    samples = []
    for index, value in enumerate(route_progress):
        x = _interpolate(xs, progress, value) - origin_x
        y = _interpolate(ys, progress, value) - origin_y
        if index + 1 < len(route_progress):
            next_value = route_progress[index + 1]
            next_x = _interpolate(xs, progress, next_value) - origin_x
            next_y = _interpolate(ys, progress, next_value) - origin_y
            yaw = math.atan2(next_y - y, next_x - x)
        else:
            prior = samples[-1]
            yaw = math.atan2(y - prior.y, x - prior.x)
        samples.append(
            LoggedCenterlineSample(
                logical_frame=index,
                log_time=value - ROUTE_START_METERS,
                x=x,
                y=y,
                yaw=yaw,
            )
        )
    return (
        LoggedCenterlineCorridor(
            samples=tuple(samples),
            half_width=1.0,
            max_heading_error=math.radians(20.0),
        ),
        progress,
    )


def simulate_drive(
    corridor: LoggedCenterlineCorridor,
    *,
    fps: int,
    speed_mps: float,
    lateral_amplitude_meters: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dt = 1.0 / fps
    vehicle = SimpleVehicleModel(
        wheelbase=2.8,
        max_steer_angle=math.radians(25.0),
        max_acceleration=3.5,
        max_braking=7.0,
        linear_drag=0.08,
        max_speed=speed_mps,
    )
    spawn_progress = 1.0
    driver = MtgsAutodriver(
        vehicle,
        MtgsAutodriverConfig(
            cruise_speed_mps=speed_mps,
            lateral_amplitude_meters=lateral_amplitude_meters,
            lookahead_meters=8.0,
            speed_gain=1.2,
            comfortable_braking_mps2=5.0,
            endpoint_margin_meters=1.0,
            spawn_progress_meters=spawn_progress,
            max_steer_rate_per_second=1.6,
        ),
    )
    state = corridor.pose_at_progress(spawn_progress)
    samples: list[dict[str, Any]] = []
    boundary_hits = 0
    endpoint_reached = False
    for frame_index in range(round(40.0 * fps)):
        measurement = corridor.measure(state)
        decision = driver.decide(state, corridor, dt)
        samples.append(
            {
                "frame_index": frame_index,
                "state": state,
                "measurement": measurement,
                "decision": decision,
                "global_progress_meters": (
                    ROUTE_START_METERS + measurement.progress
                ),
            }
        )
        if (
            decision.remaining_support_meters <= 0.15
            and state.speed <= 0.35
        ):
            endpoint_reached = True
            break
        candidate = vehicle.step(state, decision.control, dt)
        try:
            corridor.validate(candidate)
        except ValueError:
            boundary_hits += 1
            break
        state = candidate

    offsets = [item["measurement"].lateral_offset for item in samples]
    heading_errors = [item["measurement"].heading_error for item in samples]
    speeds = [item["state"].speed for item in samples]
    steers = [item["decision"].control.steer for item in samples]
    summary = {
        "endpoint_reached": endpoint_reached,
        "boundary_hits": boundary_hits,
        "frame_count": len(samples),
        "duration_seconds": samples[-1]["state"].time,
        "maximum_speed_mps": max(speeds),
        "maximum_left_meters": max(offsets),
        "maximum_right_meters": min(offsets),
        "minimum_support_margin_meters": min(
            corridor.half_width - abs(value) for value in offsets
        ),
        "maximum_abs_heading_error_degrees": max(
            abs(math.degrees(value)) for value in heading_errors
        ),
        "maximum_abs_steer": max(abs(value) for value in steers),
        "maximum_brake": max(
            item["decision"].control.brake for item in samples
        ),
    }
    summary["control_gate"] = (
        endpoint_reached
        and boundary_hits == 0
        and summary["maximum_speed_mps"] >= min(10.0, speed_mps)
        and summary["maximum_left_meters"] >= 0.25
        and summary["maximum_right_meters"] <= -0.25
        and summary["minimum_support_margin_meters"] >= 0.25
        and summary["maximum_brake"] > 0.0
    )
    if not summary["control_gate"]:
        raise RuntimeError(f"simulated drive failed before rendering: {summary}")
    return samples, summary


def _front_records(datamanager: Any) -> dict[int, Any]:
    records: dict[int, Any] = {}
    for dataset in (datamanager.train_dataset, datamanager.eval_dataset):
        for index, raw_filename in enumerate(dataset.image_filenames):
            path = Path(raw_filename)
            if REFERENCE_LOG in path.parts and path.parent.name == CAMERA_NAME:
                records[int(path.stem)] = deepcopy(dataset.cameras[index : index + 1])
    return records


def _apply_heading(camera: Any, heading_error: float, np: Any, torch: Any) -> Any:
    pose = camera.camera_to_worlds[0].detach().cpu().numpy()
    cosine = math.cos(heading_error)
    sine = math.sin(heading_error)
    local_yaw = np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    )
    pose[:, :3] = pose[:, :3] @ local_yaw
    camera.camera_to_worlds = torch.from_numpy(pose)[None].to(
        dtype=camera.camera_to_worlds.dtype
    )
    return camera


def render_tile(
    *,
    tile_id: str,
    config_path: Path,
    data_root: Path,
    samples: list[dict[str, Any]],
    active_indices: list[int],
    target_timestamps_ns: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image

    def update_config(config: Any) -> Any:
        return streamline_tbv_probe_config(config, data_root=data_root)

    _, pipeline, checkpoint_path, checkpoint_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    if checkpoint_step != 7999:
        raise RuntimeError(f"{tile_id} expected step 7999, got {checkpoint_step}")
    pipeline.model.eval()
    records = _front_records(pipeline.datamanager)
    timestamps = tuple(sorted(records))
    if len(timestamps) < 2:
        raise RuntimeError(f"{tile_id} has insufficient front-camera records")
    frame_dir = output_dir / "tile_frames" / tile_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    render_seconds: list[float] = []
    clamp_seconds: list[float] = []
    for frame_index in active_indices:
        target_ns = target_timestamps_ns[frame_index]
        clamped_ns = min(max(target_ns, timestamps[0]), timestamps[-1])
        clamp_seconds.append(abs(clamped_ns - target_ns) / 1e9)
        sample = samples[frame_index]
        camera = _interpolated_camera(
            records,
            timestamps,
            clamped_ns,
            lateral_left_meters=sample["measurement"].lateral_offset,
            np=np,
            torch=torch,
        )
        camera = _apply_heading(
            camera,
            sample["measurement"].heading_error,
            np,
            torch,
        )
        image, elapsed = _render_rgb(pipeline, camera, torch)
        Image.fromarray(image).save(
            frame_dir / f"frame_{frame_index:06d}.jpg",
            quality=95,
        )
        render_seconds.append(elapsed)
    result = {
        "tile_id": tile_id,
        "config": str(config_path.resolve()),
        "data_root": str(data_root.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(checkpoint_step),
        "rendered_frame_count": len(active_indices),
        "available_timestamp_range_ns": [timestamps[0], timestamps[-1]],
        "maximum_source_time_clamp_seconds": max(clamp_seconds, default=0.0),
        "render_seconds": distribution(render_seconds),
        "dataparser_transform": (
            pipeline.datamanager.train_dataparser_outputs
            .dataparser_transform.tolist()
        ),
        "dataparser_scale": float(
            pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        ),
    }
    del pipeline, records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _annotate_frame(
    image: Any,
    sample: dict[str, Any],
    weights: Sequence[Any],
    *,
    image_module: Any,
    draw_module: Any,
) -> Any:
    canvas = image.copy().convert("RGB")
    draw = draw_module.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 58), fill=(0, 0, 0))
    state = sample["state"]
    measurement = sample["measurement"]
    decision = sample["decision"]
    weight_text = " + ".join(
        f"{item.tile_id}:{item.weight:.2f}" for item in weights
    )
    draw.text(
        (10, 7),
        (
            f"TbV 260 m simulated drive | {sample['global_progress_meters'] - ROUTE_START_METERS:6.1f}"
            f"/260.0 m | {state.speed:4.1f} m/s"
        ),
        fill=(255, 235, 90),
    )
    draw.text(
        (10, 31),
        (
            f"left {measurement.lateral_offset:+.2f} m | steer "
            f"{decision.control.steer:+.2f} | {weight_text}"
        ),
        fill=(235, 240, 245),
    )
    return canvas


def combine_frames(
    route: SceneTileRoute,
    samples: list[dict[str, Any]],
    output_dir: Path,
    *,
    fps: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    frames_dir = output_dir / "drive_frames"
    frames_dir.mkdir()
    near_black: list[float] = []
    temporal_rgb_mae: list[float] = []
    transition_deltas: list[dict[str, object]] = []
    previous_output: Any | None = None
    transitions = 0
    prior_ids: tuple[str, ...] | None = None
    for sample in samples:
        frame_index = sample["frame_index"]
        weights = route.weights_at(sample["global_progress_meters"])
        ids = tuple(item.tile_id for item in weights)
        if prior_ids is not None and ids != prior_ids:
            transitions += 1
            transition_changed = True
        else:
            transition_changed = False
        prior_ids = ids
        arrays = [
            np.asarray(
                Image.open(
                    output_dir
                    / "tile_frames"
                    / item.tile_id
                    / f"frame_{frame_index:06d}.jpg"
                ).convert("RGB")
            ).astype(np.float32)
            for item in weights
        ]
        blended = sum(
            array * item.weight for array, item in zip(arrays, weights)
        )
        output = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        if previous_output is not None:
            delta = float(
                np.mean(
                    np.abs(
                        output.astype(np.float32)
                        - previous_output.astype(np.float32)
                    )
                )
            )
            temporal_rgb_mae.append(delta)
            if transition_changed:
                transition_deltas.append(
                    {
                        "frame_index": frame_index,
                        "active_tiles": list(ids),
                        "rgb_mae_from_previous": delta,
                    }
                )
        previous_output = output
        near_black.append(float(np.mean(np.max(output, axis=-1) < 8)))
        annotated = _annotate_frame(
            Image.fromarray(output),
            sample,
            weights,
            image_module=Image,
            draw_module=ImageDraw,
        )
        annotated.save(
            frames_dir / f"frame_{frame_index:06d}.jpg",
            quality=94,
        )
    video = _encode_video(frames_dir, output_dir / "tbv_260m_drive.mp4", fps)
    return {
        "video": video,
        "near_black_fraction": distribution(near_black),
        "temporal_rgb_mae": distribution(temporal_rgb_mae),
        "tile_weight_transition_deltas": transition_deltas,
        "tile_weight_state_changes": transitions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for tile_id in ("tile-2", "tile-3", "tile-4"):
        parser.add_argument(f"--{tile_id}-config", type=Path, required=True)
        parser.add_argument(f"--{tile_id}-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed-mps", type=float, default=12.0)
    parser.add_argument("--lateral-amplitude-meters", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(args.speed_mps) or args.speed_mps <= 0.0:
        raise ValueError("speed must be finite and positive")
    if not 0.0 < args.lateral_amplitude_meters < 1.0:
        raise ValueError("lateral amplitude must lie inside (0, 1) m")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {item.log_id: item for item in load_manifest(args.cache_dir)}
    track = load_track(metadata[REFERENCE_LOG])
    if track is None:
        raise RuntimeError("reference route track is missing")
    corridor, full_progress = build_corridor(track)
    samples, control_summary = simulate_drive(
        corridor,
        fps=args.fps,
        speed_mps=args.speed_mps,
        lateral_amplitude_meters=args.lateral_amplitude_meters,
    )
    target_timestamps_ns = [
        _interpolate(
            track.timestamps,
            full_progress,
            sample["global_progress_meters"],
        )
        * 1e9
        for sample in samples
    ]

    inputs = {
        "tile_2": (args.tile_2_config, args.tile_2_data_root),
        "tile_3": (args.tile_3_config, args.tile_3_data_root),
        "tile_4": (args.tile_4_config, args.tile_4_data_root),
    }
    active = {
        tile_id: [
            index
            for index, sample in enumerate(samples)
            if start <= sample["global_progress_meters"] <= end
        ]
        for tile_id, (start, end) in TILE_RANGES.items()
    }
    tile_results = {}
    for tile_id, (config_path, data_root) in inputs.items():
        tile_results[tile_id] = render_tile(
            tile_id=tile_id,
            config_path=config_path,
            data_root=data_root,
            samples=samples,
            active_indices=active[tile_id],
            target_timestamps_ns=target_timestamps_ns,
            output_dir=output_dir,
        )

    scene_tiles = tuple(
        SceneTile(
            tile_id=tile_id,
            route_start_meters=start,
            route_end_meters=end,
            config_path=tile_results[tile_id]["config"],
            checkpoint_path=tile_results[tile_id]["checkpoint"],
            checkpoint_step=tile_results[tile_id]["checkpoint_step"],
            data_root=tile_results[tile_id]["data_root"],
            source_log=REFERENCE_LOG,
            source_time_window_seconds=tuple(
                value / 1e9
                for value in tile_results[tile_id][
                    "available_timestamp_range_ns"
                ]
            ),
            dataparser_transform=tuple(
                tuple(float(value) for value in row)
                for row in tile_results[tile_id]["dataparser_transform"]
            ),
            support_half_width_meters=1.0,
            renderer_profile="tbv_reference_front_static_8k",
            evidence_path=str(output_dir / "tbv_three_tile_drive.json"),
        )
        for tile_id, (start, end) in TILE_RANGES.items()
    )
    route = SceneTileRoute(scene_tiles)
    composite = combine_frames(route, samples, output_dir, fps=args.fps)
    report = {
        "format": "driving_scene_reconstruction.tbv_three_tile_drive.v0",
        "scope": (
            "Sequential offline loading of three independently reconstructed "
            "static checkpoints with a real kinematic bicycle model, bounded "
            "simulated-human controls, logged source-time progression, and "
            "smooth image-space overlap blending. This is a coverage/runtime "
            "pilot, not equivalent-realism or live-streaming acceptance."
        ),
        "reference_log": REFERENCE_LOG,
        "camera": CAMERA_NAME,
        "fps": args.fps,
        "requested_cruise_speed_mps": args.speed_mps,
        "route": route.manifest(),
        "control": control_summary,
        "tiles": tile_results,
        "composite": composite,
        "technical_gates": {
            "control_gate": control_summary["control_gate"],
            "all_checkpoints_at_step_7999": all(
                item["checkpoint_step"] == 7999
                for item in tile_results.values()
            ),
            "all_frames_rendered": all(
                item["rendered_frame_count"] == len(active[tile_id])
                for tile_id, item in tile_results.items()
            ),
            "no_large_source_time_clamp": all(
                item["maximum_source_time_clamp_seconds"] <= 0.2
                for item in tile_results.values()
            ),
            "all_frames_not_mostly_black": (
                composite["near_black_fraction"]["max"] < 0.5
            ),
            "video_decodes_all_frames": (
                composite["video"]["decoded_frame_count"] == len(samples)
            ),
        },
        "visual_status": "requires_manual_review",
        "limitations": [
            "Only the front-center camera is rendered.",
            "Tile checkpoints are loaded sequentially; output is assembled offline.",
            "The static TbV treatment retains vehicle ghosts and cannot supply collision truth.",
            "The supported lateral tube remains +/-1 m for this data.",
        ],
    }
    report["technical_status"] = (
        "pass" if all(report["technical_gates"].values()) else "fail"
    )
    report_path = output_dir / "tbv_three_tile_drive.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
