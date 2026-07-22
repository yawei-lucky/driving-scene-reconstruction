#!/usr/bin/env python3
"""Probe the accepted static-8k scene from true simulated world poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    NearbyPoseLimits,
    SimpleVehicleModel,
    SplatADWorldRenderer,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/pandaset_h3/"
    "scene_040_splatad_static_8000/splatad/"
    "2026-07-19_resume_2k_to_8k/config.yml"
)
CAMERAS = (
    "front_left",
    "front",
    "front_right",
    "left",
    "back",
    "right",
)


def parse_csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected a non-empty list of finite numbers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--anchor-log-time", type=float, default=4.0)
    parser.add_argument(
        "--lateral-offsets",
        type=parse_csv_floats,
        default=parse_csv_floats("-3,-2,-1,0,1,2,3"),
    )
    parser.add_argument(
        "--yaw-degrees",
        type=parse_csv_floats,
        default=parse_csv_floats("-10,-5,0,5,10"),
    )
    parser.add_argument("--turn-steps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--initial-speed", type=float, default=2.0)
    parser.add_argument("--turn-steer", type=float, default=0.35)
    return parser.parse_args()


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


def mean_absolute_pixel_difference(first: object, second: object) -> float:
    import numpy as np

    return float(
        np.abs(first.astype(np.float32) - second.astype(np.float32)).mean()
    )


def near_black_fraction(frame: object) -> float:
    import numpy as np

    return float(np.all(frame < 5, axis=2).mean())


def all_frames_valid(observation: object) -> bool:
    import numpy as np

    return all(
        name in observation.frames
        and observation.frames[name].dtype == np.uint8
        and observation.frames[name].ndim == 3
        and observation.frames[name].shape[2] == 3
        and bool(np.isfinite(observation.frames[name]).all())
        for name in CAMERAS
    )


def labelled_contact_sheet(
    image_module: object,
    image_draw_module: object,
    labelled_frames: list[tuple[str, object]],
    *,
    columns: int = 4,
) -> object:
    cell = (480, 270)
    rows = math.ceil(len(labelled_frames) / columns)
    canvas = image_module.new("RGB", (cell[0] * columns, cell[1] * rows), "black")
    for index, (label, frame) in enumerate(labelled_frames):
        tile = image_module.fromarray(frame).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        image_draw_module.Draw(tile).text(
            (8, 7),
            label,
            fill=(255, 216, 77),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        x = (index % columns) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // columns) * cell[1] + (cell[1] - tile.height) // 2
        canvas.paste(tile, (x, y))
    return canvas


def six_camera_mosaic(image_module: object, frames: object) -> object:
    cell = (480, 270)
    canvas = image_module.new("RGB", (cell[0] * 3, cell[1] * 2), "black")
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        x = (index % 3) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // 3) * cell[1] + (cell[1] - tile.height) // 2
        canvas.paste(tile, (x, y))
    return canvas


def state_record(state: EgoState) -> dict[str, float]:
    return {
        "x_meters": state.x,
        "y_meters": state.y,
        "yaw_degrees": math.degrees(state.yaw),
        "speed_meters_per_second": state.speed,
        "simulation_time_seconds": state.time,
    }


def main() -> None:
    args = parse_args()
    if args.turn_steps < 1:
        raise ValueError("turn_steps must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(args.initial_speed) or args.initial_speed < 0.0:
        raise ValueError("initial_speed must be finite and non-negative")
    if not math.isfinite(args.turn_steer) or not -1.0 <= args.turn_steer <= 1.0:
        raise ValueError("turn_steer must be finite and in [-1, 1]")

    from PIL import Image, ImageDraw
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_lateral = max(abs(value) for value in args.lateral_offsets)
    max_yaw = max(abs(math.radians(value)) for value in args.yaw_degrees)
    limits = NearbyPoseLimits(
        max_abs_forward_meters=100.0,
        max_abs_left_meters=max(max_lateral, 10.0),
        max_abs_yaw_radians=max(max_yaw, math.pi),
    )
    renderer = SplatADWorldRenderer(
        args.config,
        output_scale=args.output_scale,
        anchor_log_time=args.anchor_log_time,
        limits=limits,
    )
    renderer.load()
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(
            f"expected accepted static step 7999, got {renderer.checkpoint_step}"
        )
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))

    # Warm all camera heads and verify exact reset repeatability.
    renderer.render("040", EgoState(), rig)
    reset_first = renderer.render("040", EgoState(), rig)
    reset_second = renderer.render("040", EgoState(), rig)
    reset_exact = all(
        np.array_equal(reset_first.frames[name], reset_second.frames[name])
        for name in CAMERAS
    )
    centre_front = reset_first.frames["front"]

    observations: list[object] = [reset_first, reset_second]
    lateral_rows: list[dict[str, object]] = []
    lateral_fronts: list[tuple[str, object]] = []
    lateral_observations: dict[float, object] = {}
    for lateral in args.lateral_offsets:
        state = EgoState(y=lateral)
        observation = renderer.render("040", state, rig)
        observations.append(observation)
        lateral_observations[lateral] = observation
        lateral_fronts.append((f"left={lateral:+.1f}m", observation.frames["front"]))
        lateral_rows.append(
            {
                "left_meters": lateral,
                "front_mean_abs_diff_from_center": mean_absolute_pixel_difference(
                    centre_front, observation.frames["front"]
                ),
                "front_near_black_fraction": near_black_fraction(
                    observation.frames["front"]
                ),
                "all_six_frames_valid": all_frames_valid(observation),
                "render_seconds": observation.metadata["render_seconds"],
            }
        )

    yaw_rows: list[dict[str, object]] = []
    yaw_fronts: list[tuple[str, object]] = []
    for yaw_degrees in args.yaw_degrees:
        state = EgoState(yaw=math.radians(yaw_degrees))
        observation = renderer.render("040", state, rig)
        observations.append(observation)
        yaw_fronts.append((f"yaw={yaw_degrees:+.1f}deg", observation.frames["front"]))
        yaw_rows.append(
            {
                "yaw_degrees": yaw_degrees,
                "front_mean_abs_diff_from_center": mean_absolute_pixel_difference(
                    centre_front, observation.frames["front"]
                ),
                "front_near_black_fraction": near_black_fraction(
                    observation.frames["front"]
                ),
                "all_six_frames_valid": all_frames_valid(observation),
                "render_seconds": observation.metadata["render_seconds"],
            }
        )

    vehicle_model = SimpleVehicleModel(max_acceleration=1.0, max_braking=4.0)
    turn_state = EgoState(speed=args.initial_speed)
    turn_rows: list[dict[str, object]] = []
    turn_fronts: list[tuple[str, object]] = []
    turn_observations: list[object] = []
    for step in range(args.turn_steps + 1):
        observation = renderer.render("040", turn_state, rig)
        observations.append(observation)
        turn_observations.append(observation)
        turn_rows.append(
            {
                "step": step,
                **state_record(turn_state),
                "all_six_frames_valid": all_frames_valid(observation),
                "front_near_black_fraction": near_black_fraction(
                    observation.frames["front"]
                ),
                "render_seconds": observation.metadata["render_seconds"],
            }
        )
        if step % max(1, args.turn_steps // 7) == 0 or step == args.turn_steps:
            turn_fronts.append(
                (
                    f"step={step} x={turn_state.x:.1f} y={turn_state.y:.1f} "
                    f"yaw={math.degrees(turn_state.yaw):.1f}deg",
                    observation.frames["front"],
                )
            )
        if step < args.turn_steps:
            turn_state = vehicle_model.step(
                turn_state,
                HumanControl(throttle=0.15, steer=args.turn_steer),
                args.dt,
            )

    labelled_contact_sheet(Image, ImageDraw, lateral_fronts).save(
        output_dir / "lateral_corridor_front.jpg", quality=95
    )
    labelled_contact_sheet(Image, ImageDraw, yaw_fronts).save(
        output_dir / "yaw_probe_front.jpg", quality=95
    )
    labelled_contact_sheet(Image, ImageDraw, turn_fronts).save(
        output_dir / "continuous_turn_front.jpg", quality=95
    )
    for lateral in (-max_lateral, 0.0, max_lateral):
        if lateral in lateral_observations:
            six_camera_mosaic(
                Image, lateral_observations[lateral].frames
            ).save(
                output_dir / f"six_camera_left_{lateral:+.1f}m.jpg",
                quality=95,
            )
    six_camera_mosaic(Image, turn_observations[-1].frames).save(
        output_dir / "six_camera_turn_final.jpg", quality=95
    )

    render_times = [
        float(observation.metadata["render_seconds"])
        for observation in observations
    ]
    anchor_frames = {
        int(observation.metadata["anchor_logical_frame"])
        for observation in observations
    }
    scene_times = {
        float(observation.metadata["scene_time_seconds"])
        for observation in observations
    }
    all_outputs_valid = all(all_frames_valid(item) for item in observations)
    plumbing_gates = {
        "accepted_static_checkpoint_step_7999": renderer.checkpoint_step == 7999,
        "reset_pixel_exact": reset_exact,
        "all_six_camera_outputs_valid": all_outputs_valid,
        "single_fixed_anchor_frame": len(anchor_frames) == 1,
        "simulation_and_scene_time_decoupled": len(scene_times) == 1
        and turn_state.time > 0.0,
        "source_log_motion_not_reused": all(
            item.metadata["source_motion_metadata_zeroed"] is True
            for item in observations
        ),
        "six_camera_poses_synchronized_to_scene_time": all(
            item.metadata["rig_pose_time_synchronized"] is True
            and float(item.metadata["rendered_camera_scene_time_spread_ms"])
            == 0.0
            for item in observations
        ),
        "vehicle_model_changes_future_heading": turn_state.yaw > 0.0,
        "vehicle_model_changes_future_lateral_position": turn_state.y > 0.0,
        "requested_lateral_offsets_rendered": set(args.lateral_offsets)
        == set(row["left_meters"] for row in lateral_rows),
    }
    report = {
        "result": "pass" if all(plumbing_gates.values()) else "fail",
        "scope": "world-pose plumbing and visual probe; not corridor acceptance",
        "visual_corridor_status": "requires_human_review",
        "scene": "040",
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(renderer.checkpoint_path),
        "checkpoint_step": renderer.checkpoint_step,
        "output_scale": args.output_scale,
        "anchor": {
            "requested_log_time_seconds": args.anchor_log_time,
            "selected_log_time_seconds": renderer.anchor_selected_time,
            "logical_frame": renderer.anchor_frame,
            "source_camera_log_times_seconds": reset_first.metadata[
                "source_camera_log_times_seconds"
            ],
            "rig_pose_sync_brackets": reset_first.metadata[
                "rig_pose_sync_brackets"
            ],
            "rendered_camera_scene_time_spread_ms": reset_first.metadata[
                "rendered_camera_scene_time_spread_ms"
            ],
        },
        "coordinate_semantics": {
            "simulation_time": "EgoState.time; advanced only by vehicle model",
            "scene_time": "frozen at selected source anchor",
            "x": "meters forward in fixed anchor-local world frame",
            "y": "meters left in fixed anchor-local world frame",
            "yaw": "radians left from fixed anchor heading",
        },
        "plumbing_gates": plumbing_gates,
        "lateral_probes": lateral_rows,
        "yaw_probes": yaw_rows,
        "continuous_turn": {
            "dt_seconds": args.dt,
            "steer": args.turn_steer,
            "initial_speed_meters_per_second": args.initial_speed,
            "final_state": state_record(turn_state),
            "states": turn_rows,
        },
        "render_seconds": distribution(render_times),
        "artifacts": {
            "lateral_front": str(output_dir / "lateral_corridor_front.jpg"),
            "yaw_front": str(output_dir / "yaw_probe_front.jpg"),
            "continuous_turn_front": str(output_dir / "continuous_turn_front.jpg"),
            "continuous_turn_six_camera_final": str(
                output_dir / "six_camera_turn_final.jpg"
            ),
        },
        "limitations": [
            "Near-black fraction is only an obvious-hole proxy, not a quality verdict.",
            "The accepted checkpoint is static; dynamic actor correctness is not tested.",
            "Rolling shutter is disabled because source-log motion is not the simulated motion.",
            "Human visual review and geometry checks are required before certifying a corridor.",
        ],
    }
    report_path = output_dir / "world_pose_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["result"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
