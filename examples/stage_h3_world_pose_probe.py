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
    parser.add_argument("--straight-steps", type=int, default=30)
    parser.add_argument("--lane-change-steps", type=int, default=40)
    parser.add_argument("--brake-steps", type=int, default=15)
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


def six_camera_mosaic(
    image_module: object,
    image_draw_module: object,
    frames: object,
    *,
    native_resolution: bool = False,
) -> object:
    sample = frames[CAMERAS[0]]
    cell = (
        (int(sample.shape[1]), int(sample.shape[0]))
        if native_resolution
        else (480, 270)
    )
    canvas = image_module.new("RGB", (cell[0] * 3, cell[1] * 2), "black")
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        image_draw_module.Draw(tile).text(
            (8, 7),
            name,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
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


def trajectory_specs(
    args: argparse.Namespace,
) -> list[tuple[str, EgoState, tuple[HumanControl, ...]]]:
    """Build symmetric world-space paths from one common reconstruction anchor."""

    throttle = 0.15
    turn = args.turn_steer
    half_lane_change = args.lane_change_steps // 2
    lane_change_tail = args.lane_change_steps - half_lane_change
    return [
        (
            "straight",
            EgoState(speed=args.initial_speed),
            tuple(
                HumanControl(throttle=throttle)
                for _ in range(args.straight_steps)
            ),
        ),
        (
            "left_turn",
            EgoState(speed=args.initial_speed),
            tuple(
                HumanControl(throttle=throttle, steer=turn)
                for _ in range(args.turn_steps)
            ),
        ),
        (
            "right_turn",
            EgoState(speed=args.initial_speed),
            tuple(
                HumanControl(throttle=throttle, steer=-turn)
                for _ in range(args.turn_steps)
            ),
        ),
        (
            "left_lane_change",
            EgoState(speed=args.initial_speed),
            tuple(
                [
                    HumanControl(throttle=throttle, steer=turn)
                    for _ in range(half_lane_change)
                ]
                + [
                    HumanControl(throttle=throttle, steer=-turn)
                    for _ in range(lane_change_tail)
                ]
            ),
        ),
        (
            "right_lane_change",
            EgoState(speed=args.initial_speed),
            tuple(
                [
                    HumanControl(throttle=throttle, steer=-turn)
                    for _ in range(half_lane_change)
                ]
                + [
                    HumanControl(throttle=throttle, steer=turn)
                    for _ in range(lane_change_tail)
                ]
            ),
        ),
        (
            "full_brake",
            EgoState(speed=3.0),
            tuple(HumanControl(brake=1.0) for _ in range(args.brake_steps)),
        ),
    ]


def mirrored_states(
    first: list[EgoState],
    second: list[EgoState],
    *,
    tolerance: float = 1e-8,
) -> bool:
    return len(first) == len(second) and all(
        abs(left.x - right.x) <= tolerance
        and abs(left.y + right.y) <= tolerance
        and abs(left.yaw + right.yaw) <= tolerance
        and abs(left.speed - right.speed) <= tolerance
        and abs(left.time - right.time) <= tolerance
        for left, right in zip(first, second)
    )


def draw_trajectory_plot(
    image_module: object,
    image_draw_module: object,
    trajectories: dict[str, list[EgoState]],
) -> object:
    """Draw a dependency-light x/y evidence plot for all continuous paths."""

    width, height, margin = 1100, 720, 70
    canvas = image_module.new("RGB", (width, height), (18, 18, 18))
    draw = image_draw_module.Draw(canvas)
    all_states = [state for states in trajectories.values() for state in states]
    x_values = [state.x for state in all_states]
    y_values = [state.y for state in all_states]
    x_min, x_max = min(x_values), max(x_values)
    y_extent = max(1.25, max(abs(value) for value in y_values) * 1.1)
    x_extent = max(1.0, x_max - x_min)

    def point(state: EgoState) -> tuple[int, int]:
        px = margin + int((state.x - x_min) / x_extent * (width - 2 * margin))
        py = height // 2 - int(state.y / y_extent * (height - 2 * margin) / 2)
        return px, py

    draw.line((margin, height // 2, width - margin, height // 2), fill=(100, 100, 100), width=2)
    draw.text((margin, 22), "world-space trajectory probe (x forward, y left)", fill=(255, 216, 77))
    colours = {
        "straight": (240, 240, 240),
        "left_turn": (255, 92, 92),
        "right_turn": (92, 160, 255),
        "left_lane_change": (255, 190, 60),
        "right_lane_change": (80, 220, 150),
        "full_brake": (190, 100, 255),
    }
    legend_y = 48
    for name, states in trajectories.items():
        colour = colours[name]
        points = [point(state) for state in states]
        if len(points) > 1:
            draw.line(points, fill=colour, width=4)
        radius = 5
        for endpoint in (points[0], points[-1]):
            draw.ellipse(
                (
                    endpoint[0] - radius,
                    endpoint[1] - radius,
                    endpoint[0] + radius,
                    endpoint[1] + radius,
                ),
                fill=colour,
            )
        draw.text((width - 330, legend_y), name, fill=colour)
        legend_y += 24
    draw.text(
        (margin, height - 38),
        f"x=[{x_min:.2f}, {x_max:.2f}] m, y=[{-y_extent:.2f}, {y_extent:.2f}] m",
        fill=(190, 190, 190),
    )
    return canvas


def main() -> None:
    args = parse_args()
    step_counts = {
        "turn_steps": args.turn_steps,
        "straight_steps": args.straight_steps,
        "lane_change_steps": args.lane_change_steps,
        "brake_steps": args.brake_steps,
    }
    if any(value < 1 for value in step_counts.values()):
        raise ValueError(f"trajectory step counts must be positive: {step_counts}")
    if args.lane_change_steps % 2:
        raise ValueError("lane_change_steps must be even for a symmetric path")
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
    time_shifted_same_pose = renderer.render("040", EgoState(time=10.0), rig)
    reset_exact = all(
        np.array_equal(reset_first.frames[name], reset_second.frames[name])
        for name in CAMERAS
    )
    centre_front = reset_first.frames["front"]

    observations: list[object] = [
        reset_first,
        reset_second,
        time_shifted_same_pose,
    ]
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
    trajectory_states: dict[str, list[EgoState]] = {}
    trajectory_observations: dict[str, list[object]] = {}
    trajectory_reports: dict[str, dict[str, object]] = {}
    trajectory_artifacts: dict[str, dict[str, object]] = {}
    for name, initial_state, controls in trajectory_specs(args):
        state = initial_state
        states: list[EgoState] = []
        path_observations: list[object] = []
        rows: list[dict[str, object]] = []
        front_samples: list[tuple[str, object]] = []
        for step in range(len(controls) + 1):
            observation = renderer.render("040", state, rig)
            observations.append(observation)
            states.append(state)
            path_observations.append(observation)
            rows.append(
                {
                    "step": step,
                    **state_record(state),
                    "all_six_frames_valid": all_frames_valid(observation),
                    "front_near_black_fraction_diagnostic_only": near_black_fraction(
                        observation.frames["front"]
                    ),
                    "render_seconds": observation.metadata["render_seconds"],
                }
            )
            sample_stride = max(1, len(controls) // 7)
            if step % sample_stride == 0 or step == len(controls):
                front_samples.append(
                    (
                        f"step={step} x={state.x:.1f} y={state.y:.1f} "
                        f"yaw={math.degrees(state.yaw):.1f}deg "
                        f"v={state.speed:.1f}m/s",
                        observation.frames["front"],
                    )
                )
            if step < len(controls):
                state = vehicle_model.step(state, controls[step], args.dt)

        frame_differences = [
            mean_absolute_pixel_difference(
                path_observations[index - 1].frames["front"],
                path_observations[index].frames["front"],
            )
            for index in range(1, len(path_observations))
        ]
        trajectory_states[name] = states
        trajectory_observations[name] = path_observations
        front_path = output_dir / f"trajectory_{name}_front.jpg"
        labelled_contact_sheet(Image, ImageDraw, front_samples).save(
            front_path, quality=95
        )
        keyframe_paths: list[str] = []
        for label, index in (
            ("start", 0),
            ("middle", len(path_observations) // 2),
            ("final", len(path_observations) - 1),
        ):
            keyframe_path = output_dir / f"trajectory_{name}_{label}_six_camera.jpg"
            six_camera_mosaic(
                Image,
                ImageDraw,
                path_observations[index].frames,
                native_resolution=True,
            ).save(keyframe_path, quality=95)
            keyframe_paths.append(str(keyframe_path))
        trajectory_artifacts[name] = {
            "front_sequence": str(front_path),
            "six_camera_keyframes": keyframe_paths,
        }
        trajectory_reports[name] = {
            "dt_seconds": args.dt,
            "step_count": len(controls),
            "initial_state": state_record(states[0]),
            "final_state": state_record(states[-1]),
            "front_consecutive_mean_abs_pixel_difference": distribution(
                frame_differences
            ),
            "states": rows,
        }

    labelled_contact_sheet(Image, ImageDraw, lateral_fronts).save(
        output_dir / "lateral_corridor_front.jpg", quality=95
    )
    labelled_contact_sheet(Image, ImageDraw, yaw_fronts).save(
        output_dir / "yaw_probe_front.jpg", quality=95
    )
    for lateral, observation in lateral_observations.items():
        six_camera_mosaic(
            Image,
            ImageDraw,
            observation.frames,
            native_resolution=True,
        ).save(
            output_dir / f"six_camera_left_{lateral:+.1f}m.jpg",
            quality=95,
        )
    # Preserve the first-run filenames as aliases while making the symmetric
    # trajectory names authoritative in the JSON report.
    labelled_contact_sheet(
        Image,
        ImageDraw,
        [
            (
                f"step={index} x={state.x:.1f} y={state.y:.1f} "
                f"yaw={math.degrees(state.yaw):.1f}deg",
                trajectory_observations["left_turn"][index].frames["front"],
            )
            for index, state in enumerate(trajectory_states["left_turn"])
            if index % max(1, args.turn_steps // 7) == 0
            or index == args.turn_steps
        ],
    ).save(output_dir / "continuous_turn_front.jpg", quality=95)
    six_camera_mosaic(
        Image,
        ImageDraw,
        trajectory_observations["left_turn"][-1].frames,
        native_resolution=True,
    ).save(output_dir / "six_camera_turn_final.jpg", quality=95)
    trajectory_plot_path = output_dir / "world_trajectory_xy.jpg"
    draw_trajectory_plot(Image, ImageDraw, trajectory_states).save(
        trajectory_plot_path, quality=95
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
        and max(state.time for states in trajectory_states.values() for state in states)
        > 0.0,
        "source_log_motion_not_reused": all(
            item.metadata["source_motion_metadata_zeroed"] is True
            for item in observations
        ),
        "six_camera_poses_synchronized_to_scene_time": all(
            item.metadata["rig_pose_time_synchronized"] is True
            and float(item.metadata["effective_camera_scene_time_spread_ms"])
            == 0.0
            for item in observations
        ),
        "learned_camera_time_adjustment_disabled": all(
            item.metadata["learned_camera_time_adjustment_disabled"] is True
            for item in observations
        ),
        "requested_lateral_offsets_rendered": set(args.lateral_offsets)
        == set(row["left_meters"] for row in lateral_rows),
    }
    different_simulation_time_same_pose_exact = all(
        np.array_equal(
            reset_first.frames[name],
            time_shifted_same_pose.frames[name],
        )
        for name in CAMERAS
    )
    left_turn = trajectory_states["left_turn"]
    right_turn = trajectory_states["right_turn"]
    left_lane_change = trajectory_states["left_lane_change"]
    right_lane_change = trajectory_states["right_lane_change"]
    straight = trajectory_states["straight"]
    braking = trajectory_states["full_brake"]
    braking_observations = trajectory_observations["full_brake"]
    brake_tail_pose_exact = all(
        state.x == braking[-1].x
        and state.y == braking[-1].y
        and state.yaw == braking[-1].yaw
        and state.speed == 0.0
        for state in braking[-3:]
    )
    brake_tail_pixels_exact = all(
        np.array_equal(observation.frames[name], braking_observations[-1].frames[name])
        for observation in braking_observations[-3:]
        for name in CAMERAS
    )
    motion_gates = {
        "same_pose_different_simulation_time_pixel_exact": different_simulation_time_same_pose_exact,
        "left_right_turn_states_are_mirrored": mirrored_states(left_turn, right_turn),
        "left_right_lane_change_states_are_mirrored": mirrored_states(
            left_lane_change, right_lane_change
        ),
        "straight_has_no_lateral_or_yaw_drift": all(
            abs(state.y) <= 1e-9 and abs(state.yaw) <= 1e-9
            for state in straight
        ),
        "left_turn_changes_future_heading_and_position": left_turn[-1].yaw > 0.0
        and left_turn[-1].y > 0.0,
        "right_turn_changes_future_heading_and_position": right_turn[-1].yaw < 0.0
        and right_turn[-1].y < 0.0,
        "lane_changes_return_heading_and_move_laterally": abs(
            math.degrees(left_lane_change[-1].yaw)
        )
        < 1.0
        and abs(math.degrees(right_lane_change[-1].yaw)) < 1.0
        and left_lane_change[-1].y > 0.5
        and right_lane_change[-1].y < -0.5,
        "full_brake_reaches_zero_speed": braking[-1].speed == 0.0,
        "brake_tail_pose_is_fixed": brake_tail_pose_exact,
        "brake_tail_pixels_are_fixed": brake_tail_pixels_exact,
        "moving_paths_change_front_pixels": all(
            mean_absolute_pixel_difference(
                trajectory_observations[name][0].frames["front"],
                trajectory_observations[name][-1].frames["front"],
            )
            > 1.0
            for name in (
                "straight",
                "left_turn",
                "right_turn",
                "left_lane_change",
                "right_lane_change",
                "full_brake",
            )
        ),
        "six_camera_renderer_p95_under_100ms": percentile(render_times, 0.95)
        < 0.1,
    }
    automated_probe_status = (
        "pass"
        if all(plumbing_gates.values()) and all(motion_gates.values())
        else "fail"
    )
    report = {
        "automated_probe_status": automated_probe_status,
        "plumbing_status": "pass" if all(plumbing_gates.values()) else "fail",
        "motion_status": "pass" if all(motion_gates.values()) else "fail",
        "scope": "world-pose plumbing and visual probe; not corridor acceptance",
        "visual_corridor_status": "requires_human_review",
        "certified_drivable_corridor": False,
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
            "common_anchor_time_range_seconds": list(
                renderer.common_anchor_time_range
            ),
            "effective_camera_scene_time_spread_ms": reset_first.metadata[
                "effective_camera_scene_time_spread_ms"
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
        "motion_gates": motion_gates,
        "lateral_probes": lateral_rows,
        "yaw_probes": yaw_rows,
        "trajectories": trajectory_reports,
        "render_seconds": distribution(render_times),
        "artifacts": {
            "lateral_front": str(output_dir / "lateral_corridor_front.jpg"),
            "yaw_front": str(output_dir / "yaw_probe_front.jpg"),
            "continuous_turn_front": str(output_dir / "continuous_turn_front.jpg"),
            "continuous_turn_six_camera_final": str(
                output_dir / "six_camera_turn_final.jpg"
            ),
            "trajectory_xy": str(trajectory_plot_path),
            "trajectories": trajectory_artifacts,
        },
        "limitations": [
            "Near-black fraction and pixel difference are diagnostics, not visual-quality gates.",
            "The accepted checkpoint is static; dynamic actor correctness is not tested.",
            "Rolling shutter is disabled because source-log motion is not the simulated motion.",
            "The +/-1m browser envelope is provisional, not a certified corridor.",
            "One anchor cross-section cannot establish coverage along the full road.",
            "Human visual review and geometry checks are required before certifying a corridor.",
        ],
    }
    report_path = output_dir / "world_pose_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if automated_probe_status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
