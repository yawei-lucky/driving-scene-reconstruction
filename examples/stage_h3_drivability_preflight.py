#!/usr/bin/env python3
"""Automated preflight for H3 logged-time drivability evidence.

This script does not replace an operator driving trial. It verifies the
repeatable backend evidence that should be true before asking a human to judge
road/lane readability in the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    LoggedEgoOffsetController,
    LoggedMovementProfile,
    SplatADLoggedRenderer,
    logged_movement_profile,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--scene", default="040")
    parser.add_argument("--probe-time", type=float, default=0.4)
    parser.add_argument("--min-counterfactual-mad", type=float, default=1.0)
    parser.add_argument(
        "--movement-profile",
        choices=("safe", "visible"),
        default="visible",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def control_for_step(step: int, movement_profile: str) -> HumanControl:
    if movement_profile == "visible":
        script = (
            HumanControl(throttle=1.0, steer=1.0),
            HumanControl(throttle=1.0, steer=1.0),
            HumanControl(throttle=1.0, steer=1.0),
            HumanControl(throttle=1.0, steer=0.6),
            HumanControl(throttle=0.8),
            HumanControl(throttle=0.8, steer=-1.0),
            HumanControl(throttle=0.6, steer=-1.0),
            HumanControl(steer=-1.0),
            HumanControl(brake=0.7, steer=-0.5),
            HumanControl(brake=0.7),
            HumanControl(throttle=0.6, steer=1.0),
            HumanControl(throttle=0.8, steer=1.0),
        )
        return script[step % len(script)]
    script = (
        HumanControl(throttle=1.0),
        HumanControl(throttle=1.0, steer=0.6),
        HumanControl(steer=0.6),
        HumanControl(throttle=0.5),
        HumanControl(steer=-0.6),
        HumanControl(steer=-0.6),
        HumanControl(brake=0.5),
        HumanControl(steer=0.4),
        HumanControl(throttle=0.3),
        HumanControl(),
    )
    return script[step % len(script)]


def state_record(step: int, state: EgoState) -> dict[str, float | int]:
    return {
        "step": step,
        "time": state.time,
        "x": state.x,
        "y": state.y,
        "yaw_degrees": math.degrees(state.yaw),
        "speed": state.speed,
    }


def scripted_states(
    controller: LoggedEgoOffsetController,
    *,
    steps: int,
    dt: float,
    movement_profile: str,
) -> list[dict[str, float | int]]:
    state = controller.reset()
    states = []
    for step in range(steps):
        if step:
            state = controller.step(
                state,
                control_for_step(step - 1, movement_profile),
                dt,
            )
        states.append(state_record(step, state))
    return states


def mean_absolute_pixel_difference(first: object, second: object) -> float:
    import numpy as np

    a = first.astype(np.float32)
    b = second.astype(np.float32)
    return float(np.abs(a - b).mean())


def frame_digest(frame: object) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def all_camera_outputs_valid(frames: dict[str, object]) -> bool:
    import numpy as np

    for name in CAMERAS:
        frame = frames.get(name)
        if frame is None:
            return False
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return False
        if not bool(np.isfinite(frame).all()):
            return False
    return True


def make_mosaic(
    image_module: Any,
    image_draw_module: Any,
    frames: dict[str, object],
    state: EgoState,
    label: str,
) -> object:
    cell = (480, 270)
    status_height = 36
    canvas = image_module.new(
        "RGB",
        (cell[0] * 3, cell[1] * 2 + status_height),
        "black",
    )
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        x = (index % 3) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // 3) * cell[1] + (cell[1] - tile.height) // 2
        image_draw_module.Draw(tile).text(
            (8, 7),
            name,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        canvas.paste(tile, (x, y))
    status = (
        f"{label}  t={state.time:.2f}s  x={state.x:+.2f}m  "
        f"y={state.y:+.2f}m  yaw={math.degrees(state.yaw):+.1f}deg"
    )
    image_draw_module.Draw(canvas).text(
        (10, cell[1] * 2 + 11),
        status,
        fill=(255, 216, 77),
    )
    return canvas


def make_front_probe_sheet(
    image_module: Any,
    image_draw_module: Any,
    probes: dict[str, object],
) -> object:
    cell = (480, 270)
    columns = 3
    rows = 2
    canvas = image_module.new("RGB", (cell[0] * columns, cell[1] * rows), "black")
    for index, (label, frame) in enumerate(probes.items()):
        tile = image_module.fromarray(frame).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        x = (index % columns) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // columns) * cell[1] + (cell[1] - tile.height) // 2
        image_draw_module.Draw(tile).text(
            (8, 7),
            label,
            fill=(255, 216, 77),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        canvas.paste(tile, (x, y))
    return canvas


def make_contact_sheet(
    image_module: Any,
    image_draw_module: Any,
    images: list[tuple[str, object]],
) -> object:
    cell = (720, 306)
    columns = 2
    rows = math.ceil(len(images) / columns)
    canvas = image_module.new("RGB", (cell[0] * columns, cell[1] * rows), "black")
    for index, (label, image) in enumerate(images):
        tile = image.copy()
        tile.thumbnail((cell[0], cell[1] - 22), image_module.Resampling.LANCZOS)
        x = (index % columns) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // columns) * cell[1] + 22 + (cell[1] - 22 - tile.height) // 2
        title_x = (index % columns) * cell[0] + 10
        title_y = (index // columns) * cell[1] + 5
        image_draw_module.Draw(canvas).text(
            (title_x, title_y),
            label,
            fill=(255, 216, 77),
        )
        canvas.paste(tile, (x, y))
    return canvas


def outside_limits_rejected(
    renderer: SplatADLoggedRenderer,
    scene: str,
    rig: CameraRig,
    profile: LoggedMovementProfile,
) -> bool:
    limit_states = (
        EgoState(x=profile.limits.max_abs_forward_meters + 0.001),
        EgoState(y=profile.limits.max_abs_left_meters + 0.001),
        EgoState(yaw=profile.limits.max_abs_yaw_radians + math.radians(0.01)),
    )
    for state in limit_states:
        try:
            renderer.render(scene, state, rig)
        except ValueError:
            continue
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if not math.isfinite(args.probe_time) or args.probe_time < 0.0:
        raise ValueError("--probe-time must be finite and non-negative")
    if (
        not math.isfinite(args.min_counterfactual_mad)
        or args.min_counterfactual_mad < 0.0
    ):
        raise ValueError("--min-counterfactual-mad must be finite and non-negative")

    from PIL import Image, ImageDraw
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    sequence_dir = output_dir / "sequence_samples"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    profile = logged_movement_profile(args.movement_profile)
    renderer = SplatADLoggedRenderer(
        args.config,
        output_scale=args.output_scale,
        limits=profile.limits,
    )
    renderer.load()
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    controller = LoggedEgoOffsetController.from_profile(
        renderer.logged_duration,
        profile,
    )
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(
            f"expected accepted static step 7999, got {renderer.checkpoint_step}"
        )

    renderer.render(args.scene, EgoState(), rig)
    reset_first = renderer.render(args.scene, controller.reset(), rig)
    reset_second = renderer.render(args.scene, controller.reset(), rig)
    reset_exact = all(
        np.array_equal(reset_first.frames[name], reset_second.frames[name])
        for name in CAMERAS
    )

    probe_time = min(args.probe_time, renderer.logged_duration)
    probe_states = {
        "centre": EgoState(time=probe_time),
        f"forward +{profile.probe_forward_meters:.2f}m": EgoState(
            x=profile.probe_forward_meters,
            time=probe_time,
        ),
        f"left +{profile.probe_left_meters:.2f}m": EgoState(
            y=profile.probe_left_meters,
            time=probe_time,
        ),
        f"yaw +{math.degrees(profile.probe_yaw_radians):.1f}deg": EgoState(
            yaw=profile.probe_yaw_radians,
            time=probe_time,
        ),
        f"yaw -{math.degrees(profile.probe_yaw_radians):.1f}deg": EgoState(
            yaw=-profile.probe_yaw_radians,
            time=probe_time,
        ),
    }
    probe_observations = {
        label: renderer.render(args.scene, state, rig)
        for label, state in probe_states.items()
    }
    centre = probe_observations["centre"]
    front_probe_mad = {
        label: mean_absolute_pixel_difference(
            centre.frames["front"],
            observation.frames["front"],
        )
        for label, observation in probe_observations.items()
        if label != "centre"
    }
    yaw_labels = [
        label for label in probe_observations if label.startswith("yaw ")
    ]
    positive_negative_yaw_mad = mean_absolute_pixel_difference(
        probe_observations[yaw_labels[0]].frames["front"],
        probe_observations[yaw_labels[1]].frames["front"],
    )
    probe_logical_frames = {
        label: int(observation.metadata["logical_frame"])
        for label, observation in probe_observations.items()
    }
    make_front_probe_sheet(
        Image,
        ImageDraw,
        {
            label: observation.frames["front"]
            for label, observation in probe_observations.items()
        },
    ).save(output_dir / "counterfactual_front_preflight.jpg", quality=92)

    first_script_states = scripted_states(
        controller,
        steps=args.steps,
        dt=args.dt,
        movement_profile=profile.name,
    )
    second_script_states = scripted_states(
        controller,
        steps=args.steps,
        dt=args.dt,
        movement_profile=profile.name,
    )
    scripted_states_repeatable = first_script_states == second_script_states

    selected_steps = {
        0,
        min(args.steps - 1, args.steps // 4),
        min(args.steps - 1, args.steps // 2),
        min(args.steps - 1, (args.steps * 3) // 4),
        args.steps - 1,
    }
    latency_ms: list[float] = []
    logical_frames: list[int] = []
    sensor_spread_ms: list[float] = []
    rendered_state_records: list[dict[str, float | int]] = []
    sampled_mosaics: list[tuple[str, object]] = []
    frame_digests: list[dict[str, str | int]] = []
    frame_shapes_by_camera: dict[str, tuple[int, ...]] = {}
    all_outputs_valid = True
    all_camera_times_present = True
    all_sources_present = True

    state = controller.reset()
    final_observation = None
    for step in range(args.steps):
        if step:
            state = controller.step(
                state,
                control_for_step(step - 1, profile.name),
                args.dt,
            )
        observation = renderer.render(args.scene, state, rig)
        final_observation = observation
        metadata = dict(observation.metadata)
        frames = dict(observation.frames)
        all_outputs_valid = all_outputs_valid and all_camera_outputs_valid(frames)
        all_camera_times_present = all_camera_times_present and set(
            metadata["camera_log_times_seconds"]
        ) == set(CAMERAS)
        all_sources_present = all_sources_present and set(
            metadata["camera_source_names"]
        ) == set(CAMERAS)
        logical_frames.append(int(metadata["logical_frame"]))
        latency_ms.append(float(metadata["render_seconds"]) * 1000.0)
        sensor_spread_ms.append(float(metadata["camera_time_spread_ms"]))
        rendered_state_records.append(state_record(step, state))
        if step in selected_steps:
            label = f"step {step:03d} logical {int(metadata['logical_frame']):03d}"
            mosaic = make_mosaic(Image, ImageDraw, frames, state, label)
            mosaic_path = sequence_dir / f"{step:03d}.jpg"
            mosaic.save(mosaic_path, quality=90)
            sampled_mosaics.append((label, mosaic))
            frame_digests.append(
                {
                    "step": step,
                    "logical_frame": int(metadata["logical_frame"]),
                    "front_sha256": frame_digest(frames["front"]),
                }
            )
            if not frame_shapes_by_camera:
                frame_shapes_by_camera = {
                    name: tuple(int(value) for value in frame.shape)
                    for name, frame in frames.items()
                }

    if final_observation is None:
        raise RuntimeError("no observations rendered")
    final_repeat = renderer.render(args.scene, state, rig)
    final_exact = all(
        np.array_equal(final_observation.frames[name], final_repeat.frames[name])
        for name in CAMERAS
    )
    make_contact_sheet(Image, ImageDraw, sampled_mosaics).save(
        output_dir / "sequence_contact_sheet.jpg",
        quality=90,
    )

    latency = distribution(latency_ms)
    sensor_spread = distribution(sensor_spread_ms)
    max_abs_yaw_degrees = max(
        abs(float(record["yaw_degrees"])) for record in rendered_state_records
    )
    max_forward = max(float(record["x"]) for record in rendered_state_records)
    max_left = max(abs(float(record["y"])) for record in rendered_state_records)
    logical_frames_non_decreasing = all(
        right >= left for left, right in zip(logical_frames, logical_frames[1:])
    )
    logical_frame_progressed = logical_frames[-1] > logical_frames[0]
    states_match_rendered_script = first_script_states == rendered_state_records
    same_logical_frame_for_counterfactuals = len(
        set(probe_logical_frames.values())
    ) == 1
    same_time_counterfactual_pixels_change = all(
        mad >= args.min_counterfactual_mad for mad in front_probe_mad.values()
    )
    steering_probe_distinct = (
        positive_negative_yaw_mad >= args.min_counterfactual_mad
    )
    script_exercises_motion = (
        max_forward >= min(0.1, profile.limits.max_abs_forward_meters)
        and max_abs_yaw_degrees >= min(
            1.0,
            math.degrees(profile.limits.max_abs_yaw_radians),
        )
    )
    automated_gates = {
        "accepted_static_8k_checkpoint": renderer.checkpoint_step == 7999,
        "all_six_camera_outputs_valid": all_outputs_valid,
        "same_time_counterfactual_pixels_change": same_time_counterfactual_pixels_change,
        "same_logical_frame_for_counterfactual_probes": same_logical_frame_for_counterfactuals,
        "positive_and_negative_yaw_are_distinct": steering_probe_distinct,
        "logical_frames_never_go_backward": logical_frames_non_decreasing,
        "logical_frame_progressed": logical_frame_progressed,
        "camera_times_and_sources_present": (
            all_camera_times_present and all_sources_present
        ),
        "sensor_spread_reported_under_100ms": sensor_spread["maximum"] <= 100.0,
        "reset_exact_pixel_repeatability": reset_exact,
        "scripted_states_repeatable": scripted_states_repeatable,
        "rendered_states_match_scripted_states": states_match_rendered_script,
        "final_state_pixel_repeatability": final_exact,
        "script_exercises_forward_and_yaw_motion": script_exercises_motion,
        "inside_profile_limits": (
            max_forward <= profile.limits.max_abs_forward_meters
            and max_left <= profile.limits.max_abs_left_meters
            and max_abs_yaw_degrees
            <= math.degrees(profile.limits.max_abs_yaw_radians)
        ),
        "outside_profile_limits_rejected": outside_limits_rejected(
            renderer,
            args.scene,
            rig,
            profile,
        ),
        "renderer_observation_p95_at_most_100ms": latency["p95"] <= 100.0,
    }
    manual_review_items = {
        "road_lane_curb_continuity": "review sequence_contact_sheet.jpg",
        "steering_direction_visual": "review counterfactual_front_preflight.jpg",
        "nearby_pose_artifacts": "review sequence samples and front probes",
        "physical_key_to_display_latency": "requires real browser/client trial",
        "dynamic_traffic_decision_impact": "requires human review of sampled frames",
    }
    report = {
        "report_type": "h3_drivability_preflight_not_human_acceptance",
        "config": str(args.config),
        "checkpoint": str(renderer.checkpoint_path),
        "checkpoint_step": renderer.checkpoint_step,
        "scene": args.scene,
        "logged_duration_seconds": renderer.logged_duration,
        "output_scale": args.output_scale,
        "movement_profile": profile.name,
        "movement_limits": {
            "max_abs_forward_meters": profile.limits.max_abs_forward_meters,
            "max_abs_left_meters": profile.limits.max_abs_left_meters,
            "max_abs_yaw_degrees": math.degrees(
                profile.limits.max_abs_yaw_radians
            ),
        },
        "steps": args.steps,
        "dt_seconds": args.dt,
        "cameras": list(CAMERAS),
        "frame_shapes_by_camera": frame_shapes_by_camera,
        "automated_gates": automated_gates,
        "all_automated_gates_passed": all(automated_gates.values()),
        "manual_review_items": manual_review_items,
        "probe_time_seconds": probe_time,
        "counterfactual_probe_logical_frames": probe_logical_frames,
        "front_counterfactual_mean_abs_diff": front_probe_mad,
        "front_positive_negative_yaw_mean_abs_diff": positive_negative_yaw_mad,
        "latency_ms": latency,
        "sensor_time_spread_ms": sensor_spread,
        "logical_frames": logical_frames,
        "motion_summary": {
            "max_forward_meters": max_forward,
            "max_abs_left_meters": max_left,
            "max_abs_yaw_degrees": max_abs_yaw_degrees,
            "final_state": rendered_state_records[-1],
        },
        "selected_frame_digests": frame_digests,
        "artifacts": {
            "front_counterfactual_sheet": str(
                output_dir / "counterfactual_front_preflight.jpg"
            ),
            "sequence_contact_sheet": str(output_dir / "sequence_contact_sheet.jpg"),
            "sequence_samples": str(sequence_dir),
        },
    }
    report_path = output_dir / "stage_h3_drivability_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE: {report_path}")
    if not report["all_automated_gates_passed"]:
        failed = [
            name for name, passed in automated_gates.items() if not passed
        ]
        raise SystemExit(f"FAIL: H3 drivability preflight gates failed: {failed}")


if __name__ == "__main__":
    main()
