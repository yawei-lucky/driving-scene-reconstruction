#!/usr/bin/env python3
"""Render a deterministic, no-browser TbV straight/right driving trial."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from driving_scene_reconstruction.sim import EgoState, HumanControl  # noqa: E402
from stage_h3_tbv_driving_adapter import (  # noqa: E402
    AUTOPLAY_END_MARGIN_METERS,
    AUTOPLAY_STOPPED_SPEED_MPS,
    DEFAULT_CONFIG,
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_OUTPUT_SCALE,
    FRONT_CAMERAS,
    CylindricalCockpitComposer,
    autoplay_control,
    make_adapter,
    make_cockpit_frame,
    route_height,
)
from stage_h3_tbv_world_pose_probe import (  # noqa: E402
    LocalWorldPose,
    TbVWorldRenderer,
)


DEFAULT_OUTPUT_DIR = Path(
    "/home/yawei/stage3_external/artifacts/"
    "tbv_headless_humanized_20260725"
)
VIDEO_FPS = 20
TAKEOVER_PROGRESS_METERS = -19.5
DEVIATION_TARGET_METERS = 0.38
DEVIATION_REACHED_METERS = 0.28
RECOVERY_METERS = 0.15
HANDOFF_MIN_MARGIN_METERS = 0.40
HANDOFF_MAX_HEADING_ERROR_DEGREES = 8.0
MINIMUM_DRIVING_MARGIN_METERS = 0.25
MAX_STEER_CHANGE_PER_SECOND = 2.4


@dataclass
class HumanizedPhase:
    name: str = "AUTO"
    started_at: float = 0.0
    previous_steer: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-scale", type=float, default=DEFAULT_OUTPUT_SCALE)
    parser.add_argument("--dt", type=float, default=1.0 / VIDEO_FPS)
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS)
    parser.add_argument("--expected-checkpoint-step", type=int, default=7999)
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def offset_following_control(
    adapter: Any,
    state: EgoState,
    target_lateral_offset: float,
    previous_steer: float,
    dt: float,
) -> HumanControl:
    """Follow a nearby offset path with bounded steering-rate changes."""

    speed_control = autoplay_control(adapter, state)
    corridor = adapter.active_route.corridor
    measurement = corridor.measure(state)
    lookahead = max(1.6, state.speed * 0.75)
    target_progress = min(
        corridor.length,
        measurement.progress + lookahead,
    )
    centre = corridor.pose_at_progress(target_progress)
    target_x = centre.x - math.sin(centre.yaw) * target_lateral_offset
    target_y = centre.y + math.cos(centre.yaw) * target_lateral_offset
    dx = target_x - state.x
    dy = target_y - state.y
    target_distance = max(1e-6, math.hypot(dx, dy))
    bearing_error = math.atan2(
        math.sin(math.atan2(dy, dx) - state.yaw),
        math.cos(math.atan2(dy, dx) - state.yaw),
    )
    desired_angle = math.atan2(
        2.0
        * adapter.vehicle_model.wheelbase
        * math.sin(bearing_error),
        target_distance,
    )
    raw_steer = _clamp(
        desired_angle / adapter.vehicle_model.max_steer_angle,
        -1.0,
        1.0,
    )
    maximum_change = MAX_STEER_CHANGE_PER_SECOND * dt
    steer = _clamp(
        raw_steer,
        previous_steer - maximum_change,
        previous_steer + maximum_change,
    )
    return HumanControl(
        steer=steer,
        throttle=speed_control.throttle,
        brake=speed_control.brake,
    )


def update_phase(
    phase: HumanizedPhase,
    *,
    state: EgoState,
    progress: float,
    lateral_offset: float,
    heading_error_degrees: float,
) -> str | None:
    """Advance the deterministic A/recover/D/recover control programme."""

    previous = phase.name
    age = state.time - phase.started_at
    if phase.name == "AUTO" and progress >= TAKEOVER_PROGRESS_METERS:
        phase.name = "A_DEVIATE"
    elif (
        phase.name == "A_DEVIATE"
        and (lateral_offset >= DEVIATION_REACHED_METERS or age >= 2.5)
    ):
        phase.name = "A_RECOVER"
    elif (
        phase.name == "A_RECOVER"
        and abs(lateral_offset) <= RECOVERY_METERS
        and age >= 0.25
    ):
        phase.name = "D_DEVIATE"
    elif (
        phase.name == "D_DEVIATE"
        and (lateral_offset <= -DEVIATION_REACHED_METERS or age >= 2.5)
    ):
        phase.name = "D_RECOVER"
    elif (
        phase.name == "D_RECOVER"
        and abs(lateral_offset) <= RECOVERY_METERS
        and abs(heading_error_degrees) <= 4.0
        and age >= 0.25
    ):
        phase.name = "CRUISE"
    if phase.name != previous:
        phase.started_at = state.time
        return f"{previous}->{phase.name}"
    return None


def phase_target(phase: str) -> float:
    if phase == "A_DEVIATE":
        return DEVIATION_TARGET_METERS
    if phase == "D_DEVIATE":
        return -DEVIATION_TARGET_METERS
    return 0.0


def phase_label(phase: str) -> tuple[str, str]:
    labels = {
        "AUTO": ("AUTO", "centreline follower"),
        "A_DEVIATE": ("HUMANIZED", "A · small left deviation"),
        "A_RECOVER": ("HUMANIZED", "release A · recover"),
        "D_DEVIATE": ("HUMANIZED", "D · small right deviation"),
        "D_RECOVER": ("HUMANIZED", "release D · recover"),
        "CRUISE": ("HUMANIZED", "manual-like centre recovery"),
        "BRANCH_SELECT": ("HUMANIZED", "branch selected"),
        "STOPPED": ("HUMANIZED", "endpoint brake-to-rest"),
    }
    return labels[phase]


def add_overlay(
    image: Any,
    image_draw: Any,
    *,
    branch: str,
    phase: str,
    control: HumanControl,
    support: dict[str, object],
    verdict: str = "RUNNING",
) -> Any:
    draw = image_draw.Draw(image)
    mode, action = phase_label(phase)
    margin = float(support["distance_margin_meters"])
    colour = (
        (70, 220, 130)
        if margin >= HANDOFF_MIN_MARGIN_METERS
        else (255, 92, 72)
    )
    draw.rectangle((8, 44, 612, 112), fill=(4, 8, 12))
    draw.text(
        (18, 52),
        f"{branch.upper()}  |  {mode}  |  {action}",
        fill=(235, 245, 250),
    )
    draw.text(
        (18, 76),
        (
            f"steer={control.steer:+.2f} throttle={control.throttle:.2f} "
            f"brake={control.brake:.2f} | margin={margin:.2f}m"
        ),
        fill=colour,
    )
    draw.rectangle((image.width - 142, image.height - 42, image.width - 8, image.height - 8), fill=(4, 8, 12))
    draw.text(
        (image.width - 130, image.height - 33),
        verdict,
        fill=(70, 220, 130) if verdict == "PASS" else (255, 216, 77),
    )
    return image


def card(image_module: Any, image_draw: Any, width: int, height: int, lines: list[str]) -> Any:
    image = image_module.new("RGB", (width, height), (5, 9, 13))
    draw = image_draw.Draw(image)
    y = 120
    for index, line in enumerate(lines):
        draw.text(
            (90, y),
            line,
            fill=(240, 246, 248) if index == 0 else (170, 205, 218),
        )
        y += 46 if index == 0 else 34
    return image


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if abs(args.dt * VIDEO_FPS - 1.0) > 1e-6:
        raise ValueError(f"--dt must equal 1/{VIDEO_FPS} for the evidence video")
    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image, ImageDraw

    renderer = TbVWorldRenderer(args.config, args.output_scale)
    renderer.load()
    if renderer.checkpoint_step != args.expected_checkpoint_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_checkpoint_step}, "
            f"got {renderer.checkpoint_step}"
        )
    adapter = make_adapter(renderer, max_speed_mps=args.max_speed_mps)
    composer = CylindricalCockpitComposer.from_renderer(renderer)
    assert renderer.checkpoint_path is not None

    frame_index = 0
    samples: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    reset_events: list[dict[str, object]] = []
    notable_frames: list[Path] = []

    def append_frame(image: Any, *, notable: bool = False) -> Path:
        nonlocal frame_index
        path = frames_dir / f"frame_{frame_index:06d}.jpg"
        image.save(path, format="JPEG", quality=90)
        frame_index += 1
        if notable:
            notable_frames.append(path)
        return path

    def append_hold(image: Any, seconds: float) -> None:
        for _ in range(round(seconds * VIDEO_FPS)):
            append_frame(image)

    width = composer.width
    height = composer.height + 48
    append_hold(
        card(
            Image,
            ImageDraw,
            width,
            height,
            [
                "TbV no-browser humanized driving trial",
                "Straight + right, 4.0 m/s cap, direct local GPU rendering",
                "AUTO -> A deviation -> recovery -> D deviation -> recovery",
                "Straight handoff requires at least 0.40 m correction margin",
            ],
        ),
        2.5,
    )

    for episode_index, branch in enumerate(("straight", "right")):
        state = adapter.reset()
        phase = HumanizedPhase()
        reset_events.append(
            {
                "episode": episode_index,
                "branch": branch,
                "simulation_time_seconds": state.time,
                "state": {
                    "x_meters": state.x,
                    "y_meters": state.y,
                    "yaw_degrees": math.degrees(state.yaw),
                    "speed_mps": state.speed,
                },
            }
        )
        append_hold(
            card(
                Image,
                ImageDraw,
                width,
                height,
                [
                    f"RESET {episode_index + 1} / 2  -  {branch.upper()}",
                    "Start on common approach at -20 m",
                    "Deterministic human-like controls; not a real human claim",
                ],
            ),
            1.5,
        )
        episode_samples: list[dict[str, object]] = []
        transitions: list[dict[str, object]] = []
        boundary_hit = False
        selected_at_sample: int | None = None
        for step_index in range(600):
            support = adapter.support(state)
            transition = update_phase(
                phase,
                state=state,
                progress=support.progress_from_anchor_meters,
                lateral_offset=support.lateral_offset_meters,
                heading_error_degrees=support.heading_error_degrees,
            )
            if transition:
                transitions.append(
                    {
                        "transition": transition,
                        "simulation_time_seconds": state.time,
                        "progress_from_anchor_meters": (
                            support.progress_from_anchor_meters
                        ),
                        "lateral_offset_meters": support.lateral_offset_meters,
                    }
                )
            branch_selected_now = False
            if support.selection_required:
                adapter.select_branch(branch, state)
                support = adapter.support(state)
                selected_at_sample = len(samples)
                branch_selected_now = True
                transitions.append(
                    {
                        "transition": f"SELECT_{branch.upper()}",
                        "simulation_time_seconds": state.time,
                        "route_support": support.as_dict(),
                    }
                )

            if phase.name in {"AUTO", "CRUISE"}:
                control = autoplay_control(adapter, state)
            else:
                control = offset_following_control(
                    adapter,
                    state,
                    phase_target(phase.name),
                    phase.previous_steer,
                    args.dt,
                )
            phase.previous_steer = control.steer
            update = adapter.step(state, control, args.dt)
            state = update.state
            support = adapter.support(state)

            render_started = time.perf_counter()
            z = route_height(
                renderer,
                support.renderer_profile,
                support.progress_from_anchor_meters,
            )
            observation = renderer.render(
                support.renderer_profile,
                LocalWorldPose(state.x, state.y, z, state.yaw),
                FRONT_CAMERAS,
            )
            camera_frames = dict(observation["frames"])
            finite = all(
                np.isfinite(frame).all() for frame in camera_frames.values()
            )
            cockpit = make_cockpit_frame(
                Image,
                ImageDraw,
                composer,
                camera_frames,
                adapter,
                state,
                support.as_dict(),
            )
            shown_phase = "BRANCH_SELECT" if branch_selected_now else phase.name
            add_overlay(
                cockpit,
                ImageDraw,
                branch=branch,
                phase=shown_phase,
                control=control,
                support=support.as_dict(),
            )
            path = append_frame(
                cockpit,
                notable=(
                    transition is not None
                    or branch_selected_now
                    or step_index == 0
                ),
            )
            elapsed_ms = (time.perf_counter() - render_started) * 1000.0
            sample = {
                "sequence": len(samples),
                "episode": episode_index,
                "branch": branch,
                "frame": path.name,
                "simulation_time_seconds": state.time,
                "control_mode": (
                    "autoplay_route_follower"
                    if phase.name == "AUTO"
                    else "humanized_script"
                ),
                "control_phase": phase.name,
                "control": {
                    "steer": control.steer,
                    "throttle": control.throttle,
                    "brake": control.brake,
                },
                "ego_pose": {
                    "x_meters": state.x,
                    "y_meters": state.y,
                    "yaw_degrees": math.degrees(state.yaw),
                    "speed_mps": state.speed,
                },
                "route_support": support.as_dict(),
                "all_camera_frames_finite": finite,
                "camera_count": len(camera_frames),
                "renderer_ms": float(observation["render_seconds"]) * 1000.0,
                "render_and_presentation_ms": elapsed_ms,
                "frame_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "boundary_hit": update.boundary_hit,
                "boundary_reason": update.boundary_reason,
            }
            samples.append(sample)
            episode_samples.append(sample)
            if update.boundary_hit:
                boundary_hit = True
                break
            remaining = max(
                0.0,
                adapter.active_route.corridor.length
                - adapter.active_route.corridor.measure(state).progress,
            )
            if (
                adapter.selected_branch is not None
                and remaining <= AUTOPLAY_END_MARGIN_METERS + 0.08
                and state.speed <= AUTOPLAY_STOPPED_SPEED_MPS
            ):
                break
        else:
            raise RuntimeError(f"{branch} trial did not finish in 600 steps")

        offsets = [
            float(sample["route_support"]["lateral_offset_meters"])  # type: ignore[index]
            for sample in episode_samples
        ]
        margins = [
            float(sample["route_support"]["distance_margin_meters"])  # type: ignore[index]
            for sample in episode_samples
        ]
        headings = [
            abs(float(sample["route_support"]["heading_error_degrees"]))  # type: ignore[index]
            for sample in episode_samples
        ]
        handoff_samples = [
            sample
            for sample in episode_samples
            if sample["route_support"]["selected_branch"] == branch  # type: ignore[index]
            and -1.0
            <= float(sample["route_support"]["progress_from_anchor_meters"])  # type: ignore[index]
            <= 8.0
        ]
        handoff_min_margin = min(
            (
                float(sample["route_support"]["distance_margin_meters"])  # type: ignore[index]
                for sample in handoff_samples
            ),
            default=1.0,
        )
        handoff_max_heading = max(
            (
                abs(float(sample["route_support"]["heading_error_degrees"]))  # type: ignore[index]
                for sample in handoff_samples
            ),
            default=0.0,
        )
        phase_names = {
            str(sample["control_phase"]) for sample in episode_samples
        }
        gates = {
            "finite_three_front_cameras": all(
                bool(sample["all_camera_frames_finite"])
                and int(sample["camera_count"]) == len(FRONT_CAMERAS)
                for sample in episode_samples
            ),
            "no_support_boundary_hit": not boundary_hit,
            "driving_retains_emergency_margin": (
                min(margins) >= MINIMUM_DRIVING_MARGIN_METERS
            ),
            "auto_to_humanized": (
                "AUTO" in phase_names and "CRUISE" in phase_names
            ),
            "a_deviation_reached": max(offsets) >= DEVIATION_REACHED_METERS,
            "d_deviation_reached": min(offsets) <= -DEVIATION_REACHED_METERS,
            "a_and_d_recovery_completed": (
                "A_RECOVER" in phase_names
                and "D_RECOVER" in phase_names
                and "CRUISE" in phase_names
            ),
            "straight_handoff_retains_correction_margin": (
                branch != "straight"
                or handoff_min_margin >= HANDOFF_MIN_MARGIN_METERS
            ),
            "straight_handoff_heading_is_correctable": (
                branch != "straight"
                or handoff_max_heading
                <= HANDOFF_MAX_HEADING_ERROR_DEGREES
            ),
            "endpoint_brake_to_rest": (
                state.speed <= AUTOPLAY_STOPPED_SPEED_MPS
            ),
        }
        passed = all(gates.values())
        render_values = [
            float(sample["renderer_ms"]) for sample in episode_samples
        ]
        episode_report = {
            "branch": branch,
            "verdict": "pass" if passed else "fail",
            "gates": gates,
            "sample_count": len(episode_samples),
            "duration_seconds": state.time,
            "selected_at_sample": selected_at_sample,
            "maximum_left_offset_meters": max(offsets),
            "maximum_right_offset_meters": min(offsets),
            "maximum_abs_lateral_offset_meters": max(map(abs, offsets)),
            "minimum_distance_margin_meters": min(margins),
            "maximum_abs_heading_error_degrees": max(headings),
            "handoff_minimum_margin_meters": handoff_min_margin,
            "handoff_maximum_heading_error_degrees": handoff_max_heading,
            "final_speed_mps": state.speed,
            "renderer_ms": {
                "p50": statistics.median(render_values),
                "p95": percentile(render_values, 0.95),
                "maximum": max(render_values),
            },
            "transitions": transitions,
        }
        episodes.append(episode_report)
        append_hold(
            card(
                Image,
                ImageDraw,
                width,
                height,
                [
                    f"{branch.upper()}  -  {'PASS' if passed else 'FAIL'}",
                    (
                        f"offset range {min(offsets):+.2f} .. {max(offsets):+.2f} m"
                    ),
                    f"minimum corridor margin {min(margins):.2f} m",
                    (
                        f"handoff margin {handoff_min_margin:.2f} m, "
                        f"heading error {handoff_max_heading:.1f} deg"
                    ),
                    f"endpoint speed {state.speed:.3f} m/s",
                ],
            ),
            2.0,
        )

    overall_passed = all(episode["verdict"] == "pass" for episode in episodes)
    append_hold(
        card(
            Image,
            ImageDraw,
            width,
            height,
            [
                f"OVERALL AUTOMATED VERDICT  -  {'PASS' if overall_passed else 'FAIL'}",
                "Both routes include AUTO -> humanized, A/D deviation, recovery and reset",
                "Visual driving judgment is recorded separately after frame review",
                "This is a deterministic human-like test, not a real-human claim",
            ],
        ),
        3.0,
    )

    video_path = output_dir / "tbv_headless_humanized_straight_right.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            str(frames_dir / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )

    sheet_width = 800
    thumb_width = sheet_width // 2
    thumb_height = round(height * thumb_width / width)
    selected = notable_frames[:10]
    sheet = Image.new(
        "RGB",
        (sheet_width, max(1, math.ceil(len(selected) / 2)) * thumb_height),
        (5, 9, 13),
    )
    for index, path in enumerate(selected):
        thumb = Image.open(path).convert("RGB")
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(
            thumb,
            ((index % 2) * thumb_width, (index // 2) * thumb_height),
        )
    sheet_path = output_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, format="JPEG", quality=92)

    report = {
        "format": "driving_scene_reconstruction.tbv_headless_humanized.v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "claim": "deterministic_human_like_test_not_real_human",
        "config_path": str(args.config.expanduser().resolve()),
        "checkpoint_path": str(renderer.checkpoint_path),
        "checkpoint_step": renderer.checkpoint_step,
        "output_scale": args.output_scale,
        "dt_seconds": args.dt,
        "video_fps": VIDEO_FPS,
        "maximum_speed_mps": args.max_speed_mps,
        "camera_names": list(FRONT_CAMERAS),
        "straight_handoff_contract": {
            "minimum_correction_margin_meters": HANDOFF_MIN_MARGIN_METERS,
            "maximum_heading_error_degrees": (
                HANDOFF_MAX_HEADING_ERROR_DEGREES
            ),
        },
        "deviation_contract": {
            "target_meters": DEVIATION_TARGET_METERS,
            "reached_meters": DEVIATION_REACHED_METERS,
            "recovery_meters": RECOVERY_METERS,
        },
        "reset_events": reset_events,
        "episodes": episodes,
        "summary": {
            "verdict": "pass" if overall_passed else "fail",
            "sample_count": len(samples),
            "video_frame_count": frame_index,
            "video_duration_seconds": frame_index / VIDEO_FPS,
            "browser_used": False,
            "http_used": False,
            "training_performed": False,
            "visual_review_status": "pending_manual_frame_review",
        },
        "artifacts": {
            "video": str(video_path),
            "contact_sheet": str(sheet_path),
            "frames": str(frames_dir),
        },
        "samples": samples,
    }
    report_path = output_dir / "tbv_headless_humanized_trial.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"video: {video_path}")
    print(f"report: {report_path}")
    print(f"contact sheet: {sheet_path}")
    print(f"automated verdict: {report['summary']['verdict']}")


if __name__ == "__main__":
    main()
