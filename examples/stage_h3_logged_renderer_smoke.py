#!/usr/bin/env python3
"""GPU smoke for logged-time H3 rendering plus bounded human offsets."""

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
    LoggedEgoOffsetController,
    SplatADLoggedRenderer,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
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


def control_for_step(step: int) -> HumanControl:
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


def mosaic(image_module: object, frames: dict[str, object]) -> object:
    cell = (480, 270)
    canvas = image_module.new("RGB", (cell[0] * 3, cell[1] * 2), "black")
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        x = (index % 3) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // 3) * cell[1] + (cell[1] - tile.height) // 2
        canvas.paste(tile, (x, y))
    return canvas


def mean_absolute_pixel_difference(first: object, second: object) -> float:
    import numpy as np

    a = first.astype(np.float32)
    b = second.astype(np.float32)
    return float(np.abs(a - b).mean())


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    from PIL import Image
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer = SplatADLoggedRenderer(
        args.config,
        output_scale=args.output_scale,
    )
    renderer.load()
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(
            f"expected accepted static step 7999, got {renderer.checkpoint_step}"
        )
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    controller = LoggedEgoOffsetController(renderer.logged_duration)

    # Warm every camera head before timing or repeatability checks.
    renderer.render("040", EgoState(), rig)
    reset_first = renderer.render("040", controller.reset(), rig)
    reset_second = renderer.render("040", controller.reset(), rig)
    reset_exact = all(
        np.array_equal(reset_first.frames[name], reset_second.frames[name])
        for name in CAMERAS
    )
    all_camera_outputs_valid = all(
        name in reset_first.frames
        and reset_first.frames[name].dtype == np.uint8
        and reset_first.frames[name].ndim == 3
        and reset_first.frames[name].shape[2] == 3
        for name in CAMERAS
    )

    probe_time = min(0.4, renderer.logged_duration)
    centre = renderer.render("040", EgoState(time=probe_time), rig)
    left = renderer.render("040", EgoState(y=0.1, time=probe_time), rig)
    yaw = renderer.render(
        "040",
        EgoState(yaw=math.radians(1.0), time=probe_time),
        rig,
    )
    pose_probe = {
        "time_seconds": probe_time,
        "front_mean_abs_diff_left": mean_absolute_pixel_difference(
            centre.frames["front"], left.frames["front"]
        ),
        "front_mean_abs_diff_yaw": mean_absolute_pixel_difference(
            centre.frames["front"], yaw.frames["front"]
        ),
    }
    for label, observation in (
        ("centre", centre),
        ("left_p0.10m", left),
        ("yaw_p1.0deg", yaw),
    ):
        mosaic(Image, dict(observation.frames)).save(
            output_dir / f"pose_probe_{label}.jpg",
            quality=92,
        )

    state = controller.reset()
    latency_ms: list[float] = []
    logical_frames: list[int] = []
    states: list[dict[str, float | int]] = []
    maximum_sensor_spread_ms = 0.0
    for step in range(args.steps):
        if step:
            state = controller.step(
                state,
                control_for_step(step - 1),
                args.dt,
            )
        observation = renderer.render("040", state, rig)
        metadata = dict(observation.metadata)
        latency_ms.append(float(metadata["render_seconds"]) * 1000.0)
        logical_frames.append(int(metadata["logical_frame"]))
        maximum_sensor_spread_ms = max(
            maximum_sensor_spread_ms,
            float(metadata["camera_time_spread_ms"]),
        )
        states.append(
            {
                "step": step,
                "time": state.time,
                "x": state.x,
                "y": state.y,
                "yaw": state.yaw,
                "speed": state.speed,
                "logical_frame": int(metadata["logical_frame"]),
            }
        )
        mosaic(Image, dict(observation.frames)).save(
            frames_dir / f"{step:03d}.jpg",
            quality=90,
        )

    latency = distribution(latency_ms)
    logical_frames_non_decreasing = all(
        right >= left
        for left, right in zip(logical_frames, logical_frames[1:])
    )
    pose_response_nonzero = (
        pose_probe["front_mean_abs_diff_left"] > 0.0
        and pose_probe["front_mean_abs_diff_yaw"] > 0.0
    )
    automated_smoke_gates = {
        "all_six_camera_outputs_valid": all_camera_outputs_valid,
        "logical_frames_never_go_backward": logical_frames_non_decreasing,
        "nearby_pose_changes_pixels": pose_response_nonzero,
        "reset_exact_pixel_repeatability": reset_exact,
        "renderer_observation_p95_at_most_100ms": latency["p95"] <= 100.0,
    }
    report = {
        "config": str(args.config),
        "checkpoint": str(renderer.checkpoint_path),
        "checkpoint_step": renderer.checkpoint_step,
        "scene": "040",
        "logged_duration_seconds": renderer.logged_duration,
        "output_scale": args.output_scale,
        "cameras": list(CAMERAS),
        "steps": args.steps,
        "dt_seconds": args.dt,
        "automated_smoke_gates": automated_smoke_gates,
        "all_automated_smoke_gates_passed": all(
            automated_smoke_gates.values()
        ),
        "reset_exact_pixel_repeatability": reset_exact,
        "logical_frames_monotonic": logical_frames_non_decreasing,
        "logical_frames": logical_frames,
        "maximum_sensor_time_spread_ms": maximum_sensor_spread_ms,
        "pose_probe": pose_probe,
        "six_camera_render_latency_ms": latency,
        "six_camera_p95_at_most_100ms": latency["p95"] <= 100.0,
        "states": states,
    }
    report_path = output_dir / "stage_h3_logged_renderer_smoke.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE: {report_path}")
    if not report["all_automated_smoke_gates_passed"]:
        failed = [
            name
            for name, passed in automated_smoke_gates.items()
            if not passed
        ]
        raise SystemExit(f"FAIL: logged-renderer smoke gates failed: {failed}")


if __name__ == "__main__":
    main()
