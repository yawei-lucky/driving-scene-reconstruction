#!/usr/bin/env python3
"""Render a small multi-station keep-or-rebuild test for the H3 corridor."""

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
    LoggedCenterlineCorridor,
    NearbyPoseLimits,
    SplatADWorldRenderer,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/pandaset_h3/"
    "scene_040_splatad_static_8000/splatad/"
    "2026-07-19_resume_2k_to_8k/config.yml"
)
CAMERAS = ("front_left", "front", "front_right", "left", "back", "right")


def csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--anchor-log-time", type=float, default=4.0)
    parser.add_argument(
        "--forward-stations",
        type=csv_floats,
        default=csv_floats("-34,-17,0,15,29"),
        help="signed metres from the anchor-local origin along the centreline",
    )
    parser.add_argument(
        "--lateral-offsets",
        type=csv_floats,
        default=csv_floats("-1,0,1"),
    )
    return parser.parse_args()


def offset_from_road(center: EgoState, left_meters: float) -> EgoState:
    return EgoState(
        x=center.x - math.sin(center.yaw) * left_meters,
        y=center.y + math.cos(center.yaw) * left_meters,
        yaw=center.yaw,
    )


def labelled_grid(image_module: object, draw_module: object, rows: list[list[tuple[str, object]]]) -> object:
    cell_width, cell_height = 640, 300
    canvas = image_module.new(
        "RGB",
        (cell_width * max(len(row) for row in rows), cell_height * len(rows)),
        "black",
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


def six_camera_mosaic(image_module: object, draw_module: object, frames: object, label: str) -> object:
    cell_width, cell_height = 480, 220
    canvas = image_module.new("RGB", (cell_width * 3, cell_height * 2 + 34), "black")
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail((cell_width, cell_height), image_module.Resampling.LANCZOS)
        draw_module.Draw(tile).text(
            (7, 6), name, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0)
        )
        x = (index % 3) * cell_width + (cell_width - tile.width) // 2
        y = (index // 3) * cell_height + (cell_height - tile.height) // 2
        canvas.paste(tile, (x, y))
    draw_module.Draw(canvas).text((8, cell_height * 2 + 8), label, fill=(255, 216, 77))
    return canvas


def main() -> None:
    args = parse_args()
    import numpy as np
    from PIL import Image, ImageDraw

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = SplatADWorldRenderer(
        args.config,
        output_scale=args.output_scale,
        anchor_log_time=args.anchor_log_time,
        limits=NearbyPoseLimits(100.0, 100.0, math.pi),
    )
    renderer.load()
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(f"expected static step 7999, got {renderer.checkpoint_step}")

    corridor = LoggedCenterlineCorridor(renderer.logged_centerline)
    reference_progress = corridor.measure(EgoState()).progress
    available_forward = corridor.length - reference_progress
    available_backward = reference_progress
    if (
        min(args.forward_stations) < -available_backward - 1e-6
        or max(args.forward_stations) > available_forward + 1e-6
    ):
        raise ValueError(
            f"stations must fit available corridor "
            f"[-{available_backward:.3f}, +{available_forward:.3f}]m"
        )

    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    renderer.render("040", EgoState(), rig)
    front_rows: list[list[tuple[str, object]]] = []
    records: list[dict[str, object]] = []
    render_seconds: list[float] = []
    all_valid = True
    for forward in args.forward_stations:
        center = corridor.pose_at_progress(reference_progress + forward)
        front_row: list[tuple[str, object]] = []
        for lateral in args.lateral_offsets:
            state = offset_from_road(center, lateral)
            observation = renderer.render("040", state, rig)
            valid = all(
                name in observation.frames
                and observation.frames[name].dtype == np.uint8
                and observation.frames[name].shape[2] == 3
                and bool(np.isfinite(observation.frames[name]).all())
                for name in CAMERAS
            )
            all_valid &= valid
            render_time = float(observation.metadata["render_seconds"])
            render_seconds.append(render_time)
            label = f"forward={forward:.1f}m  left={lateral:+.1f}m"
            front_row.append((label, observation.frames["front"]))
            mosaic_path = output_dir / (
                f"six_forward_{forward:04.1f}_left_{lateral:+.1f}.jpg"
            )
            six_camera_mosaic(Image, ImageDraw, observation.frames, label).save(
                mosaic_path, quality=94
            )
            measurement = corridor.measure(state)
            records.append(
                {
                    "forward_from_anchor_origin_meters": forward,
                    "left_from_centerline_meters": lateral,
                    "world_pose": {
                        "x_meters": state.x,
                        "y_meters": state.y,
                        "yaw_degrees": math.degrees(state.yaw),
                    },
                    "measured_distance_from_centerline_meters": measurement.distance,
                    "all_six_frames_valid": valid,
                    "render_seconds": render_time,
                    "six_camera_artifact": str(mosaic_path),
                }
            )
        front_rows.append(front_row)

    front_sheet = output_dir / "corridor_5x3_front.jpg"
    labelled_grid(Image, ImageDraw, front_rows).save(front_sheet, quality=95)
    ordered = sorted(render_seconds)
    report = {
        "automated_render_status": "pass" if all_valid else "fail",
        "visual_keep_or_rebuild_decision": "requires_human_review",
        "scene": "040",
        "checkpoint_step": renderer.checkpoint_step,
        "checkpoint": str(renderer.checkpoint_path),
        "output_scale": args.output_scale,
        "corridor_length_meters": corridor.length,
        "anchor_origin_progress_meters": reference_progress,
        "available_backward_meters": available_backward,
        "available_forward_meters": available_forward,
        "forward_stations_meters": list(args.forward_stations),
        "lateral_offsets_meters": list(args.lateral_offsets),
        "observation_count": len(records),
        "all_six_frames_valid": all_valid,
        "render_seconds": {
            "p50": statistics.median(ordered),
            "p95": ordered[round((len(ordered) - 1) * 0.95)],
            "maximum": max(ordered),
        },
        "records": records,
        "front_contact_sheet": str(front_sheet),
        "scope": (
            "cheap visual keep-or-rebuild test; no GT or LiDAR road-depth "
            "acceptance is implied"
        ),
    }
    report_path = output_dir / "corridor_sweep.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
