#!/usr/bin/env python3
"""Inspect one PandaSet sequence and preserve the H3 data/calibration gate."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pandaset import DataSet
from pandaset.geometry import projection


CAMERA_ORDER = (
    "front_left_camera",
    "front_camera",
    "front_right_camera",
    "left_camera",
    "back_camera",
    "right_camera",
)
OVERLAY_CAMERAS = (
    "front_left_camera",
    "front_camera",
    "front_right_camera",
)


def parse_frames(value: str) -> tuple[int, ...]:
    frames = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not frames:
        raise argparse.ArgumentTypeError("at least one frame is required")
    if any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError("frame indices must be non-negative")
    return frames


def label_for_camera(camera: str) -> str:
    return camera.removesuffix("_camera").replace("_", "-")


def text_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def labeled_thumbnail(
    image: Image.Image,
    label: str,
    size: tuple[int, int],
) -> Image.Image:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    left = (size[0] - thumb.width) // 2
    top = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = text_font(18)
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, box[2] + 20, box[3] + 18), fill=(0, 0, 0))
    draw.text((14, 12), label, fill=(120, 255, 120), font=font)
    return canvas


def make_contact_sheet(
    sequence: object,
    frames: Iterable[int],
    output_path: Path,
) -> None:
    frames = tuple(frames)
    cell_size = (480, 270)
    sheet = Image.new(
        "RGB",
        (cell_size[0] * 3, cell_size[1] * len(frames) * 2),
        "black",
    )
    for frame_row, frame in enumerate(frames):
        for camera_index, camera_name in enumerate(CAMERA_ORDER):
            image = sequence.camera[camera_name][frame]
            cell = labeled_thumbnail(
                image,
                f"{label_for_camera(camera_name)}  frame {frame:02d}",
                cell_size,
            )
            row = frame_row * 2 + camera_index // 3
            column = camera_index % 3
            sheet.paste(cell, (column * cell_size[0], row * cell_size[1]))
    sheet.save(output_path, quality=94)


def depth_color(depth: float, minimum: float, maximum: float) -> tuple[int, int, int]:
    unit = max(0.0, min(1.0, (depth - minimum) / (maximum - minimum)))
    hue = (1.0 - unit) * 0.68
    red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def project_pandar64(
    sequence: object,
    camera_name: str,
    frame: int,
    minimum_depth: float,
    maximum_depth: float,
    max_points: int,
) -> Image.Image:
    camera = sequence.camera[camera_name]
    image = camera[frame].convert("RGB").copy()
    lidar = sequence.lidar[frame]
    lidar = lidar.loc[lidar["d"] == 0]
    points_2d, points_3d, _ = projection(
        lidar[["x", "y", "z"]].to_numpy(),
        image,
        camera.poses[frame],
        camera.intrinsics,
    )
    depth = points_3d[:, 2]
    valid = (depth >= minimum_depth) & (depth <= maximum_depth)
    points_2d = points_2d[valid]
    depth = depth[valid]

    if len(depth) > max_points:
        keep = np.linspace(0, len(depth) - 1, max_points, dtype=np.int64)
        points_2d = points_2d[keep]
        depth = depth[keep]

    draw = ImageDraw.Draw(image)
    for point, point_depth in sorted(
        zip(points_2d, depth, strict=True),
        key=lambda value: value[1],
        reverse=True,
    ):
        x, y = float(point[0]), float(point[1])
        color = depth_color(float(point_depth), minimum_depth, maximum_depth)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return image


def make_overlay_sheet(
    sequence: object,
    frames: Iterable[int],
    output_path: Path,
    minimum_depth: float,
    maximum_depth: float,
    max_points: int,
) -> None:
    frames = tuple(frames)
    cell_size = (640, 360)
    sheet = Image.new(
        "RGB",
        (cell_size[0] * len(OVERLAY_CAMERAS), cell_size[1] * len(frames)),
        "black",
    )
    for row, frame in enumerate(frames):
        for column, camera_name in enumerate(OVERLAY_CAMERAS):
            overlay = project_pandar64(
                sequence,
                camera_name,
                frame,
                minimum_depth,
                maximum_depth,
                max_points,
            )
            cell = labeled_thumbnail(
                overlay,
                f"{label_for_camera(camera_name)}  frame {frame:02d}  Pandar64",
                cell_size,
            )
            sheet.paste(cell, (column * cell_size[0], row * cell_size[1]))
    sheet.save(output_path, quality=94)


def pose_positions(poses: list[dict[str, dict[str, float]]]) -> np.ndarray:
    return np.asarray(
        [
            [
                pose["position"]["x"],
                pose["position"]["y"],
                pose["position"]["z"],
            ]
            for pose in poses
        ],
        dtype=np.float64,
    )


def timestamp_stats(
    values: Iterable[float],
    reference: Iterable[float],
) -> dict[str, float]:
    delta_ms = (
        np.asarray(tuple(values), dtype=np.float64)
        - np.asarray(tuple(reference), dtype=np.float64)
    ) * 1000.0
    return {
        "mean_signed_ms": float(delta_ms.mean()),
        "mean_absolute_ms": float(np.abs(delta_ms).mean()),
        "max_absolute_ms": float(np.abs(delta_ms).max()),
    }


def build_report(dataset: object, sequence: object, sequence_id: str) -> dict[str, object]:
    frame_count = len(sequence.camera["front_camera"].data)
    reference_timestamps = sequence.camera["front_camera"].timestamps
    camera_report = {}
    for name in CAMERA_ORDER:
        camera = sequence.camera[name]
        camera_report[name] = {
            "images": len(camera.data),
            "poses": len(camera.poses),
            "timestamps": len(camera.timestamps),
            "source_resolution": list(camera[0].size),
            "timestamp_offset_from_front": timestamp_stats(
                camera.timestamps,
                reference_timestamps,
            ),
        }

    lidar_counts = []
    pandar64_counts = []
    pandargt_counts = []
    for frame in sequence.lidar.data:
        sensor_ids = frame["d"].to_numpy()
        lidar_counts.append(int(len(frame)))
        pandar64_counts.append(int(np.count_nonzero(sensor_ids == 0)))
        pandargt_counts.append(int(np.count_nonzero(sensor_ids == 1)))

    actor_counts = []
    dynamic_counts = []
    for frame in sequence.cuboids.data:
        if "cuboids.sensor_id" in frame:
            frame = frame.loc[frame["cuboids.sensor_id"] != 1]
        actor_counts.append(int(len(frame)))
        dynamic_counts.append(int((~frame["stationary"].astype(bool)).sum()))

    positions = pose_positions(sequence.camera["front_camera"].poses)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    gps_speed = np.asarray(
        [
            math.hypot(sample["xvel"], sample["yvel"])
            for sample in sequence.gps.data
        ],
        dtype=np.float64,
    )
    duration = float(sequence.timestamps.data[-1] - sequence.timestamps.data[0])

    return {
        "dataset_root_sequences": sorted(dataset.sequences()),
        "dataset_root_semseg_sequences": sorted(dataset.sequences(with_semseg=True)),
        "sequence": sequence_id,
        "frame_count": frame_count,
        "cameras": camera_report,
        "lidar": {
            "frames": len(sequence.lidar.data),
            "poses": len(sequence.lidar.poses),
            "timestamps": len(sequence.lidar.timestamps),
            "all_points_frame_0": lidar_counts[0],
            "pandar64_points_frame_0": pandar64_counts[0],
            "pandargt_points_frame_0": pandargt_counts[0],
            "pandar64_mean_points": float(np.mean(pandar64_counts)),
            "pandargt_mean_points": float(np.mean(pandargt_counts)),
            "timestamp_offset_from_front": timestamp_stats(
                sequence.lidar.timestamps,
                reference_timestamps,
            ),
        },
        "annotations": {
            "cuboid_frames": len(sequence.cuboids.data),
            "semseg_frames": len(sequence.semseg.data) if sequence.semseg else 0,
            "mean_cuboids_excluding_pandargt_duplicates": float(
                np.mean(actor_counts)
            ),
            "max_cuboids_excluding_pandargt_duplicates": max(actor_counts),
            "mean_dynamic_cuboids": float(np.mean(dynamic_counts)),
            "max_dynamic_cuboids": max(dynamic_counts),
        },
        "ego_motion": {
            "duration_seconds": duration,
            "front_camera_pose_path_metres": float(segment_lengths.sum()),
            "front_camera_pose_endpoint_displacement_metres": float(
                np.linalg.norm(positions[-1] - positions[0])
            ),
            "mean_gps_speed_metres_per_second": float(gps_speed.mean()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="040")
    parser.add_argument("--frames", type=parse_frames, default=(0, 40, 79))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-depth", type=float, default=1.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--max-overlay-points", type=int, default=15_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.max_overlay_points <= 0:
        raise ValueError("--max-overlay-points must be positive")

    dataset = DataSet(str(args.data_root))
    if args.sequence not in dataset.sequences():
        raise FileNotFoundError(
            f"sequence {args.sequence} not found under {args.data_root}"
        )

    sequence = dataset[args.sequence]
    sequence.load()
    frame_count = len(sequence.camera["front_camera"].data)
    if any(frame >= frame_count for frame in args.frames):
        raise IndexError(
            f"requested frames {args.frames}, but sequence has {frame_count} frames"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(dataset, sequence, args.sequence)
    report_path = args.output_dir / f"scene_{args.sequence}_data_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contact_path = args.output_dir / f"scene_{args.sequence}_six_camera_keyframes.jpg"
    make_contact_sheet(sequence, args.frames, contact_path)
    overlay_path = (
        args.output_dir / f"scene_{args.sequence}_pandar64_camera_overlay.jpg"
    )
    make_overlay_sheet(
        sequence,
        args.frames,
        overlay_path,
        args.min_depth,
        args.max_depth,
        args.max_overlay_points,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PASS: data report: {report_path}")
    print(f"PASS: six-camera contact sheet: {contact_path}")
    print(f"PASS: Pandar64 calibration overlay: {overlay_path}")


if __name__ == "__main__":
    main()
