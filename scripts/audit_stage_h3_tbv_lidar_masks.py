#!/usr/bin/env python3
"""Audit one bounded TbV window pair's image-guided LiDAR filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_h3_tbv_dataparser import TbVDataParserConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sequence", dest="sequences", action="append")
    parser.add_argument(
        "--window-start-seconds", action="append", type=float
    )
    parser.add_argument(
        "--window-end-seconds", action="append", type=float
    )
    args = parser.parse_args()
    lengths = tuple(
        len(value or ())
        for value in (
            args.sequences,
            args.window_start_seconds,
            args.window_end_seconds,
        )
    )
    if not all(lengths) or len(set(lengths)) != 1:
        parser.error(
            "--sequence, --window-start-seconds, and "
            "--window-end-seconds must be repeated equally"
        )
    output_json = args.output_json.expanduser().resolve()
    if output_json.exists():
        raise RuntimeError(f"audit output already exists: {output_json}")

    config = TbVDataParserConfig(
        data=args.data,
        mask_root=args.mask_root,
        mask_lidar_points=True,
        sequences=tuple(args.sequences),
        window_start_seconds=tuple(args.window_start_seconds),
        window_end_seconds=tuple(args.window_end_seconds),
        train_split_fraction=0.9,
    )
    outputs = config.setup().get_dataparser_outputs(split="train")
    lidar_filter = outputs.metadata["lidar_mask_filter"]
    removed_fraction = float(lidar_filter["removed_point_fraction"])
    report = {
        "format": "driving_scene_reconstruction.tbv_lidar_mask_audit.v0",
        "scope": (
            "Image-guided LiDAR return exclusion audit; not reconstruction "
            "quality or actor truth."
        ),
        "data": str(args.data.expanduser().resolve()),
        "mask_root": str(args.mask_root.expanduser().resolve()),
        "sequences": args.sequences,
        "window_start_seconds": args.window_start_seconds,
        "window_end_seconds": args.window_end_seconds,
        "lidar_mask_filter": lidar_filter,
        "technical_gates": {
            "removed_at_least_one_point": (
                lidar_filter["removed_point_count"] > 0
            ),
            "retained_at_least_one_point": (
                lidar_filter["retained_point_count"] > 0
            ),
            "removed_fraction_below_25_percent": removed_fraction < 0.25,
            "all_seven_cameras_used": (
                len(lidar_filter["camera_images_used"]) == 7
            ),
            "camera_projection_coverage_at_least_95_percent": (
                lidar_filter["camera_projection_coverage_fraction"] >= 0.95
            ),
        },
    }
    report["technical_status"] = (
        "pass" if all(report["technical_gates"].values()) else "fail"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
