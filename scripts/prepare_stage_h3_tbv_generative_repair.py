#!/usr/bin/env python3
"""Render a short observed-pose TbV sequence and align traffic-removal masks."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

from probe_stage_h3_tbv_tile_seam import (
    CAMERA_NAME,
    REFERENCE_LOG,
    render_tile,
)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("distribution requires at least one value")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        blend = position - low
        return ordered[low] * (1.0 - blend) + ordered[high] * blend

    return {
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def align_removal_mask(
    valid_mask_path: Path,
    *,
    output_width: int,
    output_height: int,
    image_module: Any,
    np: Any,
) -> Any:
    """Crop a source valid-pixel mask and return 255 for traffic removal."""

    valid = image_module.open(valid_mask_path).convert("L")
    source_crop_height = round(output_height * valid.width / output_width)
    valid = valid.crop(
        (0, 0, valid.width, min(valid.height, source_crop_height))
    )
    if valid.size != (output_width, output_height):
        valid = valid.resize(
            (output_width, output_height),
            image_module.Resampling.NEAREST,
        )
    return np.where(np.asarray(valid) < 128, 255, 0).astype(np.uint8)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"av2\.utils\.io",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--traffic-mask-root", type=Path, required=True)
    parser.add_argument("--window-start-seconds", type=float, required=True)
    parser.add_argument("--window-end-seconds", type=float, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--expected-checkpoint-step", type=int, default=7999)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_count < 2:
        parser.error("--sample-count must be at least 2")
    if (
        not math.isfinite(args.window_start_seconds)
        or not math.isfinite(args.window_end_seconds)
        or args.window_end_seconds <= args.window_start_seconds
    ):
        parser.error("window end must be finite and greater than start")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = render_tile(
        args.config,
        output_dir,
        data_root=args.data_root,
        label="static_8k",
        target_timestamps_ns=None,
        sample_count=args.sample_count,
        overlap_start_seconds=args.window_start_seconds,
        overlap_end_seconds=args.window_end_seconds,
    )
    if result["checkpoint_step"] != args.expected_checkpoint_step:
        raise RuntimeError(
            "unexpected checkpoint step "
            f"{result['checkpoint_step']} != {args.expected_checkpoint_step}"
        )

    import numpy as np
    from PIL import Image

    ground_truth_dir = output_dir / "ground_truth"
    mask_dir = output_dir / "removal_masks"
    ground_truth_dir.mkdir()
    mask_dir.mkdir()
    mask_fractions: list[float] = []
    for timestamp_ns in result["target_timestamps_ns"]:
        rendered = result["images"][timestamp_ns]
        height, width = rendered.shape[:2]
        valid_mask_path = (
            args.traffic_mask_root.expanduser().resolve()
            / REFERENCE_LOG
            / "sensors"
            / "cameras"
            / CAMERA_NAME
            / f"{timestamp_ns}.png"
        )
        if not valid_mask_path.is_file():
            raise FileNotFoundError(valid_mask_path)
        removal_mask = align_removal_mask(
            valid_mask_path,
            output_width=width,
            output_height=height,
            image_module=Image,
            np=np,
        )
        mask_fractions.append(float(np.mean(removal_mask > 0)))
        Image.fromarray(result["ground_truth"][timestamp_ns]).save(
            ground_truth_dir / f"{timestamp_ns}.jpg", quality=95
        )
        Image.fromarray(removal_mask, mode="L").save(
            mask_dir / f"{timestamp_ns}.png", optimize=True
        )

    report = {
        "format": "driving_scene_reconstruction.generative_repair_input.v0",
        "scope": (
            "Observed-pose static-8k renders plus aligned traffic masks for an "
            "offline video-inpainting smoke; not a simulator repair result."
        ),
        "reference_log": REFERENCE_LOG,
        "camera": CAMERA_NAME,
        "window_absolute_seconds": [
            args.window_start_seconds,
            args.window_end_seconds,
        ],
        "frame_count": len(result["target_timestamps_ns"]),
        "target_timestamps_ns": list(result["target_timestamps_ns"]),
        "render_resolution_height_width": list(
            next(iter(result["images"].values())).shape[:2]
        ),
        "config": result["config"],
        "checkpoint": result["checkpoint"],
        "checkpoint_step": result["checkpoint_step"],
        "render_seconds": result["render_seconds"],
        "psnr_db_against_source_with_traffic": result["psnr_db"],
        "removal_mask_fraction": distribution(mask_fractions),
        "input_frames": str(output_dir / "static_8k"),
        "ground_truth_frames": str(ground_truth_dir),
        "removal_masks": str(mask_dir),
        "limitations": [
            "The masks come from source-image traffic detections, not manual "
            "pixel-perfect ghost labels.",
            "The source ground truth still contains traffic and therefore is "
            "not valid hidden-background ground truth inside removal masks.",
        ],
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
