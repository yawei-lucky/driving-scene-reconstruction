#!/usr/bin/env python3
"""Compare two TbV checkpoints at identical observed front-camera poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings
from typing import Any

from probe_stage_h3_tbv_tile_seam import (
    CAMERA_NAME,
    REFERENCE_LOG,
    distribution,
    render_tile,
)


def resize_rgb(image: Any, shape: tuple[int, int], image_module: Any) -> Any:
    """Resize one RGB uint8 array to ``(height, width)``."""

    import numpy as np

    height, width = shape
    if image.shape[:2] == shape:
        return image
    return np.asarray(
        image_module.fromarray(image).resize(
            (width, height),
            image_module.Resampling.LANCZOS,
        )
    )


def image_quality(rendered: Any, ground_truth: Any, np: Any) -> dict[str, float]:
    """Return intensity and first-difference metrics for one image pair."""

    rendered_float = rendered.astype(np.float32)
    ground_truth_float = ground_truth.astype(np.float32)
    error = rendered_float - ground_truth_float
    mse = float(np.mean(error * error))
    psnr = (
        float("inf")
        if mse == 0.0
        else 10.0 * math.log10(255.0**2 / mse)
    )
    rendered_gradients = (
        np.diff(rendered_float, axis=0),
        np.diff(rendered_float, axis=1),
    )
    ground_truth_gradients = (
        np.diff(ground_truth_float, axis=0),
        np.diff(ground_truth_float, axis=1),
    )
    gradient_mae = float(
        sum(
            np.mean(np.abs(rendered_gradient - ground_truth_gradient))
            for rendered_gradient, ground_truth_gradient in zip(
                rendered_gradients, ground_truth_gradients
            )
        )
        / len(rendered_gradients)
    )
    rendered_detail = float(
        sum(np.mean(np.abs(value)) for value in rendered_gradients)
        / len(rendered_gradients)
    )
    ground_truth_detail = float(
        sum(np.mean(np.abs(value)) for value in ground_truth_gradients)
        / len(ground_truth_gradients)
    )
    return {
        "psnr_db": psnr,
        "gradient_mae": gradient_mae,
        "detail_retention_ratio": (
            rendered_detail / ground_truth_detail
            if ground_truth_detail > 0.0
            else 1.0
        ),
    }


def aligned_metrics(
    result: dict[str, Any],
    *,
    target_shape: tuple[int, int],
    reference_ground_truth: dict[int, Any],
    np: Any,
    image_module: Any,
) -> dict[str, dict[str, float]]:
    """Evaluate a result after matching a common display resolution."""

    per_metric: dict[str, list[float]] = {
        "psnr_db": [],
        "gradient_mae": [],
        "detail_retention_ratio": [],
    }
    for timestamp_ns in result["target_timestamps_ns"]:
        rendered = resize_rgb(
            result["images"][timestamp_ns], target_shape, image_module
        )
        metrics = image_quality(
            rendered,
            reference_ground_truth[timestamp_ns],
            np,
        )
        for name, value in metrics.items():
            per_metric[name].append(value)
    return {
        name: distribution(values) for name, values in per_metric.items()
    }


def build_contact_sheet(
    output_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    timestamps = tuple(baseline["target_timestamps_ns"])
    target_shape = next(iter(baseline["images"].values())).shape[:2]
    cell_width, cell_height = 512, 340
    labels = (
        "source GT",
        baseline_label,
        candidate_label,
        "candidate - baseline",
    )
    sheet = Image.new(
        "RGB",
        (cell_width * len(labels), cell_height * len(timestamps)),
        "black",
    )
    for row, timestamp_ns in enumerate(timestamps):
        first = baseline["images"][timestamp_ns]
        second = resize_rgb(
            candidate["images"][timestamp_ns], target_shape, Image
        )
        difference = np.abs(
            second.astype(np.int16) - first.astype(np.int16)
        ).astype(np.uint8)
        frames = (
            baseline["ground_truth"][timestamp_ns],
            first,
            second,
            difference,
        )
        for column, (label, frame) in enumerate(zip(labels, frames)):
            panel = ImageOps.contain(
                Image.fromarray(frame).convert("RGB"),
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 38), fill=(0, 0, 0))
            draw.text(
                (8, 7),
                f"{label} | {timestamp_ns}",
                fill=(255, 230, 80),
            )
            x = column * cell_width + (cell_width - panel.width) // 2
            y = row * cell_height + (cell_height - panel.height) // 2
            sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"av2\.utils\.io",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--baseline-data-root", type=Path)
    parser.add_argument("--candidate-data-root", type=Path)
    parser.add_argument("--baseline-label", default="static_8k")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--window-start-seconds", type=float, required=True)
    parser.add_argument("--window-end-seconds", type=float, required=True)
    parser.add_argument("--sample-count", type=int, default=7)
    parser.add_argument(
        "--expected-baseline-step", type=int, required=True
    )
    parser.add_argument(
        "--expected-candidate-step", type=int, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if (
        not math.isfinite(args.window_start_seconds)
        or not math.isfinite(args.window_end_seconds)
        or args.window_end_seconds <= args.window_start_seconds
    ):
        parser.error("window end must be finite and greater than start")
    for path in (args.baseline_config, args.candidate_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = render_tile(
        args.baseline_config,
        output_dir,
        data_root=args.baseline_data_root,
        label=args.baseline_label,
        target_timestamps_ns=None,
        sample_count=args.sample_count,
        overlap_start_seconds=args.window_start_seconds,
        overlap_end_seconds=args.window_end_seconds,
    )
    candidate = render_tile(
        args.candidate_config,
        output_dir,
        data_root=args.candidate_data_root,
        label=args.candidate_label,
        target_timestamps_ns=tuple(baseline["target_timestamps_ns"]),
        sample_count=args.sample_count,
        overlap_start_seconds=args.window_start_seconds,
        overlap_end_seconds=args.window_end_seconds,
    )

    import numpy as np
    from PIL import Image

    target_shape = next(iter(baseline["images"].values())).shape[:2]
    baseline_metrics = aligned_metrics(
        baseline,
        target_shape=target_shape,
        reference_ground_truth=baseline["ground_truth"],
        np=np,
        image_module=Image,
    )
    candidate_metrics = aligned_metrics(
        candidate,
        target_shape=target_shape,
        reference_ground_truth=baseline["ground_truth"],
        np=np,
        image_module=Image,
    )
    pairwise_mae = []
    for timestamp_ns in baseline["target_timestamps_ns"]:
        candidate_aligned = resize_rgb(
            candidate["images"][timestamp_ns], target_shape, Image
        )
        pairwise_mae.append(
            float(
                np.mean(
                    np.abs(
                        candidate_aligned.astype(np.float32)
                        - baseline["images"][timestamp_ns].astype(np.float32)
                    )
                )
            )
        )

    contact_sheet = output_dir / "quality_ab_contact.jpg"
    build_contact_sheet(
        contact_sheet,
        baseline,
        candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    baseline_shape = list(
        next(iter(baseline["images"].values())).shape[:2]
    )
    candidate_shape = list(
        next(iter(candidate["images"].values())).shape[:2]
    )
    report = {
        "format": "driving_scene_reconstruction.tbv_quality_ab.v0",
        "scope": (
            "Identical observed front-camera pose comparison. Metrics are "
            "aligned to the baseline display resolution; visual review is "
            "still required for vehicle ghosts and vegetation."
        ),
        "reference_log": REFERENCE_LOG,
        "camera": CAMERA_NAME,
        "window_absolute_seconds": [
            args.window_start_seconds,
            args.window_end_seconds,
        ],
        "target_timestamps_ns": list(baseline["target_timestamps_ns"]),
        "alignment": {
            "target_height_width": list(target_shape),
            "resampler": "PIL LANCZOS",
            "baseline_native_height_width": baseline_shape,
            "candidate_native_height_width": candidate_shape,
        },
        args.baseline_label: {
            key: value
            for key, value in baseline.items()
            if key not in ("images", "ground_truth")
        },
        args.candidate_label: {
            key: value
            for key, value in candidate.items()
            if key not in ("images", "ground_truth")
        },
        "aligned_quality": {
            args.baseline_label: baseline_metrics,
            args.candidate_label: candidate_metrics,
        },
        "pairwise_rgb_mae": distribution(pairwise_mae),
        "contact_sheet": str(contact_sheet),
        "technical_gates": {
            "checkpoint_steps_match": (
                baseline["checkpoint_step"]
                == args.expected_baseline_step
                and candidate["checkpoint_step"]
                == args.expected_candidate_step
            ),
            "matched_requested_world_poses": (
                len(baseline["target_timestamps_ns"])
                == min(
                    args.sample_count,
                    baseline["available_overlap_frame_count"],
                )
            ),
            "both_models_use_metre_scale": (
                baseline["dataparser_scale"] == 1.0
                and candidate["dataparser_scale"] == 1.0
            ),
            "all_views_not_mostly_black": (
                baseline["near_black_fraction"]["max"] < 0.25
                and candidate["near_black_fraction"]["max"] < 0.25
            ),
            "candidate_native_area_not_smaller": (
                candidate_shape[0] * candidate_shape[1]
                >= baseline_shape[0] * baseline_shape[1]
            ),
        },
        "visual_status": "requires_manual_review",
        "limitations": [
            (
                "PSNR includes traffic pixels even though candidate training "
                "excluded detected traffic; a visually cleaner road can score "
                "worse against a source frame containing a vehicle."
            ),
            "Only observed front-camera poses in one bounded tile are tested.",
            (
                "This is a bounded method comparison; passing technical "
                "gates does not make either checkpoint a driving-quality "
                "acceptance result."
            ),
        ],
    }
    report["technical_status"] = (
        "pass" if all(report["technical_gates"].values()) else "fail"
    )
    report_path = output_dir / "quality_ab.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["technical_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
