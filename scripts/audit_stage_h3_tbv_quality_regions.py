#!/usr/bin/env python3
"""Audit static background and safely recovered traffic regions separately."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution needs at least one value")
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def aligned_array(
    path: Path,
    *,
    width: int,
    height: int,
    grayscale: bool,
) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("L" if grayscale else "RGB")
    source_crop_height = round(height * image.width / width)
    image = image.crop(
        (0, 0, image.width, min(image.height, source_crop_height))
    )
    if image.size != (width, height):
        resampling = (
            Image.Resampling.NEAREST
            if grayscale
            else Image.Resampling.LANCZOS
        )
        image = image.resize((width, height), resampling)
    return np.asarray(image)


def region_metrics(
    rendered: Any,
    reference: Any,
    region: Any,
) -> dict[str, float]:
    import numpy as np

    count = int(region.sum())
    if count < 1:
        raise ValueError("metric region is empty")
    difference = (
        rendered[region].astype(np.float32)
        - reference[region].astype(np.float32)
    )
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference * difference))
    gradient_errors = []
    rendered_detail = []
    reference_detail = []
    for axis in (0, 1):
        if axis == 0:
            pair_region = region[:-1, :] & region[1:, :]
        else:
            pair_region = region[:, :-1] & region[:, 1:]
        rendered_gradient = np.diff(
            rendered.astype(np.float32), axis=axis
        )
        reference_gradient = np.diff(
            reference.astype(np.float32), axis=axis
        )
        if bool(pair_region.any()):
            gradient_errors.append(
                float(
                    np.mean(
                        np.abs(
                            rendered_gradient[pair_region]
                            - reference_gradient[pair_region]
                        )
                    )
                )
            )
            rendered_detail.append(
                float(np.mean(np.abs(rendered_gradient[pair_region])))
            )
            reference_detail.append(
                float(np.mean(np.abs(reference_gradient[pair_region])))
            )
    mean_rendered_detail = float(np.mean(rendered_detail))
    mean_reference_detail = float(np.mean(reference_detail))
    return {
        "pixel_count": count,
        "mae": mae,
        "psnr_db": (
            float("inf")
            if mse == 0.0
            else 10.0 * math.log10(255.0**2 / mse)
        ),
        "gradient_mae": float(np.mean(gradient_errors)),
        "detail_retention_ratio": (
            mean_rendered_detail / mean_reference_detail
            if mean_reference_detail > 0.0
            else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-ab-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--traffic-mask-root", type=Path, required=True)
    parser.add_argument("--cross-visit-root", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report_path = args.quality_ab_json.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = report_path.parent
    data_root = args.data_root.expanduser().resolve()
    traffic_root = args.traffic_mask_root.expanduser().resolve()
    cross_root = args.cross_visit_root.expanduser().resolve()
    width = int(report["alignment"]["target_height_width"][1])
    height = int(report["alignment"]["target_height_width"][0])
    log_id = report["reference_log"]
    camera = report["camera"]

    per_region: dict[str, dict[str, list[float]]] = {
        "observed_static_background": {
            f"{args.baseline_label}_mae": [],
            f"{args.baseline_label}_psnr_db": [],
            f"{args.baseline_label}_gradient_mae": [],
            f"{args.baseline_label}_detail_retention_ratio": [],
            f"{args.candidate_label}_mae": [],
            f"{args.candidate_label}_psnr_db": [],
            f"{args.candidate_label}_gradient_mae": [],
            f"{args.candidate_label}_detail_retention_ratio": [],
        },
        "cross_visit_safe_fill": {
            f"{args.baseline_label}_mae": [],
            f"{args.baseline_label}_psnr_db": [],
            f"{args.baseline_label}_gradient_mae": [],
            f"{args.baseline_label}_detail_retention_ratio": [],
            f"{args.candidate_label}_mae": [],
            f"{args.candidate_label}_psnr_db": [],
            f"{args.candidate_label}_gradient_mae": [],
            f"{args.candidate_label}_detail_retention_ratio": [],
        },
    }
    frames = []
    for timestamp_ns in report["target_timestamps_ns"]:
        relative = (
            Path(log_id)
            / "sensors"
            / "cameras"
            / camera
            / f"{timestamp_ns}.jpg"
        )
        raw_path = data_root / relative
        original_mask_path = (
            traffic_root / relative.with_suffix(".png")
        )
        recovered_rgb_path = cross_root / "rgb" / relative
        residual_mask_path = (
            cross_root / "valid_masks" / relative.with_suffix(".png")
        )
        baseline_path = (
            output_dir / args.baseline_label / f"{timestamp_ns}.jpg"
        )
        candidate_path = (
            output_dir / args.candidate_label / f"{timestamp_ns}.jpg"
        )
        required = (
            raw_path,
            original_mask_path,
            recovered_rgb_path,
            residual_mask_path,
            baseline_path,
            candidate_path,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

        raw = aligned_array(
            raw_path, width=width, height=height, grayscale=False
        )
        recovered = aligned_array(
            recovered_rgb_path,
            width=width,
            height=height,
            grayscale=False,
        )
        original_valid = (
            aligned_array(
                original_mask_path,
                width=width,
                height=height,
                grayscale=True,
            )
            >= 128
        )
        residual_valid = (
            aligned_array(
                residual_mask_path,
                width=width,
                height=height,
                grayscale=True,
            )
            >= 128
        )
        baseline = aligned_array(
            baseline_path, width=width, height=height, grayscale=False
        )
        candidate = aligned_array(
            candidate_path, width=width, height=height, grayscale=False
        )
        safe_fill = (~original_valid) & residual_valid
        static_metrics = {
            args.baseline_label: region_metrics(
                baseline, raw, original_valid
            ),
            args.candidate_label: region_metrics(
                candidate, raw, original_valid
            ),
        }
        fill_metrics = None
        if bool(safe_fill.any()):
            fill_metrics = {
                args.baseline_label: region_metrics(
                    baseline, recovered, safe_fill
                ),
                args.candidate_label: region_metrics(
                    candidate, recovered, safe_fill
                ),
            }
        frames.append(
            {
                "timestamp_ns": timestamp_ns,
                "static_pixel_count": int(original_valid.sum()),
                "safe_fill_pixel_count": int(safe_fill.sum()),
                "observed_static_background": static_metrics,
                "cross_visit_safe_fill": fill_metrics,
            }
        )
        for label, metrics in static_metrics.items():
            for metric in (
                "mae",
                "psnr_db",
                "gradient_mae",
                "detail_retention_ratio",
            ):
                per_region["observed_static_background"][
                    f"{label}_{metric}"
                ].append(metrics[metric])
        if fill_metrics is not None:
            for label, metrics in fill_metrics.items():
                for metric in (
                    "mae",
                    "psnr_db",
                    "gradient_mae",
                    "detail_retention_ratio",
                ):
                    per_region["cross_visit_safe_fill"][
                        f"{label}_{metric}"
                    ].append(metrics[metric])

    summary = {
        region: {
            metric: distribution(values)
            for metric, values in metrics.items()
        }
        for region, metrics in per_region.items()
    }
    result = {
        "format": "driving_scene_reconstruction.tbv_quality_regions.v0",
        "scope": (
            "Observed static pixels use the original source as reference. "
            "Safe-fill pixels use only cross-visit donor-supported RGB; "
            "they are evidence-backed proxies, not hidden ground truth."
        ),
        "quality_ab_json": str(report_path),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "summary": summary,
        "frames": frames,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
