#!/usr/bin/env python3
"""Compare a Stage H3 vehicle-object pilot with its fixed 8k baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERA_ORDER = (
    "front_left",
    "front",
    "front_right",
    "left",
    "back",
    "right",
)
MINIMUM_SIDES = (16, 32, 64, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("vehicle_crops"):
        raise ValueError(f"report has no vehicle crops: {path}")
    return report


def crop_key(item: dict[str, Any]) -> tuple[str, str, int]:
    return item["frame"], item["camera"], int(item["actor_id"])


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def heldout_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(
            np.mean(
                [
                    report["by_camera"][camera]["by_split"]["heldout"][metric][
                        "mean"
                    ]
                    for camera in CAMERA_ORDER
                ]
            )
        )
        for metric in ("psnr", "ssim", "lpips")
    }


def minimum_side(item: dict[str, Any]) -> int:
    left, top, right, bottom = item["bbox_xyxy"]
    return min(right - left, bottom - top)


def paired_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    minimum_side_pixels: int,
    motion: str | None = None,
) -> dict[str, Any]:
    selected = [
        pair
        for pair in pairs
        if minimum_side(pair[0]) >= minimum_side_pixels
        and (
            motion is None
            or (motion == "moving") == (not pair[0]["stationary"])
        )
    ]
    psnr_deltas = [
        candidate["vehicle_crop_psnr"] - baseline["vehicle_crop_psnr"]
        for baseline, candidate in selected
    ]
    lpips_deltas = [
        candidate["vehicle_crop_lpips"] - baseline["vehicle_crop_lpips"]
        for baseline, candidate in selected
    ]
    return {
        "minimum_side_pixels": minimum_side_pixels,
        "pairs": len(selected),
        "stationary_pairs": sum(pair[0]["stationary"] for pair in selected),
        "moving_pairs": sum(not pair[0]["stationary"] for pair in selected),
        "psnr_delta_db_candidate_minus_baseline": distribution(psnr_deltas),
        "lpips_delta_candidate_minus_baseline": distribution(lpips_deltas),
        "candidate_psnr_improvement_fraction": float(
            np.mean(np.asarray(psnr_deltas) > 0)
        )
        if psnr_deltas
        else 0.0,
        "candidate_lpips_improvement_fraction": float(
            np.mean(np.asarray(lpips_deltas) < 0)
        )
        if lpips_deltas
        else 0.0,
    }


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def comparison_tile(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> Image.Image:
    baseline_image = Image.open(baseline["comparison_image"]).convert("RGB")
    candidate_image = Image.open(candidate["comparison_image"]).convert("RGB")
    gt = baseline_image.crop((0, 0, 256, 256))
    baseline_prediction = baseline_image.crop((256, 0, 512, 256))
    candidate_prediction = candidate_image.crop((256, 0, 512, 256))
    canvas = Image.new("RGB", (768, 294), "black")
    canvas.paste(gt, (0, 0))
    canvas.paste(baseline_prediction, (256, 0))
    canvas.paste(candidate_prediction, (512, 0))
    draw = ImageDraw.Draw(canvas)
    header_font = font(16)
    draw.text((8, 7), "GT", fill=(255, 230, 60), font=header_font)
    draw.text((264, 7), "static 8k", fill=(255, 230, 60), font=header_font)
    draw.text((520, 7), "candidate 8k", fill=(255, 230, 60), font=header_font)
    label = (
        f"{baseline['camera']} f{baseline['frame']} actor {baseline['actor_id']} "
        f"{'stationary' if baseline['stationary'] else 'moving'}  "
        f"LPIPS {baseline['vehicle_crop_lpips']:.3f}→"
        f"{candidate['vehicle_crop_lpips']:.3f}  "
        f"PSNR {baseline['vehicle_crop_psnr']:.1f}→"
        f"{candidate['vehicle_crop_psnr']:.1f}"
    )
    draw.text((8, 266), label, fill="white", font=header_font)
    return canvas


def contact_sheet(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    output_path: Path,
    title: str,
) -> None:
    tiles = [comparison_tile(baseline, candidate) for baseline, candidate in pairs]
    width = 768 * 2
    height = 48 + 294 * ((len(tiles) + 1) // 2)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    ImageDraw.Draw(canvas).text(
        (16, 10), title, fill="white", font=font(24)
    )
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 2) * 768, 48 + (index // 2) * 294))
    canvas.save(output_path, quality=94)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_report(args.baseline_report)
    candidate = load_report(args.candidate_report)

    baseline_crops = {
        crop_key(item): item for item in baseline["vehicle_crops"]
    }
    candidate_crops = {
        crop_key(item): item for item in candidate["vehicle_crops"]
    }
    if set(baseline_crops) != set(candidate_crops):
        raise RuntimeError("baseline and candidate vehicle crop keys differ")
    pairs = [
        (baseline_crops[key], candidate_crops[key])
        for key in sorted(baseline_crops)
    ]
    for baseline_crop, candidate_crop in pairs:
        if baseline_crop["bbox_xyxy"] != candidate_crop["bbox_xyxy"]:
            raise RuntimeError(f"crop bounds differ: {crop_key(baseline_crop)}")

    near_pairs = [pair for pair in pairs if minimum_side(pair[0]) >= 64]
    regressions = sorted(
        near_pairs,
        key=lambda pair: (
            pair[1]["vehicle_crop_lpips"] - pair[0]["vehicle_crop_lpips"]
        ),
        reverse=True,
    )[:12]
    improvements = sorted(
        near_pairs,
        key=lambda pair: (
            pair[1]["vehicle_crop_lpips"] - pair[0]["vehicle_crop_lpips"]
        ),
    )[:12]
    moving_pairs = [
        pair
        for pair in pairs
        if minimum_side(pair[0]) >= 32 and not pair[0]["stationary"]
    ]
    moving_regressions = sorted(
        moving_pairs,
        key=lambda pair: (
            pair[1]["vehicle_crop_lpips"] - pair[0]["vehicle_crop_lpips"]
        ),
        reverse=True,
    )[:12]
    regression_sheet = args.output_dir / "vehicle_crop_worst_regressions.jpg"
    improvement_sheet = args.output_dir / "vehicle_crop_best_improvements.jpg"
    moving_regression_sheet = (
        args.output_dir / "moving_vehicle_crop_worst_regressions.jpg"
    )
    contact_sheet(
        regressions,
        regression_sheet,
        "Largest near-vehicle LPIPS regressions (candidate minus baseline)",
    )
    contact_sheet(
        improvements,
        improvement_sheet,
        "Largest near-vehicle LPIPS improvements (candidate minus baseline)",
    )
    contact_sheet(
        moving_regressions,
        moving_regression_sheet,
        "Largest moving-vehicle LPIPS regressions (candidate minus baseline)",
    )

    baseline_heldout = heldout_metrics(baseline)
    candidate_heldout = heldout_metrics(candidate)
    report = {
        "baseline_report": str(args.baseline_report),
        "candidate_report": str(args.candidate_report),
        "crop_pair_count": len(pairs),
        "full_heldout_48_view": {
            "baseline": baseline_heldout,
            "candidate": candidate_heldout,
            "candidate_minus_baseline": {
                metric: candidate_heldout[metric] - baseline_heldout[metric]
                for metric in baseline_heldout
            },
        },
        "paired_vehicle_crops": {
            str(side): paired_summary(pairs, side)
            for side in MINIMUM_SIDES
        },
        "paired_moving_vehicle_crops": {
            str(side): paired_summary(pairs, side, motion="moving")
            for side in MINIMUM_SIDES
        },
        "paired_stationary_vehicle_crops": {
            str(side): paired_summary(pairs, side, motion="stationary")
            for side in MINIMUM_SIDES
        },
        "temporal_excess_warp_p95": {
            camera: {
                "baseline": baseline["by_camera"][camera]["excess_warp_mae"][
                    "p95"
                ],
                "candidate": candidate["by_camera"][camera]["excess_warp_mae"][
                    "p95"
                ],
            }
            for camera in CAMERA_ORDER
        },
        "finite_views": {
            "baseline": baseline["finite_failures"] == 0,
            "candidate": candidate["finite_failures"] == 0,
        },
        "render_latency_p95_ms": {
            "baseline": baseline["render_latency_ms"]["p95"],
            "candidate": candidate["render_latency_ms"]["p95"],
        },
        "visuals": {
            "worst_regressions": str(regression_sheet),
            "best_improvements": str(improvement_sheet),
            "moving_worst_regressions": str(moving_regression_sheet),
        },
        "decision": {
            "candidate_accepted": False,
            "reason": (
                "The candidate does not improve paired vehicle crops without "
                "regressing full held-out quality; it is not accepted as the "
                "new baseline."
            ),
        },
    }
    output_path = args.output_dir / "vehicle_pilot_comparison.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE: {output_path}")


if __name__ == "__main__":
    main()
