#!/usr/bin/env python3
"""Recover masked TbV background pixels from a second observed traversal.

This is intentionally conservative: a traffic pixel is promoted back into the
RGB loss only when the other traversal has an unmasked observation and a
feature-based homography passes explicit alignment gates. Pixels without safe
donor support remain masked instead of being synthetically invented.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Sequence
import warnings


CAMERA_TO_BOTTOM_CROP = {
    "ring_front_center": 250,
    "ring_front_left": 0,
    "ring_front_right": 0,
    "ring_rear_left": 250,
    "ring_rear_right": 250,
    "ring_side_left": 0,
    "ring_side_right": 0,
}


def evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    count = min(length, count)
    if count == 1:
        return (length // 2,)
    return tuple(
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    )


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution needs at least one value")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def angle_difference_degrees(first: float, second: float) -> float:
    delta = (first - second + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def mask_path_for_image(
    image_path: Path,
    *,
    data_root: Path,
    mask_root: Path,
) -> Path:
    relative = image_path.absolute().relative_to(data_root.absolute())
    return mask_root.absolute() / relative.with_suffix(".png")


def output_path_for_image(
    image_path: Path,
    *,
    data_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    relative = image_path.absolute().relative_to(data_root.absolute())
    if suffix == ".jpg":
        return output_root.absolute() / "rgb" / relative
    return output_root.absolute() / "valid_masks" / relative.with_suffix(
        suffix
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose_record(
    loader: Any,
    image_path: Path,
    sequence: str,
) -> dict[str, Any]:
    import numpy as np

    timestamp_ns = int(image_path.stem)
    transform = loader.get_city_SE3_ego(
        sequence, timestamp_ns
    ).transform_matrix
    return {
        "path": image_path,
        "timestamp_ns": timestamp_ns,
        "xy": np.asarray(transform[:2, 3], dtype=np.float64),
        "heading_rad": math.atan2(transform[1, 0], transform[0, 0]),
    }


def _candidate_pose_score(
    target: dict[str, Any],
    donor: dict[str, Any],
) -> tuple[float, float, float]:
    import numpy as np

    distance = float(np.linalg.norm(target["xy"] - donor["xy"]))
    yaw = angle_difference_degrees(
        target["heading_rad"], donor["heading_rad"]
    )
    return distance + 0.10 * yaw, distance, yaw


def _estimate_alignment(
    target_image: Any,
    donor_image: Any,
    target_valid: Any,
    donor_valid: Any,
    *,
    max_features: int,
    ratio_threshold: float,
    ransac_threshold: float,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    kernel = np.ones((11, 11), dtype=np.uint8)
    target_features_valid = cv2.erode(target_valid, kernel)
    donor_features_valid = cv2.erode(donor_valid, kernel)
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.02,
        edgeThreshold=12,
    )
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    donor_gray = cv2.cvtColor(donor_image, cv2.COLOR_BGR2GRAY)
    target_keypoints, target_descriptors = sift.detectAndCompute(
        target_gray, target_features_valid
    )
    donor_keypoints, donor_descriptors = sift.detectAndCompute(
        donor_gray, donor_features_valid
    )
    if target_descriptors is None or donor_descriptors is None:
        return {"accepted": False, "reason": "no_descriptors"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(donor_descriptors, target_descriptors, k=2)
    matches = [
        first
        for first, second in pairs
        if first.distance < ratio_threshold * second.distance
    ]
    if len(matches) < 20:
        return {
            "accepted": False,
            "reason": "too_few_ratio_matches",
            "ratio_match_count": len(matches),
        }
    donor_points = np.float32(
        [donor_keypoints[match.queryIdx].pt for match in matches]
    )
    target_points = np.float32(
        [target_keypoints[match.trainIdx].pt for match in matches]
    )
    homography, inliers = cv2.findHomography(
        donor_points,
        target_points,
        cv2.RANSAC,
        ransac_threshold,
    )
    if homography is None or inliers is None:
        return {
            "accepted": False,
            "reason": "homography_failed",
            "ratio_match_count": len(matches),
        }
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_fraction = inlier_count / len(matches)
    projected = cv2.perspectiveTransform(
        donor_points[inlier_mask, None, :], homography
    )[:, 0, :]
    errors = np.linalg.norm(
        projected - target_points[inlier_mask], axis=1
    )
    median_error = float(np.median(errors))
    accepted = (
        inlier_count >= 18
        and (inlier_fraction >= 0.35 or inlier_count >= 35)
        and median_error <= 2.5
    )
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "alignment_gate_failed",
        "homography": homography,
        "ratio_match_count": len(matches),
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
        "median_reprojection_error_px": median_error,
    }


def _warp_and_fill(
    target_image: Any,
    donor_image: Any,
    target_valid: Any,
    donor_valid: Any,
    homography: Any,
    *,
    training_height: int,
    seam_guard_pixels: float,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    height, width = target_valid.shape
    warped_donor = cv2.warpPerspective(
        donor_image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_valid = cv2.warpPerspective(
        donor_valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    coverage = cv2.warpPerspective(
        np.full_like(donor_valid, 255),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    donor_safe = cv2.erode(
        np.where((warped_valid >= 128) & (coverage >= 128), 255, 0).astype(
            np.uint8
        ),
        np.ones((7, 7), dtype=np.uint8),
    )
    training_region = np.zeros_like(target_valid, dtype=bool)
    training_region[:training_height] = True
    target_excluded = (target_valid < 128) & training_region
    possible_fill = target_excluded & (donor_safe >= 128)
    distance = cv2.distanceTransform(
        possible_fill.astype(np.uint8), cv2.DIST_L2, 5
    )
    accepted_fill = possible_fill & (distance >= seam_guard_pixels)

    static_overlap = (
        (target_valid >= 128)
        & (donor_safe >= 128)
        & training_region
    )
    color_offset = np.zeros(3, dtype=np.float32)
    if int(static_overlap.sum()) >= 1000:
        differences = (
            target_image[static_overlap].astype(np.float32)
            - warped_donor[static_overlap].astype(np.float32)
        )
        color_offset = np.clip(
            np.median(differences, axis=0), -20.0, 20.0
        )
    adjusted_donor = np.clip(
        warped_donor.astype(np.float32) + color_offset,
        0.0,
        255.0,
    ).astype(np.uint8)
    composite = target_image.copy()
    composite[accepted_fill] = adjusted_donor[accepted_fill]
    output_valid = target_valid.copy()
    output_valid[accepted_fill] = 255

    if bool(static_overlap.any()):
        photometric_mae = float(
            np.mean(
                np.abs(
                    target_image[static_overlap].astype(np.float32)
                    - adjusted_donor[static_overlap].astype(np.float32)
                )
            )
        )
    else:
        photometric_mae = float("inf")
    excluded_count = int(target_excluded.sum())
    filled_count = int(accepted_fill.sum())
    return {
        "composite": composite,
        "output_valid": output_valid,
        "warped_donor": adjusted_donor,
        "target_excluded_count": excluded_count,
        "filled_count": filled_count,
        "filled_fraction_of_excluded": (
            filled_count / excluded_count if excluded_count else 1.0
        ),
        "residual_excluded_fraction": float(
            np.mean((output_valid[:training_height] < 128))
        ),
        "photometric_mae": photometric_mae,
        "color_offset_bgr": color_offset.tolist(),
    }


def _write_image(path: Path, image: Any, parameters: Sequence[int]) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.part{path.suffix}")
    if not cv2.imwrite(str(temporary), image, list(parameters)):
        raise RuntimeError(f"failed to write {temporary}")
    os.replace(temporary, path)


def _reuse_existing_target(
    target: dict[str, Any],
    *,
    data_root: Path,
    mask_root: Path,
    output_root: Path,
) -> dict[str, Any] | None:
    import cv2
    import numpy as np

    target_path = target["path"]
    target_mask_path = mask_path_for_image(
        target_path, data_root=data_root, mask_root=mask_root
    )
    rgb_path = output_path_for_image(
        target_path,
        data_root=data_root,
        output_root=output_root,
        suffix=".jpg",
    )
    valid_path = output_path_for_image(
        target_path,
        data_root=data_root,
        output_root=output_root,
        suffix=".png",
    )
    if not rgb_path.is_file() or not valid_path.is_file():
        return None
    target_valid = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
    output_valid = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
    if target_valid is None or output_valid is None:
        return None
    camera_name = target_path.parent.name
    training_height = (
        target_valid.shape[0] - CAMERA_TO_BOTTOM_CROP[camera_name]
    )
    target_excluded = target_valid[:training_height] < 128
    residual_excluded = output_valid[:training_height] < 128
    target_excluded_count = int(target_excluded.sum())
    filled_count = target_excluded_count - int(residual_excluded.sum())
    return {
        "source": str(target_path),
        "source_mask": str(target_mask_path),
        "output_rgb": str(rgb_path),
        "output_valid_mask": str(valid_path),
        "status": (
            "reused_with_fill"
            if filled_count > 0
            else "reused_without_fill"
        ),
        "selected_donor": None,
        "selected_donors": [],
        "selected_attempt": None,
        "attempts": [],
        "target_excluded_fraction": float(np.mean(target_excluded)),
        "filled_count": filled_count,
        "filled_fraction_of_excluded": (
            filled_count / target_excluded_count
            if target_excluded_count
            else 1.0
        ),
        "residual_excluded_fraction": float(
            np.mean(residual_excluded)
        ),
        "elapsed_seconds": 0.0,
        "rgb_bytes": rgb_path.stat().st_size,
        "rgb_sha256": _sha256(rgb_path),
        "mask_bytes": valid_path.stat().st_size,
        "mask_sha256": _sha256(valid_path),
        "_warped_donor": None,
    }


def _process_target(
    target: dict[str, Any],
    donors: Sequence[dict[str, Any]],
    *,
    data_root: Path,
    mask_root: Path,
    output_root: Path,
    max_pose_distance: float,
    max_yaw_degrees: float,
    donor_candidates: int,
    max_features: int,
    ratio_threshold: float,
    ransac_threshold: float,
    seam_guard_pixels: float,
    jpeg_quality: int,
    keep_preview: bool,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    started = time.perf_counter()
    target_path = target["path"]
    target_mask_path = mask_path_for_image(
        target_path, data_root=data_root, mask_root=mask_root
    )
    target_image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    target_valid = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
    if target_image is None or target_valid is None:
        raise RuntimeError(f"failed to decode {target_path}")
    if target_image.shape[:2] != target_valid.shape:
        raise RuntimeError(f"image/mask shape mismatch for {target_path}")
    camera_name = target_path.parent.name
    training_height = (
        target_image.shape[0] - CAMERA_TO_BOTTOM_CROP[camera_name]
    )

    ranked = sorted(
        (
            (*_candidate_pose_score(target, donor), donor)
            for donor in donors
        ),
        key=lambda item: item[0],
    )
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for _, pose_distance, yaw_degrees, donor in ranked[:donor_candidates]:
        attempt: dict[str, Any] = {
            "donor": str(donor["path"]),
            "pose_distance_m": pose_distance,
            "yaw_difference_degrees": yaw_degrees,
        }
        if (
            pose_distance > max_pose_distance
            or yaw_degrees > max_yaw_degrees
        ):
            attempt.update(accepted=False, reason="pose_gate_failed")
            attempts.append(attempt)
            continue
        donor_mask_path = mask_path_for_image(
            donor["path"], data_root=data_root, mask_root=mask_root
        )
        donor_image = cv2.imread(str(donor["path"]), cv2.IMREAD_COLOR)
        donor_valid = cv2.imread(
            str(donor_mask_path), cv2.IMREAD_GRAYSCALE
        )
        if donor_image is None or donor_valid is None:
            raise RuntimeError(f"failed to decode {donor['path']}")
        alignment = _estimate_alignment(
            target_image,
            donor_image,
            target_valid,
            donor_valid,
            max_features=max_features,
            ratio_threshold=ratio_threshold,
            ransac_threshold=ransac_threshold,
        )
        homography = alignment.pop("homography", None)
        attempt.update(alignment)
        if not alignment["accepted"]:
            attempts.append(attempt)
            continue
        fill = _warp_and_fill(
            target_image,
            donor_image,
            target_valid,
            donor_valid,
            homography,
            training_height=training_height,
            seam_guard_pixels=seam_guard_pixels,
        )
        attempt.update(
            {
                key: value
                for key, value in fill.items()
                if key
                not in {
                    "composite",
                    "output_valid",
                    "warped_donor",
                }
            }
        )
        attempt["_images"] = (
            fill["composite"],
            fill["output_valid"],
            fill["warped_donor"],
        )
        attempts.append(attempt)
        accepted.append(attempt)

    if accepted:
        ordered_accepted = sorted(
            accepted,
            key=lambda attempt: (
                attempt["photometric_mae"],
                -attempt["filled_count"],
            ),
        )
        selected = max(
            accepted,
            key=lambda attempt: (
                attempt["filled_count"],
                -attempt["photometric_mae"],
            ),
        )
        composite = target_image.copy()
        output_valid = target_valid.copy()
        warped_donor = selected["_images"][2]
        selected_donors = []
        for attempt in ordered_accepted:
            candidate_composite, candidate_valid, _ = attempt["_images"]
            new_support = (
                (output_valid < 128) & (candidate_valid >= 128)
            )
            if bool(new_support.any()):
                composite[new_support] = candidate_composite[new_support]
                output_valid[new_support] = 255
                selected_donors.append(attempt["donor"])
        for attempt in accepted:
            attempt.pop("_images", None)
        status = "accepted"
    else:
        selected = None
        selected_donors = []
        composite = target_image
        output_valid = target_valid
        warped_donor = np.zeros_like(target_image)
        status = "no_safe_donor"

    rgb_path = output_path_for_image(
        target_path,
        data_root=data_root,
        output_root=output_root,
        suffix=".jpg",
    )
    valid_path = output_path_for_image(
        target_path,
        data_root=data_root,
        output_root=output_root,
        suffix=".png",
    )
    _write_image(
        rgb_path,
        composite,
        (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality),
    )
    _write_image(valid_path, output_valid, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    target_excluded = (
        target_valid[:training_height] < 128
    )
    residual_excluded = (
        output_valid[:training_height] < 128
    )
    target_excluded_count = int(target_excluded.sum())
    combined_filled_count = target_excluded_count - int(
        residual_excluded.sum()
    )
    combined_filled_fraction = (
        combined_filled_count / target_excluded_count
        if target_excluded_count
        else 1.0
    )
    return {
        "source": str(target_path),
        "source_mask": str(target_mask_path),
        "output_rgb": str(rgb_path),
        "output_valid_mask": str(valid_path),
        "status": status,
        "selected_donor": (
            selected["donor"] if selected is not None else None
        ),
        "selected_donors": selected_donors,
        "selected_attempt": selected,
        "attempts": attempts,
        "target_excluded_fraction": float(
            np.mean(target_excluded)
        ),
        "filled_count": combined_filled_count,
        "filled_fraction_of_excluded": combined_filled_fraction,
        "residual_excluded_fraction": float(
            np.mean(residual_excluded)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "rgb_bytes": rgb_path.stat().st_size,
        "rgb_sha256": _sha256(rgb_path),
        "mask_bytes": valid_path.stat().st_size,
        "mask_sha256": _sha256(valid_path),
        "_warped_donor": warped_donor if keep_preview else None,
    }


def build_contact_sheet(
    output_path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    selected = [
        record
        for record in records
        if record["_warped_donor"] is not None
    ]
    if not selected:
        return
    cell_width, cell_height = 400, 310
    sheet = Image.new(
        "RGB", (cell_width * 5, cell_height * len(selected)), "black"
    )
    for row, record in enumerate(selected):
        source = Image.open(record["source"]).convert("RGB")
        valid = Image.open(record["source_mask"]).convert("L")
        excluded = valid.point(lambda value: 0 if value >= 128 else 180)
        overlay = Image.new("RGB", source.size, (255, 35, 35))
        masked = Image.composite(overlay, source, excluded)
        warped = Image.fromarray(
            record.pop("_warped_donor")[:, :, ::-1]
        ).convert("RGB")
        output = Image.open(record["output_rgb"]).convert("RGB")
        residual_valid = Image.open(
            record["output_valid_mask"]
        ).convert("L")
        residual_excluded = residual_valid.point(
            lambda value: 0 if value >= 128 else 180
        )
        supervised = Image.composite(
            overlay, output, residual_excluded
        )
        donor_name = (
            Path(record["selected_donor"]).stem
            if record["selected_donor"]
            else "none"
        )
        labels = (
            "source",
            "traffic mask",
            f"aligned other visit | {donor_name}",
            (
                f"safe fill {record['filled_fraction_of_excluded']:.1%} | "
                f"left {record['residual_excluded_fraction']:.1%}"
            ),
            "RGB supervision | residual red",
        )
        for column, (label, image) in enumerate(
            zip(labels, (source, masked, warped, output, supervised))
        ):
            panel = ImageOps.contain(
                image,
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 36), fill=(0, 0, 0))
            draw.text(
                (8, 7),
                f"{label} | {Path(record['source']).stem}",
                fill=(255, 230, 80),
            )
            x = column * cell_width + (cell_width - panel.width) // 2
            y = row * cell_height + (cell_height - panel.height) // 2
            sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"av2\.utils\.io",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument(
        "--window-start-seconds", action="append", type=float, required=True
    )
    parser.add_argument(
        "--window-end-seconds", action="append", type=float, required=True
    )
    parser.add_argument("--camera")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pose-distance", type=float, default=1.5)
    parser.add_argument("--max-yaw-degrees", type=float, default=6.0)
    parser.add_argument("--donor-candidates", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--ratio-threshold", type=float, default=0.72)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--seam-guard-pixels", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete RGB/mask pairs in a partial output directory",
    )
    args = parser.parse_args()
    lengths = (
        len(args.sequence),
        len(args.window_start_seconds),
        len(args.window_end_seconds),
    )
    if len(set(lengths)) != 1 or lengths[0] != 2:
        parser.error("exactly two equally bounded traversals are required")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.donor_candidates < 1 or args.max_features < 20:
        parser.error("candidate and feature counts must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0.0 < args.ratio_threshold < 1.0:
        parser.error("--ratio-threshold must be within (0, 1)")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be within [1, 100]")

    data_root = args.data_root.expanduser().absolute()
    mask_root = args.mask_root.expanduser().absolute()
    output_root = args.output_root.expanduser().absolute()
    if not data_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("data root and mask root must exist")
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not args.resume
    ):
        raise RuntimeError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    from av2.datasets.sensor.av2_sensor_dataloader import (
        AV2SensorDataLoader,
    )

    loader = AV2SensorDataLoader(data_root, data_root)
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for sequence, start, end in zip(
        args.sequence,
        args.window_start_seconds,
        args.window_end_seconds,
    ):
        records: list[dict[str, Any]] = []
        camera_names = (
            (args.camera,)
            if args.camera is not None
            else tuple(CAMERA_TO_BOTTOM_CROP)
        )
        for camera_name in camera_names:
            for image_path in loader.get_ordered_log_cam_fpaths(
                sequence, camera_name
            ):
                timestamp_seconds = int(image_path.stem) / 1e9
                if start <= timestamp_seconds <= end:
                    records.append(
                        _pose_record(loader, image_path, sequence)
                    )
        by_sequence[sequence] = sorted(
            records, key=lambda record: str(record["path"])
        )

    targets = sorted(
        (
            record
            for records in by_sequence.values()
            for record in records
        ),
        key=lambda record: str(record["path"]),
    )
    if args.limit is not None:
        targets = [
            targets[index]
            for index in evenly_spaced_indices(len(targets), args.limit)
        ]
    front_targets = [
        target
        for target in targets
        if target["path"].parent.name == "ring_front_center"
    ]
    preview_targets = front_targets if front_targets else targets
    preview_paths = {
        preview_targets[index]["path"]
        for index in evenly_spaced_indices(
            len(preview_targets), min(8, len(preview_targets))
        )
    }

    def process(target: dict[str, Any]) -> dict[str, Any]:
        if args.resume and target["path"] not in preview_paths:
            reused = _reuse_existing_target(
                target,
                data_root=data_root,
                mask_root=mask_root,
                output_root=output_root,
            )
            if reused is not None:
                return reused
        target_sequence = target["path"].parents[3].name
        donor_sequence = next(
            sequence
            for sequence in args.sequence
            if sequence != target_sequence
        )
        same_camera_donors = [
            donor
            for donor in by_sequence[donor_sequence]
            if donor["path"].parent.name == target["path"].parent.name
        ]
        return _process_target(
            target,
            same_camera_donors,
            data_root=data_root,
            mask_root=mask_root,
            output_root=output_root,
            max_pose_distance=args.max_pose_distance,
            max_yaw_degrees=args.max_yaw_degrees,
            donor_candidates=args.donor_candidates,
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            ransac_threshold=args.ransac_threshold,
            seam_guard_pixels=args.seam_guard_pixels,
            jpeg_quality=args.jpeg_quality,
            keep_preview=target["path"] in preview_paths,
        )

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, target): target for target in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if (
                index % 50 == 0
                or index == len(targets)
                or len(targets) <= 20
            ):
                print(
                    f"cross-visit {index}/{len(targets)} "
                    f"{Path(record['source']).stem}: {record['status']}, "
                    f"filled={record['filled_fraction_of_excluded']:.1%}"
                )
    records.sort(key=lambda record: record["source"])

    contact_sheet = output_root / "cross_visit_contact.jpg"
    build_contact_sheet(contact_sheet, records)
    for record in records:
        record.pop("_warped_donor", None)
    accepted = [
        record
        for record in records
        if record["status"] in {"accepted", "reused_with_fill"}
    ]
    reused_count = sum(
        record["status"].startswith("reused") for record in records
    )
    report = {
        "format": "driving_scene_reconstruction.tbv_cross_visit_rgb.v0",
        "scope": (
            "Conservative observed-background transfer between two TbV "
            "traversals; residual unsupported traffic pixels stay masked."
        ),
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "output_root": str(output_root),
        "sequences": args.sequence,
        "window_start_seconds": args.window_start_seconds,
        "window_end_seconds": args.window_end_seconds,
        "camera_filter": args.camera,
        "target_count": len(records),
        "accepted_count": len(accepted),
        "accepted_fraction": len(accepted) / len(records),
        "reused_count": reused_count,
        "target_excluded_fraction": distribution(
            [record["target_excluded_fraction"] for record in records]
        ),
        "filled_fraction_of_excluded": distribution(
            [record["filled_fraction_of_excluded"] for record in records]
        ),
        "residual_excluded_fraction": distribution(
            [record["residual_excluded_fraction"] for record in records]
        ),
        "elapsed_seconds": distribution(
            [record["elapsed_seconds"] for record in records]
        ),
        "contact_sheet": str(contact_sheet),
        "records": records,
    }
    manifest = output_root / "cross_visit_manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "records"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
