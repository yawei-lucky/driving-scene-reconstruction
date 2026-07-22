#!/usr/bin/env python3
"""Find repeat, adjacent, and multi-direction PandaSet trajectories in a ZIP.

The scan reads only small GPS, timestamp, and front-camera pose JSON entries.
It does not extract sensor payloads or require the PandaSet development kit.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EARTH_RADIUS_METRES = 6_371_008.8
DEFAULT_ARCHIVE = Path(
    "/home/yawei/stage3_external/artifacts/"
    "pandaset-e2e123aea3b3132c67f4b395ec6120f63e190271.zip"
)


@dataclass(frozen=True)
class Track:
    scene: str
    xy_metres: tuple[tuple[float, float], ...]
    timestamps: tuple[float, ...]
    headings_radians: tuple[float, ...]
    motion_per_sample_metres: tuple[float, ...]
    has_semseg: bool
    pose_alignment: dict[str, float]


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def angle_difference_degrees(first: float, second: float) -> float:
    difference = (first - second + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(difference))


def project_gps(
    samples: Sequence[dict[str, float]],
    reference_latitude_degrees: float,
    reference_longitude_degrees: float,
) -> tuple[tuple[float, float], ...]:
    latitude_scale = EARTH_RADIUS_METRES * math.pi / 180.0
    longitude_scale = latitude_scale * math.cos(
        math.radians(reference_latitude_degrees)
    )
    return tuple(
        (
            (float(sample["long"]) - reference_longitude_degrees) * longitude_scale,
            (float(sample["lat"]) - reference_latitude_degrees) * latitude_scale,
        )
        for sample in samples
    )


def derive_headings_and_motion(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(points) < 2:
        raise ValueError("a trajectory needs at least two GPS samples")
    headings: list[float] = []
    motion: list[float] = []
    for index, point in enumerate(points):
        previous = points[max(0, index - 1)]
        following = points[min(len(points) - 1, index + 1)]
        divisor = 1 if index in (0, len(points) - 1) else 2
        dx = (following[0] - previous[0]) / divisor
        dy = (following[1] - previous[1]) / divisor
        headings.append(math.atan2(dy, dx))
        motion.append(math.hypot(dx, dy))
    return tuple(headings), tuple(motion)


def fit_rigid_pose_to_gps(
    pose_points: Sequence[tuple[float, float]],
    gps_points: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Fit one 2D rigid transform from scene-local poses to shared GPS ENU."""
    if len(pose_points) != len(gps_points) or len(pose_points) < 2:
        raise ValueError("pose and GPS tracks must have the same length >= 2")
    pose_mean = (
        statistics.fmean(point[0] for point in pose_points),
        statistics.fmean(point[1] for point in pose_points),
    )
    gps_mean = (
        statistics.fmean(point[0] for point in gps_points),
        statistics.fmean(point[1] for point in gps_points),
    )
    dot_sum = 0.0
    cross_sum = 0.0
    pose_energy = 0.0
    for pose, gps in zip(pose_points, gps_points, strict=True):
        px, py = pose[0] - pose_mean[0], pose[1] - pose_mean[1]
        gx, gy = gps[0] - gps_mean[0], gps[1] - gps_mean[1]
        dot_sum += px * gx + py * gy
        cross_sum += px * gy - py * gx
        pose_energy += px * px + py * py
    rotation = math.atan2(cross_sum, dot_sum)
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    translation = (
        gps_mean[0] - (cosine * pose_mean[0] - sine * pose_mean[1]),
        gps_mean[1] - (sine * pose_mean[0] + cosine * pose_mean[1]),
    )
    residuals = []
    aligned_dot_sum = 0.0
    for pose, gps in zip(pose_points, gps_points, strict=True):
        x = cosine * pose[0] - sine * pose[1] + translation[0]
        y = sine * pose[0] + cosine * pose[1] + translation[1]
        residuals.append(math.hypot(x - gps[0], y - gps[1]))
        px, py = pose[0] - pose_mean[0], pose[1] - pose_mean[1]
        gx, gy = gps[0] - gps_mean[0], gps[1] - gps_mean[1]
        aligned_dot_sum += (cosine * px - sine * py) * gx
        aligned_dot_sum += (sine * px + cosine * py) * gy
    fitted_scale = aligned_dot_sum / pose_energy if pose_energy else float("nan")
    return {
        "rotation_degrees": math.degrees(rotation),
        "translation_x_metres": translation[0],
        "translation_y_metres": translation[1],
        "fitted_similarity_scale": fitted_scale,
        "residual_p50_metres": percentile(residuals, 50.0),
        "residual_p95_metres": percentile(residuals, 95.0),
        "residual_max_metres": max(residuals),
    }


def trajectory_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def covered_length(
    points: Sequence[tuple[float, float]], nearest_distances: Sequence[float], threshold: float
) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second, first_distance, second_distance in zip(
            points, points[1:], nearest_distances, nearest_distances[1:]
        )
        if first_distance <= threshold and second_distance <= threshold
    )


def nearest_matches(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> tuple[list[float], list[int], list[float], list[int]]:
    first_distances = [float("inf")] * len(first)
    first_indices = [-1] * len(first)
    second_distances = [float("inf")] * len(second)
    second_indices = [-1] * len(second)
    for first_index, first_point in enumerate(first):
        for second_index, second_point in enumerate(second):
            distance = math.hypot(
                first_point[0] - second_point[0], first_point[1] - second_point[1]
            )
            if distance < first_distances[first_index]:
                first_distances[first_index] = distance
                first_indices[first_index] = second_index
            if distance < second_distances[second_index]:
                second_distances[second_index] = distance
                second_indices[second_index] = first_index
    return first_distances, first_indices, second_distances, second_indices


def pair_metrics(
    first: Track,
    second: Track,
    *,
    overlap_distance_metres: float,
    direction_distance_metres: float,
    minimum_motion_per_sample_metres: float,
    minimum_overlap_samples: int,
) -> dict[str, object]:
    first_distances, first_indices, second_distances, second_indices = nearest_matches(
        first.xy_metres, second.xy_metres
    )
    heading_differences: list[float] = []
    for distances, indices, source, target in (
        (first_distances, first_indices, first, second),
        (second_distances, second_indices, second, first),
    ):
        for source_index, (distance, target_index) in enumerate(zip(distances, indices)):
            if distance > direction_distance_metres:
                continue
            if (
                source.motion_per_sample_metres[source_index]
                < minimum_motion_per_sample_metres
                or target.motion_per_sample_metres[target_index]
                < minimum_motion_per_sample_metres
            ):
                continue
            heading_differences.append(
                angle_difference_degrees(
                    source.headings_radians[source_index],
                    target.headings_radians[target_index],
                )
            )

    first_overlap = [value for value in first_distances if value <= overlap_distance_metres]
    second_overlap = [value for value in second_distances if value <= overlap_distance_metres]
    combined_overlap = first_overlap + second_overlap
    first_overlap_length = covered_length(
        first.xy_metres, first_distances, overlap_distance_metres
    )
    second_overlap_length = covered_length(
        second.xy_metres, second_distances, overlap_distance_metres
    )
    minimum_overlap_count = min(len(first_overlap), len(second_overlap))
    heading_median = (
        percentile(heading_differences, 50.0) if heading_differences else None
    )
    heading_p95 = (
        percentile(heading_differences, 95.0) if heading_differences else None
    )
    heading_over_threshold_count = sum(
        difference > 20.0 for difference in heading_differences
    )
    heading_over_threshold_fraction = (
        heading_over_threshold_count / len(heading_differences)
        if heading_differences
        else None
    )
    has_path_overlap = (
        minimum_overlap_count >= minimum_overlap_samples
        and min(first_overlap_length, second_overlap_length) >= 10.0
    )
    same_direction_repeat = (
        has_path_overlap
        and len(heading_differences) >= 3
        and heading_median is not None
        and heading_median <= 20.0
    )
    opposite_direction = (
        min(first_distances) <= direction_distance_metres
        and len(heading_differences) >= 3
        and heading_median is not None
        and heading_median >= 140.0
    )
    direction_change_review = (
        min(first_distances) <= direction_distance_metres
        and len(heading_differences) >= 3
        and (
            opposite_direction
            or (
                heading_over_threshold_fraction is not None
                and heading_over_threshold_fraction >= 0.20
            )
            or (heading_p95 is not None and heading_p95 >= 45.0)
        )
    )
    overlap_distance_median = (
        percentile(combined_overlap, 50.0) if combined_overlap else None
    )
    adjacent_or_offset = (
        same_direction_repeat
        and overlap_distance_median is not None
        and overlap_distance_median >= 1.5
    )
    first_length = trajectory_length(first.xy_metres)
    second_length = trajectory_length(second.xy_metres)
    estimated_union_length = first_length + second_length - min(
        first_overlap_length, second_overlap_length
    )
    alignment_initialization_pass = all(
        track.pose_alignment.get("residual_p95_metres", float("inf")) <= 1.0
        and 0.95
        <= track.pose_alignment.get("fitted_similarity_scale", float("inf"))
        <= 1.05
        for track in (first, second)
    )
    return {
        "scenes": [first.scene, second.scene],
        "minimum_distance_metres": min(first_distances),
        "overlap_distance_threshold_metres": overlap_distance_metres,
        "overlap_samples": {
            first.scene: len(first_overlap),
            second.scene: len(second_overlap),
        },
        "overlap_length_metres": {
            first.scene: first_overlap_length,
            second.scene: second_overlap_length,
        },
        "overlap_nearest_distance_p50_metres": overlap_distance_median,
        "overlap_nearest_distance_p95_metres": (
            percentile(combined_overlap, 95.0) if combined_overlap else None
        ),
        "heading_difference_p50_degrees": heading_median,
        "heading_difference_p95_degrees": heading_p95,
        "heading_difference_over_20_degrees_count": heading_over_threshold_count,
        "heading_difference_over_20_degrees_fraction": heading_over_threshold_fraction,
        "valid_heading_match_count": len(heading_differences),
        "start_time_gap_hours": abs(first.timestamps[0] - second.timestamps[0])
        / 3600.0,
        "estimated_union_route_length_metres": estimated_union_length,
        "both_have_semseg": first.has_semseg and second.has_semseg,
        "classification": {
            "has_bidirectional_path_overlap": has_path_overlap,
            "same_direction_repeat": same_direction_repeat,
            "adjacent_or_offset_repeat": adjacent_or_offset,
            "direction_change_review_candidate": direction_change_review,
            "opposite_direction_candidate": opposite_direction,
            "pose_gps_alignment_initialization_pass": alignment_initialization_pass,
            "extends_longest_input_by_10m": estimated_union_length
            >= max(first_length, second_length) + 10.0,
        },
    }


def _pose_xy(poses: Sequence[dict[str, dict[str, float]]]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(pose["position"]["x"]), float(pose["position"]["y"]))
        for pose in poses
    )


def load_tracks(archive: Path) -> tuple[list[Track], int]:
    with zipfile.ZipFile(archive) as dataset_zip:
        names = dataset_zip.namelist()
        gps_names = sorted(
            name
            for name in names
            if name.startswith("pandaset/") and name.endswith("/meta/gps.json")
        )
        if not gps_names:
            raise ValueError(f"no PandaSet GPS metadata found in {archive}")
        gps_by_scene = {
            name.split("/")[1]: json.loads(dataset_zip.read(name))
            for name in gps_names
        }
        reference_latitude = statistics.fmean(
            float(sample["lat"])
            for samples in gps_by_scene.values()
            for sample in samples
        )
        reference_longitude = statistics.fmean(
            float(sample["long"])
            for samples in gps_by_scene.values()
            for sample in samples
        )
        semseg_scenes = {
            name.split("/")[1]
            for name in names
            if "/annotations/semseg/" in name and name.endswith(".pkl.gz")
        }
        tracks: list[Track] = []
        for scene, gps_samples in sorted(gps_by_scene.items()):
            timestamps = json.loads(
                dataset_zip.read(f"pandaset/{scene}/meta/timestamps.json")
            )
            poses = json.loads(
                dataset_zip.read(
                    f"pandaset/{scene}/camera/front_camera/poses.json"
                )
            )
            if not (len(gps_samples) == len(timestamps) == len(poses)):
                raise ValueError(f"scene {scene} metadata lengths do not match")
            gps_points = project_gps(
                gps_samples, reference_latitude, reference_longitude
            )
            headings, motion = derive_headings_and_motion(gps_points)
            tracks.append(
                Track(
                    scene=scene,
                    xy_metres=gps_points,
                    timestamps=tuple(float(value) for value in timestamps),
                    headings_radians=headings,
                    motion_per_sample_metres=motion,
                    has_semseg=scene in semseg_scenes,
                    pose_alignment=fit_rigid_pose_to_gps(_pose_xy(poses), gps_points),
                )
            )
    return tracks, len(names)


def track_summary(track: Track) -> dict[str, object]:
    return {
        "scene": track.scene,
        "samples": len(track.xy_metres),
        "trajectory_length_metres": trajectory_length(track.xy_metres),
        "endpoint_displacement_metres": math.hypot(
            track.xy_metres[-1][0] - track.xy_metres[0][0],
            track.xy_metres[-1][1] - track.xy_metres[0][1],
        ),
        "duration_seconds": track.timestamps[-1] - track.timestamps[0],
        "has_semseg": track.has_semseg,
        "pose_to_gps_rigid_alignment": track.pose_alignment,
    }


def build_report(
    archive: Path,
    *,
    focus_scene: str = "040",
    overlap_distance_metres: float = 5.0,
    direction_distance_metres: float = 15.0,
    minimum_motion_per_sample_metres: float = 0.2,
    minimum_overlap_samples: int = 20,
) -> dict[str, object]:
    tracks, archive_entries = load_tracks(archive)
    track_by_scene = {track.scene: track for track in tracks}
    if focus_scene not in track_by_scene:
        raise ValueError(f"focus scene {focus_scene} is not in the archive")
    pairs = [
        pair_metrics(
            first,
            second,
            overlap_distance_metres=overlap_distance_metres,
            direction_distance_metres=direction_distance_metres,
            minimum_motion_per_sample_metres=minimum_motion_per_sample_metres,
            minimum_overlap_samples=minimum_overlap_samples,
        )
        for index, first in enumerate(tracks)
        for second in tracks[index + 1 :]
    ]
    repeats = [
        pair
        for pair in pairs
        if pair["classification"]["same_direction_repeat"]
    ]
    repeats.sort(
        key=lambda pair: (
            -min(pair["overlap_samples"].values()),
            pair["overlap_nearest_distance_p50_metres"],
            pair["scenes"],
        )
    )
    expansion_candidates = [
        pair
        for pair in repeats
        if pair["classification"]["adjacent_or_offset_repeat"]
        and pair["classification"]["extends_longest_input_by_10m"]
        and pair["classification"]["pose_gps_alignment_initialization_pass"]
    ]
    expansion_candidates.sort(
        key=lambda pair: (
            not pair["both_have_semseg"],
            -min(pair["overlap_length_metres"].values()),
            -pair["estimated_union_route_length_metres"],
            pair["scenes"],
        )
    )
    direction_change_review = [
        pair
        for pair in pairs
        if pair["classification"]["direction_change_review_candidate"]
    ]
    direction_change_review.sort(
        key=lambda pair: (
            -pair["heading_difference_over_20_degrees_fraction"],
            pair["minimum_distance_metres"],
            pair["scenes"],
        )
    )
    opposite_direction = [
        pair
        for pair in direction_change_review
        if pair["classification"]["opposite_direction_candidate"]
    ]
    focus_neighbors = []
    for pair in pairs:
        if focus_scene not in pair["scenes"]:
            continue
        other = pair["scenes"][0] if pair["scenes"][1] == focus_scene else pair["scenes"][1]
        focus_neighbors.append(
            {"scene": other, "minimum_distance_metres": pair["minimum_distance_metres"]}
        )
    focus_neighbors.sort(key=lambda item: (item["minimum_distance_metres"], item["scene"]))
    return {
        "schema_version": 2,
        "archive": {
            "path": str(archive.resolve()),
            "size_bytes": archive.stat().st_size,
            "entry_count": archive_entries,
            "scene_count": len(tracks),
        },
        "method": {
            "distance_projection": "local equirectangular ENU at all-scene mean latitude/longitude",
            "overlap_distance_metres": overlap_distance_metres,
            "direction_search_distance_metres": direction_distance_metres,
            "minimum_motion_per_sample_metres": minimum_motion_per_sample_metres,
            "minimum_bidirectional_overlap_samples": minimum_overlap_samples,
            "direction_change_review_heading_threshold_degrees": 20.0,
            "direction_change_review_minimum_fraction": 0.20,
            "notes": [
                "GPS is a candidate-selection signal, not final reconstruction registration.",
                "Pose alignment is a per-scene 2D rigid fit of front-camera x/y to GPS ENU.",
                "The union route length subtracts the smaller covered length and is an estimate.",
                "Direction-change pairs require visual/shape review; they are not automatically verified branches.",
            ],
        },
        "focus_scene": {
            "scene": focus_scene,
            "nearest_scenes": focus_neighbors[:10],
            "has_path_overlap_candidate": any(
                focus_scene in pair["scenes"]
                and pair["classification"]["has_bidirectional_path_overlap"]
                for pair in pairs
            ),
        },
        "candidate_counts": {
            "same_direction_repeats": len(repeats),
            "offset_route_expansions": len(expansion_candidates),
            "direction_change_review": len(direction_change_review),
            "opposite_direction": len(opposite_direction),
        },
        "top_geometric_expansion_candidate": (
            expansion_candidates[0] if expansion_candidates else None
        ),
        "offset_route_expansion_candidates": expansion_candidates[:20],
        "same_direction_repeat_candidates": repeats[:20],
        "direction_change_review_candidates": direction_change_review[:20],
        "opposite_direction_candidates": opposite_direction[:20],
        "tracks": [track_summary(track) for track in tracks],
    }


def parse_csv_integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def write_front_contact_sheet(
    archive: Path,
    output_path: Path,
    scenes: Iterable[str],
    frames: Iterable[int],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - depends on the H3 environment
        raise RuntimeError("contact-sheet output requires Pillow") from error
    scenes = tuple(scenes)
    frames = tuple(frames)
    cell = (384, 216)
    sheet = Image.new("RGB", (cell[0] * len(frames), cell[1] * len(scenes)), "black")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    with zipfile.ZipFile(archive) as dataset_zip:
        for row, scene in enumerate(scenes):
            for column, frame in enumerate(frames):
                entry = f"pandaset/{scene}/camera/front_camera/{frame:02d}.jpg"
                image = Image.open(io.BytesIO(dataset_zip.read(entry))).convert("RGB")
                image.thumbnail(cell, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", cell, "black")
                canvas.paste(image, ((cell[0] - image.width) // 2, (cell[1] - image.height) // 2))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((5, 5, 120, 31), fill="black")
                draw.text((10, 8), f"{scene}  frame {frame:02d}", font=font, fill=(120, 255, 120))
                sheet.paste(canvas, (column * cell[0], row * cell[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--focus-scene", default="040")
    parser.add_argument("--overlap-distance", type=float, default=5.0)
    parser.add_argument("--direction-distance", type=float, default=15.0)
    parser.add_argument("--minimum-overlap-samples", type=int, default=20)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--contact-scenes", type=parse_csv_strings, default=("003", "057", "032", "070", "040")
    )
    parser.add_argument(
        "--contact-frames", type=parse_csv_integers, default=(0, 20, 40, 60, 79)
    )
    args = parser.parse_args()
    if args.overlap_distance <= 0 or args.direction_distance <= 0:
        parser.error("distance thresholds must be positive")
    if args.minimum_overlap_samples < 1:
        parser.error("--minimum-overlap-samples must be positive")

    report = build_report(
        args.archive,
        focus_scene=args.focus_scene,
        overlap_distance_metres=args.overlap_distance,
        direction_distance_metres=args.direction_distance,
        minimum_overlap_samples=args.minimum_overlap_samples,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.contact_sheet:
        write_front_contact_sheet(
            args.archive,
            args.contact_sheet,
            args.contact_scenes,
            args.contact_frames,
        )

    focus = report["focus_scene"]
    recommendation = report["top_geometric_expansion_candidate"]
    print(
        f"scanned {report['archive']['scene_count']} scenes; "
        f"focus {focus['scene']} nearest={focus['nearest_scenes'][0]['scene']} "
        f"at {focus['nearest_scenes'][0]['minimum_distance_metres']:.1f} m"
    )
    print(
        "direction-change pairs requiring review: "
        f"{report['candidate_counts']['direction_change_review']}"
    )
    if recommendation:
        scenes = "+".join(recommendation["scenes"])
        print(
            f"top geometric expansion candidate: {scenes}; "
            f"estimated union {recommendation['estimated_union_route_length_metres']:.1f} m"
        )
    if args.output_json:
        print(f"report: {args.output_json}")
    if args.contact_sheet:
        print(f"contact sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()
