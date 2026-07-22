#!/usr/bin/env python3
"""Inventory Argoverse TbV trajectories without downloading sensor payloads.

The public S3 layout exposes one small city-frame pose Feather file per log.
This tool lists the logs, identifies their cities from object names, downloads
only those pose files, and searches for repeated routes and shared approaches
that diverge into different branches. Camera and LiDAR files are never fetched.

Run the metadata preparation and analysis with the H3 environment because the
analysis step needs PyArrow and SciPy::

    /home/yawei/stage3_external/envs/h3_splatad/bin/python \
      scripts/analyze_stage_h3_tbv_trajectories.py --prepare
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from analyze_stage_h3_pandaset_trajectories import (
    Track,
    angle_difference_degrees,
    derive_headings_and_motion,
    percentile,
    trajectory_length,
)


S3_ENDPOINT = "https://argoverse.s3.amazonaws.com/"
TBV_PREFIX = "datasets/av2/tbv/"
XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_CACHE = Path(
    "/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/metadata"
)
DEFAULT_REPORT = Path(
    "/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/"
    "trajectory_inventory.json"
)
CITY_PATTERN = re.compile(r"____([A-Z]{3})(?:_city_|\.npy)")


@dataclass(frozen=True)
class LogMetadata:
    log_id: str
    city: str
    pose_path: Path


def _s3_url(**parameters: str) -> str:
    return S3_ENDPOINT + "?" + urllib.parse.urlencode(parameters)


def _fetch_bytes(url: str, *, attempts: int = 3, timeout: float = 30.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "stage-h3-tbv-inventory/1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # network errors vary by Python release
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"failed to read {url}") from last_error


def parse_log_prefixes(xml_bytes: bytes) -> tuple[list[str], str | None]:
    root = ET.fromstring(xml_bytes)
    prefixes = [
        element.text or ""
        for element in root.findall("s3:CommonPrefixes/s3:Prefix", XML_NAMESPACE)
    ]
    log_ids = [prefix.rstrip("/").split("/")[-1] for prefix in prefixes]
    continuation = root.findtext("s3:NextContinuationToken", namespaces=XML_NAMESPACE)
    return log_ids, continuation


def list_tbv_logs() -> list[str]:
    logs: list[str] = []
    continuation: str | None = None
    while True:
        parameters = {
            "list-type": "2",
            "prefix": TBV_PREFIX,
            "delimiter": "/",
        }
        if continuation:
            parameters["continuation-token"] = continuation
        page_logs, continuation = parse_log_prefixes(_fetch_bytes(_s3_url(**parameters)))
        logs.extend(page_logs)
        if not continuation:
            return sorted(set(logs))


def parse_city_from_map_listing(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    keys = [
        element.text or "" for element in root.findall("s3:Contents/s3:Key", XML_NAMESPACE)
    ]
    cities = {match.group(1) for key in keys if (match := CITY_PATTERN.search(key))}
    if len(cities) != 1:
        raise ValueError(f"expected one city code in map listing, found {sorted(cities)}")
    return cities.pop()


def _map_listing(log_id: str) -> bytes:
    return _fetch_bytes(
        _s3_url(
            **{
                "list-type": "2",
                "prefix": f"{TBV_PREFIX}{log_id}/map/",
                "delimiter": "/",
            }
        )
    )


def _pose_url(log_id: str) -> str:
    key = f"{TBV_PREFIX}{log_id}/city_SE3_egovehicle.feather"
    return S3_ENDPOINT + urllib.parse.quote(key, safe="/")


def _download_pose(log_id: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    payload = _fetch_bytes(_pose_url(log_id), timeout=60.0)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def prepare_metadata(cache_dir: Path, *, workers: int = 24) -> list[LogMetadata]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = cache_dir / "poses"
    pose_dir.mkdir(exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    existing: dict[str, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {item["log_id"]: item["city"] for item in manifest["logs"]}

    log_ids = list_tbv_logs()

    def prepare_one(log_id: str) -> LogMetadata:
        city = existing.get(log_id)
        if city is None:
            city = parse_city_from_map_listing(_map_listing(log_id))
        pose_path = pose_dir / f"{log_id}.feather"
        _download_pose(log_id, pose_path)
        return LogMetadata(log_id=log_id, city=city, pose_path=pose_path)

    metadata: list[LogMetadata] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare_one, log_id): log_id for log_id in log_ids}
        for future in as_completed(futures):
            metadata.append(future.result())
    metadata.sort(key=lambda item: item.log_id)
    manifest = {
        "schema_version": 1,
        "source": S3_ENDPOINT,
        "dataset_prefix": TBV_PREFIX,
        "download_scope": "city map object names and city_SE3_egovehicle.feather only",
        "logs": [
            {
                "log_id": item.log_id,
                "city": item.city,
                "pose_file": str(item.pose_path),
                "pose_size_bytes": item.pose_path.stat().st_size,
            }
            for item in metadata
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_manifest(cache_dir: Path) -> list[LogMetadata]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run with --prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        LogMetadata(
            log_id=item["log_id"], city=item["city"], pose_path=Path(item["pose_file"])
        )
        for item in manifest["logs"]
    ]


def _season(log_id: str) -> str | None:
    return log_id.rsplit("__", 1)[-1] if "__" in log_id else None


def _downsample_pose_rows(
    timestamps_ns: Sequence[int],
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    minimum_time_seconds: float = 0.10,
    minimum_spacing_metres: float = 0.50,
) -> tuple[list[tuple[float, float]], list[float]]:
    if not (len(timestamps_ns) == len(x_values) == len(y_values)) or not timestamps_ns:
        raise ValueError("pose columns must have the same non-zero length")
    selected = [0]
    last_time = int(timestamps_ns[0])
    last_x = float(x_values[0])
    last_y = float(y_values[0])
    minimum_time_ns = int(minimum_time_seconds * 1e9)
    for index in range(1, len(timestamps_ns) - 1):
        timestamp = int(timestamps_ns[index])
        x = float(x_values[index])
        y = float(y_values[index])
        if timestamp - last_time < minimum_time_ns:
            continue
        if math.hypot(x - last_x, y - last_y) < minimum_spacing_metres:
            continue
        selected.append(index)
        last_time, last_x, last_y = timestamp, x, y
    if len(timestamps_ns) > 1 and selected[-1] != len(timestamps_ns) - 1:
        selected.append(len(timestamps_ns) - 1)
    points = [(float(x_values[index]), float(y_values[index])) for index in selected]
    times = [float(timestamps_ns[index]) / 1e9 for index in selected]
    return points, times


def load_track(item: LogMetadata) -> Track | None:
    try:
        import pyarrow.ipc as ipc
    except ImportError as error:  # pragma: no cover - H3 environment dependency
        raise RuntimeError("trajectory analysis requires PyArrow") from error
    table = (
        ipc.open_file(item.pose_path)
        .read_all()
        .select(["timestamp_ns", "tx_m", "ty_m"])
        .to_pydict()
    )
    points, timestamps = _downsample_pose_rows(
        table["timestamp_ns"], table["tx_m"], table["ty_m"]
    )
    if len(points) < 3 or trajectory_length(points) < 5.0:
        return None
    headings, motion = derive_headings_and_motion(points)
    return Track(
        scene=item.log_id,
        xy_metres=tuple(points),
        timestamps=tuple(timestamps),
        headings_radians=headings,
        motion_per_sample_metres=motion,
        has_semseg=False,
        pose_alignment={"residual_p95_metres": 0.0, "fitted_similarity_scale": 1.0},
    )


def _bbox(track: Track) -> tuple[float, float, float, float]:
    xs = [point[0] for point in track.xy_metres]
    ys = [point[1] for point in track.xy_metres]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _covered_length(
    points: Sequence[tuple[float, float]], distances: Sequence[float], threshold: float
) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second, first_distance, second_distance in zip(
            points, points[1:], distances, distances[1:]
        )
        if first_distance <= threshold and second_distance <= threshold
    )


def _fast_pair_metrics(
    first: Track,
    second: Track,
    *,
    overlap_distance_metres: float = 3.0,
    direction_distance_metres: float = 12.0,
) -> dict[str, object]:
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - H3 environment dependency
        raise RuntimeError("trajectory analysis requires NumPy and SciPy") from error
    first_points = np.asarray(first.xy_metres)
    second_points = np.asarray(second.xy_metres)
    first_distances, first_indices = cKDTree(second_points).query(first_points, workers=1)
    second_distances, second_indices = cKDTree(first_points).query(second_points, workers=1)
    heading_differences: list[float] = []
    for distances, indices, source, target in (
        (first_distances, first_indices, first, second),
        (second_distances, second_indices, second, first),
    ):
        for source_index, (distance, target_index) in enumerate(zip(distances, indices)):
            if distance > direction_distance_metres:
                continue
            if source.motion_per_sample_metres[source_index] < 0.2:
                continue
            if target.motion_per_sample_metres[int(target_index)] < 0.2:
                continue
            heading_differences.append(
                angle_difference_degrees(
                    source.headings_radians[source_index],
                    target.headings_radians[int(target_index)],
                )
            )
    first_overlap = first_distances[first_distances <= overlap_distance_metres].tolist()
    second_overlap = second_distances[second_distances <= overlap_distance_metres].tolist()
    combined_overlap = first_overlap + second_overlap
    first_covered = _covered_length(first.xy_metres, first_distances, overlap_distance_metres)
    second_covered = _covered_length(second.xy_metres, second_distances, overlap_distance_metres)
    first_length = trajectory_length(first.xy_metres)
    second_length = trajectory_length(second.xy_metres)
    heading_p50 = percentile(heading_differences, 50.0) if heading_differences else None
    heading_p95 = percentile(heading_differences, 95.0) if heading_differences else None
    changed_fraction = (
        sum(value > 20.0 for value in heading_differences) / len(heading_differences)
        if heading_differences
        else None
    )
    has_shared_path = (
        min(first_covered, second_covered) >= 15.0
        and min(len(first_overlap), len(second_overlap)) >= 12
    )
    opposite = bool(has_shared_path and heading_p50 is not None and heading_p50 >= 140.0)
    direction_change = bool(
        has_shared_path
        and not opposite
        and heading_p50 is not None
        and heading_p50 <= 35.0
        and (
            (heading_p95 is not None and heading_p95 >= 45.0)
            or (changed_fraction is not None and changed_fraction >= 0.20)
        )
    )
    first_outside = max(0.0, first_length - first_covered)
    second_outside = max(0.0, second_length - second_covered)
    branch = bool(direction_change and min(first_outside, second_outside) >= 15.0)
    same_direction = bool(
        has_shared_path and heading_p50 is not None and heading_p50 <= 20.0
    )
    return {
        "logs": [first.scene, second.scene],
        "minimum_distance_metres": float(min(first_distances.min(), second_distances.min())),
        "route_lengths_metres": {first.scene: first_length, second.scene: second_length},
        "overlap_lengths_metres": {first.scene: first_covered, second.scene: second_covered},
        "outside_overlap_lengths_metres": {
            first.scene: first_outside,
            second.scene: second_outside,
        },
        "overlap_distance_p50_metres": (
            percentile(combined_overlap, 50.0) if combined_overlap else None
        ),
        "heading_difference_p50_degrees": heading_p50,
        "heading_difference_p95_degrees": heading_p95,
        "heading_difference_over_20_degrees_fraction": changed_fraction,
        "estimated_union_route_length_metres": (
            first_length + second_length - min(first_covered, second_covered)
        ),
        "classification": {
            "shared_path": has_shared_path,
            "same_direction_repeat": same_direction,
            "opposite_direction_repeat": opposite,
            "direction_change_review": direction_change,
            "branch_candidate": branch,
        },
    }


def _track_summary(track: Track, city: str) -> dict[str, object]:
    return {
        "log_id": track.scene,
        "city": city,
        "season": _season(track.scene),
        "samples_after_downsampling": len(track.xy_metres),
        "duration_seconds": track.timestamps[-1] - track.timestamps[0],
        "trajectory_length_metres": trajectory_length(track.xy_metres),
    }


def build_report(metadata: Iterable[LogMetadata]) -> dict[str, object]:
    metadata = list(metadata)
    city_by_log = {item.log_id: item.city for item in metadata}
    log_city_counts: dict[str, int] = {}
    for item in metadata:
        log_city_counts[item.city] = log_city_counts.get(item.city, 0) + 1
    tracks = [track for item in metadata if (track := load_track(item)) is not None]
    bboxes = {track.scene: _bbox(track) for track in tracks}
    tracks_by_city: dict[str, list[Track]] = {}
    for track in tracks:
        tracks_by_city.setdefault(city_by_log[track.scene], []).append(track)
    pairs: list[dict[str, object]] = []
    prefilter_count = 0
    for city_tracks in tracks_by_city.values():
        for first, second in combinations(city_tracks, 2):
            if _bbox_distance(bboxes[first.scene], bboxes[second.scene]) > 12.0:
                continue
            prefilter_count += 1
            result = _fast_pair_metrics(first, second)
            if result["minimum_distance_metres"] <= 12.0:
                result["city"] = city_by_log[first.scene]
                result["seasons"] = [_season(first.scene), _season(second.scene)]
                pairs.append(result)
    repeats = [pair for pair in pairs if pair["classification"]["same_direction_repeat"]]
    repeats.sort(
        key=lambda pair: (
            -min(pair["overlap_lengths_metres"].values()),
            pair["overlap_distance_p50_metres"],
        )
    )
    branches = [pair for pair in pairs if pair["classification"]["branch_candidate"]]
    branches.sort(
        key=lambda pair: (
            -min(pair["outside_overlap_lengths_metres"].values()),
            -min(pair["overlap_lengths_metres"].values()),
        )
    )
    opposites = [
        pair for pair in pairs if pair["classification"]["opposite_direction_repeat"]
    ]
    opposites.sort(key=lambda pair: -min(pair["overlap_lengths_metres"].values()))
    return {
        "schema_version": 1,
        "dataset": "Argoverse Trust, but Verify (TbV)",
        "metadata_scope": "S3 object names plus city_SE3_egovehicle.feather; no sensor payloads",
        "method": {
            "pose_downsampling": "at least 0.10 s and 0.50 m between selected samples",
            "same_city_pairing_only": True,
            "bbox_prefilter_distance_metres": 12.0,
            "overlap_distance_metres": 3.0,
            "shared_path_minimum_length_metres": 15.0,
            "branch_minimum_nonshared_length_each_metres": 15.0,
            "notes": [
                "Candidates require image/map inspection before claiming a verified branch.",
                "Global city poses are a candidate signal, not final static-LiDAR registration.",
            ],
        },
        "log_count": len(metadata),
        "usable_moving_track_count": len(tracks),
        "log_city_counts": dict(sorted(log_city_counts.items())),
        "moving_track_city_counts": {
            city: len(city_tracks) for city, city_tracks in sorted(tracks_by_city.items())
        },
        "bbox_prefilter_pair_count": prefilter_count,
        "nearby_pair_count": len(pairs),
        "candidate_counts": {
            "same_direction_repeats": len(repeats),
            "branch_review": len(branches),
            "opposite_direction_repeats": len(opposites),
        },
        "branch_review_candidates": branches,
        "same_direction_repeat_candidates": repeats[:50],
        "opposite_direction_repeat_candidates": opposites[:50],
        "tracks": [
            _track_summary(track, city_by_log[track.scene])
            for track in sorted(tracks, key=lambda item: item.scene)
        ],
    }


def write_candidate_plot(
    metadata: Iterable[LogMetadata], report: dict[str, object], output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:  # pragma: no cover - H3 environment dependency
        raise RuntimeError("candidate plotting requires Matplotlib and NumPy") from error
    metadata_by_log = {item.log_id: item for item in metadata}
    candidates = report["branch_review_candidates"][:12]
    if not candidates:
        return
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for axis, candidate in zip(axes.flat, candidates):
        tracks = [load_track(metadata_by_log[log_id]) for log_id in candidate["logs"]]
        if any(track is None for track in tracks):
            continue
        origin = np.mean(
            np.concatenate([np.asarray(track.xy_metres) for track in tracks]), axis=0
        )
        for track, color in zip(tracks, ("#1f77b4", "#d62728")):
            points = np.asarray(track.xy_metres) - origin
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
            axis.scatter(points[0, 0], points[0, 1], color=color, marker="o", s=28)
            axis.scatter(points[-1, 0], points[-1, 1], color=color, marker="x", s=34)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        short_ids = [log_id.split("__", 1)[0][:7] for log_id in candidate["logs"]]
        axis.set_title(
            f"{candidate['city']} {short_ids[0]} / {short_ids[1]}\n"
            f"shared {min(candidate['overlap_lengths_metres'].values()):.0f} m, "
            f"heading p50/p95 "
            f"{candidate['heading_difference_p50_degrees']:.0f}/"
            f"{candidate['heading_difference_p95_degrees']:.0f} deg",
            fontsize=9,
        )
    for axis in axes.flat[len(candidates) :]:
        axis.set_visible(False)
    figure.suptitle("TbV metadata-only branch review (circle=start, x=end)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--candidate-plot", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    metadata = (
        prepare_metadata(args.cache_dir, workers=args.workers)
        if args.prepare
        else load_manifest(args.cache_dir)
    )
    report = build_report(metadata)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.candidate_plot:
        write_candidate_plot(metadata, report, args.candidate_plot)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "log_count",
                    "usable_moving_track_count",
                    "log_city_counts",
                    "candidate_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
