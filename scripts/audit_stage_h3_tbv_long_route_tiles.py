#!/usr/bin/env python3
"""Audit one long repeated TbV route and a genuinely adjacent tile pair."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_stage_h3_tbv_trajectories import (  # noqa: E402
    DEFAULT_CACHE,
    load_manifest,
    load_track,
)
from download_stage_h3_tbv_window import (  # noqa: E402
    CAMERAS,
    S3Object,
    TBV_PREFIX,
    Window,
    download_object,
    list_objects,
    plan_window,
    timestamp_ns,
)


REFERENCE_LOG = "V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020"
REPEAT_LOG = "cTrSOEc1gW3XqELP562UlUFJYCmlRoa9__Spring_2020"
CAMERA_NAME = "ring_front_center"
DEFAULT_OUTPUT_DIR = Path(
    "/home/yawei/stage3_external/artifacts/"
    "tbv_long_route_tile_audit_20260725_v3"
)
SUPPORT_DISTANCE_METERS = 3.0
SUPPORT_HEADING_DEGREES = 15.0
TILE_LENGTH_METERS = 100.0
TILE_OVERLAP_METERS = 20.0


@dataclass(frozen=True)
class SupportedRun:
    start_index: int
    end_index: int
    start_progress_meters: float
    end_progress_meters: float

    @property
    def length_meters(self) -> float:
        return self.end_progress_meters - self.start_progress_meters


@dataclass(frozen=True)
class TileRange:
    index: int
    start_progress_meters: float
    end_progress_meters: float


def cumulative_distances(
    points: Sequence[tuple[float, float]],
) -> list[float]:
    if len(points) < 2:
        raise ValueError("trajectory needs at least two points")
    values = [0.0]
    for left, right in zip(points, points[1:]):
        segment = math.hypot(right[0] - left[0], right[1] - left[1])
        if not math.isfinite(segment) or segment <= 0.0:
            raise ValueError("trajectory contains an invalid segment")
        values.append(values[-1] + segment)
    return values


def longest_supported_run(
    supported: Sequence[bool],
    progress_meters: Sequence[float],
) -> SupportedRun:
    if len(supported) != len(progress_meters) or not supported:
        raise ValueError("support/progress must have equal non-zero length")
    candidates: list[SupportedRun] = []
    start: int | None = None
    for index, value in enumerate((*supported, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            candidates.append(
                SupportedRun(
                    start_index=start,
                    end_index=end,
                    start_progress_meters=float(progress_meters[start]),
                    end_progress_meters=float(progress_meters[end]),
                )
            )
            start = None
    if not candidates:
        raise ValueError("no supported trajectory run")
    return max(candidates, key=lambda item: item.length_meters)


def make_tile_ranges(
    start_progress_meters: float,
    end_progress_meters: float,
    *,
    tile_length_meters: float = TILE_LENGTH_METERS,
    tile_overlap_meters: float = TILE_OVERLAP_METERS,
) -> list[TileRange]:
    values = (
        start_progress_meters,
        end_progress_meters,
        tile_length_meters,
        tile_overlap_meters,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tile parameters must be finite")
    if tile_length_meters <= 0.0:
        raise ValueError("tile length must be positive")
    if not 0.0 <= tile_overlap_meters < tile_length_meters:
        raise ValueError("tile overlap must lie in [0, tile length)")
    if end_progress_meters - start_progress_meters < tile_length_meters:
        raise ValueError("supported run is shorter than one tile")
    stride = tile_length_meters - tile_overlap_meters
    ranges = []
    tile_start = start_progress_meters
    while tile_start + tile_length_meters <= end_progress_meters + 1e-9:
        ranges.append(
            TileRange(
                index=len(ranges),
                start_progress_meters=tile_start,
                end_progress_meters=tile_start + tile_length_meters,
            )
        )
        tile_start += stride
    return ranges


def _percentile(values: object, fraction: float, np: object) -> float:
    return float(np.percentile(values, fraction))


def _angle_difference_degrees(first: float, second: float) -> float:
    return abs(
        math.degrees(
            math.atan2(math.sin(first - second), math.cos(first - second))
        )
    )


def _nearest_camera_object(
    objects: Sequence[S3Object],
    target_seconds: float,
) -> tuple[S3Object, float]:
    timestamped = [
        (obj, stamp)
        for obj in objects
        if (stamp := timestamp_ns(obj)) is not None
    ]
    if not timestamped:
        raise RuntimeError("camera listing contains no timestamped images")
    target_ns = round(target_seconds * 1e9)
    obj, stamp = min(timestamped, key=lambda item: abs(item[1] - target_ns))
    return obj, abs(stamp - target_ns) / 1e9


def _make_contact_sheet(
    image_records: list[dict[str, object]],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    stations = sorted(
        {float(item["station_progress_meters"]) for item in image_records}
    )
    logs = (REFERENCE_LOG, REPEAT_LOG)
    images = {
        (str(item["log_id"]), float(item["station_progress_meters"])): item
        for item in image_records
    }
    opened = Image.open(str(image_records[0]["path"])).convert("RGB")
    tile_width, tile_height = opened.size
    sheet = Image.new(
        "RGB",
        (tile_width * len(stations), tile_height * len(logs)),
        color=(0, 0, 0),
    )
    for row, log_id in enumerate(logs):
        for column, station in enumerate(stations):
            record = images[(log_id, station)]
            image = Image.open(str(record["path"])).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
            draw.text(
                (10, 10),
                f"{log_id[:8]} | route {station:.0f} m",
                fill=(255, 255, 255),
            )
            sheet.paste(image, (column * tile_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-source-images", action="store_true")
    parser.add_argument("--plan-tile-payload", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "run this audit with the accepted h3_splatad environment"
        ) from error

    metadata = {item.log_id: item for item in load_manifest(args.cache_dir)}
    reference = load_track(metadata[REFERENCE_LOG])
    repeat = load_track(metadata[REPEAT_LOG])
    if reference is None or repeat is None:
        raise RuntimeError("selected long-route track is missing")
    reference_points = np.asarray(reference.xy_metres)
    repeat_points = np.asarray(repeat.xy_metres)
    nearest_distances, nearest_indices = cKDTree(repeat_points).query(
        reference_points,
        workers=1,
    )
    heading_differences = np.asarray(
        [
            _angle_difference_degrees(
                reference.headings_radians[index],
                repeat.headings_radians[int(repeat_index)],
            )
            for index, repeat_index in enumerate(nearest_indices)
        ]
    )
    monotonic = np.ones(len(nearest_indices), dtype=bool)
    monotonic[1:] = nearest_indices[1:] >= nearest_indices[:-1]
    supported = (
        (nearest_distances <= SUPPORT_DISTANCE_METERS)
        & (heading_differences <= SUPPORT_HEADING_DEGREES)
        & monotonic
    )
    reference_progress = cumulative_distances(reference.xy_metres)
    run = longest_supported_run(supported.tolist(), reference_progress)
    tiles = make_tile_ranges(
        run.start_progress_meters,
        run.end_progress_meters,
    )
    if len(tiles) < 2:
        raise RuntimeError("long route does not contain an adjacent tile pair")
    selected_pair_index = max(0, len(tiles) // 2 - 1)
    selected_pair = tiles[selected_pair_index : selected_pair_index + 2]

    tile_records = []
    for tile in tiles:
        mask = (
            (np.asarray(reference_progress) >= tile.start_progress_meters)
            & (np.asarray(reference_progress) <= tile.end_progress_meters)
        )
        tile_distances = nearest_distances[mask]
        tile_headings = heading_differences[mask]
        tile_supported = supported[mask]
        reference_indices = np.flatnonzero(mask)
        repeat_indices = nearest_indices[mask].astype(int)
        record = {
            **asdict(tile),
            "reference_sample_count": int(mask.sum()),
            "support_fraction": float(tile_supported.mean()),
            "nearest_distance_p50_meters": _percentile(
                tile_distances, 50.0, np
            ),
            "nearest_distance_p95_meters": _percentile(
                tile_distances, 95.0, np
            ),
            "nearest_distance_max_meters": float(tile_distances.max()),
            "heading_difference_p50_degrees": _percentile(
                tile_headings, 50.0, np
            ),
            "heading_difference_p95_degrees": _percentile(
                tile_headings, 95.0, np
            ),
            "reference_time_window_seconds": [
                reference.timestamps[int(reference_indices.min())],
                reference.timestamps[int(reference_indices.max())],
            ],
            "repeat_time_window_seconds": [
                min(repeat.timestamps[index] for index in repeat_indices),
                max(repeat.timestamps[index] for index in repeat_indices),
            ],
        }
        record["metadata_gate"] = (
            record["support_fraction"] >= 0.95
            and record["nearest_distance_p95_meters"] <= 1.5
            and record["heading_difference_p95_degrees"] <= 5.0
        )
        tile_records.append(record)

    pair_overlap_start = selected_pair[1].start_progress_meters
    pair_overlap_end = selected_pair[0].end_progress_meters
    station_progresses = (
        selected_pair[0].start_progress_meters,
        pair_overlap_start,
        pair_overlap_end,
        selected_pair[1].end_progress_meters,
    )
    source_image_records: list[dict[str, object]] = []
    if args.download_source_images:
        camera_objects = {
            log_id: list_objects(
                f"{TBV_PREFIX}{log_id}/sensors/cameras/{CAMERA_NAME}/"
            )
            for log_id in (REFERENCE_LOG, REPEAT_LOG)
        }
        source_root = output_dir / "source_images"
        for station in station_progresses:
            reference_index = min(
                range(len(reference_progress)),
                key=lambda index: abs(reference_progress[index] - station),
            )
            repeat_index = int(nearest_indices[reference_index])
            targets = (
                (REFERENCE_LOG, reference.timestamps[reference_index]),
                (REPEAT_LOG, repeat.timestamps[repeat_index]),
            )
            for log_id, target_time in targets:
                obj, delta = _nearest_camera_object(
                    camera_objects[log_id],
                    target_time,
                )
                download_object(source_root, obj)
                local_path = (
                    source_root / Path(obj.key).relative_to(TBV_PREFIX)
                )
                source_image_records.append(
                    {
                        "log_id": log_id,
                        "station_progress_meters": station,
                        "target_pose_time_seconds": target_time,
                        "camera_time_delta_seconds": delta,
                        "object_key": obj.key,
                        "object_size_bytes": obj.size,
                        "path": str(local_path),
                    }
                )
        _make_contact_sheet(
            source_image_records,
            output_dir / "adjacent_tile_source_contact_sheet.jpg",
        )

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.plot(
        reference_points[:, 0],
        reference_points[:, 1],
        label=f"reference {REFERENCE_LOG[:8]}",
        linewidth=2,
    )
    axis.plot(
        repeat_points[:, 0],
        repeat_points[:, 1],
        label=f"repeat {REPEAT_LOG[:8]}",
        linewidth=1.5,
    )
    colors = ("#f59e0b", "#ef4444")
    for tile, color in zip(selected_pair, colors):
        mask = (
            (np.asarray(reference_progress) >= tile.start_progress_meters)
            & (np.asarray(reference_progress) <= tile.end_progress_meters)
        )
        axis.plot(
            reference_points[mask, 0],
            reference_points[mask, 1],
            color=color,
            linewidth=6,
            alpha=0.65,
            label=(
                f"tile {tile.index}: {tile.start_progress_meters:.0f}-"
                f"{tile.end_progress_meters:.0f} m"
            ),
        )
    axis.set_aspect("equal")
    axis.set_title("TbV 610 m repeated route and selected adjacent tiles")
    axis.set_xlabel("city x (m)")
    axis.set_ylabel("city y (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    route_plot_path = output_dir / "long_route_tiles.png"
    figure.savefig(route_plot_path, dpi=160)
    plt.close(figure)

    run_slice = slice(run.start_index, run.end_index + 1)
    selected_records = [
        tile_records[tile.index] for tile in selected_pair
    ]
    payload_records = []
    unique_payload_objects: dict[str, S3Object] = {}
    if args.plan_tile_payload:
        for tile_record in selected_records:
            time_windows = (
                (
                    REFERENCE_LOG,
                    tile_record["reference_time_window_seconds"],
                ),
                (
                    REPEAT_LOG,
                    tile_record["repeat_time_window_seconds"],
                ),
            )
            for log_id, time_window in time_windows:
                window = Window(log_id, *time_window)
                objects = plan_window(window, camera_stride=2)
                unique_payload_objects.update(
                    {item.key: item for item in objects}
                )
                payload_records.append(
                    {
                        "tile_index": tile_record["index"],
                        "log_id": log_id,
                        "start_seconds": window.start_seconds,
                        "end_seconds": window.end_seconds,
                        "duration_seconds": (
                            window.end_seconds - window.start_seconds
                        ),
                        "object_count": len(objects),
                        "bytes": sum(item.size for item in objects),
                        "lidar_count": sum(
                            "/sensors/lidar/" in item.key
                            for item in objects
                        ),
                        "camera_counts": {
                            camera: sum(
                                f"/sensors/cameras/{camera}/" in item.key
                                for item in objects
                            )
                            for camera in CAMERAS
                        },
                    }
                )
    report = {
        "format": "driving_scene_reconstruction.tbv_long_route_tiles.v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "dataset": "Argoverse Trust, but Verify (TbV)",
        "metadata_only_until_source_image_sample": True,
        "logs": {
            "reference": REFERENCE_LOG,
            "repeat": REPEAT_LOG,
            "camera": CAMERA_NAME,
        },
        "support_contract": {
            "maximum_nearest_distance_meters": SUPPORT_DISTANCE_METERS,
            "maximum_heading_difference_degrees": (
                SUPPORT_HEADING_DEGREES
            ),
            "nearest_repeat_indices_must_be_monotonic": True,
        },
        "longest_supported_run": {
            **asdict(run),
            "length_meters": run.length_meters,
            "sample_count": run.end_index - run.start_index + 1,
            "nearest_distance_p50_meters": _percentile(
                nearest_distances[run_slice], 50.0, np
            ),
            "nearest_distance_p95_meters": _percentile(
                nearest_distances[run_slice], 95.0, np
            ),
            "nearest_distance_max_meters": float(
                nearest_distances[run_slice].max()
            ),
            "heading_difference_p50_degrees": _percentile(
                heading_differences[run_slice], 50.0, np
            ),
            "heading_difference_p95_degrees": _percentile(
                heading_differences[run_slice], 95.0, np
            ),
            "heading_difference_max_degrees": float(
                heading_differences[run_slice].max()
            ),
        },
        "tiling": {
            "tile_length_meters": TILE_LENGTH_METERS,
            "tile_overlap_meters": TILE_OVERLAP_METERS,
            "tile_stride_meters": (
                TILE_LENGTH_METERS - TILE_OVERLAP_METERS
            ),
            "tile_count": len(tiles),
            "tiles": tile_records,
        },
        "selected_adjacent_pair": {
            "tile_indices": [tile.index for tile in selected_pair],
            "overlap_progress_meters": [
                pair_overlap_start,
                pair_overlap_end,
            ],
            "overlap_length_meters": (
                pair_overlap_end - pair_overlap_start
            ),
            "stations_meters": list(station_progresses),
            "both_tiles_pass_metadata_gate": all(
                bool(item["metadata_gate"]) for item in selected_records
            ),
        },
        "source_images": source_image_records,
        "selected_pair_payload_plan": {
            "planned": bool(payload_records),
            "camera_stride": 2,
            "windows": payload_records,
            "unique_object_count": len(unique_payload_objects),
            "unique_bytes": sum(
                item.size for item in unique_payload_objects.values()
            ),
        },
        "artifacts": {
            "route_plot": str(route_plot_path),
            "contact_sheet": (
                str(output_dir / "adjacent_tile_source_contact_sheet.jpg")
                if source_image_records
                else None
            ),
        },
        "summary": {
            "metadata_gate": (
                run.length_meters >= 500.0
                and len(tiles) >= 6
                and all(bool(item["metadata_gate"]) for item in tile_records)
            ),
            "source_image_review": (
                "pending_manual_review"
                if source_image_records
                else "not_sampled"
            ),
            "source_image_samples_downloaded": bool(source_image_records),
            "full_tile_sensor_payload_downloaded": False,
            "selected_pair_payload_planned": bool(payload_records),
            "training_performed": False,
            "reconstruction_tiles_built": False,
        },
    }
    report_path = output_dir / "tbv_long_route_tile_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {report_path}")
    print(f"route plot: {route_plot_path}")
    if source_image_records:
        print(
            "contact sheet: "
            f"{output_dir / 'adjacent_tile_source_contact_sheet.jpg'}"
        )
    print(
        json.dumps(
            {
                "supported_run_meters": run.length_meters,
                "tile_count": len(tiles),
                "selected_tile_indices": [
                    tile.index for tile in selected_pair
                ],
                "metadata_gate": report["summary"]["metadata_gate"],
                "source_image_count": len(source_image_records),
                "planned_unique_payload_bytes": sum(
                    item.size for item in unique_payload_objects.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
