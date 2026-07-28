#!/usr/bin/env python3
"""Build exact per-return cross-visit persistence masks for one TbV tile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
import warnings


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from stage_h3_tbv_persistence import (  # noqa: E402
    distribution,
    encode_voxel_array,
    persistence_mask_path,
    save_persistence_mask,
    supported_unique_keys,
)


@dataclass(frozen=True)
class Window:
    sequence: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class LidarRecord:
    sequence: str
    timestamp_ns: int
    path: Path
    ego_to_city: Any


def _records(loader: Any, window: Window) -> tuple[LidarRecord, ...]:
    start_ns = round(window.start_seconds * 1e9)
    end_ns = round(window.end_seconds * 1e9)
    records = []
    for timestamp_ns in loader.get_ordered_log_lidar_timestamps(
        window.sequence
    ):
        if not start_ns <= timestamp_ns <= end_ns:
            continue
        records.append(
            LidarRecord(
                sequence=window.sequence,
                timestamp_ns=timestamp_ns,
                path=loader.get_lidar_fpath(
                    window.sequence,
                    timestamp_ns,
                ).resolve(),
                ego_to_city=loader.get_city_SE3_ego(
                    window.sequence,
                    timestamp_ns,
                ).transform_matrix,
            )
        )
    if not records:
        raise RuntimeError(f"no LiDAR frames in {window}")
    return tuple(records)


def _read_xyz(path: Path) -> Any:
    import numpy as np
    from av2.utils.io import read_feather
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        frame = read_feather(path)
    xyz = frame.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"non-finite LiDAR point in {path}")
    return xyz


def _point_keys(
    record: LidarRecord,
    *,
    origin: Any,
    voxel_meters: float,
) -> Any:
    import numpy as np

    xyz = _read_xyz(record.path)
    transform = np.asarray(record.ego_to_city, dtype=np.float64)
    city = xyz @ transform[:3, :3].T + transform[:3, 3]
    voxels = np.floor((city - origin) / voxel_meters).astype(np.int64)
    return encode_voxel_array(voxels)


def _occupied_voxels(
    records: tuple[LidarRecord, ...],
    *,
    origin: Any,
    voxel_meters: float,
    label: str,
) -> tuple[Any, int]:
    import numpy as np

    frame_keys = []
    point_count = 0
    for index, record in enumerate(records, start=1):
        keys = _point_keys(
            record,
            origin=origin,
            voxel_meters=voxel_meters,
        )
        point_count += len(keys)
        frame_keys.append(np.unique(keys))
        if index % 25 == 0 or index == len(records):
            print(
                f"{label}: indexed {index}/{len(records)} LiDAR frames",
                flush=True,
            )
    occupied = np.unique(np.concatenate(frame_keys))
    return occupied, point_count


def _write_masks(
    records: tuple[LidarRecord, ...],
    *,
    data_root: Path,
    output_dir: Path,
    origin: Any,
    voxel_meters: float,
    occupied: Any,
    supported_voxels: Any,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    total_points = 0
    total_persistent = 0
    fractions: list[float] = []
    files = []
    for index, record in enumerate(records, start=1):
        keys = _point_keys(
            record,
            origin=origin,
            voxel_meters=voxel_meters,
        )
        positions = np.searchsorted(occupied, keys)
        if (
            np.any(positions >= len(occupied))
            or np.any(occupied[positions] != keys)
        ):
            raise RuntimeError("point voxel missing from its visit index")
        persistent = supported_voxels[positions]
        destination = persistence_mask_path(
            data_root=data_root,
            persistence_root=output_dir,
            lidar_path=record.path,
        )
        save_persistence_mask(destination, persistent)
        persistent_count = int(persistent.sum())
        total_points += len(persistent)
        total_persistent += persistent_count
        fraction = persistent_count / len(persistent) if len(persistent) else 0.0
        fractions.append(fraction)
        files.append(
            {
                "sequence": record.sequence,
                "timestamp_ns": record.timestamp_ns,
                "source": str(record.path),
                "mask": str(destination),
                "point_count": len(persistent),
                "persistent_point_count": persistent_count,
                "persistent_fraction": fraction,
            }
        )
        if index % 25 == 0 or index == len(records):
            print(
                f"{label}: wrote {index}/{len(records)} masks",
                flush=True,
            )
    return (
        {
            "lidar_frame_count": len(records),
            "point_count": total_points,
            "persistent_point_count": total_persistent,
            "persistent_point_fraction": (
                total_persistent / total_points if total_points else 0.0
            ),
            "per_frame_persistent_fraction": distribution(fractions),
        },
        files,
    )


def build(
    *,
    data_root: Path,
    output_dir: Path,
    windows: tuple[Window, Window],
    voxel_meters: float,
    neighbor_voxels: int,
) -> dict[str, Any]:
    import numpy as np
    from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

    data_root = data_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    if not math.isfinite(voxel_meters) or voxel_meters <= 0.0:
        raise ValueError("voxel size must be positive and finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = AV2SensorDataLoader(data_root, data_root)
    available = set(loader.get_log_ids())
    missing = {window.sequence for window in windows} - available
    if missing:
        raise FileNotFoundError(f"TbV logs not found: {sorted(missing)}")
    records = tuple(_records(loader, window) for window in windows)
    first_translation = np.asarray(
        records[0][0].ego_to_city[:3, 3],
        dtype=np.float64,
    )
    origin = np.floor(first_translation)

    occupied = []
    input_counts = []
    for index, visit_records in enumerate(records):
        visit_occupied, point_count = _occupied_voxels(
            visit_records,
            origin=origin,
            voxel_meters=voxel_meters,
            label=f"visit {index}",
        )
        occupied.append(visit_occupied)
        input_counts.append(point_count)
    supported = (
        supported_unique_keys(
            occupied[0],
            occupied[1],
            radius_voxels=neighbor_voxels,
        ),
        supported_unique_keys(
            occupied[1],
            occupied[0],
            radius_voxels=neighbor_voxels,
        ),
    )

    sequence_stats = {}
    file_records = []
    for index, visit_records in enumerate(records):
        stats, files = _write_masks(
            visit_records,
            data_root=data_root,
            output_dir=output_dir,
            origin=origin,
            voxel_meters=voxel_meters,
            occupied=occupied[index],
            supported_voxels=supported[index],
            label=f"visit {index}",
        )
        stats.update(
            {
                "occupied_voxel_count": len(occupied[index]),
                "supported_voxel_count": int(supported[index].sum()),
                "supported_voxel_fraction": float(supported[index].mean()),
            }
        )
        sequence_stats[windows[index].sequence] = stats
        file_records.extend(files)

    report = {
        "format": "driving_scene_reconstruction.tbv_cross_visit_persistence.v0",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "policy": (
            "A point is persistent when the other visit occupies a voxel "
            "inside the configured Chebyshev neighborhood. Training may use "
            "this only to rescue recurrent geometry from a projected traffic "
            "mask; persistence alone is not semantic actor truth."
        ),
        "voxel_meters": voxel_meters,
        "neighbor_voxels": neighbor_voxels,
        "origin_city_meters": origin.tolist(),
        "windows": [
            {
                "sequence": window.sequence,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
            }
            for window in windows
        ],
        "sequences": sequence_stats,
        "files": file_records,
    }
    manifest_path = output_dir / "persistence_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        key: value for key, value in report.items() if key != "files"
    }
    summary["manifest"] = str(manifest_path)
    summary["manifest_sha256"] = digest
    summary["mask_file_count"] = len(file_records)
    print(json.dumps(summary, indent=2))
    return report


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"av2\.utils\.io",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument(
        "--window-start-seconds",
        action="append",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--window-end-seconds",
        action="append",
        type=float,
        required=True,
    )
    parser.add_argument("--voxel-meters", type=float, default=0.20)
    parser.add_argument("--neighbor-voxels", type=int, default=1)
    args = parser.parse_args()
    lengths = {
        len(args.sequence),
        len(args.window_start_seconds),
        len(args.window_end_seconds),
    }
    if lengths != {2}:
        parser.error(
            "exactly two equally repeated sequence/start/end windows are required"
        )
    windows = tuple(
        Window(sequence, start, end)
        for sequence, start, end in zip(
            args.sequence,
            args.window_start_seconds,
            args.window_end_seconds,
        )
    )
    if any(
        not math.isfinite(window.start_seconds)
        or not math.isfinite(window.end_seconds)
        or window.end_seconds <= window.start_seconds
        for window in windows
    ):
        parser.error("window bounds must be finite and ordered")
    build(
        data_root=args.data_root,
        output_dir=args.output_dir,
        windows=windows,  # type: ignore[arg-type]
        voxel_meters=args.voxel_meters,
        neighbor_voxels=args.neighbor_voxels,
    )


if __name__ == "__main__":
    main()
