#!/usr/bin/env python3
"""Validate the bounded TbV pair and its city-frame LiDAR registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from stage_h3_tbv_dataparser import TbVDataParserConfig


def _voxelized_shared_cloud(
    point_clouds, lidars, indices: np.ndarray, *, voxel_size: float
) -> np.ndarray:
    chunks = []
    for index in indices:
        points = point_clouds[int(index)][:, :3]
        radius = torch.linalg.norm(points[:, :2], dim=-1)
        points = points[
            (radius > 3.0)
            & (radius < 55.0)
            & (points[:, 2] > -3.0)
            & (points[:, 2] < 6.0)
        ][::5]
        homogeneous = torch.cat(
            (points, torch.ones((len(points), 1), dtype=points.dtype)), dim=1
        )
        transform = torch.eye(4, dtype=points.dtype)
        transform[:3, :4] = lidars.lidar_to_worlds[int(index)]
        chunks.append((homogeneous @ transform.T)[:, :3].numpy())
    cloud = np.concatenate(chunks)
    voxel = np.floor(cloud / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel, axis=0, return_index=True)
    return cloud[unique_indices]


def inspect(data: Path) -> dict[str, object]:
    outputs = (
        TbVDataParserConfig(data=data, train_split_fraction=0.9)
        .setup()
        .get_dataparser_outputs("train")
    )
    cameras = outputs.cameras
    lidars = outputs.metadata["lidars"]
    point_clouds = outputs.metadata["point_clouds"]
    lidar_sensor_ids = sorted(
        int(index)
        for index, name in outputs.metadata["sensor_idx_to_name"].items()
        if name.endswith("/lidar")
    )
    if len(lidar_sensor_ids) != 2:
        raise ValueError(f"expected two traversal LiDAR IDs, got {lidar_sensor_ids}")

    sensor_ids = lidars.metadata["sensor_idxs"].squeeze(-1).numpy()
    origins = lidars.lidar_to_worlds[:, :3, 3].numpy()
    shared_indices = []
    shared_counts = []
    for sensor_id, other_id in (
        (lidar_sensor_ids[0], lidar_sensor_ids[1]),
        (lidar_sensor_ids[1], lidar_sensor_ids[0]),
    ):
        indices = np.flatnonzero(sensor_ids == sensor_id)
        other_indices = np.flatnonzero(sensor_ids == other_id)
        nearest = cKDTree(origins[other_indices, :2]).query(origins[indices, :2])[0]
        selected = indices[nearest < 3.0]
        shared_indices.append(selected)
        shared_counts.append(int(len(selected)))

    clouds = [
        _voxelized_shared_cloud(point_clouds, lidars, indices, voxel_size=0.20)
        for indices in shared_indices
    ]
    distances = np.concatenate(
        (
            cKDTree(clouds[1]).query(clouds[0], workers=-1)[0],
            cKDTree(clouds[0]).query(clouds[1], workers=-1)[0],
        )
    )
    alignment = {
        "method": "symmetric nearest neighbour over shared-origin sweeps; 0.20 m voxel",
        "shared_sweeps": shared_counts,
        "voxel_points": [int(len(cloud)) for cloud in clouds],
        "samples": int(len(distances)),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "fraction_within_0_5_m": float(np.mean(distances <= 0.5)),
    }
    finite = {
        "camera_poses": bool(torch.isfinite(cameras.camera_to_worlds).all()),
        "lidar_poses": bool(torch.isfinite(lidars.lidar_to_worlds).all()),
        "point_clouds": bool(all(torch.isfinite(pc).all() for pc in point_clouds)),
    }
    gates = {
        "fourteen_camera_sensors": len(
            cameras.metadata["sensor_idxs"].unique()
        )
        == 14,
        "two_lidar_sensors": len(lidars.metadata["sensor_idxs"].unique()) == 2,
        "empty_actor_set": len(outputs.metadata["trajectories"]) == 0,
        "finite": all(finite.values()),
        "alignment_p50_at_most_0_20_m": alignment["p50_m"] <= 0.20,
        "alignment_p90_at_most_0_50_m": alignment["p90_m"] <= 0.50,
    }
    return {
        "data": str(data),
        "split": "train at 0.9 linspace per traversal-specific sensor",
        "images": len(outputs.image_filenames),
        "lidar_sweeps": len(lidars),
        "lidar_points": int(sum(len(pc) for pc in point_clouds)),
        "duration_seconds": float(outputs.metadata["duration"]),
        "camera_sensor_ids": cameras.metadata["sensor_idxs"].unique().tolist(),
        "lidar_sensor_ids": lidars.metadata["sensor_idxs"].unique().tolist(),
        "sensor_idx_to_name": outputs.metadata["sensor_idx_to_name"],
        "actor_trajectories": len(outputs.metadata["trajectories"]),
        "finite": finite,
        "alignment": alignment,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/yawei/stage3_external/data/tbv_branch_pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/yawei/stage3_external/artifacts/tbv_branch_pair_data_gate.json"
        ),
    )
    args = parser.parse_args()
    report = inspect(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
