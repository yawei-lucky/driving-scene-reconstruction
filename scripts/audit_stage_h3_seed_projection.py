#!/usr/bin/env python3
"""Compare scan-centre and per-point LiDAR actor seed assignment.

The audit operates on raw PandaSet world points and calibrated moving cuboid
trajectories.  It uses point-cloud semantics as the quantitative independent
signal and draws both seed sets over the real back-camera image.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pandaset import DataSet
from scipy.spatial.transform import Rotation, Slerp
import yaml

from driving_scene_reconstruction.h3.vehicle_object_layer import (
    PandaSetVehicleObjectParserConfig,
)
from nerfstudio.data.dataparsers import pandaset_dataparser


PANDAR64_ID = 0
VEHICLE_SEMANTIC_ID = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actors", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[19, 39, 59, 69, 78],
    )
    return parser.parse_args()


def load_parser_config(path: Path) -> PandaSetVehicleObjectParserConfig:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    source = loaded.pipeline.datamanager.dataparser
    kwargs = {
        definition.name: deepcopy(getattr(source, definition.name))
        for definition in fields(PandaSetVehicleObjectParserConfig)
        if (
            definition.init
            and definition.name != "_target"
            and hasattr(source, definition.name)
        )
    }
    config = PandaSetVehicleObjectParserConfig(**kwargs)
    config.include_stationary_rigid_actors = False
    config.include_deformable_actors = False
    config.use_calibrated_lidar_frame_for_cuboid_time = True
    return config


def pandaset_pose(pose: dict[str, Any]) -> np.ndarray:
    return pandaset_dataparser._pandaset_pose_to_matrix(pose)


class PoseInterpolator:
    """Position interpolation plus quaternion SLERP."""

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=np.float64)[order]
        self.poses = np.asarray(poses, dtype=np.float64)[order]
        self.positions = self.poses[:, :3, 3]
        self.slerp = Slerp(
            self.times,
            Rotation.from_matrix(self.poses[:, :3, :3]),
        )

    def query(
        self,
        query_times: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.atleast_1d(np.asarray(query_times, dtype=np.float64))
        clipped = np.clip(raw, self.times[0], self.times[-1])
        positions = np.stack(
            [
                np.interp(clipped, self.times, self.positions[:, axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        rotations = self.slerp(clipped).as_matrix()
        present = (raw >= self.times[0]) & (raw <= self.times[-1])
        return rotations, positions, present


def points_to_local(
    points_world: np.ndarray,
    query_times: np.ndarray,
    actor: PoseInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    rotations, positions, present = actor.query(query_times)
    local = np.einsum(
        "ni,nij->nj",
        points_world - positions,
        rotations,
    )
    return local, present


def project_world(
    points_world: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
    intrinsics: Any,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = np.einsum(
        "ni,nij->nj",
        points_world - positions,
        rotations,
    )
    valid = camera_points[:, 2] > 1e-6
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    uv[:, 0] = (
        intrinsics.fx * camera_points[:, 0] / camera_points[:, 2]
        + intrinsics.cx
    )
    uv[:, 1] = (
        intrinsics.fy * camera_points[:, 1] / camera_points[:, 2]
        + intrinsics.cy
    )
    return uv, valid


def rolling_project(
    local_points: np.ndarray,
    actor: PoseInterpolator,
    camera: PoseInterpolator,
    image_time: float,
    intrinsics: Any,
    height: int,
    rolling_shutter_time: float,
    time_to_center_pixel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_times = np.full(
        len(local_points),
        image_time + time_to_center_pixel,
        dtype=np.float64,
    )
    convergence = np.full(len(local_points), np.inf, dtype=np.float64)
    uv = np.full((len(local_points), 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(local_points), dtype=np.bool_)
    for _ in range(5):
        actor_rot, actor_pos, actor_present = actor.query(row_times)
        world = (
            np.einsum("ni,nji->nj", local_points, actor_rot)
            + actor_pos
        )
        camera_rot, camera_pos, camera_present = camera.query(row_times)
        uv, in_front = project_world(
            world,
            camera_rot,
            camera_pos,
            intrinsics,
        )
        new_times = (
            image_time
            + time_to_center_pixel
            + (
                np.clip(uv[:, 1], 0, height - 1) / max(height - 1, 1)
                - 0.5
            )
            * rolling_shutter_time
        )
        convergence = np.abs(new_times - row_times)
        row_times = new_times
        valid = actor_present & camera_present & in_front
    return uv, valid, convergence


def actor_uuids(sequence: Any) -> list[str]:
    ordered: list[str] = []
    for frame_index in range(pandaset_dataparser.PANDASET_SEQ_LEN):
        current = sequence.cuboids[frame_index]
        for _, item in current.iterrows():
            if bool(item["stationary"]):
                continue
            if item["label"] not in pandaset_dataparser.ALLOWED_RIGID_CLASSES:
                continue
            if int(item["cuboids.sensor_id"]) == 1:
                continue
            uuid = str(item["uuid"])
            if uuid not in ordered:
                ordered.append(uuid)
    return ordered


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(
    image: Image.Image,
    centre_uv: np.ndarray,
    centre_valid: np.ndarray,
    exact_uv: np.ndarray,
    exact_valid: np.ndarray,
    title: str,
    left_label: str = "scan-centre seeds",
    right_label: str = "per-point-time seeds",
) -> Image.Image:
    base = image.convert("RGB")
    visible_sets = [
        uv[valid]
        for uv, valid in (
            (centre_uv, centre_valid),
            (exact_uv, exact_valid),
        )
        if valid.any()
    ]
    crop_left = 0
    crop_top = 0
    if visible_sets:
        visible = np.concatenate(visible_sets, axis=0)
        centre = np.median(visible, axis=0)
        crop_width = min(base.width, 560)
        crop_height = min(base.height, 360)
        crop_left = int(
            np.clip(
                centre[0] - crop_width / 2,
                0,
                base.width - crop_width,
            )
        )
        crop_top = int(
            np.clip(
                centre[1] - crop_height / 2,
                0,
                base.height - crop_height,
            )
        )
        base = base.crop(
            (
                crop_left,
                crop_top,
                crop_left + crop_width,
                crop_top + crop_height,
            )
        )
    panels = []
    for name, uv, valid, color in (
        (left_label, centre_uv, centre_valid, (50, 180, 255)),
        (right_label, exact_uv, exact_valid, (255, 220, 30)),
    ):
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        for raw_x, raw_y in uv[valid]:
            x = raw_x - crop_left
            y = raw_y - crop_top
            if 0 <= x < panel.width and 0 <= y < panel.height:
                draw.ellipse(
                    (x - 3, y - 3, x + 3, y + 3),
                    fill=color,
                    outline=(0, 0, 0),
                    width=1,
                )
        draw.rectangle((0, 0, panel.width, 54), fill=(0, 0, 0))
        draw.text(
            (14, 11),
            f"{title} | {name}",
            fill=color,
            font=font(25),
        )
        panels.append(panel)
    panels = [
        panel.resize(
            (panel.width * 2, panel.height * 2),
            Image.Resampling.LANCZOS,
        )
        for panel in panels
    ]
    combined = Image.new(
        "RGB",
        (panels[0].width * 2, panels[0].height),
    )
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (panels[0].width, 0))
    return combined


def finite_projection_mask(
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return (
        valid
        & np.isfinite(uv).all(axis=-1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_parser_config(args.source_config)
    dataset = DataSet(str(config.data.absolute()))
    sequence = dataset[config.sequence]
    sequence.load()
    sequence.load_semseg()

    parser = config.setup()
    parser.sequence = sequence
    trajectories = parser._get_actor_trajectories()
    uuids = actor_uuids(sequence)
    if len(uuids) != len(trajectories):
        raise RuntimeError(
            f"actor UUID/trajectory mismatch: {len(uuids)} != "
            f"{len(trajectories)}"
        )

    back_camera = sequence.camera["back_camera"]
    camera_poses = np.stack([pandaset_pose(p) for p in back_camera.poses])
    camera_interp = PoseInterpolator(
        np.asarray(back_camera.timestamps),
        camera_poses,
    )
    train_indices = set(
        np.linspace(
            0,
            pandaset_dataparser.PANDASET_SEQ_LEN - 1,
            int(np.ceil(config.train_split_fraction * 80)),
            dtype=np.int64,
        ).tolist()
    )
    padding = np.asarray([0.25, 0.25, 0.1], dtype=np.float64)
    records: list[dict[str, Any]] = []
    overlays: list[str] = []

    for actor_index in args.actors:
        trajectory = trajectories[actor_index]
        actor_interp = PoseInterpolator(
            trajectory["timestamps"].numpy(),
            trajectory["poses"].numpy(),
        )
        half_extent = (
            trajectory["dims"].numpy().astype(np.float64) + padding
        ) / 2
        for frame_index in args.frames:
            raw = sequence.lidar[frame_index].to_numpy()
            lidar_mask = raw[:, -1].astype(np.int64) == PANDAR64_ID
            points = raw[lidar_mask, :3].astype(np.float64)
            point_times = raw[lidar_mask, 4].astype(np.float64)
            semantics = (
                sequence.semseg[frame_index]
                .to_numpy()
                .reshape(-1)[lidar_mask]
            )
            scan_time = float(
                sequence.camera["front_camera"].timestamps[frame_index]
            )
            point_time_offsets = point_times - scan_time
            if (
                point_time_offsets.min() < -0.06
                or point_time_offsets.max() > 0.06
            ):
                raise RuntimeError(
                    "unexpected Pandar64 point-time offsets: "
                    f"{point_time_offsets.min():.6f}, "
                    f"{point_time_offsets.max():.6f}"
                )
            centre_times = np.full(len(points), scan_time)
            centre_local, centre_present = points_to_local(
                points,
                centre_times,
                actor_interp,
            )
            exact_local, exact_present = points_to_local(
                points,
                point_times,
                actor_interp,
            )
            centre_assignment = centre_present & (
                np.abs(centre_local) < half_extent
            ).all(axis=-1)
            exact_assignment = exact_present & (
                np.abs(exact_local) < half_extent
            ).all(axis=-1)
            union = centre_assignment | exact_assignment
            intersection = centre_assignment & exact_assignment

            image = back_camera[frame_index].crop(
                (0, 0, back_camera[frame_index].width, 820)
            )
            image_time = float(back_camera.timestamps[frame_index])
            centre_uv, centre_rs_valid, centre_convergence = rolling_project(
                centre_local[centre_assignment],
                actor_interp,
                camera_interp,
                image_time,
                back_camera.intrinsics,
                image.height,
                float(config.rolling_shutter_time),
                float(config.time_to_center_pixel),
            )
            exact_uv, exact_rs_valid, exact_convergence = rolling_project(
                exact_local[exact_assignment],
                actor_interp,
                camera_interp,
                image_time,
                back_camera.intrinsics,
                image.height,
                float(config.rolling_shutter_time),
                float(config.time_to_center_pixel),
            )
            centre_visible = finite_projection_mask(
                centre_uv,
                centre_rs_valid,
                image.width,
                image.height,
            )
            exact_visible = finite_projection_mask(
                exact_uv,
                exact_rs_valid,
                image.width,
                image.height,
            )
            camera_rotation, camera_position, _ = camera_interp.query(
                image_time
            )
            current_paint_uv, current_paint_valid = project_world(
                points[exact_assignment],
                np.repeat(camera_rotation, exact_assignment.sum(), axis=0),
                np.repeat(camera_position, exact_assignment.sum(), axis=0),
                back_camera.intrinsics,
            )
            current_paint_visible = finite_projection_mask(
                current_paint_uv,
                current_paint_valid,
                image.width,
                image.height,
            )
            paint_pair_valid = current_paint_visible & exact_visible
            paint_displacement = np.linalg.norm(
                current_paint_uv[paint_pair_valid]
                - exact_uv[paint_pair_valid],
                axis=-1,
            )
            image_array = np.asarray(image)
            if paint_pair_valid.any():
                current_pixels = np.rint(
                    current_paint_uv[paint_pair_valid]
                ).astype(np.int64)
                exact_pixels = np.rint(
                    exact_uv[paint_pair_valid]
                ).astype(np.int64)
                current_pixels[:, 0] = np.clip(
                    current_pixels[:, 0], 0, image.width - 1
                )
                current_pixels[:, 1] = np.clip(
                    current_pixels[:, 1], 0, image.height - 1
                )
                exact_pixels[:, 0] = np.clip(
                    exact_pixels[:, 0], 0, image.width - 1
                )
                exact_pixels[:, 1] = np.clip(
                    exact_pixels[:, 1], 0, image.height - 1
                )
                current_rgb = image_array[
                    current_pixels[:, 1], current_pixels[:, 0]
                ].astype(np.float32)
                exact_rgb = image_array[
                    exact_pixels[:, 1], exact_pixels[:, 0]
                ].astype(np.float32)
                paint_rgb_l1 = np.abs(current_rgb - exact_rgb).mean(
                    axis=-1
                )
            else:
                paint_rgb_l1 = np.empty(0, dtype=np.float32)

            overlay_path = (
                args.output_dir
                / f"frame_{frame_index:02d}_actor_{actor_index:02d}.jpg"
            )
            draw_panel(
                image,
                centre_uv,
                centre_visible,
                exact_uv,
                exact_visible,
                (
                    f"frame {frame_index:02d} "
                    f"actor {actor_index} "
                    f"{'train' if frame_index in train_indices else 'heldout'}"
                ),
            ).save(overlay_path, quality=94)
            overlays.append(str(overlay_path))
            paint_overlay_path = (
                args.output_dir
                / (
                    f"frame_{frame_index:02d}_actor_{actor_index:02d}"
                    "_painting.jpg"
                )
            )
            draw_panel(
                image,
                current_paint_uv,
                current_paint_visible,
                exact_uv,
                exact_visible,
                (
                    f"frame {frame_index:02d} "
                    f"actor {actor_index} seed painting"
                ),
                left_label="current fixed-pose paint",
                right_label="point-time + RS paint",
            ).save(paint_overlay_path, quality=94)
            overlays.append(str(paint_overlay_path))

            centre_semantics = semantics[centre_assignment]
            exact_semantics = semantics[exact_assignment]
            records.append(
                {
                    "actor_index": actor_index,
                    "actor_uuid": uuids[actor_index],
                    "frame": frame_index,
                    "split": (
                        "train"
                        if frame_index in train_indices
                        else "heldout"
                    ),
                    "camera": "back_camera",
                    "point_time_offset_seconds": {
                        "minimum": float(point_time_offsets.min()),
                        "median": float(np.median(point_time_offsets)),
                        "maximum": float(point_time_offsets.max()),
                    },
                    "scan_centre_assigned": int(centre_assignment.sum()),
                    "per_point_assigned": int(exact_assignment.sum()),
                    "assignment_intersection": int(intersection.sum()),
                    "assignment_union": int(union.sum()),
                    "changed_over_union": (
                        float((union & ~intersection).sum() / union.sum())
                        if union.any()
                        else None
                    ),
                    "scan_centre_vehicle_semantic_precision": (
                        float(
                            (centre_semantics == VEHICLE_SEMANTIC_ID).mean()
                        )
                        if len(centre_semantics)
                        else None
                    ),
                    "per_point_vehicle_semantic_precision": (
                        float((exact_semantics == VEHICLE_SEMANTIC_ID).mean())
                        if len(exact_semantics)
                        else None
                    ),
                    "scan_centre_visible_points": int(
                        centre_visible.sum()
                    ),
                    "per_point_visible_points": int(exact_visible.sum()),
                    "paint_comparable_points": int(
                        paint_pair_valid.sum()
                    ),
                    "paint_projection_displacement_pixels": {
                        "median": (
                            float(np.median(paint_displacement))
                            if len(paint_displacement)
                            else None
                        ),
                        "p95": (
                            float(np.quantile(paint_displacement, 0.95))
                            if len(paint_displacement)
                            else None
                        ),
                        "maximum": (
                            float(paint_displacement.max())
                            if len(paint_displacement)
                            else None
                        ),
                    },
                    "paint_source_rgb_l1_255": {
                        "mean": (
                            float(paint_rgb_l1.mean())
                            if len(paint_rgb_l1)
                            else None
                        ),
                        "p95": (
                            float(np.quantile(paint_rgb_l1, 0.95))
                            if len(paint_rgb_l1)
                            else None
                        ),
                    },
                    "scan_centre_rs_converged_fraction": (
                        float((centre_convergence < 1e-5).mean())
                        if len(centre_convergence)
                        else None
                    ),
                    "per_point_rs_converged_fraction": (
                        float((exact_convergence < 1e-5).mean())
                        if len(exact_convergence)
                        else None
                    ),
                    "overlay": str(overlay_path),
                    "paint_overlay": str(paint_overlay_path),
                }
            )

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        centre_count = sum(
            record["scan_centre_assigned"] for record in selected
        )
        exact_count = sum(
            record["per_point_assigned"] for record in selected
        )
        weighted_centre_vehicle = sum(
            record["scan_centre_assigned"]
            * (record["scan_centre_vehicle_semantic_precision"] or 0)
            for record in selected
        )
        weighted_exact_vehicle = sum(
            record["per_point_assigned"]
            * (record["per_point_vehicle_semantic_precision"] or 0)
            for record in selected
        )
        return {
            "records": len(selected),
            "nonempty_records": sum(
                record["scan_centre_assigned"] > 0
                or record["per_point_assigned"] > 0
                for record in selected
            ),
            "scan_centre_assigned": centre_count,
            "per_point_assigned": exact_count,
            "scan_centre_vehicle_semantic_precision": (
                weighted_centre_vehicle / centre_count
                if centre_count
                else None
            ),
            "per_point_vehicle_semantic_precision": (
                weighted_exact_vehicle / exact_count
                if exact_count
                else None
            ),
        }

    def painting_aggregate(
        selected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected = [
            record
            for record in selected
            if record["paint_comparable_points"] > 0
        ]
        count = sum(record["paint_comparable_points"] for record in selected)

        def weighted(field: str, statistic: str) -> float | None:
            if count == 0:
                return None
            return sum(
                record["paint_comparable_points"]
                * record[field][statistic]
                for record in selected
            ) / count

        return {
            "nonempty_records": len(selected),
            "comparable_points": count,
            "weighted_frame_median_displacement_pixels": weighted(
                "paint_projection_displacement_pixels",
                "median",
            ),
            "weighted_frame_p95_displacement_pixels": weighted(
                "paint_projection_displacement_pixels",
                "p95",
            ),
            "weighted_source_rgb_l1_255": weighted(
                "paint_source_rgb_l1_255",
                "mean",
            ),
        }

    summary = {
        "source_config": str(args.source_config),
        "data": str(config.data),
        "sequence": config.sequence,
        "actors": args.actors,
        "actor_uuid_map": {
            str(index): uuid for index, uuid in enumerate(uuids)
        },
        "frames": args.frames,
        "camera": "back_camera",
        "camera_height_after_parser_crop": 820,
        "rolling_shutter_seconds": float(config.rolling_shutter_time),
        "time_to_center_pixel_seconds": float(
            config.time_to_center_pixel
        ),
        "vehicle_semantic_id": VEHICLE_SEMANTIC_ID,
        "records": records,
        "aggregate": aggregate(records),
        "aggregate_by_split": {
            split: aggregate(
                [record for record in records if record["split"] == split]
            )
            for split in ("train", "heldout")
        },
        "painting_aggregate_by_split": {
            split: painting_aggregate(
                [record for record in records if record["split"] == split]
            )
            for split in ("train", "heldout")
        },
        "aggregate_by_actor": {
            str(actor_index): aggregate(
                [
                    record
                    for record in records
                    if record["actor_index"] == actor_index
                ]
            )
            for actor_index in args.actors
        },
        "scope": (
            "Quantitative precision uses PandaSet point-cloud semantics. "
            "Source-image overlays are qualitative because PandaSet does not "
            "provide dense 2D vehicle masks for this sequence."
        ),
        "overlays": overlays,
    }
    report_path = args.output_dir / "seed_projection_audit.json"
    report_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"WROTE: {report_path}")


if __name__ == "__main__":
    main()
