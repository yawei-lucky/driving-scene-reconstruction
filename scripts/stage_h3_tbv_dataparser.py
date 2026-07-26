"""Minimal multi-traversal Argoverse TbV parser for the Stage H3 pilot.

TbV shares AV2 camera calibration and city-frame ego poses, but its LiDAR
Feathers contain one already ego-motion-compensated aggregate sweep.  In
particular, ``laser_number`` is 0..31 and must not be interpreted as AV2
Sensor's upper/lower LiDAR split.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.io import read_feather

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)

from stage_h3_tbv_lidar_masking import (
    closest_timestamp_index,
    projected_mask_exclusion,
)


AVAILABLE_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)
CAMERA_TO_BOTTOM_CROP = {
    "ring_front_center": 250,
    "ring_front_left": 0,
    "ring_front_right": 0,
    "ring_rear_left": 250,
    "ring_rear_right": 250,
    "ring_side_left": 0,
    "ring_side_right": 0,
}
DEFAULT_SEQUENCES = (
    "OCaNX1bQSmlP3jEQH80C0TZYzZhKLV81__Spring_2020",
    "QMnNKZiFaxnuGQmxpGkZFdM2EE7uWqDQ__Spring_2020",
)
DEFAULT_WINDOW_STARTS = (315972566.15, 315968138.15)
MAX_REFLECTANCE_VALUE = 255.0
HORIZONTAL_BEAM_DIVERGENCE = 3e-3
VERTICAL_BEAM_DIVERGENCE = 1.5e-3


@dataclass
class TbVDataParserConfig(ADDataParserConfig):
    """Configuration for the bounded two-traversal TbV pilot."""

    _target: Type = field(default_factory=lambda: TbV)
    data: Path = Path("/home/yawei/stage3_external/data/tbv_branch_pilot")
    sequence: str = "tbv_miami_branch_pair"
    sequences: Tuple[str, ...] = DEFAULT_SEQUENCES
    window_start_seconds: Tuple[float, ...] = DEFAULT_WINDOW_STARTS
    window_end_seconds: Tuple[float, ...] = ()
    cameras: Tuple[
        Literal[
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_left",
            "ring_side_right",
            "all",
            "none",
        ],
        ...,
    ] = ("all",)
    lidars: Tuple[Literal["lidar", "none"], ...] = ("lidar",)
    annotation_interval: float = 0.1
    load_cuboids: bool = False
    add_missing_points: bool = False
    allow_per_point_times: bool = False
    min_lidar_dist: Tuple[float, float, float] = (1.0, 2.0, 2.0)
    mask_root: Path | None = None
    mask_lidar_points: bool = False
    lidar_mask_max_camera_delta_ms: float = 50.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mask_lidar_points and self.mask_root is None:
            raise ValueError("mask_lidar_points requires mask_root")
        if (
            not math.isfinite(self.lidar_mask_max_camera_delta_ms)
            or self.lidar_mask_max_camera_delta_ms <= 0.0
        ):
            raise ValueError(
                "lidar_mask_max_camera_delta_ms must be positive and finite"
            )
        if len(self.sequences) != len(self.window_start_seconds):
            raise ValueError(
                "sequences and window_start_seconds must have equal length"
            )
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("sequences must be unique")
        if self.window_end_seconds:
            if len(self.window_end_seconds) != len(self.sequences):
                raise ValueError(
                    "window_end_seconds and sequences must have equal length"
                )
            for start, end in zip(
                self.window_start_seconds, self.window_end_seconds
            ):
                if not all(math.isfinite(value) for value in (start, end)):
                    raise ValueError("window bounds must be finite")
                if end <= start:
                    raise ValueError("window end must be greater than start")


@dataclass
class TbV(ADDataParser):
    """Load several TbV windows once in their shared city coordinate frame."""

    config: TbVDataParserConfig

    def _camera_names(self) -> Tuple[str, ...]:
        if "all" in self.config.cameras:
            return AVAILABLE_CAMERAS
        return tuple(self.config.cameras)

    def _local_time(self, traversal_idx: int, timestamp_ns: int) -> float:
        start_ns = round(self.config.window_start_seconds[traversal_idx] * 1e9)
        return (timestamp_ns - start_ns) / 1e9

    def _timestamp_in_window(
        self, traversal_idx: int, timestamp_ns: int
    ) -> bool:
        if not self.config.window_end_seconds:
            return True
        start_ns = round(
            self.config.window_start_seconds[traversal_idx] * 1e9
        )
        end_ns = round(
            self.config.window_end_seconds[traversal_idx] * 1e9
        )
        return start_ns <= timestamp_ns <= end_ns

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        filenames: List[Path] = []
        times: list[float] = []
        intrinsics: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        idxs: list[int] = []
        heights: list[int] = []
        widths: list[int] = []
        camera_names = self._camera_names()

        for traversal_idx, sequence in enumerate(self.config.sequences):
            for camera_idx, camera_name in enumerate(camera_names):
                camera = self.av2.get_log_pinhole_camera(sequence, camera_name)
                camera_paths = self.av2.get_ordered_log_cam_fpaths(
                    sequence, camera_name
                )
                for path in camera_paths:
                    timestamp_ns = int(path.stem)
                    if not self._timestamp_in_window(
                        traversal_idx, timestamp_ns
                    ):
                        continue
                    ego_to_world = self.av2.get_city_SE3_ego(
                        sequence, timestamp_ns
                    )
                    camera_to_ego = camera.ego_SE3_cam.transform_matrix.copy()
                    camera_to_ego[:3, :3] = (
                        camera_to_ego[:3, :3] @ OPENCV_TO_NERFSTUDIO
                    )
                    filenames.append(path)
                    times.append(self._local_time(traversal_idx, timestamp_ns))
                    intrinsics.append(camera.intrinsics.K)
                    poses.append(ego_to_world.transform_matrix @ camera_to_ego)
                    idxs.append(traversal_idx * len(camera_names) + camera_idx)
                    heights.append(
                        camera.intrinsics.height_px
                        - CAMERA_TO_BOTTOM_CROP[camera_name]
                    )
                    widths.append(camera.intrinsics.width_px)

        intrinsics_tensor = torch.from_numpy(np.asarray(intrinsics)).float()
        pose_tensor = torch.from_numpy(np.asarray(poses)).float()
        cameras = Cameras(
            fx=intrinsics_tensor[:, 0, 0],
            fy=intrinsics_tensor[:, 1, 1],
            cx=intrinsics_tensor[:, 0, 2],
            cy=intrinsics_tensor[:, 1, 2],
            height=torch.tensor(heights, dtype=torch.int32),
            width=torch.tensor(widths, dtype=torch.int32),
            camera_to_worlds=pose_tensor[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(times, dtype=torch.float64),
            metadata={
                "sensor_idxs": torch.tensor(idxs, dtype=torch.int32).unsqueeze(-1)
            },
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        filenames: List[Path] = []
        times: list[float] = []
        poses: list[np.ndarray] = []
        idxs: list[int] = []
        lidar_index_base = len(self.config.sequences) * len(self._camera_names())

        for traversal_idx, sequence in enumerate(self.config.sequences):
            for timestamp_ns in self.av2.get_ordered_log_lidar_timestamps(sequence):
                if not self._timestamp_in_window(
                    traversal_idx, timestamp_ns
                ):
                    continue
                path = self.av2.get_lidar_fpath(sequence, timestamp_ns)
                ego_to_world = self.av2.get_city_SE3_ego(sequence, timestamp_ns)
                filenames.append(path)
                times.append(self._local_time(traversal_idx, timestamp_ns))
                poses.append(ego_to_world.transform_matrix)
                idxs.append(lidar_index_base + traversal_idx)

        pose_tensor = torch.from_numpy(np.asarray(poses)).float()
        lidars = Lidars(
            lidar_to_worlds=pose_tensor[:, :3, :4],
            lidar_type=LidarType.VELODYNE_VLP32C,
            times=torch.tensor(times, dtype=torch.float64),
            metadata={
                "sensor_idxs": torch.tensor(idxs, dtype=torch.int32).unsqueeze(-1)
            },
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
            assume_ego_compensated=True,
        )
        return lidars, filenames

    def _read_lidars(
        self, lidars: Lidars, filepaths: List[Path]
    ) -> List[torch.Tensor]:
        from PIL import Image

        point_clouds = []
        input_points = 0
        removed_points = 0
        frames_with_removals = 0
        camera_mask_hits: Counter[str] = Counter()
        camera_images_used: Counter[str] = Counter()
        camera_images_missing: Counter[str] = Counter()
        camera_delta_ms: list[float] = []
        data_root = self.config.data.expanduser().resolve()
        mask_root = (
            self.config.mask_root.expanduser().resolve()
            if self.config.mask_root is not None
            else None
        )
        camera_paths: dict[tuple[str, str], tuple[Path, ...]] = {}
        camera_timestamps: dict[tuple[str, str], tuple[int, ...]] = {}
        if self.config.mask_lidar_points:
            for sequence in self.config.sequences:
                for camera_name in self._camera_names():
                    key = (sequence, camera_name)
                    paths = tuple(
                        self.av2.get_ordered_log_cam_fpaths(
                            sequence, camera_name
                        )
                    )
                    camera_paths[key] = paths
                    camera_timestamps[key] = tuple(
                        int(camera_path.stem) for camera_path in paths
                    )
        max_delta_ns = round(
            self.config.lidar_mask_max_camera_delta_ms * 1e6
        )
        for path in filepaths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                frame = read_feather(path)
            xyz = frame.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
            intensity = (
                frame["intensity"].to_numpy(dtype=np.float32)
                / MAX_REFLECTANCE_VALUE
            )
            relative_time = np.zeros(len(frame), dtype=np.float32)
            point_cloud = np.column_stack(
                (xyz, intensity, relative_time)
            ).astype(np.float32, copy=False)
            input_points += len(point_cloud)
            if self.config.mask_lidar_points:
                assert mask_root is not None
                sequence = path.parents[2].name
                lidar_timestamp_ns = int(path.stem)
                excluded = np.zeros(len(point_cloud), dtype=bool)
                for camera_name in self._camera_names():
                    key = (sequence, camera_name)
                    camera_index = closest_timestamp_index(
                        camera_timestamps[key],
                        lidar_timestamp_ns,
                        max_delta_ns,
                    )
                    if camera_index is None:
                        camera_images_missing[camera_name] += 1
                        continue
                    image_path = camera_paths[key][camera_index]
                    mask_path = (
                        mask_root
                        / image_path.resolve()
                        .relative_to(data_root)
                        .with_suffix(".png")
                    )
                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            f"TbV LiDAR projection mask is missing: {mask_path}"
                        )
                    with Image.open(mask_path) as mask_image:
                        valid_pixel_mask = np.asarray(
                            mask_image.convert("L")
                        )
                    camera_timestamp_ns = int(image_path.stem)
                    camera_delta_ms.append(
                        abs(camera_timestamp_ns - lidar_timestamp_ns) / 1e6
                    )
                    image_points, _, projection_valid = (
                        self.av2.project_ego_to_img_motion_compensated(
                            points_lidar_time=xyz,
                            cam_name=camera_name,
                            cam_timestamp_ns=camera_timestamp_ns,
                            lidar_timestamp_ns=lidar_timestamp_ns,
                            log_id=sequence,
                        )
                    )
                    camera_excluded = projected_mask_exclusion(
                        image_points,
                        projection_valid,
                        valid_pixel_mask,
                    )
                    excluded |= camera_excluded
                    camera_images_used[camera_name] += 1
                    camera_mask_hits[camera_name] += int(
                        camera_excluded.sum()
                    )
                removed = int(excluded.sum())
                if removed:
                    frames_with_removals += 1
                removed_points += removed
                point_cloud = point_cloud[~excluded]
            point_clouds.append(torch.from_numpy(point_cloud))
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        self._lidar_mask_filter_stats = {
            "enabled": self.config.mask_lidar_points,
            "checkpoint_trained_with_filter": bool(
                self.config.mask_lidar_points
                or getattr(
                    self.config,
                    "_checkpoint_trained_with_lidar_mask_filter",
                    False,
                )
            ),
            "evaluation_filter_skipped": bool(
                not self.config.mask_lidar_points
                and getattr(
                    self.config,
                    "_checkpoint_trained_with_lidar_mask_filter",
                    False,
                )
            ),
            "input_frame_count": len(filepaths),
            "input_point_count": input_points,
            "removed_point_count": removed_points,
            "retained_point_count": input_points - removed_points,
            "removed_point_fraction": (
                removed_points / input_points if input_points else 0.0
            ),
            "frames_with_removals": frames_with_removals,
            "camera_images_used": dict(sorted(camera_images_used.items())),
            "camera_images_missing": dict(
                sorted(camera_images_missing.items())
            ),
            "camera_mask_hits": dict(sorted(camera_mask_hits.items())),
            "camera_projection_coverage_fraction": (
                sum(camera_images_used.values())
                / (len(filepaths) * len(self._camera_names()))
                if filepaths
                else 0.0
            ),
            "max_camera_delta_ms": max(camera_delta_ms, default=0.0),
            "configured_max_camera_delta_ms": (
                self.config.lidar_mask_max_camera_delta_ms
            ),
            "policy": (
                "exclude a return if any closest synchronized camera "
                "projects it onto a zero-valued traffic-mask pixel"
            ),
        }
        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        return []

    def _sensor_idx_to_name(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        camera_names = self._camera_names()
        for traversal_idx, sequence in enumerate(self.config.sequences):
            short_sequence = sequence.split("__", 1)[0][:4]
            for camera_idx, camera_name in enumerate(camera_names):
                mapping[traversal_idx * len(camera_names) + camera_idx] = (
                    f"{short_sequence}/{camera_name}"
                )
        lidar_index_base = len(self.config.sequences) * len(camera_names)
        for traversal_idx, sequence in enumerate(self.config.sequences):
            short_sequence = sequence.split("__", 1)[0][:4]
            mapping[lidar_index_base + traversal_idx] = f"{short_sequence}/lidar"
        return mapping

    def _generate_dataparser_outputs(self, split: str = "train"):
        self.av2 = AV2SensorDataLoader(self.config.data, self.config.data)
        available = set(self.av2.get_log_ids())
        missing = set(self.config.sequences) - available
        if missing:
            raise FileNotFoundError(f"TbV log directories not found: {sorted(missing)}")
        outputs = super()._generate_dataparser_outputs(split=split)
        outputs.metadata["sensor_idx_to_name"] = self._sensor_idx_to_name()
        outputs.metadata["lidar_mask_filter"] = dict(
            self._lidar_mask_filter_stats
        )
        outputs.metadata["lidar_mask_filter"]["output_split"] = split
        outputs.metadata["lidar_mask_filter"]["output_frame_count"] = len(
            outputs.metadata["point_clouds"]
        )
        outputs.metadata["lidar_mask_filter"]["output_point_count"] = sum(
            len(point_cloud)
            for point_cloud in outputs.metadata["point_clouds"]
        )
        if self.config.mask_root is not None:
            data_root = self.config.data.expanduser().resolve()
            mask_root = self.config.mask_root.expanduser().resolve()
            mask_filenames = [
                mask_root
                / path.resolve().relative_to(data_root).with_suffix(".png")
                for path in outputs.image_filenames
            ]
            missing_masks = [
                path for path in mask_filenames if not path.is_file()
            ]
            if missing_masks:
                preview = ", ".join(str(path) for path in missing_masks[:3])
                raise FileNotFoundError(
                    f"{len(missing_masks)} TbV masks are missing: {preview}"
                )
            outputs.mask_filenames = mask_filenames
            outputs.metadata["mask_root"] = str(mask_root)
        del self.av2
        return outputs
