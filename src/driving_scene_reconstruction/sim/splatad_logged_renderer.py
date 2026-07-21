"""Logged-time PandaSet SplatAD renderer for the drivable H3 loop.

Heavy H3 dependencies are imported only by :meth:`load`, preserving the
dependency-free simulator interfaces and tests.
"""

from __future__ import annotations

import bisect
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .renderer_interface import CameraRig, RenderedObservation
from .scene_coordinates import NearbyPoseLimits, SceneReferenceFrame
from .state import EgoState


PANDASET_CAMERA_ORDER = (
    "front_left",
    "front",
    "front_right",
    "left",
    "back",
    "right",
)
H3_NEARBY_POSE_LIMITS = NearbyPoseLimits(
    max_abs_forward_meters=0.5,
    max_abs_left_meters=0.25,
    max_abs_yaw_radians=math.radians(2.0),
)


class SplatADLoggedRenderer:
    """Render a PandaSet SplatAD checkpoint along its logged trajectory.

    ``EgoState.time`` is elapsed log time in seconds. ``x``, ``y``, and
    ``yaw`` are respectively a small forward, left, and left-yaw offset from
    the selected logged ego pose. All requested cameras receive the same
    offset while retaining their calibrated per-sensor poses and timestamps.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        output_scale: float = 0.5,
        limits: NearbyPoseLimits | None = None,
        lazy: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not 0.0 < output_scale <= 1.0:
            raise ValueError("output_scale must be in (0, 1]")
        self.output_scale = output_scale
        self.limits = limits or H3_NEARBY_POSE_LIMITS
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._records: dict[int, dict[str, Any]] = {}
        self._reference_frames: dict[int, SceneReferenceFrame] = {}
        self._frame_numbers: tuple[int, ...] = ()
        self._frame_times: tuple[float, ...] = ()
        self._camera_times: dict[int, dict[str, float]] = {}
        self._checkpoint_path: Path | None = None
        self._checkpoint_step: int | None = None
        self._sequence: str | None = None
        self._sensor_source_names: dict[str, str] = {}
        if not lazy:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def camera_names(self) -> tuple[str, ...]:
        if not self.is_loaded:
            self.load()
        return PANDASET_CAMERA_ORDER

    @property
    def logged_duration(self) -> float:
        if not self.is_loaded:
            self.load()
        return self._frame_times[-1]

    @property
    def checkpoint_path(self) -> Path:
        if not self.is_loaded:
            self.load()
        assert self._checkpoint_path is not None
        return self._checkpoint_path

    @property
    def checkpoint_step(self) -> int:
        if not self.is_loaded:
            self.load()
        assert self._checkpoint_step is not None
        return self._checkpoint_step

    @staticmethod
    def _nearest_index(times: tuple[float, ...], query: float) -> int:
        if not times:
            raise ValueError("times cannot be empty")
        if not math.isfinite(query):
            raise ValueError("query time must be finite")
        insertion = bisect.bisect_left(times, query)
        if insertion == 0:
            return 0
        if insertion == len(times):
            return len(times) - 1
        before = insertion - 1
        if query - times[before] <= times[insertion] - query:
            return before
        return insertion

    def load(self) -> None:
        """Load the static checkpoint and merge train/held-out camera splits."""

        if self.is_loaded:
            return
        try:
            import torch
            from nerfstudio.scripts.render import streamline_ad_config
            from nerfstudio.utils.eval_utils import eval_setup
        except ImportError as error:
            raise RuntimeError(
                "SplatADLoggedRenderer requires the pinned Stage H3 "
                "environment with nerfstudio and torch"
            ) from error

        config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
            self.config_path,
            test_mode="test",
            update_config_callback=streamline_ad_config,
        )
        pipeline.model.eval()
        datamanager = pipeline.datamanager
        sensor_idx_to_name = datamanager.train_dataparser_outputs.metadata[
            "sensor_idx_to_name"
        ]
        canonical_by_index: dict[int, str] = {}
        sensor_source_names: dict[str, str] = {}
        for raw_index, source_name in sensor_idx_to_name.items():
            canonical = str(source_name).removesuffix("_camera")
            if canonical in PANDASET_CAMERA_ORDER:
                canonical_by_index[int(raw_index)] = canonical
                sensor_source_names[canonical] = str(source_name)
        if set(canonical_by_index.values()) != set(PANDASET_CAMERA_ORDER):
            raise RuntimeError(
                "unexpected PandaSet camera mapping: "
                f"{sensor_idx_to_name}"
            )
        split_sources = (
            (datamanager.train_dataset, datamanager.cached_train),
            (datamanager.eval_dataset, datamanager.cached_eval),
        )
        records: dict[int, dict[str, Any]] = {}
        for dataset, cache in split_sources:
            for index, filename in enumerate(dataset.image_filenames):
                # Accessing the cache applies the upstream image crop and
                # updates the dataset camera intrinsics before it is copied.
                _ = cache[index]
                frame = int(Path(filename).stem)
                camera = deepcopy(dataset.cameras[index : index + 1]).to(
                    datamanager.device
                )
                sensor_index = int(camera.metadata["sensor_idxs"].item())
                camera_name = canonical_by_index[sensor_index]
                if camera_name in records.setdefault(frame, {}):
                    raise RuntimeError(
                        f"duplicate PandaSet camera: {frame}/{camera_name}"
                    )
                records[frame][camera_name] = camera

        expected = set(PANDASET_CAMERA_ORDER)
        if len(records) != 80:
            raise RuntimeError(f"expected 80 logical frames, found {len(records)}")
        for frame, cameras in records.items():
            if set(cameras) != expected:
                raise RuntimeError(
                    f"logical frame {frame} has cameras {sorted(cameras)}, "
                    f"expected {sorted(expected)}"
                )

        frame_numbers = tuple(sorted(records))
        first_front_time = float(
            records[frame_numbers[0]]["front"].times.item()
        )
        frame_times = tuple(
            float(records[frame]["front"].times.item()) - first_front_time
            for frame in frame_numbers
        )
        if any(
            right <= left
            for left, right in zip(frame_times, frame_times[1:])
        ):
            raise RuntimeError(
                f"front-camera log times are not strictly increasing: {frame_times}"
            )
        camera_times = {
            frame: {
                name: float(camera.times.item()) - first_front_time
                for name, camera in cameras.items()
            }
            for frame, cameras in records.items()
        }
        reference_frames: dict[int, SceneReferenceFrame] = {}
        scale = float(
            datamanager.train_dataset._dataparser_outputs.dataparser_scale
        )
        rig_origins: dict[int, list[float]] = {}
        for frame, cameras in records.items():
            centers = torch.stack(
                [
                    camera.camera_to_worlds[0, :, 3].detach().cpu()
                    for camera in cameras.values()
                ]
            )
            rig_origins[frame] = centers.mean(dim=0).tolist()
        for frame_index, frame in enumerate(frame_numbers):
            before = frame_numbers[max(frame_index - 1, 0)]
            after = frame_numbers[min(frame_index + 1, len(frame_numbers) - 1)]
            origin = rig_origins[frame]
            tangent = tuple(
                rig_origins[after][axis] - rig_origins[before][axis]
                for axis in range(3)
            )
            horizontal_length = math.hypot(tangent[0], tangent[1])
            if horizontal_length <= 1e-9:
                raise RuntimeError(f"logged path tangent is zero at frame {frame}")
            forward = (
                tangent[0] / horizontal_length,
                tangent[1] / horizontal_length,
                0.0,
            )
            reference_frames[frame] = SceneReferenceFrame(
                origin=origin,
                forward=forward,
                left=(-forward[1], forward[0], 0.0),
                scene_units_per_meter=scale,
            )

        self._torch = torch
        self._pipeline = pipeline
        self._records = records
        self._reference_frames = reference_frames
        self._frame_numbers = frame_numbers
        self._frame_times = frame_times
        self._camera_times = camera_times
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_step = int(checkpoint_step)
        self._sequence = str(config.pipeline.datamanager.dataparser.sequence)
        self._sensor_source_names = sensor_source_names

    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        """Render one complete logical observation at ``ego_state.time``."""

        self.limits.validate(ego_state)
        if not self.is_loaded:
            self.load()
        assert self._pipeline is not None
        assert self._torch is not None
        assert self._sequence is not None

        if str(scene) != self._sequence:
            raise ValueError(
                f"renderer is configured for scene {self._sequence!r}, "
                f"received {str(scene)!r}"
            )
        if ego_state.time < 0.0 or ego_state.time > self.logged_duration:
            raise ValueError(
                f"log time {ego_state.time:.3f}s is outside "
                f"[0, {self.logged_duration:.3f}]s"
            )
        unknown = sorted(
            set(camera_rig.camera_names) - set(PANDASET_CAMERA_ORDER)
        )
        if unknown:
            raise KeyError(
                f"unknown PandaSet cameras {unknown}; "
                f"available={list(PANDASET_CAMERA_ORDER)}"
            )

        logical_index = self._nearest_index(self._frame_times, ego_state.time)
        frame = self._frame_numbers[logical_index]
        selected_time = self._frame_times[logical_index]
        reference = self._reference_frames[frame]
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        frames: dict[str, object] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        with torch.no_grad():
            for spec in camera_rig.cameras:
                camera = deepcopy(self._records[frame][spec.name])
                transformed = reference.transform_camera(
                    camera.camera_to_worlds[0].detach().cpu().tolist(),
                    ego_state,
                )
                camera.camera_to_worlds = torch.tensor(
                    [transformed],
                    dtype=camera.camera_to_worlds.dtype,
                    device=camera.camera_to_worlds.device,
                )
                if self.output_scale != 1.0:
                    camera.rescale_output_resolution(self.output_scale)
                outputs = self._pipeline.model.get_outputs_for_camera(camera)
                if "rgb" not in outputs:
                    raise RuntimeError(
                        f"SplatAD did not return rgb for {spec.name}"
                    )
                rgb_tensor = outputs["rgb"]
                if not bool(torch.isfinite(rgb_tensor).all()):
                    raise RuntimeError(
                        f"SplatAD returned non-finite RGB for {spec.name}"
                    )
                rgb = (
                    rgb_tensor.detach()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                frames[spec.name] = rgb
                shapes[spec.name] = tuple(int(value) for value in rgb.shape)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        requested_camera_times = {
            name: self._camera_times[frame][name]
            for name in camera_rig.camera_names
        }
        sensor_spread = (
            max(requested_camera_times.values())
            - min(requested_camera_times.values())
        )
        return RenderedObservation(
            timestamp=ego_state.time,
            frames=frames,
            metadata={
                "backend": "h3_splatad_logged",
                "scene": self._sequence,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self._checkpoint_path),
                "checkpoint_step": self._checkpoint_step,
                "logical_frame": frame,
                "requested_log_time_seconds": ego_state.time,
                "selected_log_time_seconds": selected_time,
                "frame_selection_error_ms": abs(
                    selected_time - ego_state.time
                )
                * 1000.0,
                "camera_log_times_seconds": requested_camera_times,
                "camera_time_spread_ms": sensor_spread * 1000.0,
                "camera_source_names": {
                    name: self._sensor_source_names[name]
                    for name in camera_rig.camera_names
                },
                "offset_pose": {
                    "forward_meters": ego_state.x,
                    "left_meters": ego_state.y,
                    "yaw_radians": ego_state.yaw,
                },
                "output_scale": self.output_scale,
                "frame_shapes": shapes,
                "render_seconds": elapsed,
                "observation_fps": 1.0 / elapsed if elapsed > 0.0 else None,
                "logged_duration_seconds": self.logged_duration,
            },
        )
