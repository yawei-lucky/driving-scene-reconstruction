"""World-pose PandaSet SplatAD renderer for free-driving probes.

The accepted H3 checkpoint is static, so this renderer freezes one source-log
scene time and moves a fixed six-camera rig from an independently simulated
ego pose.  Heavy H3 dependencies remain lazily imported by the parent loader.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .drivable_corridor import LoggedCenterlineSample
from .renderer_interface import CameraRig, RenderedObservation
from .scene_coordinates import NearbyPoseLimits, SceneReferenceFrame
from .splatad_logged_renderer import PANDASET_CAMERA_ORDER, SplatADLoggedRenderer
from .state import EgoState


H3_WORLD_POSE_PROBE_LIMITS = NearbyPoseLimits(
    max_abs_forward_meters=100.0,
    max_abs_left_meters=3.0,
    max_abs_yaw_radians=math.pi,
)


class SplatADWorldRenderer(SplatADLoggedRenderer):
    """Render a fixed SplatAD scene from an independently simulated ego pose.

    ``EgoState.time`` is simulation time. ``x`` and ``y`` are metric forward
    and left coordinates in a fixed local world frame anchored at
    ``anchor_log_time``; ``yaw`` is relative to that frame. Source-log time is
    frozen and reported separately in the observation metadata.

    The source cameras at the anchor frame define one fixed calibrated rig.
    Every requested camera receives the same planar rigid transform, so the
    vehicle path no longer follows the recorded trajectory.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        output_scale: float = 0.5,
        anchor_log_time: float = 4.0,
        limits: NearbyPoseLimits | None = None,
        lazy: bool = True,
    ) -> None:
        if not math.isfinite(anchor_log_time) or anchor_log_time < 0.0:
            raise ValueError("anchor_log_time must be finite and non-negative")
        self.anchor_log_time = float(anchor_log_time)
        self._anchor_frame: int | None = None
        self._anchor_selected_time: float | None = None
        self._world_reference: SceneReferenceFrame | None = None
        self._synchronized_anchor_poses: dict[str, Any] = {}
        self._rig_sync_brackets: dict[str, dict[str, float | int]] = {}
        self._common_anchor_time_range: tuple[float, float] | None = None
        self._logged_centerline: tuple[LoggedCenterlineSample, ...] = ()
        super().__init__(
            config_path,
            output_scale=output_scale,
            limits=limits or H3_WORLD_POSE_PROBE_LIMITS,
            lazy=True,
        )
        if not lazy:
            self.load()

    @property
    def anchor_frame(self) -> int:
        if self._anchor_frame is None:
            self.load()
        assert self._anchor_frame is not None
        return self._anchor_frame

    @property
    def is_world_ready(self) -> bool:
        """Whether both the parent pipeline and world anchor are initialized."""

        return (
            self.is_loaded
            and self._anchor_frame is not None
            and self._anchor_selected_time is not None
            and self._world_reference is not None
            and bool(self._synchronized_anchor_poses)
            and bool(self._rig_sync_brackets)
            and self._common_anchor_time_range is not None
        )

    @property
    def anchor_selected_time(self) -> float:
        if self._anchor_selected_time is None:
            self.load()
        assert self._anchor_selected_time is not None
        return self._anchor_selected_time

    @property
    def common_anchor_time_range(self) -> tuple[float, float]:
        """Return the interval where every source camera can be interpolated."""

        if not self.is_loaded or self._common_anchor_time_range is None:
            self.load()
        assert self._common_anchor_time_range is not None
        return self._common_anchor_time_range

    @property
    def logged_centerline(self) -> tuple[LoggedCenterlineSample, ...]:
        """Return the recorded rig path in this renderer's world coordinates."""

        if not self.is_world_ready:
            self.load()
        if self._logged_centerline:
            return self._logged_centerline
        assert self._torch is not None
        assert self._world_reference is not None
        assert self._anchor_frame is not None

        anchor_front = self._synchronized_anchor_poses["front"]
        anchor_forward = self._horizontal_camera_forward(anchor_front)
        samples: list[LoggedCenterlineSample] = []
        common_start, common_end = self.common_anchor_time_range
        for frame, log_time in zip(self._frame_numbers, self._frame_times):
            if not common_start <= log_time <= common_end:
                continue
            poses = {
                name: self._camera_pose_at_log_time(name, log_time)
                for name in PANDASET_CAMERA_ORDER
            }
            centre = self._torch.stack(
                [pose[:, 3].detach().cpu() for pose in poses.values()]
            ).mean(dim=0)
            delta = [
                float(centre[index]) - self._world_reference.origin[index]
                for index in range(3)
            ]
            scale = self._world_reference.scene_units_per_meter
            x = sum(
                delta[index] * self._world_reference.forward[index]
                for index in range(3)
            ) / scale
            y = sum(
                delta[index] * self._world_reference.left[index]
                for index in range(3)
            ) / scale
            forward = self._horizontal_camera_forward(poses["front"])
            yaw = math.atan2(
                anchor_forward[0] * forward[1]
                - anchor_forward[1] * forward[0],
                anchor_forward[0] * forward[0]
                + anchor_forward[1] * forward[1],
            )
            if frame == self._anchor_frame:
                x, y, yaw = 0.0, 0.0, 0.0
            samples.append(
                LoggedCenterlineSample(
                    logical_frame=frame,
                    log_time=log_time,
                    x=x,
                    y=y,
                    yaw=yaw,
                )
            )
        self._logged_centerline = tuple(samples)
        return self._logged_centerline

    @staticmethod
    def _horizontal_camera_forward(pose: Any) -> tuple[float, float]:
        x = -float(pose[0, 2])
        y = -float(pose[1, 2])
        length = math.hypot(x, y)
        if length <= 1e-9:
            raise RuntimeError("front camera has no horizontal viewing direction")
        return x / length, y / length

    def _camera_pose_at_log_time(self, camera_name: str, log_time: float) -> Any:
        """Interpolate one camera pose at elapsed front-camera log time."""

        assert self._torch is not None
        camera_times = tuple(
            self._camera_times[frame][camera_name]
            for frame in self._frame_numbers
        )
        nearest = self._nearest_index(camera_times, log_time)
        if camera_times[nearest] <= log_time:
            left_index = nearest
            right_index = min(nearest + 1, len(camera_times) - 1)
        else:
            left_index = max(nearest - 1, 0)
            right_index = nearest
        left_time = camera_times[left_index]
        right_time = camera_times[right_index]
        if not left_time <= log_time <= right_time:
            raise ValueError(
                f"cannot interpolate {camera_name} at {log_time:.6f}s"
            )
        left_pose = self._records[self._frame_numbers[left_index]][
            camera_name
        ].camera_to_worlds[0]
        right_pose = self._records[self._frame_numbers[right_index]][
            camera_name
        ].camera_to_worlds[0]
        if left_time == right_time:
            return left_pose.clone()
        fraction = (log_time - left_time) / (right_time - left_time)
        return self._interpolate_pose(
            self._torch,
            left_pose,
            right_pose,
            fraction,
        )

    def load(self) -> None:
        """Load the checkpoint and select the immutable source-scene anchor."""

        super().load()
        if self._anchor_frame is not None:
            if (
                self._anchor_selected_time is None
                or self._world_reference is None
                or not self._synchronized_anchor_poses
                or not self._rig_sync_brackets
                or self._common_anchor_time_range is None
            ):
                raise RuntimeError("world renderer anchor is only partially initialized")
            return
        if self.anchor_log_time > self.logged_duration:
            raise ValueError(
                f"anchor log time {self.anchor_log_time:.3f}s is outside "
                f"[0, {self.logged_duration:.3f}]s"
            )
        anchor_index = self._nearest_index(self._frame_times, self.anchor_log_time)
        anchor_frame = self._frame_numbers[anchor_index]
        anchor_selected_time = self._frame_times[anchor_index]
        common_start = max(
            self._camera_times[self._frame_numbers[0]][camera_name]
            for camera_name in PANDASET_CAMERA_ORDER
        )
        common_end = min(
            self._camera_times[self._frame_numbers[-1]][camera_name]
            for camera_name in PANDASET_CAMERA_ORDER
        )
        if not common_start <= anchor_selected_time <= common_end:
            raise ValueError(
                f"selected anchor time {anchor_selected_time:.6f}s is outside the "
                f"six-camera interpolation interval [{common_start:.6f}, "
                f"{common_end:.6f}]s"
            )
        assert self._torch is not None
        synchronized_poses: dict[str, Any] = {}
        sync_brackets: dict[str, dict[str, float | int]] = {}
        for camera_name in PANDASET_CAMERA_ORDER:
            camera_times = tuple(
                self._camera_times[frame][camera_name]
                for frame in self._frame_numbers
            )
            insertion = self._nearest_index(camera_times, anchor_selected_time)
            if camera_times[insertion] <= anchor_selected_time:
                left_index = insertion
                right_index = min(insertion + 1, len(camera_times) - 1)
            else:
                left_index = max(insertion - 1, 0)
                right_index = insertion
            left_time = camera_times[left_index]
            right_time = camera_times[right_index]
            if not left_time <= anchor_selected_time <= right_time:
                raise RuntimeError(
                    f"cannot synchronize {camera_name} at "
                    f"{anchor_selected_time:.6f}s from "
                    f"[{left_time:.6f}, {right_time:.6f}]s"
                )
            left_frame = self._frame_numbers[left_index]
            right_frame = self._frame_numbers[right_index]
            left_pose = self._records[left_frame][camera_name].camera_to_worlds[0]
            right_pose = self._records[right_frame][camera_name].camera_to_worlds[0]
            if right_time == left_time:
                fraction = 0.0
                synchronized = left_pose.clone()
            else:
                fraction = (
                    (anchor_selected_time - left_time)
                    / (right_time - left_time)
                )
                synchronized = self._interpolate_pose(
                    self._torch,
                    left_pose,
                    right_pose,
                    fraction,
                )
            synchronized_poses[camera_name] = synchronized
            sync_brackets[camera_name] = {
                "left_logical_frame": left_frame,
                "right_logical_frame": right_frame,
                "left_time_seconds": left_time,
                "right_time_seconds": right_time,
                "interpolation_fraction": fraction,
            }

        centres = self._torch.stack(
            [pose[:, 3].detach().cpu() for pose in synchronized_poses.values()]
        )
        origin = centres.mean(dim=0).tolist()
        logged_reference = self._reference_frames[anchor_frame]
        world_reference = SceneReferenceFrame(
            origin=origin,
            forward=logged_reference.forward,
            left=logged_reference.left,
            scene_units_per_meter=logged_reference.scene_units_per_meter,
        )
        assert self._pipeline is not None
        model = self._pipeline.model
        if not hasattr(model.config, "compensate_rs_camera"):
            raise RuntimeError("SplatAD model has no camera RS compensation switch")
        model.config.compensate_rs_camera = False
        if not hasattr(model, "rs_editing"):
            raise RuntimeError("SplatAD model has no rolling-shutter editing state")
        for name in (
            "rs_time",
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
        ):
            model.rs_editing[name] = 0.0

        # Commit the world-specific state only after every validation and model
        # adjustment succeeds, so a failed load cannot leave a false ready flag.
        self._anchor_frame = anchor_frame
        self._anchor_selected_time = anchor_selected_time
        self._world_reference = world_reference
        self._synchronized_anchor_poses = synchronized_poses
        self._rig_sync_brackets = sync_brackets
        self._common_anchor_time_range = (common_start, common_end)

    @staticmethod
    def _interpolate_pose(
        torch_module: Any,
        left_pose: Any,
        right_pose: Any,
        fraction: float,
    ) -> Any:
        """Interpolate translation and project blended rotation onto SO(3)."""

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("pose interpolation fraction must be in [0, 1]")
        translation = (
            left_pose[:, 3] * (1.0 - fraction)
            + right_pose[:, 3] * fraction
        )
        blended_rotation = (
            left_pose[:, :3] * (1.0 - fraction)
            + right_pose[:, :3] * fraction
        )
        u, _, vh = torch_module.linalg.svd(blended_rotation)
        rotation = u @ vh
        if float(torch_module.linalg.det(rotation)) < 0.0:
            u = u.clone()
            u[:, -1] *= -1.0
            rotation = u @ vh
        return torch_module.cat((rotation, translation[:, None]), dim=1)

    @staticmethod
    def _zero_source_motion_metadata(camera: Any) -> None:
        """Disable recorded ego motion for the frozen-world static probe."""

        if camera.metadata is None:
            return
        for name in (
            "velocities",
            "linear_velocities_local",
            "angular_velocities_local",
            "rolling_shutter_time",
            "time_to_center_pixel",
        ):
            value = camera.metadata.get(name)
            if value is None:
                continue
            if hasattr(value, "clone") and hasattr(value, "zero_"):
                camera.metadata[name] = value.clone().zero_()
            else:
                camera.metadata[name] = 0.0

    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        """Render one observation from an absolute local-world ego pose."""

        self.limits.validate(ego_state)
        if ego_state.time < 0.0:
            raise ValueError("simulation time must be non-negative")
        if not self.is_world_ready:
            self.load()
        assert self._pipeline is not None
        assert self._torch is not None
        assert self._sequence is not None
        assert self._anchor_frame is not None
        assert self._anchor_selected_time is not None
        assert self._world_reference is not None

        if str(scene) != self._sequence:
            raise ValueError(
                f"renderer is configured for scene {self._sequence!r}, "
                f"received {str(scene)!r}"
            )
        unknown = sorted(set(camera_rig.camera_names) - set(PANDASET_CAMERA_ORDER))
        if unknown:
            raise KeyError(
                f"unknown PandaSet cameras {unknown}; "
                f"available={list(PANDASET_CAMERA_ORDER)}"
            )

        frame = self._anchor_frame
        reference = self._world_reference
        synchronized_scene_time = self._records[frame]["front"].times.clone()
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
                    self._synchronized_anchor_poses[spec.name]
                    .detach()
                    .cpu()
                    .tolist(),
                    ego_state,
                )
                camera.camera_to_worlds = torch.tensor(
                    [transformed],
                    dtype=camera.camera_to_worlds.dtype,
                    device=camera.camera_to_worlds.device,
                )
                # The accepted checkpoint is static. Keep one explicit scene
                # time for all cameras and remove the recorded vehicle motion
                # instead of pretending it describes the simulated trajectory.
                camera.times = synchronized_scene_time.clone()
                self._zero_source_motion_metadata(camera)
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
        source_camera_times = {
            name: self._camera_times[frame][name]
            for name in camera_rig.camera_names
        }
        model_scene_time = float(synchronized_scene_time.item())
        effective_model_camera_times = {
            name: model_scene_time for name in camera_rig.camera_names
        }
        return RenderedObservation(
            timestamp=ego_state.time,
            frames=frames,
            metadata={
                "backend": "h3_splatad_world",
                "scene": self._sequence,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self._checkpoint_path),
                "checkpoint_step": self._checkpoint_step,
                "simulation_time_seconds": ego_state.time,
                "scene_time_mode": "frozen_static_anchor",
                "requested_anchor_log_time_seconds": self.anchor_log_time,
                "scene_time_seconds": self._anchor_selected_time,
                "anchor_logical_frame": frame,
                "common_anchor_time_range_seconds": list(
                    self.common_anchor_time_range
                ),
                "source_camera_log_times_seconds": source_camera_times,
                "rig_pose_time_synchronized": True,
                "model_scene_time_seconds": model_scene_time,
                "effective_camera_model_times_seconds": effective_model_camera_times,
                "effective_camera_scene_time_spread_ms": 0.0,
                "rig_pose_sync_brackets": {
                    name: self._rig_sync_brackets[name]
                    for name in camera_rig.camera_names
                },
                "source_motion_metadata_zeroed": True,
                "learned_camera_time_adjustment_disabled": True,
                "rolling_shutter_mode": "disabled_for_static_world_probe",
                "world_pose_frame": "anchor_local_metric_forward_left",
                "ego_reference_point": "synchronized_six_camera_centroid",
                "world_pose": {
                    "forward_meters": ego_state.x,
                    "left_meters": ego_state.y,
                    "yaw_radians": ego_state.yaw,
                    "speed_meters_per_second": ego_state.speed,
                },
                "output_scale": self.output_scale,
                "frame_shapes": shapes,
                "render_seconds": elapsed,
                "observation_fps": 1.0 / elapsed if elapsed > 0.0 else None,
            },
        )
