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
        if not self.is_loaded:
            self.load()
        assert self._anchor_frame is not None
        return self._anchor_frame

    @property
    def anchor_selected_time(self) -> float:
        if not self.is_loaded:
            self.load()
        assert self._anchor_selected_time is not None
        return self._anchor_selected_time

    def load(self) -> None:
        """Load the checkpoint and select the immutable source-scene anchor."""

        super().load()
        if self._anchor_frame is not None:
            return
        if self.anchor_log_time > self.logged_duration:
            raise ValueError(
                f"anchor log time {self.anchor_log_time:.3f}s is outside "
                f"[0, {self.logged_duration:.3f}]s"
            )
        anchor_index = self._nearest_index(self._frame_times, self.anchor_log_time)
        self._anchor_frame = self._frame_numbers[anchor_index]
        self._anchor_selected_time = self._frame_times[anchor_index]
        assert self._torch is not None
        synchronized_poses: dict[str, Any] = {}
        sync_brackets: dict[str, dict[str, float | int]] = {}
        for camera_name in PANDASET_CAMERA_ORDER:
            camera_times = tuple(
                self._camera_times[frame][camera_name]
                for frame in self._frame_numbers
            )
            insertion = self._nearest_index(camera_times, self._anchor_selected_time)
            if camera_times[insertion] <= self._anchor_selected_time:
                left_index = insertion
                right_index = min(insertion + 1, len(camera_times) - 1)
            else:
                left_index = max(insertion - 1, 0)
                right_index = insertion
            left_time = camera_times[left_index]
            right_time = camera_times[right_index]
            if not left_time <= self._anchor_selected_time <= right_time:
                raise RuntimeError(
                    f"cannot synchronize {camera_name} at "
                    f"{self._anchor_selected_time:.6f}s from "
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
                    (self._anchor_selected_time - left_time)
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
        logged_reference = self._reference_frames[self._anchor_frame]
        self._world_reference = SceneReferenceFrame(
            origin=origin,
            forward=logged_reference.forward,
            left=logged_reference.left,
            scene_units_per_meter=logged_reference.scene_units_per_meter,
        )
        self._synchronized_anchor_poses = synchronized_poses
        self._rig_sync_brackets = sync_brackets

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
        if not self.is_loaded:
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
                "source_camera_log_times_seconds": source_camera_times,
                "rig_pose_time_synchronized": True,
                "rendered_camera_scene_time_spread_ms": 0.0,
                "rig_pose_sync_brackets": {
                    name: self._rig_sync_brackets[name]
                    for name in camera_rig.camera_names
                },
                "source_motion_metadata_zeroed": True,
                "rolling_shutter_mode": "disabled_for_static_world_probe",
                "world_pose_frame": "anchor_local_metric_forward_left",
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
