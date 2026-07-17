"""Nerfstudio reconstruction renderer for nearby ego-pose queries.

The module intentionally imports Nerfstudio and Torch only when the backend is
loaded. The lightweight simulator package therefore remains usable with the
Python standard library alone.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .renderer_interface import CameraRig, RenderedObservation
from .scene_coordinates import NearbyPoseLimits, SceneReferenceFrame
from .state import EgoState


class NerfstudioRenderer:
    """Render a dataset camera rig from a trained Nerfstudio checkpoint.

    Each requested camera name must match a camera directory in the training
    dataset, such as ``front-forward``. Intrinsics, fisheye distortion, and rig
    extrinsics are cloned from ``reference_frame_index``. ``EgoState.x`` and
    ``EgoState.y`` are interpreted as forward and left displacement in meters
    from that reference rig; positive yaw turns left.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        reference_frame_index: int = 100,
        output_scale: float = 0.25,
        limits: NearbyPoseLimits | None = None,
        front_camera_name: str = "front-forward",
        lazy: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if reference_frame_index < 0:
            raise ValueError("reference_frame_index cannot be negative")
        if not 0.0 < output_scale <= 1.0:
            raise ValueError("output_scale must be in (0, 1]")
        if not front_camera_name:
            raise ValueError("front_camera_name cannot be empty")
        self.reference_frame_index = reference_frame_index
        self.output_scale = output_scale
        self.limits = limits or NearbyPoseLimits()
        self.front_camera_name = front_camera_name
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._camera_records: dict[str, list[tuple[Path, Any]]] = {}
        self._reference_frame: SceneReferenceFrame | None = None
        self._checkpoint_path: Path | None = None
        self._checkpoint_step: int | None = None
        if not lazy:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def camera_names(self) -> tuple[str, ...]:
        if not self.is_loaded:
            self.load()
        return tuple(sorted(self._camera_records))

    @staticmethod
    def _frame_sort_key(item: tuple[Path, Any]) -> tuple[int, int | str, str]:
        stem = item[0].stem
        if stem.isdigit():
            return (0, int(stem), item[0].name)
        return (1, stem, item[0].name)

    def load(self) -> None:
        """Load the configured pipeline and index its train and test cameras."""

        if self.is_loaded:
            return
        try:
            import torch
            from nerfstudio.utils.eval_utils import eval_setup
        except ImportError as error:
            raise RuntimeError(
                "NerfstudioRenderer requires the existing Nerfstudio environment; "
                "run it with the Python interpreter that provides nerfstudio and torch"
            ) from error

        _, pipeline, checkpoint_path, checkpoint_step = eval_setup(
            self.config_path,
            test_mode="test",
        )
        grouped: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
        datamanager = pipeline.datamanager
        for dataset in (datamanager.train_dataset, datamanager.eval_dataset):
            for index, filename in enumerate(dataset.image_filenames):
                path = Path(filename)
                grouped[path.parent.name].append(
                    (path, dataset.cameras[index : index + 1])
                )
        for records in grouped.values():
            records.sort(key=self._frame_sort_key)

        if self.front_camera_name not in grouped:
            raise RuntimeError(
                f"front camera {self.front_camera_name!r} not found; "
                f"available={sorted(grouped)}"
            )
        too_short = {
            name: len(records)
            for name, records in grouped.items()
            if self.reference_frame_index >= len(records)
        }
        if too_short:
            raise IndexError(
                f"reference frame {self.reference_frame_index} is unavailable: {too_short}"
            )

        reference_cameras = [
            records[self.reference_frame_index][1] for records in grouped.values()
        ]
        centers = torch.stack(
            [camera.camera_to_worlds[0, :, 3].cpu() for camera in reference_cameras]
        )
        rig_origin = centers.mean(dim=0).tolist()
        front_pose = (
            grouped[self.front_camera_name][self.reference_frame_index][1]
            .camera_to_worlds[0]
            .cpu()
            .tolist()
        )
        dataparser_scale = float(
            datamanager.train_dataset._dataparser_outputs.dataparser_scale
        )

        self._torch = torch
        self._pipeline = pipeline
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_step = int(checkpoint_step)
        self._camera_records = dict(grouped)
        self._reference_frame = SceneReferenceFrame.from_front_camera(
            front_pose,
            rig_origin,
            dataparser_scale,
        )

    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        """Render RGB uint8 arrays for the requested dataset cameras."""

        del scene
        self.limits.validate(ego_state)
        if not self.is_loaded:
            self.load()
        assert self._pipeline is not None
        assert self._reference_frame is not None
        assert self._torch is not None

        unknown = sorted(set(camera_rig.camera_names) - set(self._camera_records))
        if unknown:
            raise KeyError(
                f"camera names are not present in the checkpoint dataset: {unknown}; "
                f"available={sorted(self._camera_records)}"
            )

        torch = self._torch
        started = time.perf_counter()
        frames: dict[str, object] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for spec in camera_rig.cameras:
            reference_camera = self._camera_records[spec.name][
                self.reference_frame_index
            ][1]
            camera = reference_camera.to("cpu")
            transformed = self._reference_frame.transform_camera(
                camera.camera_to_worlds[0].tolist(),
                ego_state,
            )
            camera.camera_to_worlds = torch.tensor(
                [transformed],
                dtype=torch.float32,
            )
            if self.output_scale != 1.0:
                camera.rescale_output_resolution(self.output_scale)
            outputs = self._pipeline.model.get_outputs_for_camera(camera)
            if "rgb" not in outputs:
                raise RuntimeError(
                    f"Nerfstudio model did not return an rgb output for {spec.name}"
                )
            rgb = (
                outputs["rgb"]
                .detach()
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
        return RenderedObservation(
            timestamp=ego_state.time,
            frames=frames,
            metadata={
                "backend": "nerfstudio",
                "config_path": str(self.config_path),
                "checkpoint_path": str(self._checkpoint_path),
                "checkpoint_step": self._checkpoint_step,
                "reference_frame_index": self.reference_frame_index,
                "output_scale": self.output_scale,
                "pose": {
                    "forward_meters": ego_state.x,
                    "left_meters": ego_state.y,
                    "yaw_radians": ego_state.yaw,
                },
                "frame_shapes": shapes,
                "render_seconds": elapsed,
                "fps_equivalent": len(frames) / elapsed if elapsed > 0.0 else None,
            },
        )
