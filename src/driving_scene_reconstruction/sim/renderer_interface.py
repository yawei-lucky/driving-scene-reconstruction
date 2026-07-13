"""Renderer-neutral camera and observation interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from .state import EgoState


@dataclass(frozen=True)
class CameraSpec:
    """One camera's orientation relative to the ego frame."""

    name: str
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    horizontal_fov_degrees: float = 90.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name cannot be empty")
        if not 0.0 < self.horizontal_fov_degrees <= 360.0:
            raise ValueError("horizontal_fov_degrees must be in (0, 360]")


@dataclass(frozen=True)
class CameraRig:
    """A fixed collection of cameras mounted in the ego frame."""

    cameras: tuple[CameraSpec, ...]

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("camera rig must contain at least one camera")
        names = self.camera_names
        if len(names) != len(set(names)):
            raise ValueError("camera names must be unique within a rig")

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)


@dataclass(frozen=True)
class RenderedObservation:
    """Renderer output keyed by camera name plus backend metadata."""

    timestamp: float
    frames: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)


class Renderer(Protocol):
    """Interface implemented by replay, panorama, or reconstruction backends."""

    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        """Render the camera rig for one scene and ego state."""

        ...
