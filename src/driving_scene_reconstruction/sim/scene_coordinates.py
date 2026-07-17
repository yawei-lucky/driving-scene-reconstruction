"""Coordinate helpers for nearby-pose reconstruction rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .state import EgoState

Vector3 = tuple[float, float, float]
CameraToWorld = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _finite_vector(values: Sequence[float], label: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _normalize_xy(vector: Vector3, label: str) -> Vector3:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        raise ValueError(f"{label} has no usable horizontal component")
    return (vector[0] / length, vector[1] / length, 0.0)


def _camera_to_world(value: Sequence[Sequence[float]]) -> CameraToWorld:
    if len(value) != 3 or any(len(row) != 4 for row in value):
        raise ValueError("camera_to_world must have shape 3x4")
    result = tuple(
        tuple(float(component) for component in row)
        for row in value
    )
    if not all(math.isfinite(component) for row in result for component in row):
        raise ValueError("camera_to_world must contain only finite values")
    return result  # type: ignore[return-value]


def _rotate_z(vector: Vector3, angle: float) -> Vector3:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
        vector[2],
    )


@dataclass(frozen=True)
class NearbyPoseLimits:
    """Safety envelope for queries around one reconstructed reference pose."""

    max_abs_forward_meters: float = 2.0
    max_abs_left_meters: float = 0.5
    max_abs_yaw_radians: float = math.radians(5.0)

    def __post_init__(self) -> None:
        values = (
            self.max_abs_forward_meters,
            self.max_abs_left_meters,
            self.max_abs_yaw_radians,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("nearby-pose limits must be finite and non-negative")

    def validate(self, state: EgoState) -> None:
        """Reject a state outside the reconstruction's declared safe envelope."""

        state_values = (state.x, state.y, state.yaw, state.speed, state.time)
        if not all(math.isfinite(value) for value in state_values):
            raise ValueError("ego state values must be finite")
        if abs(state.x) > self.max_abs_forward_meters:
            raise ValueError(
                f"forward displacement {state.x:.3f}m exceeds "
                f"{self.max_abs_forward_meters:.3f}m"
            )
        if abs(state.y) > self.max_abs_left_meters:
            raise ValueError(
                f"left displacement {state.y:.3f}m exceeds "
                f"{self.max_abs_left_meters:.3f}m"
            )
        if abs(state.yaw) > self.max_abs_yaw_radians:
            raise ValueError(
                f"yaw displacement {math.degrees(state.yaw):.3f}deg exceeds "
                f"{math.degrees(self.max_abs_yaw_radians):.3f}deg"
            )


@dataclass(frozen=True)
class SceneReferenceFrame:
    """Planar ego basis expressed in the reconstruction's world coordinates.

    ``scene_units_per_meter`` converts physical ego displacement to the scaled
    coordinates used by the reconstruction model.
    """

    origin: Vector3
    forward: Vector3
    left: Vector3
    scene_units_per_meter: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _finite_vector(self.origin, "origin"))
        object.__setattr__(
            self,
            "forward",
            _normalize_xy(_finite_vector(self.forward, "forward"), "forward"),
        )
        object.__setattr__(
            self,
            "left",
            _normalize_xy(_finite_vector(self.left, "left"), "left"),
        )
        if not math.isfinite(self.scene_units_per_meter) or self.scene_units_per_meter <= 0.0:
            raise ValueError("scene_units_per_meter must be finite and positive")
        dot = self.forward[0] * self.left[0] + self.forward[1] * self.left[1]
        if abs(dot) > 1e-5:
            raise ValueError("forward and left axes must be perpendicular")

    @classmethod
    def from_front_camera(
        cls,
        camera_to_world: Sequence[Sequence[float]],
        rig_origin: Sequence[float],
        scene_units_per_meter: float,
    ) -> "SceneReferenceFrame":
        """Infer the horizontal ego basis from an OpenGL-style front camera."""

        pose = _camera_to_world(camera_to_world)
        # Nerfstudio cameras look along local -Z. Project that direction onto
        # the z-up scene plane, then construct a left-handed planar companion.
        forward = _normalize_xy(
            (-pose[0][2], -pose[1][2], 0.0),
            "front camera direction",
        )
        left = (-forward[1], forward[0], 0.0)
        return cls(
            origin=_finite_vector(rig_origin, "rig_origin"),
            forward=forward,
            left=left,
            scene_units_per_meter=scene_units_per_meter,
        )

    def transform_camera(
        self,
        camera_to_world: Sequence[Sequence[float]],
        state: EgoState,
    ) -> CameraToWorld:
        """Apply a nearby planar ego displacement to one camera in the rig."""

        pose = _camera_to_world(camera_to_world)
        rotation_columns = tuple(
            _rotate_z((pose[0][column], pose[1][column], pose[2][column]), state.yaw)
            for column in range(3)
        )
        camera_center = (pose[0][3], pose[1][3], pose[2][3])
        relative_center = tuple(
            camera_center[index] - self.origin[index] for index in range(3)
        )
        rotated_center = _rotate_z(relative_center, state.yaw)
        scale = self.scene_units_per_meter
        translation = tuple(
            scale * (state.x * self.forward[index] + state.y * self.left[index])
            for index in range(3)
        )
        new_center = tuple(
            self.origin[index] + rotated_center[index] + translation[index]
            for index in range(3)
        )
        return tuple(
            (
                rotation_columns[0][row],
                rotation_columns[1][row],
                rotation_columns[2][row],
                new_center[row],
            )
            for row in range(3)
        )  # type: ignore[return-value]
