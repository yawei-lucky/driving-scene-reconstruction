"""Data-supported driving corridor around one recorded ego centreline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .state import EgoState


@dataclass(frozen=True)
class LoggedCenterlineSample:
    """One recorded rig pose expressed in the simulator's world frame."""

    logical_frame: int
    log_time: float
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class CorridorMeasurement:
    """Nearest relationship between an ego pose and a logged centreline."""

    progress: float
    lateral_offset: float
    distance: float
    heading_error: float


@dataclass(frozen=True)
class LoggedCenterlineCorridor:
    """Provisional tube around the part of the world observed by the log.

    The centreline is used only as a reconstruction-support boundary. It does
    not pull the vehicle back to the recorded path; the vehicle model still
    owns every future pose.
    """

    samples: tuple[LoggedCenterlineSample, ...]
    half_width: float = 1.0
    max_heading_error: float = math.radians(30.0)

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise ValueError("logged centreline needs at least two samples")
        if not math.isfinite(self.half_width) or self.half_width <= 0.0:
            raise ValueError("corridor half-width must be finite and positive")
        if (
            not math.isfinite(self.max_heading_error)
            or not 0.0 < self.max_heading_error <= math.pi
        ):
            raise ValueError(
                "corridor heading error must be finite and in (0, pi]"
            )
        if any(
            not all(
                math.isfinite(value)
                for value in (sample.log_time, sample.x, sample.y, sample.yaw)
            )
            for sample in self.samples
        ):
            raise ValueError("centreline samples must be finite")
        if any(
            math.hypot(right.x - left.x, right.y - left.y) <= 1e-6
            for left, right in zip(self.samples, self.samples[1:])
        ):
            raise ValueError("centreline contains a zero-length segment")

    @property
    def length(self) -> float:
        return sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(self.samples, self.samples[1:])
        )

    def measure(self, state: EgoState) -> CorridorMeasurement:
        """Project ``state`` onto the closest recorded centreline segment."""

        best: CorridorMeasurement | None = None
        progress_before = 0.0
        for left, right in zip(self.samples, self.samples[1:]):
            dx = right.x - left.x
            dy = right.y - left.y
            segment_length = math.hypot(dx, dy)
            tangent_x = dx / segment_length
            tangent_y = dy / segment_length
            along = (state.x - left.x) * tangent_x + (
                state.y - left.y
            ) * tangent_y
            clamped_along = min(segment_length, max(0.0, along))
            projected_x = left.x + tangent_x * clamped_along
            projected_y = left.y + tangent_y * clamped_along
            residual_x = state.x - projected_x
            residual_y = state.y - projected_y
            lateral = -tangent_y * residual_x + tangent_x * residual_y
            distance = math.hypot(residual_x, residual_y)
            road_heading = math.atan2(dy, dx)
            heading_error = math.atan2(
                math.sin(state.yaw - road_heading),
                math.cos(state.yaw - road_heading),
            )
            measurement = CorridorMeasurement(
                progress=progress_before + clamped_along,
                lateral_offset=lateral,
                distance=distance,
                heading_error=heading_error,
            )
            if best is None or measurement.distance < best.distance:
                best = measurement
            progress_before += segment_length
        assert best is not None
        return best

    def validate(self, state: EgoState) -> None:
        """Reject poses outside the provisional observed-path tube."""

        values = (state.x, state.y, state.yaw, state.speed, state.time)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ego state values must be finite")
        measurement = self.measure(state)
        if measurement.distance > self.half_width:
            raise ValueError(
                f"distance from logged road centreline "
                f"{measurement.distance:.3f}m exceeds {self.half_width:.3f}m"
            )
        if abs(measurement.heading_error) > self.max_heading_error:
            raise ValueError(
                f"heading differs from logged road by "
                f"{math.degrees(measurement.heading_error):.1f}deg, exceeding "
                f"{math.degrees(self.max_heading_error):.1f}deg"
            )
