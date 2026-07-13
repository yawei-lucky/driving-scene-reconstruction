"""State types for the simulator loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EgoState:
    """Ego pose and motion state for the initial human-drivable loop.

    Position is measured in meters, yaw in radians, speed in meters per second,
    and time in seconds.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    time: float = 0.0
