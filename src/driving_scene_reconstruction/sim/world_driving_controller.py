"""Bounded controller for the first world-space reconstruction browser."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .control import HumanControl
from .drivable_corridor import CorridorMeasurement, LoggedCenterlineCorridor
from .scene_coordinates import NearbyPoseLimits
from .state import EgoState
from .vehicle_model import SimpleVehicleModel


H3_PROVISIONAL_WORLD_DRIVING_LIMITS = NearbyPoseLimits(
    max_abs_forward_meters=6.0,
    max_abs_left_meters=1.0,
    max_abs_yaw_radians=math.radians(15.0),
)


@dataclass(frozen=True)
class WorldDrivingUpdate:
    """One bounded vehicle update and any reconstruction-boundary event."""

    state: EgoState
    boundary_hit: bool = False
    boundary_reason: str | None = None


@dataclass(frozen=True)
class WorldDrivingController:
    """Advance a real vehicle state while containing it to a reviewed probe area.

    The controller never clamps a pose onto the edge of the reconstruction,
    because that would silently change vehicle dynamics. If a requested step
    crosses the provisional envelope, it keeps the last valid pose, stops the
    vehicle, advances simulation time, and reports the boundary to the caller.
    The operator can then reset instead of driving into unsupported imagery.
    """

    limits: NearbyPoseLimits = field(
        default_factory=lambda: H3_PROVISIONAL_WORLD_DRIVING_LIMITS
    )
    corridor: LoggedCenterlineCorridor | None = None
    vehicle_model: SimpleVehicleModel = field(
        default_factory=lambda: SimpleVehicleModel(
            max_steer_angle=math.radians(15.0),
            max_acceleration=1.5,
            max_braking=4.0,
            max_speed=2.0,
        )
    )

    def reset(self) -> EgoState:
        """Return the fixed world-space spawn state."""

        state = EgoState()
        self._validate(state)
        return state

    def corridor_measurement(
        self,
        state: EgoState,
    ) -> CorridorMeasurement | None:
        """Return progress/deviation when a logged corridor is active."""

        if self.corridor is None:
            return None
        return self.corridor.measure(state)

    def _validate(self, state: EgoState) -> None:
        if self.corridor is not None:
            self.corridor.validate(state)
        else:
            self.limits.validate(state)

    def step(
        self,
        state: EgoState,
        control: HumanControl,
        dt: float,
    ) -> WorldDrivingUpdate:
        """Advance one step, stopping at the provisional reconstruction edge."""

        self._validate(state)
        candidate = self.vehicle_model.step(state, control, dt)
        try:
            self._validate(candidate)
        except ValueError as error:
            stopped = EgoState(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                speed=0.0,
                time=candidate.time,
            )
            return WorldDrivingUpdate(
                state=stopped,
                boundary_hit=True,
                boundary_reason=str(error),
            )
        return WorldDrivingUpdate(state=candidate)
