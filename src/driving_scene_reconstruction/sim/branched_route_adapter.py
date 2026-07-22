"""Fail-closed route adapter for a shared approach with selectable branches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

from .control import HumanControl
from .drivable_corridor import LoggedCenterlineCorridor
from .state import EgoState
from .vehicle_model import SimpleVehicleModel
from .world_driving_controller import WorldDrivingController


@dataclass(frozen=True)
class SupportedRoute:
    """One route whose centreline bounds the reconstructed-data support."""

    name: str
    renderer_profile: str
    corridor: LoggedCenterlineCorridor
    start_progress_from_anchor: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("route name cannot be empty")
        if not self.renderer_profile.strip():
            raise ValueError("renderer profile cannot be empty")
        if not math.isfinite(self.start_progress_from_anchor):
            raise ValueError("route start progress must be finite")

    @property
    def end_progress_from_anchor(self) -> float:
        return self.start_progress_from_anchor + self.corridor.length


@dataclass(frozen=True)
class RouteSupportEvidence:
    """Task-facing support evidence for one simulated ego pose."""

    phase: str
    active_route: str
    renderer_profile: str
    selected_branch: str | None
    selection_required: bool
    progress_from_anchor_meters: float
    lateral_offset_meters: float
    distance_to_centerline_meters: float
    heading_error_degrees: float
    half_width_meters: float
    distance_margin_meters: float
    heading_limit_degrees: float
    heading_margin_degrees: float
    within_declared_support: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BranchedRouteUpdate:
    """One route-constrained update plus its explicit stopping condition."""

    state: EgoState
    support: RouteSupportEvidence
    boundary_hit: bool = False
    boundary_reason: str | None = None
    selection_required: bool = False


@dataclass
class BranchedRouteDrivingAdapter:
    """Drive a shared approach, stop near its anchor, then select a branch.

    The adapter never snaps an invalid vehicle pose onto a route. Crossing a
    reconstruction boundary keeps the last valid pose and stops the vehicle.
    The shared approach likewise stops shortly before its anchor until an
    explicit branch is selected.
    """

    common_route: SupportedRoute
    branches: Mapping[str, SupportedRoute]
    spawn_state: EgoState
    selection_window_meters: float = 0.5
    vehicle_model: SimpleVehicleModel = field(
        default_factory=lambda: SimpleVehicleModel(
            max_steer_angle=math.radians(15.0),
            max_acceleration=1.5,
            max_braking=4.0,
            max_speed=2.0,
        )
    )
    selected_branch: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("at least one branch is required")
        if any(name != route.name for name, route in self.branches.items()):
            raise ValueError("branch keys must match route names")
        if (
            not math.isfinite(self.selection_window_meters)
            or self.selection_window_meters <= 0.0
        ):
            raise ValueError("selection window must be finite and positive")
        self.common_route.corridor.validate(self.spawn_state)

    @property
    def active_route(self) -> SupportedRoute:
        if self.selected_branch is None:
            return self.common_route
        return self.branches[self.selected_branch]

    def reset(self) -> EgoState:
        self.selected_branch = None
        self.common_route.corridor.validate(self.spawn_state)
        return EgoState(
            x=self.spawn_state.x,
            y=self.spawn_state.y,
            yaw=self.spawn_state.yaw,
        )

    def support(self, state: EgoState) -> RouteSupportEvidence:
        return self._support_for_route(
            self.active_route,
            state,
            phase_override=None,
            selected_branch=self.selected_branch,
        )

    def _support_for_route(
        self,
        route: SupportedRoute,
        state: EgoState,
        *,
        phase_override: str | None,
        selected_branch: str | None,
    ) -> RouteSupportEvidence:
        measurement = route.corridor.measure(state)
        progress = route.start_progress_from_anchor + measurement.progress
        heading_error_degrees = math.degrees(measurement.heading_error)
        heading_limit_degrees = math.degrees(route.corridor.max_heading_error)
        within_support = (
            measurement.distance <= route.corridor.half_width
            and abs(measurement.heading_error)
            <= route.corridor.max_heading_error
        )
        selection_required = (
            selected_branch is None
            and route is self.common_route
            and progress
            >= self.common_route.end_progress_from_anchor
            - self.selection_window_meters
        )
        return RouteSupportEvidence(
            phase=phase_override or (
                "branch_selection"
                if selection_required
                else "common_approach"
                if selected_branch is None
                else "selected_branch"
            ),
            active_route=route.name,
            renderer_profile=route.renderer_profile,
            selected_branch=selected_branch,
            selection_required=selection_required,
            progress_from_anchor_meters=progress,
            lateral_offset_meters=measurement.lateral_offset,
            distance_to_centerline_meters=measurement.distance,
            heading_error_degrees=heading_error_degrees,
            half_width_meters=route.corridor.half_width,
            distance_margin_meters=route.corridor.half_width
            - measurement.distance,
            heading_limit_degrees=heading_limit_degrees,
            heading_margin_degrees=heading_limit_degrees
            - abs(heading_error_degrees),
            within_declared_support=within_support,
        )

    def branch_support(
        self, branch: str, state: EgoState
    ) -> RouteSupportEvidence:
        """Measure whether the current anchor pose can enter one branch."""

        if branch not in self.branches:
            raise ValueError(f"unknown branch {branch!r}")
        return self._support_for_route(
            self.branches[branch],
            state,
            phase_override="branch_candidate",
            selected_branch=branch,
        )

    def select_branch(self, branch: str, state: EgoState) -> RouteSupportEvidence:
        if branch not in self.branches:
            raise ValueError(
                f"unknown branch {branch!r}; choose from "
                f"{', '.join(sorted(self.branches))}"
            )
        if not self.support(state).selection_required:
            raise ValueError("branch can only be selected at the shared anchor")
        target = self.branches[branch]
        target.corridor.validate(state)
        self.selected_branch = branch
        return self.support(state)

    def step(
        self,
        state: EgoState,
        control: HumanControl,
        dt: float,
    ) -> BranchedRouteUpdate:
        current_support = self.support(state)
        if current_support.selection_required:
            # Validate control and dt while deliberately holding at the gate.
            control.clamped()
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be finite and positive")
            held = EgoState(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                speed=0.0,
                time=state.time + dt,
            )
            support = self.support(held)
            return BranchedRouteUpdate(
                state=held,
                support=support,
                boundary_reason="select straight or right at the shared anchor",
                selection_required=True,
            )

        controller = WorldDrivingController(
            corridor=self.active_route.corridor,
            spawn_state=self.spawn_state,
            vehicle_model=self.vehicle_model,
        )
        update = controller.step(state, control, dt)
        support = self.support(update.state)
        if update.boundary_hit:
            return BranchedRouteUpdate(
                state=update.state,
                support=support,
                boundary_hit=True,
                boundary_reason=update.boundary_reason,
            )
        if self.selected_branch is None and support.selection_required:
            stopped = EgoState(
                x=update.state.x,
                y=update.state.y,
                yaw=update.state.yaw,
                speed=0.0,
                time=update.state.time,
            )
            support = self.support(stopped)
            return BranchedRouteUpdate(
                state=stopped,
                support=support,
                boundary_reason="select straight or right at the shared anchor",
                selection_required=True,
            )
        return BranchedRouteUpdate(state=update.state, support=support)
