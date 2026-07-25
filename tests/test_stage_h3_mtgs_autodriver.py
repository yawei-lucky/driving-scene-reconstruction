"""Dependency-light tests for the MTGS simulated-human route follower."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "stage_h3_mtgs_autodriver.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage_h3_mtgs_autodriver", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from driving_scene_reconstruction.sim import (  # noqa: E402
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    SimpleVehicleModel,
    WorldDrivingController,
)


def straight_corridor() -> LoggedCenterlineCorridor:
    return LoggedCenterlineCorridor(
        samples=tuple(
            LoggedCenterlineSample(
                logical_frame=index,
                log_time=float(x),
                x=float(x),
                y=0.0,
                yaw=0.0,
            )
            for index, x in enumerate(range(0, 101, 5))
        ),
        half_width=5.0,
    )


def simulate() -> tuple[list[object], list[object]]:
    corridor = straight_corridor()
    model = SimpleVehicleModel(
        wheelbase=2.8,
        max_steer_angle=math.radians(25.0),
        max_acceleration=4.0,
        max_braking=8.0,
        linear_drag=0.05,
        max_speed=15.0,
    )
    controller = WorldDrivingController(
        corridor=corridor,
        spawn_state=corridor.pose_at_progress(5.0),
        vehicle_model=model,
    )
    driver = MODULE.MtgsAutodriver(model)
    state = controller.reset()
    states = [state]
    decisions = []
    for _ in range(400):
        decision = driver.decide(state, corridor, 0.05)
        update = controller.step(state, decision.control, 0.05)
        decisions.append(decision)
        states.append(update.state)
        state = update.state
        measurement = corridor.measure(state)
        if state.speed < 0.15 and measurement.progress > 98.5:
            break
    return states, decisions


class MtgsAutodriverTests(unittest.TestCase):
    def test_lateral_programme_visits_both_sides(self) -> None:
        values = [
            MODULE.humanized_lateral_target(
                progress,
                spawn_progress_meters=5.0,
                route_end_meters=99.5,
                amplitude_meters=3.0,
            )
            for progress in (5.0, 31.46, 44.69, 59.81, 74.93, 99.5)
        ]

        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 3.0, places=2)
        self.assertAlmostEqual(values[2], 0.0, places=2)
        self.assertAlmostEqual(values[3], -3.0, places=2)
        self.assertAlmostEqual(values[4], 0.0, places=2)
        self.assertAlmostEqual(values[5], 0.0)

    def test_closed_loop_accelerates_changes_lane_and_stops(self) -> None:
        states, _ = simulate()
        corridor = straight_corridor()
        measurements = [corridor.measure(state) for state in states]

        self.assertGreater(max(state.speed for state in states), 10.0)
        self.assertGreater(max(item.lateral_offset for item in measurements), 2.0)
        self.assertLess(min(item.lateral_offset for item in measurements), -2.0)
        self.assertLess(max(item.distance for item in measurements), 4.0)
        self.assertGreater(measurements[-1].progress, 98.5)
        self.assertLess(states[-1].speed, 0.15)

    def test_controls_are_rate_limited_and_repeatable(self) -> None:
        first_states, first_decisions = simulate()
        second_states, second_decisions = simulate()
        first_steers = [item.control.steer for item in first_decisions]
        second_steers = [item.control.steer for item in second_decisions]

        self.assertEqual(first_states, second_states)
        self.assertEqual(first_steers, second_steers)
        self.assertTrue(
            all(
                abs(right - left) <= 1.8 * 0.05 + 1e-9
                for left, right in zip(first_steers, first_steers[1:])
            )
        )


if __name__ == "__main__":
    unittest.main()
