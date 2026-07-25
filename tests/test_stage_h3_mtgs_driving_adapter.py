"""Tests for the minimal MTGS route-control adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "stage_h3_mtgs_driving_adapter.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage_h3_mtgs_driving_adapter", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Matrix:
    def __init__(self, x: float, y: float) -> None:
        self.values = (
            (1.0, 0.0, 0.0, x),
            (0.0, 1.0, 0.0, y),
            (0.0, 0.0, 1.0, 1.5),
        )

    def __getitem__(self, key: tuple[int, int]) -> float:
        row, column = key
        return self.values[row][column]


def straight_route() -> tuple[list[Matrix], list[float]]:
    distances = [float(value) for value in range(0, 101, 5)]
    return [Matrix(value, 0.0) for value in distances], distances


class MtgsDrivingAdapterTests(unittest.TestCase):
    def test_spawns_with_full_wide_corridor_margin(self) -> None:
        poses, distances = straight_route()
        adapter = MODULE.make_mtgs_driving_adapter(poses, distances)

        query = adapter.render_query(adapter.reset())

        self.assertAlmostEqual(query.progress_meters, 5.0)
        self.assertAlmostEqual(query.lateral_meters, 0.0)
        self.assertAlmostEqual(query.support_margin_meters, 5.0)

    def test_accelerates_past_ten_mps_but_respects_speed_cap(self) -> None:
        poses, distances = straight_route()
        adapter = MODULE.make_mtgs_driving_adapter(poses, distances)
        state = adapter.reset()
        update = None
        for _ in range(100):
            update = adapter.step(
                state,
                MODULE.HumanControl(throttle=1.0),
                0.05,
            )
            state = update.state
        assert update is not None

        self.assertGreater(state.speed, 10.0)
        self.assertLessEqual(state.speed, MODULE.MTGS_MAX_SPEED_MPS)
        self.assertFalse(update.boundary_hit)

    def test_crossing_wide_support_fails_closed(self) -> None:
        poses, distances = straight_route()
        adapter = MODULE.make_mtgs_driving_adapter(poses, distances)
        state = MODULE.EgoState(x=20.0, y=4.95, yaw=0.5, speed=8.0)

        update = adapter.step(
            state,
            MODULE.HumanControl(steer=1.0, throttle=1.0),
            0.1,
        )

        self.assertTrue(update.boundary_hit)
        self.assertEqual(update.state.speed, 0.0)
        self.assertGreaterEqual(update.render_query.support_margin_meters, 0.0)

    def test_endpoint_stops_without_leaving_observed_route(self) -> None:
        poses, distances = straight_route()
        adapter = MODULE.make_mtgs_driving_adapter(poses, distances)
        state = MODULE.EgoState(x=99.4, y=0.0, yaw=0.0, speed=5.0)

        update = adapter.step(
            state,
            MODULE.HumanControl(throttle=1.0),
            0.05,
        )

        self.assertTrue(update.endpoint_reached)
        self.assertEqual(update.state.speed, 0.0)


if __name__ == "__main__":
    unittest.main()
