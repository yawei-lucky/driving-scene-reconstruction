"""Dependency-light tests for the MTGS continuous-drive path profile."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_stage_h3_mtgs_continuous_drive.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_stage_h3_mtgs_continuous_drive", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MtgsContinuousDriveTests(unittest.TestCase):
    def test_lane_change_visits_both_sides_and_returns_to_center(self) -> None:
        values = [
            MODULE.lane_change_profile(
                fraction,
                amplitude_meters=4.0,
                distance_meters=72.0,
            )[0]
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 4.0)
        self.assertAlmostEqual(values[2], 0.0)
        self.assertAlmostEqual(values[3], -4.0)
        self.assertAlmostEqual(values[4], 0.0)

    def test_lane_change_heading_is_bounded_for_default_drive(self) -> None:
        _, maximum_slope = MODULE.lane_change_profile(
            0.125,
            amplitude_meters=MODULE.DEFAULT_LATERAL_AMPLITUDE_METERS,
            distance_meters=(
                MODULE.DEFAULT_SPEED_MPS * MODULE.DEFAULT_DURATION_SECONDS
            ),
        )

        self.assertLess(math.degrees(math.atan(maximum_slope)), 20.0)

    def test_lane_change_heading_is_zero_at_each_target(self) -> None:
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, slope = MODULE.lane_change_profile(
                fraction,
                amplitude_meters=4.0,
                distance_meters=72.0,
            )
            self.assertAlmostEqual(slope, 0.0)

    def test_profile_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.lane_change_profile(
                1.1,
                amplitude_meters=4.0,
                distance_meters=72.0,
            )

    def test_default_video_includes_both_simulation_endpoints(self) -> None:
        state_count = (
            round(MODULE.DEFAULT_DURATION_SECONDS * MODULE.DEFAULT_FPS) + 1
        )
        self.assertEqual(state_count, 121)
        self.assertAlmostEqual(
            (state_count - 1) / MODULE.DEFAULT_FPS,
            MODULE.DEFAULT_DURATION_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
