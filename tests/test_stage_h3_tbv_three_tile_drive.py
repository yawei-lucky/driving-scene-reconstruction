"""Dependency-light checks for the 260 m three-tile drive pilot."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_stage_h3_tbv_three_tile_drive.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_stage_h3_tbv_three_tile_drive", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVThreeTileDriveTests(unittest.TestCase):
    def test_interpolation_uses_route_progress(self) -> None:
        self.assertAlmostEqual(
            MODULE._interpolate(
                [10.0, 20.0, 50.0],
                [0.0, 5.0, 20.0],
                10.0,
            ),
            30.0,
        )

    def test_vehicle_completes_260_meter_humanized_drive(self) -> None:
        samples = tuple(
            MODULE.LoggedCenterlineSample(
                logical_frame=index,
                log_time=float(index),
                x=20.0 * index,
                y=2.0 * math.sin(index / 4.0),
                yaw=0.0,
            )
            for index in range(14)
        )
        corridor = MODULE.LoggedCenterlineCorridor(
            samples=samples,
            half_width=1.0,
            max_heading_error=math.radians(20.0),
        )

        drive, result = MODULE.simulate_drive(
            corridor,
            fps=20,
            speed_mps=12.0,
            lateral_amplitude_meters=0.55,
        )

        self.assertTrue(result["control_gate"])
        self.assertTrue(result["endpoint_reached"])
        self.assertEqual(result["boundary_hits"], 0)
        self.assertGreater(len(drive), 400)
        self.assertGreater(result["minimum_support_margin_meters"], 0.25)


if __name__ == "__main__":
    unittest.main()
