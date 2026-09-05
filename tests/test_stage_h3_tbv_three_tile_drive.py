"""Dependency-light checks for the 260 m three-tile drive pilot."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
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

    def test_generic_tile_arguments_are_sorted_by_route_progress(self) -> None:
        args = SimpleNamespace(
            tile=[
                MODULE.parse_tile("tile_5,400,500,/tmp/5.yml,/tmp/data5"),
                MODULE.parse_tile("tile_4,320,420,/tmp/4.yml,/tmp/data4"),
            ],
            tile_2_config=None,
            tile_2_data_root=None,
            tile_3_config=None,
            tile_3_data_root=None,
            tile_4_config=None,
            tile_4_data_root=None,
        )

        tiles = MODULE.resolve_tiles(args)

        self.assertEqual([item[0] for item in tiles], ["tile_4", "tile_5"])
        self.assertEqual(tiles[0][1:3], (320.0, 420.0))

    def test_vehicle_completes_420_meter_generic_drive(self) -> None:
        samples = tuple(
            MODULE.LoggedCenterlineSample(
                logical_frame=index,
                log_time=float(index),
                x=20.0 * index,
                y=1.5 * math.sin(index / 5.0),
                yaw=0.0,
            )
            for index in range(22)
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
            route_start_meters=160.0,
        )

        self.assertTrue(result["control_gate"])
        self.assertTrue(result["endpoint_reached"])
        self.assertEqual(result["boundary_hits"], 0)
        self.assertGreater(len(drive), 650)


if __name__ == "__main__":
    unittest.main()
