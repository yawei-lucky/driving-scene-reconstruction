"""Tests for the dependency-light long-route tiling contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "audit_stage_h3_tbv_long_route_tiles.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_stage_h3_tbv_long_route_tiles", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVLongRouteTileTests(unittest.TestCase):
    def test_longest_supported_run_excludes_disconnected_support(self) -> None:
        run = MODULE.longest_supported_run(
            [True, True, False, True, True, True, True],
            [0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0],
        )

        self.assertEqual(run.start_index, 3)
        self.assertEqual(run.end_index, 6)
        self.assertAlmostEqual(run.length_meters, 60.0)

    def test_six_hundred_meter_run_yields_seven_overlapping_tiles(self) -> None:
        tiles = MODULE.make_tile_ranges(0.0, 610.0)

        self.assertEqual(len(tiles), 7)
        self.assertEqual(
            [
                (tile.start_progress_meters, tile.end_progress_meters)
                for tile in tiles
            ],
            [
                (0.0, 100.0),
                (80.0, 180.0),
                (160.0, 260.0),
                (240.0, 340.0),
                (320.0, 420.0),
                (400.0, 500.0),
                (480.0, 580.0),
            ],
        )
        self.assertTrue(
            all(
                left.end_progress_meters - right.start_progress_meters
                == 20.0
                for left, right in zip(tiles, tiles[1:])
            )
        )

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.make_tile_ranges(
                0.0,
                610.0,
                tile_length_meters=100.0,
                tile_overlap_meters=100.0,
            )


if __name__ == "__main__":
    unittest.main()
