"""Dependency-light tests for the adjacent TbV tile seam probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "probe_stage_h3_tbv_tile_seam.py"
)
SPEC = importlib.util.spec_from_file_location(
    "probe_stage_h3_tbv_tile_seam", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVTileSeamProbeTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self) -> None:
        self.assertEqual(
            MODULE.evenly_spaced_indices(11, 5),
            (0, 2, 5, 8, 10),
        )

    def test_requested_count_is_clamped_to_available_values(self) -> None:
        self.assertEqual(
            MODULE.evenly_spaced_indices(3, 5),
            (0, 1, 2),
        )

    def test_distribution_interpolates_percentiles(self) -> None:
        values = MODULE.distribution([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(values["p50"], 2.5)
        self.assertAlmostEqual(values["p95"], 3.85)


if __name__ == "__main__":
    unittest.main()
