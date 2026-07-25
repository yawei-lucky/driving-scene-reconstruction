"""Dependency-light tests for the continuous TbV tile seam probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "probe_stage_h3_tbv_tile_continuous.py"
)
SPEC = importlib.util.spec_from_file_location(
    "probe_stage_h3_tbv_tile_continuous", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVTileContinuousProbeTests(unittest.TestCase):
    def test_interpolation_bracket_handles_endpoints_and_middle(self) -> None:
        timestamps = (100, 200, 400)

        self.assertEqual(
            MODULE.interpolation_bracket(timestamps, 100),
            (0, 1, 0.0),
        )
        self.assertEqual(
            MODULE.interpolation_bracket(timestamps, 300),
            (1, 2, 0.5),
        )
        self.assertEqual(
            MODULE.interpolation_bracket(timestamps, 400),
            (1, 2, 1.0),
        )

    def test_interpolation_clamps_only_tiny_endpoint_rounding(self) -> None:
        timestamps = (315970604799927217, 315970606899927213)

        self.assertEqual(
            MODULE.interpolation_bracket(
                timestamps,
                float(timestamps[-1]),
            )[2],
            1.0,
        )
        with self.assertRaises(ValueError):
            MODULE.interpolation_bracket(
                timestamps,
                timestamps[-1] + 1_000,
            )

    def test_smoothstep_blend_has_fixed_ends_and_soft_middle(self) -> None:
        self.assertEqual(MODULE.smoothstep_blend(0.2), 0.0)
        self.assertEqual(MODULE.smoothstep_blend(0.5), 0.5)
        self.assertEqual(MODULE.smoothstep_blend(0.8), 1.0)

    def test_frame_count_uses_speed_and_includes_endpoints(self) -> None:
        self.assertEqual(MODULE.frame_count_for_speed(18.0, 12.0, 20), 31)

    def test_invalid_numeric_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.frame_count_for_speed(18.0, 0.0, 20)
        with self.assertRaises(ValueError):
            MODULE.interpolation_bracket((100,), 100)
        with self.assertRaises(ValueError):
            MODULE.smoothstep_blend(0.5, 0.8, 0.2)


if __name__ == "__main__":
    unittest.main()
