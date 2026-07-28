"""Dependency-light tests for conservative TbV cross-visit filling."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate_stage_h3_tbv_cross_visit_rgb.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_stage_h3_tbv_cross_visit_rgb", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVCrossVisitRGBTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self) -> None:
        self.assertEqual(
            MODULE.evenly_spaced_indices(11, 5),
            (0, 2, 5, 8, 10),
        )

    def test_angle_difference_wraps(self) -> None:
        self.assertAlmostEqual(
            MODULE.angle_difference_degrees(
                math.radians(179.0), math.radians(-179.0)
            ),
            2.0,
        )

    def test_parallel_output_paths_are_separate(self) -> None:
        image = Path(
            "/data/log/sensors/cameras/ring_front_center/1.jpg"
        )
        self.assertEqual(
            MODULE.output_path_for_image(
                image,
                data_root=Path("/data"),
                output_root=Path("/filled"),
                suffix=".jpg",
            ),
            Path(
                "/filled/rgb/log/sensors/cameras/"
                "ring_front_center/1.jpg"
            ),
        )
        self.assertEqual(
            MODULE.output_path_for_image(
                image,
                data_root=Path("/data"),
                output_root=Path("/filled"),
                suffix=".png",
            ),
            Path(
                "/filled/valid_masks/log/sensors/cameras/"
                "ring_front_center/1.png"
            ),
        )

    def test_resume_is_an_explicit_mode(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument(\n        "--resume"', source)
        self.assertIn('"reused_with_fill"', source)


if __name__ == "__main__":
    unittest.main()
