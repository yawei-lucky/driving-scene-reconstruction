"""Dependency-light tests for the TbV route-driving browser entry point."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).parents[1] / "examples" / "stage_h3_tbv_driving_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("stage_h3_tbv_driving_adapter", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVDrivingAdapterEntryPointTests(unittest.TestCase):
    def test_builds_exact_bounded_route_segment(self) -> None:
        route = tuple(
            MODULE.RouteSample(float(i), float(i), float(i), 0.0, 0.0, 0.0)
            for i in range(-3, 4)
        )

        supported = MODULE.supported_route(
            name="straight",
            renderer_profile="test",
            route=route,
            start=-2.5,
            end=2.5,
        )

        self.assertAlmostEqual(supported.start_progress_from_anchor, -2.5)
        self.assertAlmostEqual(supported.end_progress_from_anchor, 2.5)
        self.assertAlmostEqual(supported.corridor.samples[0].x, -2.5)
        self.assertAlmostEqual(supported.corridor.samples[-1].x, 2.5)
        self.assertAlmostEqual(supported.corridor.half_width, 1.0)
        self.assertAlmostEqual(
            math.degrees(supported.corridor.max_heading_error), 30.0
        )

    def test_page_exposes_branch_choice_and_evidence(self) -> None:
        page = MODULE.render_web_page(0.1)

        self.assertIn('data-branch="straight"', page)
        self.assertIn('data-branch="right"', page)
        self.assertIn("/evidence.json", page)
        self.assertIn("/evidence-sample", page)
        self.assertIn("/evidence-route-timing", page)
        self.assertIn("100.000000", page)

    def test_mosaic_preserves_heterogeneous_camera_aspect_ratios(self) -> None:
        image_module = mock.Mock()
        image_module.new.return_value = mock.Mock()
        image_module.fromarray.side_effect = lambda frame: frame
        image_module.Resampling.LANCZOS = object()
        draw_module = mock.Mock()
        frames = {}
        for index, name in enumerate(MODULE.CAMERAS):
            tile = mock.Mock()
            tile.convert.return_value = tile
            tile.width = 300 if name == "ring_front_center" else 390
            tile.height = 520 if name == "ring_front_center" else 200
            frames[name] = tile
        support = {
            "phase": "common_approach",
            "selected_branch": None,
            "progress_from_anchor_meters": -20.0,
            "lateral_offset_meters": 0.0,
        }

        MODULE.make_mosaic(
            image_module, draw_module, frames, MODULE.EgoState(), support
        )

        for tile in frames.values():
            tile.thumbnail.assert_called_once()
        frames["ring_front_center"].thumbnail.assert_called_once_with(
            (520, 520), image_module.Resampling.LANCZOS
        )


if __name__ == "__main__":
    unittest.main()
