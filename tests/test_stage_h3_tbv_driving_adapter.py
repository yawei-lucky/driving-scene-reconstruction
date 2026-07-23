"""Dependency-light tests for the TbV route-driving browser entry point."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
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

    def test_adapter_uses_faster_bounded_driving_speed(self) -> None:
        route = tuple(
            MODULE.RouteSample(float(i), float(i), float(i), 0.0, 0.0, 0.0)
            for i in range(-30, 51)
        )
        renderer = SimpleNamespace(
            routes={
                MODULE.RIGHT_TRAVERSAL: route,
                MODULE.STRAIGHT_TRAVERSAL: route,
            }
        )

        adapter = MODULE.make_adapter(renderer, max_speed_mps=4.0)

        self.assertAlmostEqual(adapter.vehicle_model.max_speed, 4.0)
        self.assertAlmostEqual(adapter.vehicle_model.max_acceleration, 2.0)
        self.assertAlmostEqual(adapter.vehicle_model.max_braking, 5.0)

    def test_page_exposes_branch_choice_and_evidence(self) -> None:
        page = MODULE.render_web_page(0.1)

        self.assertIn('data-branch="straight"', page)
        self.assertIn('data-branch="right"', page)
        self.assertIn("/evidence.json", page)
        self.assertIn("/evidence-sample", page)
        self.assertIn("/evidence-route-timing", page)
        self.assertIn('href="/diagnostic"', page)
        self.assertIn("前向环视", page)
        self.assertIn("原相机图", page)
        self.assertIn("cylindrical driving view", page)
        self.assertIn("100.000000", page)

    def test_diagnostic_page_is_separate_from_driving_cockpit(self) -> None:
        self.assertIn("/diagnostic.jpg", MODULE.DIAGNOSTIC_PAGE)
        self.assertIn("不作为真人驾驶视图", MODULE.DIAGNOSTIC_PAGE)
        self.assertIn('id="refresh"', MODULE.DIAGNOSTIC_PAGE)
        self.assertNotIn("setInterval", MODULE.DIAGNOSTIC_PAGE)
        self.assertIn("<h1>原相机图</h1>", MODULE.DIAGNOSTIC_PAGE)
        self.assertIn('href="/"', MODULE.DIAGNOSTIC_PAGE)

    def test_diagnostic_mosaic_preserves_camera_aspect_ratios(self) -> None:
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

        MODULE.make_diagnostic_mosaic(
            image_module, draw_module, frames, MODULE.EgoState(), support
        )

        for tile in frames.values():
            tile.thumbnail.assert_called_once()
        frames["ring_front_center"].thumbnail.assert_called_once_with(
            (520, 520), image_module.Resampling.LANCZOS
        )

    def test_bev_uses_route_geometry_and_marks_it_non_rgb(self) -> None:
        image_module = mock.Mock()
        image_module.new.return_value = mock.Mock()
        draw = mock.Mock()
        draw_module = mock.Mock()
        draw_module.Draw.return_value = draw
        samples = (
            SimpleNamespace(x=-1.0, y=0.0),
            SimpleNamespace(x=0.0, y=0.0),
            SimpleNamespace(x=10.0, y=0.0),
        )
        corridor = SimpleNamespace(samples=samples, half_width=1.0)
        straight = SimpleNamespace(corridor=corridor)
        right = SimpleNamespace(corridor=corridor)
        adapter = SimpleNamespace(
            selected_branch=None,
            branches={"straight": straight, "right": right},
            active_route=straight,
        )
        support = {
            "distance_margin_meters": 0.7,
            "lateral_offset_meters": 0.3,
        }

        MODULE.make_trajectory_bev(
            image_module,
            draw_module,
            adapter,
            MODULE.EgoState(),
            support,
            size=240,
        )

        self.assertTrue(draw.line.called)
        self.assertTrue(draw.polygon.called)
        labels = [call.args[1] for call in draw.text.call_args_list]
        self.assertIn("not overhead RGB", labels)

    def test_overhead_assist_is_visual_only_and_masks_vehicle(self) -> None:
        image_module = mock.Mock()
        canvas = mock.Mock()
        canvas.convert.return_value = canvas
        image_module.fromarray.return_value = canvas
        draw = mock.Mock()
        draw_module = mock.Mock()
        draw_module.Draw.return_value = draw
        samples = (
            SimpleNamespace(x=0.0, y=0.0),
            SimpleNamespace(x=10.0, y=0.0),
        )
        corridor = SimpleNamespace(samples=samples, half_width=1.0)
        route = SimpleNamespace(corridor=corridor)
        adapter = SimpleNamespace(
            selected_branch=None,
            branches={"straight": route, "right": route},
            active_route=route,
        )
        support = {
            "distance_margin_meters": 0.8,
            "lateral_offset_meters": 0.2,
        }
        composer = SimpleNamespace(
            size=240,
            forward_meters=12.0,
            rear_meters=4.0,
            side_meters=8.0,
            screen=lambda forward, left: (
                int(round(120 - 10 * left)),
                int(round(180 - 10 * forward)),
            ),
        )

        MODULE.make_trajectory_bev(
            image_module,
            draw_module,
            adapter,
            MODULE.EgoState(),
            support,
            size=240,
            overhead=object(),
            overhead_composer=composer,
        )

        labels = [call.args[1] for call in draw.text.call_args_list]
        self.assertIn("OVERHEAD  ·  VISUAL ONLY", labels)
        self.assertIn("black = no camera coverage", labels)
        self.assertTrue(
            any(
                call.kwargs.get("fill") == (24, 29, 34)
                for call in draw.polygon.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
