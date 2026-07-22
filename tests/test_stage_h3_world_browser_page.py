"""Lightweight checks for the restricted H3 world-browser contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage_h3_world_browser import (  # noqa: E402
    boundary_help_text,
    control_for_keys,
    render_web_page,
    should_step_world,
)


class StageH3WorldBrowserPageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.page = render_web_page(0.1)

    def test_controls_use_throttle_brake_and_left_positive_steering(self) -> None:
        left = control_for_keys({"w", "a"})
        right = control_for_keys({"w", "d"})
        braking = control_for_keys({"w", "s"})

        self.assertEqual(left.throttle, 1.0)
        self.assertEqual(left.steer, 1.0)
        self.assertEqual(right.steer, -1.0)
        self.assertEqual(braking.throttle, 0.0)
        self.assertEqual(braking.brake, 1.0)

    def test_tick_rule_coasts_until_stopped(self) -> None:
        self.assertFalse(should_step_world(set(), 0.0))
        self.assertFalse(should_step_world({"a"}, 0.0))
        self.assertTrue(should_step_world({"w"}, 0.0))
        self.assertTrue(should_step_world(set(), 0.2))
        self.assertTrue(should_step_world({"s"}, 0.2))

    def test_page_starts_stopped_and_has_no_autoplay(self) -> None:
        self.assertIn("默认静止", self.page)
        self.assertNotIn('id="autoplay"', self.page)
        self.assertNotIn("自动播放", self.page)
        self.assertNotIn("tick(generation);\n  </script>", self.page)

    def test_page_supports_arrow_keys_and_world_controls(self) -> None:
        self.assertIn('["arrowup", "w"]', self.page)
        self.assertIn('["arrowdown", "s"]', self.page)
        self.assertIn('["arrowleft", "a"]', self.page)
        self.assertIn('["arrowright", "d"]', self.page)
        self.assertIn("↑ / W 油门", self.page)
        self.assertIn("↓ / S 刹车", self.page)

    def test_page_marks_provisional_boundary_and_reset_behavior(self) -> None:
        self.assertIn("不是已认证道路", boundary_help_text())
        self.assertIn("原始真实轨迹", self.page)
        self.assertIn("最多 1m", self.page)
        self.assertIn("boundary_hit", self.page)
        self.assertIn("已触及实验边界", self.page)
        self.assertIn("按 R 重置", self.page)
        self.assertIn('fetch("/reset"', self.page)

    def test_page_reports_corridor_progress_and_deviation(self) -> None:
        self.assertIn("道路进度", self.page)
        self.assertIn("偏离", self.page)
        self.assertIn("result.corridor", self.page)

    def test_page_prioritizes_native_large_view_and_hides_shortcuts(self) -> None:
        self.assertIn("width: calc(100vw - 8px)", self.page)
        self.assertIn("max-height: calc(100vh - 82px)", self.page)
        self.assertIn('id="controls" hidden', self.page)
        self.assertIn('id="toggle-controls"', self.page)


if __name__ == "__main__":
    unittest.main()
