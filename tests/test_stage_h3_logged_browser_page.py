"""Lightweight checks for the H3 logged browser page contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage_h3_logged_browser import (  # noqa: E402
    WEB_PAGE,
    mode_help_text,
    should_advance_log_time,
)


class StageH3LoggedBrowserPageTest(unittest.TestCase):
    def test_page_exposes_trial_report(self) -> None:
        self.assertIn('href="/trial.json"', WEB_PAGE)
        self.assertIn("试驾记录 JSON", WEB_PAGE)

    def test_tick_records_browser_latency_sample(self) -> None:
        self.assertIn("browser_request_to_image_ms", WEB_PAGE)
        self.assertIn("browser_input_to_image_ms", WEB_PAGE)
        self.assertIn('fetch("/trial-sample"', WEB_PAGE)

    def test_manual_time_mode_does_not_auto_tick_on_page_load(self) -> None:
        startup_block = WEB_PAGE.split("async function tick", maxsplit=1)[1].split(
            "</script>",
            maxsplit=1,
        )[0]

        self.assertIn('let autoplay = timeMode === "auto";', WEB_PAGE)
        self.assertIn("if (autoplay)", startup_block)
        self.assertIn("人工控制", startup_block)
        self.assertIn("!autoplay && held.size === 0", WEB_PAGE)

    def test_time_mode_advance_rules(self) -> None:
        self.assertFalse(should_advance_log_time("manual", set()))
        self.assertTrue(should_advance_log_time("manual", {"w"}))
        self.assertTrue(should_advance_log_time("manual", {"a"}))
        self.assertTrue(should_advance_log_time("auto", set()))
        self.assertTrue(should_advance_log_time("auto", {"s"}))
        self.assertTrue(should_advance_log_time("manual", set(), autoplay=True))
        with self.assertRaises(ValueError):
            should_advance_log_time("free_roam", set())

    def test_mode_help_text_documents_manual_vs_auto(self) -> None:
        self.assertIn("默认人工控制", mode_help_text("manual"))
        self.assertIn("轨迹按固定节奏播放", mode_help_text("auto"))
        with self.assertRaises(ValueError):
            mode_help_text("invalid")

    def test_page_supports_arrow_keys_and_autoplay_toggle(self) -> None:
        self.assertIn('["arrowup", "w"]', WEB_PAGE)
        self.assertIn('["arrowdown", "s"]', WEB_PAGE)
        self.assertIn('["arrowleft", "a"]', WEB_PAGE)
        self.assertIn('["arrowright", "d"]', WEB_PAGE)
        self.assertIn('id="autoplay"', WEB_PAGE)
        self.assertIn('autoplayButton.addEventListener("click"', WEB_PAGE)
        self.assertIn('"&autoplay=" + autoplayQuery', WEB_PAGE)
        self.assertIn("↑ / W 加速", WEB_PAGE)
        self.assertIn("↓ / S 减速", WEB_PAGE)

    def test_page_records_manual_drivability_review(self) -> None:
        self.assertIn("人工试驾验收", WEB_PAGE)
        self.assertIn('fetch("/trial-review"', WEB_PAGE)
        self.assertIn(
            'data-review-gate="road_lane_curb_continuity"',
            WEB_PAGE,
        )
        self.assertIn(
            'data-review-gate="steering_response_direction"',
            WEB_PAGE,
        )
        self.assertIn(
            'data-review-gate="dynamic_traffic_decision_impact"',
            WEB_PAGE,
        )

    def test_reset_does_not_pollute_next_input_latency(self) -> None:
        reset_block = WEB_PAGE.split("async function reset()", maxsplit=1)[1].split(
            "async function tick",
            maxsplit=1,
        )[0]

        self.assertIn("pendingInputStartedAt = null;", reset_block)
        self.assertIn("重置→图像加载", reset_block)


if __name__ == "__main__":
    unittest.main()
