"""Lightweight checks for the H3 logged browser page contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage_h3_logged_browser import WEB_PAGE  # noqa: E402


class StageH3LoggedBrowserPageTest(unittest.TestCase):
    def test_page_exposes_trial_report(self) -> None:
        self.assertIn('href="/trial.json"', WEB_PAGE)
        self.assertIn("试驾记录 JSON", WEB_PAGE)

    def test_tick_records_browser_latency_sample(self) -> None:
        self.assertIn("browser_request_to_image_ms", WEB_PAGE)
        self.assertIn("browser_input_to_image_ms", WEB_PAGE)
        self.assertIn('fetch("/trial-sample"', WEB_PAGE)

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
