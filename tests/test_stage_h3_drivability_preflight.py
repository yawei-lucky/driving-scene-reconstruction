"""Dependency-light tests for the H3 drivability preflight script."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    LoggedEgoOffsetController,
    logged_movement_profile,
)
from stage_h3_drivability_preflight import (  # noqa: E402
    control_for_step,
    distribution,
    scripted_states,
)


class StageH3DrivabilityPreflightTest(unittest.TestCase):
    def test_distribution_reports_percentiles(self) -> None:
        result = distribution([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result["count"], 5)
        self.assertAlmostEqual(result["p50"], 3.0)
        self.assertAlmostEqual(result["p95"], 4.8)
        self.assertAlmostEqual(result["maximum"], 5.0)

    def test_scripted_states_are_repeatable(self) -> None:
        profile = logged_movement_profile("visible")
        controller = LoggedEgoOffsetController.from_profile(1.0, profile)

        first = scripted_states(
            controller,
            steps=8,
            dt=0.1,
            movement_profile=profile.name,
        )
        second = scripted_states(
            controller,
            steps=8,
            dt=0.1,
            movement_profile=profile.name,
        )

        self.assertEqual(first, second)

    def test_visible_script_exercises_more_motion_than_safe(self) -> None:
        safe = logged_movement_profile("safe")
        visible = logged_movement_profile("visible")
        safe_controller = LoggedEgoOffsetController.from_profile(1.0, safe)
        visible_controller = LoggedEgoOffsetController.from_profile(1.0, visible)

        safe_states = scripted_states(
            safe_controller,
            steps=8,
            dt=0.1,
            movement_profile=safe.name,
        )
        visible_states = scripted_states(
            visible_controller,
            steps=8,
            dt=0.1,
            movement_profile=visible.name,
        )

        self.assertGreater(visible_states[-1]["x"], safe_states[-1]["x"])
        self.assertGreater(
            abs(visible_states[-1]["yaw_degrees"]),
            abs(safe_states[-1]["yaw_degrees"]),
        )

    def test_controls_are_normalized_objects(self) -> None:
        control = control_for_step(0, "visible")

        self.assertLessEqual(abs(control.steer), 1.0)
        self.assertLessEqual(control.throttle, 1.0)
        self.assertGreaterEqual(control.brake, 0.0)


if __name__ == "__main__":
    unittest.main()
