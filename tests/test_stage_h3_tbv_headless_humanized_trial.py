"""Dependency-light tests for the no-browser TbV humanized trial."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "stage_h3_tbv_headless_humanized_trial.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage_h3_tbv_headless_humanized_trial", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVHeadlessHumanizedTrialTests(unittest.TestCase):
    def test_programme_requires_a_d_recovery_and_heading_settle(self) -> None:
        phase = MODULE.HumanizedPhase()

        self.assertIsNone(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=0.1),
                progress=-19.8,
                lateral_offset=0.0,
                heading_error_degrees=0.0,
            )
        )
        self.assertEqual(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=0.2),
                progress=-19.4,
                lateral_offset=0.0,
                heading_error_degrees=0.0,
            ),
            "AUTO->A_DEVIATE",
        )
        self.assertEqual(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=0.6),
                progress=-18.0,
                lateral_offset=0.29,
                heading_error_degrees=2.0,
            ),
            "A_DEVIATE->A_RECOVER",
        )
        self.assertEqual(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=1.0),
                progress=-16.0,
                lateral_offset=0.10,
                heading_error_degrees=-2.0,
            ),
            "A_RECOVER->D_DEVIATE",
        )
        self.assertEqual(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=1.4),
                progress=-14.0,
                lateral_offset=-0.29,
                heading_error_degrees=-2.0,
            ),
            "D_DEVIATE->D_RECOVER",
        )
        self.assertIsNone(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=1.8),
                progress=-12.0,
                lateral_offset=-0.10,
                heading_error_degrees=5.0,
            )
        )
        self.assertEqual(
            MODULE.update_phase(
                phase,
                state=MODULE.EgoState(time=1.9),
                progress=-11.5,
                lateral_offset=-0.10,
                heading_error_degrees=3.0,
            ),
            "D_RECOVER->CRUISE",
        )

    def test_video_dt_matches_real_twenty_fps(self) -> None:
        self.assertEqual(MODULE.VIDEO_FPS, 20)
        self.assertAlmostEqual(1.0 / MODULE.VIDEO_FPS, 0.05)


if __name__ == "__main__":
    unittest.main()
