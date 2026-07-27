from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load() -> object:
    path = ROOT / "scripts" / "build_stage_h3_mtgs_app_control_demo.py"
    spec = importlib.util.spec_from_file_location(
        "mtgs_app_control_demo_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEMO = _load()


class MtgsAppControlDemoTests(unittest.TestCase):
    def test_phase_boundaries_cover_the_complete_schedule(self) -> None:
        expected = (
            (0, "auto"),
            (DEMO.AUTO_END_FRAME - 1, "auto"),
            (DEMO.AUTO_END_FRAME, "remote"),
            (DEMO.REMOTE_END_FRAME - 1, "remote"),
            (DEMO.REMOTE_END_FRAME, "estop"),
            (DEMO.ESTOP_END_FRAME - 1, "estop"),
            (DEMO.ESTOP_END_FRAME, "reset"),
            (DEMO.RESET_END_FRAME - 1, "reset"),
            (DEMO.RESET_END_FRAME, "auto_restarted"),
            (DEMO.TOTAL_FRAMES - 1, "auto_restarted"),
        )
        for frame, phase in expected:
            with self.subTest(frame=frame):
                self.assertEqual(DEMO.demo_phase(frame), phase)

    def test_only_applied_remote_controls_light_keys(self) -> None:
        control = DEMO.HumanControl(
            steer=0.25,
            throttle=0.8,
            brake=0.0,
        )
        self.assertEqual(
            DEMO.active_control_keys(control, "remote"),
            frozenset(("W", "A")),
        )
        self.assertEqual(
            DEMO.active_control_keys(control, "auto"),
            frozenset(),
        )

    def test_right_and_brake_mapping(self) -> None:
        control = DEMO.HumanControl(
            steer=-0.25,
            throttle=0.0,
            brake=0.7,
        )
        self.assertEqual(
            DEMO.active_control_keys(control, "remote"),
            frozenset(("S", "D")),
        )

    def test_invalid_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DEMO.demo_phase(-1)
        with self.assertRaises(ValueError):
            DEMO.demo_phase(DEMO.TOTAL_FRAMES)


if __name__ == "__main__":
    unittest.main()
