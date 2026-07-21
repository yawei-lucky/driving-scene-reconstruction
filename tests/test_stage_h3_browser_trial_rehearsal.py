"""Tests for the scripted H3 browser trial rehearsal client."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import BrowserTrialRecorder  # noqa: E402
from stage_h3_browser_trial_rehearsal import (  # noqa: E402
    ALLOWED_REHEARSAL_FAILURES,
    control_keys_for_step,
    run_rehearsal,
)


class FakeBrowserService:
    def __init__(self, output_path: Path) -> None:
        self.duration = 7.8992390632629395
        self.dt = 0.1
        self.time = 0.0
        self.logical_frame = 0
        self.recorder = BrowserTrialRecorder(
            scene="040",
            movement_profile="visible",
            output_scale=0.25,
            dt_seconds=self.dt,
            checkpoint_step=7999,
            logged_duration_seconds=self.duration,
            output_path=output_path,
        )

    def get_json(self, path: str) -> dict[str, object]:
        if path != "/trial.json":
            raise AssertionError(f"unexpected GET JSON path: {path}")
        return self.recorder.report()

    def get_bytes(self, path: str) -> bytes:
        parsed = urlparse(path)
        if parsed.path != "/frame.jpg":
            raise AssertionError(f"unexpected GET bytes path: {path}")
        return b"fake-jpeg"

    def post_json(
        self,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        parsed = urlparse(path)
        if parsed.path == "/tick":
            keys = parse_qs(parsed.query).get("keys", [""])[0]
            self.time = min(self.time + self.dt, self.duration)
            self.logical_frame = min(self.logical_frame + 1, 79)
            return {
                "time": self.time,
                "duration": self.duration,
                "x": 0.1 if "w" in keys else 0.0,
                "y": 0.1 if "a" in keys else 0.0,
                "yaw_degrees": 2.0 if "a" in keys else -2.0 if "d" in keys else 0.0,
                "speed": 0.5 if "w" in keys else 0.0,
                "logical_frame": self.logical_frame,
                "renderer_ms": 60.0,
                "server_control_to_jpeg_ms": 65.0,
                "frame_selection_error_ms": 0.0,
                "camera_time_spread_ms": 81.0,
                "movement_profile": "visible",
            }
        if parsed.path == "/trial-sample":
            assert payload is not None
            return {"summary": self.recorder.record_sample(payload)}
        if parsed.path == "/reset":
            self.time = 0.0
            self.logical_frame = 0
            payload = {
                "time": 0.0,
                "x": 0.0,
                "y": 0.0,
                "yaw_degrees": 0.0,
                "speed": 0.0,
                "logical_frame": 0,
                "server_control_to_jpeg_ms": 65.0,
            }
            self.recorder.record_reset(payload)
            return payload
        if parsed.path == "/trial-review":
            assert payload is not None
            return {"summary": self.recorder.record_manual_review(payload)}
        raise AssertionError(f"unexpected POST path: {path}")


class StageH3BrowserTrialRehearsalTest(unittest.TestCase):
    def test_control_schedule_exercises_all_driving_keys(self) -> None:
        observed = set("".join(control_keys_for_step(index) for index in range(40)))

        self.assertEqual(observed, {"w", "a", "d", "s"})

    def test_rehearsal_passes_when_only_manual_gates_are_unsure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "rehearsal.json"
            acceptance_output_path = Path(tmp) / "acceptance.json"
            service = FakeBrowserService(Path(tmp) / "trial.json")

            report = run_rehearsal(
                service,
                steps=80,
                expected_movement_profile="visible",
                output_path=output_path,
                acceptance_output_path=acceptance_output_path,
            )
            output_exists = output_path.exists()
            acceptance_output_exists = acceptance_output_path.exists()

        self.assertTrue(report["passed"], report["unexpected_acceptance_failures"])
        self.assertEqual(report["steps_recorded"], 79)
        self.assertEqual(report["unexpected_acceptance_failures"], [])
        self.assertEqual(report["missing_expected_acceptance_failures"], [])
        self.assertEqual(
            set(report["acceptance_result"]["failures"]),
            ALLOWED_REHEARSAL_FAILURES,
        )
        self.assertTrue(output_exists)
        self.assertTrue(acceptance_output_exists)

    def test_rehearsal_fails_when_non_manual_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeBrowserService(Path(tmp) / "trial.json")

            report = run_rehearsal(
                service,
                steps=10,
                expected_movement_profile="visible",
            )

        self.assertFalse(report["passed"])
        self.assertIn("enough_trial_samples", report["unexpected_acceptance_failures"])


if __name__ == "__main__":
    unittest.main()
