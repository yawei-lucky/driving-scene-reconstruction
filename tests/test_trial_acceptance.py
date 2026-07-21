"""Tests for H3 browser trial acceptance checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    BrowserTrialRecorder,
    TrialAcceptanceConfig,
    evaluate_trial_report,
)
from driving_scene_reconstruction.sim.trial_recorder import (  # noqa: E402
    MANUAL_REVIEW_GATES,
)


def sample_payload(
    *,
    index: int,
    time_seconds: float,
    duration_seconds: float,
    keys: str = "",
    input_to_image_ms: float | None = None,
    request_to_image_ms: float = 80.0,
    server_to_jpeg_ms: float = 75.0,
) -> dict[str, object]:
    return {
        "sequence": index + 1,
        "keys": keys,
        "client_unix_ms": float(index),
        "browser_request_to_image_ms": request_to_image_ms,
        "browser_input_to_image_ms": input_to_image_ms,
        "server": {
            "time": time_seconds,
            "duration": duration_seconds,
            "x": 0.05 * index,
            "y": 0.0,
            "yaw_degrees": 0.2 * index,
            "speed": 0.5 if keys else 0.0,
            "logical_frame": index,
            "renderer_ms": 70.0,
            "server_control_to_jpeg_ms": server_to_jpeg_ms,
            "frame_selection_error_ms": 0.0,
            "camera_time_spread_ms": 81.0,
            "movement_profile": "visible",
        },
    }


class TrialAcceptanceTest(unittest.TestCase):
    def make_recorder(self, path: Path) -> BrowserTrialRecorder:
        return BrowserTrialRecorder(
            scene="040",
            movement_profile="visible",
            output_scale=0.25,
            dt_seconds=0.1,
            checkpoint_step=7999,
            logged_duration_seconds=7.8992390632629395,
            output_path=path,
        )

    def record_complete_trial(
        self,
        recorder: BrowserTrialRecorder,
        *,
        input_to_image_ms: float | None = 90.0,
        request_to_image_ms: float = 80.0,
        server_to_jpeg_ms: float = 75.0,
        review_statuses: dict[str, str] | None = None,
        reset: bool = True,
    ) -> dict[str, object]:
        duration = recorder.logged_duration_seconds
        for index in range(80):
            recorder.record_sample(
                sample_payload(
                    index=index,
                    time_seconds=min(index * 0.1, duration),
                    duration_seconds=duration,
                    keys="wa" if index == 3 else "",
                    input_to_image_ms=input_to_image_ms if index == 3 else None,
                    request_to_image_ms=request_to_image_ms,
                    server_to_jpeg_ms=server_to_jpeg_ms,
                )
            )
        if reset:
            recorder.record_reset(
                {
                    "time": 0.0,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_degrees": 0.0,
                    "speed": 0.0,
                    "logical_frame": 0,
                    "server_control_to_jpeg_ms": server_to_jpeg_ms,
                }
            )
        recorder.record_manual_review(
            {
                "gates": review_statuses
                if review_statuses is not None
                else {gate: "pass" for gate in MANUAL_REVIEW_GATES}
            }
        )
        return recorder.report()

    def assertGateFailed(self, result: dict[str, object], gate: str) -> None:
        self.assertFalse(result["passed"])
        self.assertIn(gate, result["failures"])

    def test_complete_trial_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            report = self.record_complete_trial(recorder)

        result = evaluate_trial_report(report)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["failures"], [])

    def test_latest_manual_review_must_happen_after_complete_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            recorder.record_manual_review(
                {"gates": {gate: "pass" for gate in MANUAL_REVIEW_GATES}}
            )
            duration = recorder.logged_duration_seconds
            for index in range(80):
                recorder.record_sample(
                    sample_payload(
                        index=index,
                        time_seconds=min(index * 0.1, duration),
                        duration_seconds=duration,
                        keys="wa" if index == 3 else "",
                        input_to_image_ms=90.0 if index == 3 else None,
                    )
                )
            recorder.record_reset(
                {
                    "time": 0.0,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_degrees": 0.0,
                    "speed": 0.0,
                    "logical_frame": 0,
                    "server_control_to_jpeg_ms": 75.0,
                }
            )
            report = recorder.report()

        result = evaluate_trial_report(report)

        self.assertGateFailed(result, "manual_review_after_enough_samples")
        self.assertIn("manual_review_after_completed_log", result["failures"])

    def test_missing_reset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            report = self.record_complete_trial(recorder, reset=False)

        result = evaluate_trial_report(report)

        self.assertGateFailed(result, "reset_recorded")

    def test_missing_input_latency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            report = self.record_complete_trial(recorder, input_to_image_ms=None)

        result = evaluate_trial_report(report)

        self.assertGateFailed(result, "browser_input_to_image_p95")

    def test_high_latency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            report = self.record_complete_trial(
                recorder,
                input_to_image_ms=140.0,
                request_to_image_ms=125.0,
            )

        result = evaluate_trial_report(report)

        self.assertGateFailed(result, "browser_request_to_image_p95")
        self.assertIn("browser_input_to_image_p95", result["failures"])

    def test_manual_unsure_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            statuses = {gate: "pass" for gate in MANUAL_REVIEW_GATES}
            statuses["dynamic_traffic_decision_impact"] = "unsure"
            report = self.record_complete_trial(
                recorder,
                review_statuses=statuses,
            )

        result = evaluate_trial_report(report)

        self.assertGateFailed(result, "manual_review_all_passed")
        self.assertIn("all_manual_gate_statuses_pass", result["failures"])

    def test_expected_safe_profile_can_be_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            report = self.record_complete_trial(recorder)

        result = evaluate_trial_report(
            report,
            TrialAcceptanceConfig(expected_movement_profile="safe"),
        )

        self.assertGateFailed(result, "expected_movement_profile")


if __name__ == "__main__":
    unittest.main()
