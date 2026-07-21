"""Tests for browser-side H3 trial recording."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import BrowserTrialRecorder  # noqa: E402
from driving_scene_reconstruction.sim.trial_recorder import (  # noqa: E402
    MANUAL_REVIEW_GATES,
)


def sample_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "sequence": 1,
        "keys": "wa",
        "client_unix_ms": 123.0,
        "browser_request_to_image_ms": 80.0,
        "browser_input_to_image_ms": 95.0,
        "server": {
            "time": 0.1,
            "duration": 1.0,
            "x": 0.04,
            "y": 0.01,
            "yaw_degrees": 2.4,
            "speed": 0.4,
            "logical_frame": 1,
            "renderer_ms": 70.0,
            "server_control_to_jpeg_ms": 78.0,
            "frame_selection_error_ms": 0.0,
            "camera_time_spread_ms": 81.0,
            "movement_profile": "visible",
        },
    }
    payload.update(overrides)
    return payload


class BrowserTrialRecorderTest(unittest.TestCase):
    def make_recorder(self, path: Path) -> BrowserTrialRecorder:
        return BrowserTrialRecorder(
            scene="040",
            movement_profile="visible",
            output_scale=0.25,
            dt_seconds=0.1,
            checkpoint_step=7999,
            logged_duration_seconds=1.0,
            output_path=path,
        )

    def test_record_sample_persists_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trial.json"
            recorder = self.make_recorder(output_path)

            summary = recorder.record_sample(sample_payload())
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["observed_key_sets"], ["aw"])
        self.assertEqual(report["summary"]["checkpoint_step"], 7999)
        self.assertEqual(report["samples"][0]["logical_frame"], 1)
        self.assertEqual(
            report["summary"]["browser_input_to_image_ms"]["p50"],
            95.0,
        )

    def test_record_reset_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")

            summary = recorder.record_reset(
                {
                    "time": 0.0,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_degrees": 0.0,
                    "speed": 0.0,
                    "logical_frame": 0,
                    "server_control_to_jpeg_ms": 70.0,
                }
            )

        self.assertEqual(summary["reset_count"], 1)

    def test_invalid_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")

            with self.assertRaises(ValueError):
                recorder.record_sample(sample_payload(keys="wx"))

    def test_non_integer_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")

            with self.assertRaises(ValueError):
                recorder.record_sample(sample_payload(sequence=1.5))

    def test_profile_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            payload = sample_payload()
            server = dict(payload["server"])  # type: ignore[arg-type]
            server["movement_profile"] = "safe"
            payload["server"] = server

            with self.assertRaises(ValueError):
                recorder.record_sample(payload)

    def test_manual_review_persists_drivability_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trial.json"
            recorder = self.make_recorder(output_path)

            summary = recorder.record_manual_review(
                {
                    "client_unix_ms": 456.0,
                    "reviewer": "operator",
                    "gates": {gate: "pass" for gate in MANUAL_REVIEW_GATES},
                    "notes": "lane, steering, latency, and traffic residue looked usable",
                }
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["manual_review_count"], 1)
        self.assertTrue(summary["manual_review_completed"])
        self.assertTrue(summary["manual_review_all_passed"])
        self.assertEqual(summary["manual_review_blocking_gates"], {})
        self.assertEqual(report["manual_reviews"][0]["reviewer"], "operator")
        self.assertEqual(
            report["manual_reviews"][0]["statuses"][
                "road_lane_curb_continuity"
            ],
            "pass",
        )
        self.assertIn("manual_review_scope", report["trial"])

    def test_manual_review_unsure_gate_blocks_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            gates = {gate: "pass" for gate in MANUAL_REVIEW_GATES}
            gates["dynamic_traffic_decision_impact"] = "unsure"

            summary = recorder.record_manual_review({"gates": gates})

        self.assertFalse(summary["manual_review_all_passed"])
        self.assertEqual(
            summary["manual_review_blocking_gates"],
            {"dynamic_traffic_decision_impact": "unsure"},
        )

    def test_manual_review_requires_every_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            gates = {gate: "pass" for gate in MANUAL_REVIEW_GATES}
            gates.pop("physical_input_display_latency")

            with self.assertRaises(ValueError):
                recorder.record_manual_review({"gates": gates})

    def test_manual_review_rejects_unknown_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")
            gates = {gate: "pass" for gate in MANUAL_REVIEW_GATES}
            gates["photorealism_score"] = "pass"

            with self.assertRaises(ValueError):
                recorder.record_manual_review({"gates": gates})

    def test_manual_review_rejects_long_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp) / "trial.json")

            with self.assertRaises(ValueError):
                recorder.record_manual_review(
                    {
                        "gates": {gate: "pass" for gate in MANUAL_REVIEW_GATES},
                        "notes": "x" * 2049,
                    }
                )


if __name__ == "__main__":
    unittest.main()
