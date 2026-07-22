"""Dependency-light tests for the TbV route adapter and evidence outlet."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    BranchedRouteDrivingAdapter,
    EgoState,
    HumanControl,
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    RouteDrivingEvidenceRecorder,
    SupportedRoute,
)


def corridor(points: tuple[tuple[float, float], ...]) -> LoggedCenterlineCorridor:
    return LoggedCenterlineCorridor(
        tuple(
            LoggedCenterlineSample(index, index * 0.1, x, y, 0.0)
            for index, (x, y) in enumerate(points)
        ),
        half_width=1.0,
        max_heading_error=math.radians(30.0),
    )


class BranchedRouteDrivingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        common = SupportedRoute(
            "common",
            "right_traversal",
            corridor(((-20.0, 0.0), (-10.0, 0.0), (0.0, 0.0))),
            -20.0,
        )
        straight = SupportedRoute(
            "straight",
            "straight_traversal",
            corridor(((-20.0, 0.0), (0.0, 0.0), (10.0, 0.0))),
            -20.0,
        )
        right = SupportedRoute(
            "right",
            "right_traversal",
            corridor(((-20.0, 0.0), (0.0, 0.0), (5.0, -2.0))),
            -20.0,
        )
        self.adapter = BranchedRouteDrivingAdapter(
            common_route=common,
            branches={"straight": straight, "right": right},
            spawn_state=EgoState(x=-20.0),
            selection_window_meters=0.5,
        )

    def test_spawns_at_minus_twenty_and_requires_anchor_selection(self) -> None:
        self.assertEqual(self.adapter.reset(), EgoState(x=-20.0))
        near_anchor = EgoState(x=-0.4, speed=0.5)

        update = self.adapter.step(
            near_anchor, HumanControl(throttle=1.0), dt=0.1
        )

        self.assertTrue(update.selection_required)
        self.assertEqual(update.state.speed, 0.0)
        self.assertEqual(update.support.phase, "branch_selection")
        self.assertIsNone(update.support.selected_branch)

    def test_branch_cannot_be_selected_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "shared anchor"):
            self.adapter.select_branch("straight", EgoState(x=-5.0))

    def test_selected_branch_changes_support_and_renderer_profile(self) -> None:
        state = EgoState(x=-0.4)

        support = self.adapter.select_branch("straight", state)
        update = self.adapter.step(
            state, HumanControl(throttle=1.0), dt=0.1
        )

        self.assertEqual(support.selected_branch, "straight")
        self.assertEqual(support.renderer_profile, "straight_traversal")
        self.assertFalse(update.selection_required)
        self.assertGreater(update.state.x, state.x)

    def test_anchor_reports_each_branch_support_before_selection(self) -> None:
        state = EgoState(x=-0.4)

        straight = self.adapter.branch_support("straight", state)
        right = self.adapter.branch_support("right", state)

        self.assertEqual(straight.phase, "branch_candidate")
        self.assertEqual(straight.selected_branch, "straight")
        self.assertTrue(straight.within_declared_support)
        self.assertTrue(right.within_declared_support)
        self.assertIsNone(self.adapter.selected_branch)

    def test_route_boundary_stops_without_clamping_pose(self) -> None:
        state = EgoState(x=-0.4)
        self.adapter.select_branch("straight", state)
        edge = EgoState(x=10.9, speed=2.0)

        update = self.adapter.step(edge, HumanControl(throttle=1.0), dt=0.1)

        self.assertTrue(update.boundary_hit)
        self.assertAlmostEqual(update.state.x, edge.x)
        self.assertEqual(update.state.speed, 0.0)


class RouteDrivingEvidenceRecorderTests(unittest.TestCase):
    def test_persists_server_and_browser_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            recorder = RouteDrivingEvidenceRecorder(
                scene="tbv_branch_pair",
                config_path="config.yml",
                checkpoint_path="step.ckpt",
                checkpoint_step=7999,
                output_scale=0.5,
                dt_seconds=0.1,
                camera_names=("front", "left"),
                route_contract={"lateral_limit_meters": 1.0},
                limitations=("not certified",),
                output_path=output,
            )
            recorder.record_server_sample(
                {
                    "sequence": 0,
                    "control_keys": "aw",
                    "simulation_time_seconds": 0.1,
                    "x_meters": -19.99,
                    "y_meters": 0.0,
                    "yaw_degrees": 0.1,
                    "speed_mps": 0.15,
                    "route_support": {
                        "phase": "common_approach",
                        "active_route": "common",
                        "renderer_profile": "right",
                        "selected_branch": None,
                        "selection_required": False,
                        "progress_from_anchor_meters": -19.99,
                        "lateral_offset_meters": 0.0,
                        "distance_to_centerline_meters": 0.0,
                        "heading_error_degrees": 0.1,
                        "half_width_meters": 1.0,
                        "distance_margin_meters": 1.0,
                        "heading_limit_degrees": 30.0,
                        "heading_margin_degrees": 29.9,
                        "within_declared_support": True,
                    },
                    "renderer_profile": "right",
                    "frozen_scene_time_seconds": 0.5,
                    "camera_count": 2,
                    "all_camera_frames_finite": True,
                    "renderer_ms": 60.0,
                    "server_control_to_jpeg_ms": 66.0,
                    "frame_sha256": "a" * 64,
                    "boundary_hit": False,
                    "boundary_reason": None,
                }
            )
            recorder.record_browser_timing(
                sequence=0,
                browser_request_to_image_ms=70.0,
                browser_input_to_image_ms=73.0,
                client_unix_ms=1000.0,
            )
            event = recorder.record_route_event(
                "branch_selected", {"branch": "straight"}
            )
            recorder.record_route_browser_timing(
                event_index=event["event_index"],
                browser_selection_to_image_ms=80.0,
                client_unix_ms=1001.0,
            )

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["format"], recorder.FORMAT)
        self.assertEqual(report["summary"]["sample_count"], 1)
        self.assertEqual(report["summary"]["browser_timing_coverage"], 1.0)
        self.assertEqual(report["summary"]["support_violation_count"], 0)
        self.assertEqual(
            report["summary"]["browser_branch_selection_to_image_ms"]["p50"],
            80.0,
        )
        self.assertEqual(
            report["summary"]["claim_status"], "evidence_only_not_certified"
        )


if __name__ == "__main__":
    unittest.main()
