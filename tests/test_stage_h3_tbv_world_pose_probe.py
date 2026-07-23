"""Dependency-light tests for the TbV shared-world probe geometry."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "examples" / "stage_h3_tbv_world_pose_probe.py"
)
SPEC = importlib.util.spec_from_file_location("stage_h3_tbv_world_pose_probe", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVWorldPoseProbeTests(unittest.TestCase):
    def test_camera_subset_is_non_empty_unique_and_ordered(self) -> None:
        requested = (
            "ring_front_left",
            "ring_front_center",
            "ring_front_right",
        )

        self.assertEqual(MODULE.validated_camera_names(requested), requested)
        for invalid in (
            (),
            ("ring_front_center", "ring_front_center"),
            ("unknown",),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MODULE.validated_camera_names(invalid)

    def test_left_offset_uses_pose_heading(self) -> None:
        forward = MODULE.offset_pose(
            MODULE.LocalWorldPose(10.0, 2.0, 0.5, 0.0), 1.0
        )
        turned = MODULE.offset_pose(
            MODULE.LocalWorldPose(10.0, 2.0, 0.5, math.pi / 2.0), 1.0
        )

        self.assertAlmostEqual(forward.x, 10.0)
        self.assertAlmostEqual(forward.y, 3.0)
        self.assertAlmostEqual(turned.x, 9.0)
        self.assertAlmostEqual(turned.y, 2.0)
        self.assertAlmostEqual(turned.z, 0.5)

    def test_branch_anchor_retreats_from_last_shared_sample(self) -> None:
        straight = tuple(
            MODULE.RouteSample(float(i), float(i), float(i), 0.0, 0.0, 0.0)
            for i in range(23)
        )
        shared = [
            MODULE.RouteSample(float(i), float(i), float(i), 0.0, 0.0, 0.0)
            for i in range(11)
        ]
        turn = [
            MODULE.RouteSample(
                10.0 + i,
                10.0 + 2.0 * i,
                10.0,
                -2.0 * i,
                0.0,
                -math.pi / 2.0,
            )
            for i in range(1, 7)
        ]

        match = MODULE.select_branch_anchor(tuple(shared + turn), straight)

        self.assertEqual(match.right_index, 8)
        self.assertEqual(match.straight_index, 8)
        self.assertAlmostEqual(match.distance, 0.0)
        self.assertEqual(match.shared_match_count, 11)

    def test_route_pose_interpolates_progress_and_wrapped_yaw(self) -> None:
        route = (
            MODULE.RouteSample(0.0, -1.0, 0.0, 0.0, 0.0, math.radians(170)),
            MODULE.RouteSample(1.0, 1.0, 2.0, 4.0, 1.0, math.radians(-170)),
        )

        pose = MODULE.pose_at_progress(route, 0.0)

        self.assertAlmostEqual(pose.x, 1.0)
        self.assertAlmostEqual(pose.y, 2.0)
        self.assertAlmostEqual(pose.z, 0.5)
        self.assertAlmostEqual(abs(math.degrees(pose.yaw)), 180.0)


if __name__ == "__main__":
    unittest.main()
