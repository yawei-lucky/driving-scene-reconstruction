"""Lightweight checks for the PandaSet multi-trajectory inventory."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_stage_h3_pandaset_trajectories import (  # noqa: E402
    Track,
    derive_headings_and_motion,
    fit_rigid_pose_to_gps,
    pair_metrics,
)


def make_track(scene: str, points: list[tuple[float, float]]) -> Track:
    headings, motion = derive_headings_and_motion(points)
    return Track(
        scene=scene,
        xy_metres=tuple(points),
        timestamps=tuple(float(index) for index in range(len(points))),
        headings_radians=headings,
        motion_per_sample_metres=motion,
        has_semseg=True,
        pose_alignment={},
    )


def metrics(first: Track, second: Track) -> dict[str, object]:
    return pair_metrics(
        first,
        second,
        overlap_distance_metres=5.0,
        direction_distance_metres=15.0,
        minimum_motion_per_sample_metres=0.2,
        minimum_overlap_samples=20,
    )


class PandaSetTrajectoryAnalysisTest(unittest.TestCase):
    def test_rigid_pose_to_gps_fit_recovers_rotation_and_translation(self) -> None:
        pose = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        gps = [(10.0, 5.0), (10.0, 6.0), (10.0, 7.0), (10.0, 8.0)]

        fit = fit_rigid_pose_to_gps(pose, gps)

        self.assertAlmostEqual(fit["rotation_degrees"], 90.0, places=9)
        self.assertAlmostEqual(fit["translation_x_metres"], 10.0, places=9)
        self.assertAlmostEqual(fit["translation_y_metres"], 5.0, places=9)
        self.assertAlmostEqual(fit["fitted_similarity_scale"], 1.0, places=9)
        self.assertLess(fit["residual_max_metres"], 1e-9)

    def test_parallel_offset_tracks_are_an_adjacent_repeat(self) -> None:
        first = make_track("001", [(float(x), 0.0) for x in range(30)])
        second = make_track("002", [(float(x), 3.0) for x in range(30)])

        result = metrics(first, second)

        self.assertTrue(result["classification"]["same_direction_repeat"])
        self.assertTrue(result["classification"]["adjacent_or_offset_repeat"])
        self.assertFalse(result["classification"]["direction_change_review_candidate"])
        self.assertAlmostEqual(result["overlap_nearest_distance_p50_metres"], 3.0)

    def test_opposite_tracks_are_multi_direction_not_same_direction(self) -> None:
        first = make_track("001", [(float(x), 0.0) for x in range(30)])
        second = make_track("002", [(float(x), 0.0) for x in reversed(range(30))])

        result = metrics(first, second)

        self.assertFalse(result["classification"]["same_direction_repeat"])
        self.assertTrue(result["classification"]["direction_change_review_candidate"])
        self.assertTrue(result["classification"]["opposite_direction_candidate"])
        self.assertTrue(
            math.isclose(result["heading_difference_p50_degrees"], 180.0)
        )

    def test_partial_overlap_identifies_route_extension(self) -> None:
        first = make_track("001", [(float(x), 0.0) for x in range(50)])
        second = make_track("002", [(float(x), 3.0) for x in range(20, 80)])

        result = metrics(first, second)

        self.assertTrue(result["classification"]["same_direction_repeat"])
        self.assertTrue(result["classification"]["adjacent_or_offset_repeat"])
        self.assertTrue(result["classification"]["extends_longest_input_by_10m"])
        self.assertGreaterEqual(min(result["overlap_samples"].values()), 30)

    def test_common_approach_then_turn_is_flagged_for_direction_review(self) -> None:
        first = make_track("001", [(float(x), 0.0) for x in range(50)])
        second_points = [(float(x), 0.0) for x in range(30)]
        second_points.extend((29.0, float(y)) for y in range(1, 21))
        second = make_track("002", second_points)

        result = metrics(first, second)

        self.assertTrue(result["classification"]["has_bidirectional_path_overlap"])
        self.assertTrue(result["classification"]["direction_change_review_candidate"])
        self.assertGreaterEqual(
            result["heading_difference_over_20_degrees_fraction"], 0.20
        )

    def test_endpoint_touch_is_not_a_repeated_path(self) -> None:
        first = make_track("001", [(float(x), 0.0) for x in range(30)])
        second = make_track("002", [(float(x), 0.0) for x in range(29, 59)])

        result = metrics(first, second)

        self.assertFalse(result["classification"]["has_bidirectional_path_overlap"])
        self.assertFalse(result["classification"]["same_direction_repeat"])


if __name__ == "__main__":
    unittest.main()
