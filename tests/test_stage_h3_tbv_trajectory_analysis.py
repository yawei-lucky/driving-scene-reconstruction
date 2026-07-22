"""Lightweight checks for the TbV metadata trajectory inventory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_stage_h3_tbv_trajectories import (  # noqa: E402
    _bbox_distance,
    _downsample_pose_rows,
    parse_city_from_map_listing,
    parse_log_prefixes,
)


S3_HEADER = b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'


class TbvTrajectoryAnalysisTest(unittest.TestCase):
    def test_log_prefixes_and_continuation_are_parsed(self) -> None:
        payload = S3_HEADER + b"""
          <CommonPrefixes><Prefix>datasets/av2/tbv/log_b/</Prefix></CommonPrefixes>
          <CommonPrefixes><Prefix>datasets/av2/tbv/log_a/</Prefix></CommonPrefixes>
          <NextContinuationToken>next+page=</NextContinuationToken>
        </ListBucketResult>"""

        logs, continuation = parse_log_prefixes(payload)

        self.assertEqual(logs, ["log_b", "log_a"])
        self.assertEqual(continuation, "next+page=")

    def test_city_is_parsed_from_ground_raster_key(self) -> None:
        payload = S3_HEADER + b"""
          <Contents><Key>
            datasets/av2/tbv/log/map/log_ground_height_surface____WDC.npy
          </Key></Contents>
        </ListBucketResult>"""

        self.assertEqual(parse_city_from_map_listing(payload), "WDC")

    def test_pose_rows_are_reduced_by_time_and_distance(self) -> None:
        timestamps = [0, 50_000_000, 100_000_000, 200_000_000, 300_000_000]
        x_values = [0.0, 0.3, 0.6, 0.9, 1.2]
        y_values = [0.0] * 5

        points, times = _downsample_pose_rows(timestamps, x_values, y_values)

        self.assertEqual(points, [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0)])
        self.assertEqual(times, [0.0, 0.1, 0.3])

    def test_bbox_distance_is_zero_for_intersection(self) -> None:
        first = (0.0, 0.0, 10.0, 2.0)
        crossing = (5.0, -5.0, 7.0, 8.0)
        separate = (13.0, 6.0, 20.0, 9.0)

        self.assertEqual(_bbox_distance(first, crossing), 0.0)
        self.assertEqual(_bbox_distance(first, separate), 5.0)


if __name__ == "__main__":
    unittest.main()
