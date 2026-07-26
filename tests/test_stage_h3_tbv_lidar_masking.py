"""Dependency-light tests for projected TbV LiDAR traffic masks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "stage_h3_tbv_lidar_masking.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage_h3_tbv_lidar_masking", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVLidarMaskingTests(unittest.TestCase):
    def test_closest_timestamp_is_bounded_and_tie_breaks_earlier(self) -> None:
        timestamps = (100, 200, 300)

        self.assertEqual(
            MODULE.closest_timestamp_index(timestamps, 250, 50), 1
        )
        self.assertIsNone(
            MODULE.closest_timestamp_index(timestamps, 351, 50)
        )

    def test_closest_timestamp_requires_strict_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            MODULE.closest_timestamp_index((100, 100), 100, 10)

    @unittest.skipIf(np is None, "NumPy is available in the H3 environment")
    def test_excludes_only_valid_projection_on_zero_pixel(self) -> None:
        assert np is not None
        mask = np.full((4, 5), 255, dtype=np.uint8)
        mask[2, 1] = 0
        result = MODULE.projected_mask_exclusion(
            np.asarray(((1.2, 1.8), (1.2, 1.8), (9.0, 9.0))),
            np.asarray((True, False, True)),
            mask,
        )

        np.testing.assert_array_equal(result, (True, False, False))

    @unittest.skipIf(np is None, "NumPy is available in the H3 environment")
    def test_rejects_misaligned_projection_validity(self) -> None:
        assert np is not None
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.projected_mask_exclusion(
                np.zeros((2, 2)),
                np.ones(3, dtype=bool),
                np.ones((2, 2), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
