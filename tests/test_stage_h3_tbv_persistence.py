"""Dependency-light checks for cross-visit TbV persistence masks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "stage_h3_tbv_persistence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage_h3_tbv_persistence", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVPersistenceTests(unittest.TestCase):
    def test_voxel_encoding_is_unique_and_sortable(self) -> None:
        values = [
            MODULE.encode_voxel_triplet(-1, 0, 0),
            MODULE.encode_voxel_triplet(0, 0, 0),
            MODULE.encode_voxel_triplet(0, 0, 1),
            MODULE.encode_voxel_triplet(0, 1, 0),
            MODULE.encode_voxel_triplet(1, 0, 0),
        ]

        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(values))

    def test_neighbor_offsets_have_expected_extent(self) -> None:
        self.assertEqual(MODULE.neighbor_key_offsets(0), (0,))
        offsets = MODULE.neighbor_key_offsets(1)
        self.assertEqual(len(offsets), 27)
        self.assertIn(1, offsets)
        self.assertIn(-(1 << MODULE.VOXEL_X_SHIFT), offsets)

    def test_persistence_path_preserves_dataset_layout(self) -> None:
        result = MODULE.persistence_mask_path(
            data_root=Path("/data"),
            persistence_root=Path("/masks"),
            lidar_path=Path("/data/log/sensors/lidar/123.feather"),
        )

        self.assertEqual(
            result,
            Path("/masks/log/sensors/lidar/123.npz"),
        )

    def test_out_of_range_voxel_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "21-bit"):
            MODULE.encode_voxel_triplet(
                MODULE.VOXEL_AXIS_BIAS,
                0,
                0,
            )


if __name__ == "__main__":
    unittest.main()
