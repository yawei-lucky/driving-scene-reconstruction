"""Dependency-light tests for the TbV traffic-mask generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate_stage_h3_tbv_vehicle_masks.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_stage_h3_tbv_vehicle_masks", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVVehicleMaskTests(unittest.TestCase):
    def test_mask_path_preserves_dataset_layout(self) -> None:
        data = Path("/data")
        image = data / "log" / "sensors/cameras/front/123.jpg"

        self.assertEqual(
            MODULE.mask_path_for_image(image, data, Path("/masks")),
            Path("/masks/log/sensors/cameras/front/123.png"),
        )

    def test_selection_requires_dynamic_label_and_score(self) -> None:
        self.assertEqual(
            MODULE.selected_indices(
                (1, 3, 8, 15),
                (0.9, 0.4, 0.8, 0.99),
                dynamic_label_ids={1, 3, 8},
                score_threshold=0.6,
            ),
            (0, 2),
        )

    def test_distribution_interpolates_percentiles(self) -> None:
        result = MODULE.distribution((0.0, 1.0, 2.0))

        self.assertEqual(result["p50"], 1.0)
        self.assertEqual(result["p95"], 1.9)


if __name__ == "__main__":
    unittest.main()
