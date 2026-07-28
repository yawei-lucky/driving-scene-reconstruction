"""Source-level gate for separate TbV RGB and LiDAR mask roots."""

from __future__ import annotations

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).parents[1] / "scripts" / "stage_h3_tbv_dataparser.py"
).read_text(encoding="utf-8")


class TbVImageMaskRootTests(unittest.TestCase):
    def test_image_mask_root_overrides_lidar_mask_root_for_rgb(self) -> None:
        self.assertIn("image_mask_root: Path | None = None", SOURCE)
        self.assertIn(
            "self.config.image_mask_root"
            "\n            if self.config.image_mask_root is not None"
            "\n            else self.config.mask_root",
            SOURCE,
        )
        self.assertIn('outputs.metadata["lidar_mask_root"]', SOURCE)


if __name__ == "__main__":
    unittest.main()
