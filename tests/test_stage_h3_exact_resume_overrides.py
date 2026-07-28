"""Source-level gates for bounded exact-resume data overrides."""

from __future__ import annotations

from pathlib import Path
import unittest


RESUME_SOURCE = (
    Path(__file__).parents[1] / "scripts" / "resume_stage_h3_exact.py"
).read_text(encoding="utf-8")
PROBE_SOURCE = (
    Path(__file__).parents[1]
    / "scripts"
    / "probe_stage_h3_tbv_tile_seam.py"
).read_text(encoding="utf-8")


class ExactResumeOverrideTests(unittest.TestCase):
    def test_resume_can_pin_separate_rgb_and_lidar_inputs(self) -> None:
        for argument in (
            "--rgb-root",
            "--image-mask-root",
            "--lidar-mask-root",
            "--lidar-persistence-root",
            "--downsample-factor",
            "--use-mask-aligned-model",
        ):
            self.assertIn(argument, RESUME_SOURCE)

    def test_quality_probe_forces_original_rgb(self) -> None:
        self.assertIn("dataparser.rgb_root = None", PROBE_SOURCE)
        self.assertIn("dataparser.image_mask_root = None", PROBE_SOURCE)
        self.assertIn("dataparser.load_image_masks = False", PROBE_SOURCE)


if __name__ == "__main__":
    unittest.main()
