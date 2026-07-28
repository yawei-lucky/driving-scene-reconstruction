"""Dependency-light tests for the bounded TbV RGB inpainting tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate_stage_h3_tbv_inpainted_rgb.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_stage_h3_tbv_inpainted_rgb", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVInpaintedRGBTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self) -> None:
        self.assertEqual(
            MODULE.evenly_spaced_indices(11, 5),
            (0, 2, 5, 8, 10),
        )

    def test_parallel_paths_preserve_relative_layout(self) -> None:
        data = Path("/data")
        image = data / "log" / "sensors" / "cameras" / "front" / "1.jpg"

        self.assertEqual(
            MODULE.mask_path_for_image(
                image,
                data_root=data,
                mask_root=Path("/masks"),
            ),
            Path("/masks/log/sensors/cameras/front/1.png"),
        )
        self.assertEqual(
            MODULE.output_path_for_image(
                image,
                data_root=data,
                output_root=Path("/rgb"),
            ),
            Path("/rgb/log/sensors/cameras/front/1.jpg"),
        )


if __name__ == "__main__":
    unittest.main()
