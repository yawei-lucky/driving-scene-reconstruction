"""Dependency-light tests for TbV parallel RGB path remapping."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE_PATH = (
    Path(__file__).parents[1] / "scripts" / "stage_h3_tbv_dataparser.py"
)


def load_path_functions():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"remap_rgb_path", "relative_rgb_path"}
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


class TbVRGBRemapTests(unittest.TestCase):
    def test_source_path_is_unchanged_without_override(self) -> None:
        remap = load_path_functions()["remap_rgb_path"]
        source = Path("/data/log/sensors/cameras/front/1.jpg")

        self.assertEqual(
            remap(source, data_root=Path("/data"), rgb_root=None),
            source,
        )

    def test_parallel_root_preserves_relative_path(self) -> None:
        remap = load_path_functions()["remap_rgb_path"]
        source = Path("/data/log/sensors/cameras/front/1.jpg")

        self.assertEqual(
            remap(
                source,
                data_root=Path("/data"),
                rgb_root=Path("/inpainted"),
            ),
            Path("/inpainted/log/sensors/cameras/front/1.jpg"),
        )

    def test_relative_path_uses_active_rgb_root(self) -> None:
        relative = load_path_functions()["relative_rgb_path"]
        self.assertEqual(
            relative(
                Path("/inpainted/log/sensors/cameras/front/1.jpg"),
                data_root=Path("/data"),
                rgb_root=Path("/inpainted"),
            ),
            Path("log/sensors/cameras/front/1.jpg"),
        )


if __name__ == "__main__":
    unittest.main()
