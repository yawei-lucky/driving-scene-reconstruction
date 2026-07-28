"""Tests for freezing checkpoint-compatible TbV manifest windows."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "materialize_stage_h3_tbv_manifest_subset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_stage_h3_tbv_manifest_subset", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ManifestSubsetTests(unittest.TestCase):
    def test_materialize_and_reuse_exact_hardlinked_view(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source_dir = root / "source"
            source_dir.mkdir()
            first = source_dir / "first.bin"
            second = source_dir / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            manifest_path = root / "selection_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_prefix": "dataset/",
                        "logs": [
                            {
                                "objects": [
                                    {
                                        "key": "dataset/first.bin",
                                        "path": str(first),
                                        "size": first.stat().st_size,
                                    }
                                ]
                            },
                            {
                                "objects": [
                                    {
                                        "key": "dataset/second.bin",
                                        "path": str(second),
                                        "size": second.stat().st_size,
                                    }
                                ]
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "frozen"

            created = MODULE.materialize(
                manifest_path,
                output_dir,
                (0, 1),
            )
            reused = MODULE.materialize(
                manifest_path,
                output_dir,
                (0, 1),
            )

            self.assertFalse(created["reused_existing"])
            self.assertTrue(reused["reused_existing"])
            self.assertEqual(created["linked_objects"], 2)
            self.assertEqual(
                os.stat(first).st_ino,
                os.stat(output_dir / "first.bin").st_ino,
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                MODULE.materialize(manifest_path, output_dir, (0,))


if __name__ == "__main__":
    unittest.main()
