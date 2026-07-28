"""Dependency-light tests for the TbV region quality audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "audit_stage_h3_tbv_quality_regions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_stage_h3_tbv_quality_regions", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVQualityRegionTests(unittest.TestCase):
    def test_empty_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.distribution(())

    def test_region_audit_declares_proxy_limit(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("proxies, not hidden ground truth", source)


if __name__ == "__main__":
    unittest.main()
