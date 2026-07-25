import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_stage_h3_tbv_window.py"
SPEC = importlib.util.spec_from_file_location("download_stage_h3_tbv_window", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TbVWindowDownloadTests(unittest.TestCase):
    def test_custom_window_is_parsed(self):
        window = MODULE.parse_window("log_id,1.25,2.5")

        self.assertEqual(window, MODULE.Window("log_id", 1.25, 2.5))

    def test_invalid_custom_window_is_rejected(self):
        with self.assertRaises(MODULE.argparse.ArgumentTypeError):
            MODULE.parse_window("log_id,2.5,1.25")
        with self.assertRaises(MODULE.argparse.ArgumentTypeError):
            MODULE.parse_window("log_id,nan,2.5")

    def test_manifest_totals_deduplicate_overlapping_windows(self):
        shared = MODULE.S3Object(
            "datasets/av2/tbv/log/calibration/shared.feather", 10, "a"
        )
        first_only = MODULE.S3Object(
            "datasets/av2/tbv/log/sensors/lidar/1.feather", 20, "b"
        )
        second_only = MODULE.S3Object(
            "datasets/av2/tbv/log/sensors/lidar/2.feather", 30, "c"
        )
        plans = {
            MODULE.Window("log", 1.0, 2.0): [shared, first_only],
            MODULE.Window("log", 1.5, 2.5): [shared, second_only],
        }

        manifest = MODULE.build_manifest(plans, Path("/tmp/data"), 2)

        self.assertEqual(manifest["window_object_references"], 4)
        self.assertEqual(manifest["total_objects"], 3)
        self.assertEqual(manifest["total_bytes"], 60)

    def test_parse_object_page_and_continuation(self):
        xml = b'''<?xml version="1.0" encoding="UTF-8"?>
        <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
          <Contents><Key>datasets/av2/tbv/log/sensors/lidar/100.feather</Key><ETag>"abc"</ETag><Size>42</Size></Contents>
          <NextContinuationToken>next page</NextContinuationToken>
        </ListBucketResult>'''
        objects, continuation = MODULE.parse_object_page(xml)
        self.assertEqual(
            objects,
            [
                MODULE.S3Object(
                    key="datasets/av2/tbv/log/sensors/lidar/100.feather",
                    size=42,
                    etag="abc",
                )
            ],
        )
        self.assertEqual(continuation, "next page")

    def test_window_selection_is_inclusive_sorted_and_strided(self):
        window = MODULE.Window("log", 1.0, 2.0)
        objects = [
            MODULE.S3Object(f"root/{stamp}.jpg", 1, "")
            for stamp in (
                2_100_000_000,
                1_500_000_000,
                1_000_000_000,
                2_000_000_000,
            )
        ]
        selected = MODULE.select_sensor_objects(objects, window, stride=2)
        self.assertEqual(
            [MODULE.timestamp_ns(obj) for obj in selected],
            [1_000_000_000, 2_000_000_000],
        )

    def test_local_path_preserves_official_log_layout(self):
        obj = MODULE.S3Object(
            "datasets/av2/tbv/log/calibration/intrinsics.feather", 1, ""
        )
        self.assertEqual(
            MODULE.local_path(Path("/tmp/out"), obj),
            Path("/tmp/out/log/calibration/intrinsics.feather"),
        )


if __name__ == "__main__":
    unittest.main()
