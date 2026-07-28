from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import SceneTile, SceneTileRoute


def tile(index: int, start: float, end: float) -> SceneTile:
    return SceneTile(
        tile_id=f"tile_{index}",
        route_start_meters=start,
        route_end_meters=end,
        config_path=f"/configs/tile_{index}.yml",
        checkpoint_path=f"/checkpoints/tile_{index}.ckpt",
        checkpoint_step=7999,
        data_root=f"/data/tile_{index}",
        source_log="reference-log",
        source_time_window_seconds=(float(index), float(index + 1)),
        dataparser_transform=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        support_half_width_meters=1.0,
        renderer_profile="tbv_reference",
        evidence_path=f"/evidence/tile_{index}.json",
    )


class SceneTileRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = SceneTileRoute(
            (
                tile(2, 160.0, 260.0),
                tile(3, 240.0, 340.0),
                tile(4, 320.0, 420.0),
            )
        )

    def test_three_tiles_form_a_260_meter_route(self) -> None:
        self.assertEqual(self.route.route_start_meters, 160.0)
        self.assertEqual(self.route.route_end_meters, 420.0)
        self.assertEqual(self.route.route_length_meters, 260.0)
        manifest = self.route.manifest()
        self.assertEqual(
            [
                (overlap["start_meters"], overlap["end_meters"])
                for overlap in manifest["overlaps"]
            ],
            [(240.0, 260.0), (320.0, 340.0)],
        )

    def test_overlap_midpoints_blend_adjacent_tiles_equally(self) -> None:
        first = self.route.weights_at(250.0)
        second = self.route.weights_at(330.0)
        self.assertEqual(
            [(item.tile_id, item.weight) for item in first],
            [("tile_2", 0.5), ("tile_3", 0.5)],
        )
        self.assertEqual(
            [(item.tile_id, item.weight) for item in second],
            [("tile_3", 0.5), ("tile_4", 0.5)],
        )

    def test_non_overlap_progress_uses_one_tile(self) -> None:
        self.assertEqual(self.route.weights_at(200.0)[0].tile_id, "tile_2")
        self.assertEqual(self.route.weights_at(300.0)[0].tile_id, "tile_3")
        self.assertEqual(self.route.weights_at(400.0)[0].tile_id, "tile_4")

    def test_gap_and_triple_overlap_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive overlap"):
            SceneTileRoute((tile(0, 0.0, 100.0), tile(1, 101.0, 201.0)))
        with self.assertRaisesRegex(ValueError, "triple-overlap"):
            SceneTileRoute(
                (
                    tile(0, 0.0, 100.0),
                    tile(1, 60.0, 160.0),
                    tile(2, 90.0, 190.0),
                )
            )
        with self.assertRaisesRegex(ValueError, "triple-overlap"):
            SceneTileRoute(
                (
                    tile(0, 0.0, 100.0),
                    tile(1, 60.0, 160.0),
                    tile(2, 100.0, 200.0),
                )
            )

    def test_progress_outside_route_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "route progress"):
            self.route.weights_at(159.9)


if __name__ == "__main__":
    unittest.main()
