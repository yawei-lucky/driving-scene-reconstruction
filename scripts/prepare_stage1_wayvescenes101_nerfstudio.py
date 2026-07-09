#!/usr/bin/env python3
"""Prepare one WayveScenes101 scene for Nerfstudio training.

This wraps the upstream WayveScenes101 COLMAP-to-Nerfstudio adapter and adds a
small compatibility fix for current Nerfstudio versions: when all frames share
the same camera model, also write that camera model at the transforms.json root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wayve_scenes.utils import colmap_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Unzipped WayveScenes101 scene directory, e.g. .../wayve_scenes_101/scene_094",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not include Wayve mask paths in transforms.json.",
    )
    return parser.parse_args()


def add_top_level_camera_model(transforms_path: Path) -> str | None:
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_models = sorted({frame.get("camera_model") for frame in data.get("frames", [])})

    if len(camera_models) == 1 and camera_models[0]:
        data["camera_model"] = camera_models[0]
        transforms_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        return camera_models[0]

    return None


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    recon_dir = scene_dir / "colmap_sparse" / "rig"

    if not recon_dir.is_dir():
        raise FileNotFoundError(f"Missing COLMAP rig directory: {recon_dir}")

    num_frames = colmap_utils.colmap_to_json(
        recon_dir=recon_dir,
        output_dir=scene_dir,
        use_masks=not args.no_masks,
    )

    transforms_path = scene_dir / "transforms.json"
    camera_model = add_top_level_camera_model(transforms_path)

    print(f"scene_dir={scene_dir}")
    print(f"frames={num_frames}")
    print(f"transforms={transforms_path}")
    print(f"top_level_camera_model={camera_model or 'not_set'}")


if __name__ == "__main__":
    main()
