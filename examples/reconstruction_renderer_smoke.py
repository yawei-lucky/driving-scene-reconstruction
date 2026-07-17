#!/usr/bin/env python3
"""Render one nearby pose from the Stage H1 Nerfstudio checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    NerfstudioRenderer,
)

DEFAULT_CONFIG = Path(
    "/home/yawei/stage1_external/outputs/wayvescenes101_h1/"
    "scene_094_h1_big/splatfacto/run_v2/config.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/dsr_stage_h2"))
    parser.add_argument("--reference-frame", type=int, default=100)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--forward", type=float, default=0.0, help="Meters")
    parser.add_argument("--left", type=float, default=0.0, help="Meters")
    parser.add_argument("--yaw-degrees", type=float, default=0.0)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["front-forward"],
        help="Wayve camera directory names to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to write smoke-render images") from error

    renderer = NerfstudioRenderer(
        args.config,
        reference_frame_index=args.reference_frame,
        output_scale=args.output_scale,
    )
    rig = CameraRig(tuple(CameraSpec(name) for name in args.cameras))
    state = EgoState(
        x=args.forward,
        y=args.left,
        yaw=math.radians(args.yaw_degrees),
    )
    observation = renderer.render("scene_094", state, rig)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, frame in observation.frames.items():
        output_path = output_dir / f"{camera_name}.png"
        Image.fromarray(frame).save(output_path)
        print(f"frame={output_path}")
    print(f"metadata={dict(observation.metadata)}")


if __name__ == "__main__":
    main()
