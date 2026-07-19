#!/usr/bin/env python3
"""Build compact visual summaries from Stage H3 dataset renders."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "front_left_camera",
    "front_camera",
    "front_right_camera",
    "left_camera",
    "back_camera",
    "right_camera",
)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def tile(path: Path, size: tuple[int, int], label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(17)
    bounds = draw.textbbox((0, 0), label, font=label_font)
    draw.rectangle((6, 6, bounds[2] + 16, bounds[3] + 14), fill="black")
    draw.text((11, 9), label, fill=(255, 230, 70), font=label_font)
    return canvas


def comparison(
    smoke_root: Path,
    pilot_root: Path,
    frame: str,
    output_path: Path,
) -> None:
    size = (320, 180)
    sheet = Image.new("RGB", (size[0] * len(CAMERAS), size[1] * 3), "black")
    rows = (
        ("GT", pilot_root / "test" / "gt-rgb"),
        ("100 steps", smoke_root / "test" / "rgb"),
        ("2,000 steps", pilot_root / "test" / "rgb"),
    )
    for row, (row_label, root) in enumerate(rows):
        for column, camera in enumerate(CAMERAS):
            name = camera.removesuffix("_camera").replace("_", "-")
            cell = tile(root / camera / f"{frame}.jpg", size, f"{row_label}  {name}")
            sheet.paste(cell, (column * size[0], row * size[1]))
    sheet.save(output_path, quality=95)


def front_progression(
    pilot_root: Path,
    frames: tuple[str, ...],
    output_path: Path,
) -> None:
    size = (640, 360)
    rows = ("gt-rgb", "rgb", "depth")
    labels = {"gt-rgb": "GT", "rgb": "2,000 steps", "depth": "depth"}
    sheet = Image.new("RGB", (size[0] * len(frames), size[1] * len(rows)), "black")
    for row, output_name in enumerate(rows):
        for column, frame in enumerate(frames):
            path = pilot_root / "test" / output_name / "front_camera" / f"{frame}.jpg"
            cell = tile(path, size, f"{labels[output_name]}  frame {frame}")
            sheet.paste(cell, (column * size[0], row * size[1]))
    sheet.save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--frame", default="39")
    parser.add_argument("--progression-frames", default="09,39,69")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = tuple(item.strip() for item in args.progression_frames.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = (
        args.output_dir / f"scene_040_frame_{args.frame}_gt_100_2000.jpg"
    )
    progression_path = args.output_dir / "scene_040_front_2000_progression.jpg"
    comparison(args.smoke_root, args.pilot_root, args.frame, comparison_path)
    front_progression(args.pilot_root, frames, progression_path)
    print(f"PASS: {comparison_path}")
    print(f"PASS: {progression_path}")


if __name__ == "__main__":
    main()
