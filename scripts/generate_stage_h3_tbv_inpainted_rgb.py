#!/usr/bin/env python3
"""Fill detected traffic pixels to provide bounded background supervision."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Sequence


def evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    count = min(length, count)
    if count == 1:
        return (length // 2,)
    return tuple(
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    )


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution needs at least one value")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def mask_path_for_image(
    image_path: Path,
    *,
    data_root: Path,
    mask_root: Path,
) -> Path:
    relative = image_path.absolute().relative_to(data_root.absolute())
    return mask_root.absolute() / relative.with_suffix(".png")


def output_path_for_image(
    image_path: Path,
    *,
    data_root: Path,
    output_root: Path,
) -> Path:
    relative = image_path.absolute().relative_to(data_root.absolute())
    return output_root.absolute() / relative


def _write_inpainted(
    image_path: Path,
    *,
    data_root: Path,
    mask_root: Path,
    output_root: Path,
    method: str,
    radius: float,
    jpeg_quality: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    mask_path = mask_path_for_image(
        image_path,
        data_root=data_root,
        mask_root=mask_root,
    )
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or valid_mask is None:
        raise RuntimeError(f"failed to decode {image_path} or {mask_path}")
    if image.shape[:2] != valid_mask.shape:
        raise RuntimeError(
            f"image/mask shape mismatch for {image_path}: "
            f"{image.shape[:2]} != {valid_mask.shape}"
        )
    excluded = np.where(valid_mask < 128, 255, 0).astype(np.uint8)
    flag = (
        cv2.INPAINT_TELEA
        if method == "telea"
        else cv2.INPAINT_NS
    )
    started = time.perf_counter()
    result = (
        cv2.inpaint(image, excluded, radius, flag)
        if bool(excluded.any())
        else image
    )
    elapsed = time.perf_counter() - started
    destination = output_path_for_image(
        image_path,
        data_root=data_root,
        output_root=output_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.jpg")
    if not cv2.imwrite(
        str(temporary),
        result,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    ):
        raise RuntimeError(f"failed to write {temporary}")
    os.replace(temporary, destination)
    return {
        "source": str(image_path),
        "mask": str(mask_path),
        "output": str(destination),
        "excluded_fraction": float(np.mean(excluded > 0)),
        "inpaint_seconds": elapsed,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def build_contact_sheet(
    output_path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    selected = [
        record
        for record in records
        if Path(record["source"]).parent.name == "ring_front_center"
    ]
    if not selected:
        selected = list(records)
    selected = [
        selected[index]
        for index in evenly_spaced_indices(len(selected), min(6, len(selected)))
    ]
    cell_width, cell_height = 500, 340
    sheet = Image.new(
        "RGB",
        (cell_width * 3, cell_height * len(selected)),
        "black",
    )
    for row, record in enumerate(selected):
        source = Image.open(record["source"]).convert("RGB")
        valid_mask = Image.open(record["mask"]).convert("L")
        excluded = valid_mask.point(lambda value: 0 if value >= 128 else 180)
        overlay = Image.new("RGB", source.size, (255, 35, 35))
        masked = Image.composite(overlay, source, excluded)
        inpainted = Image.open(record["output"]).convert("RGB")
        for column, (label, image) in enumerate(
            (
                ("source", source),
                ("traffic mask", masked),
                ("inpainted background", inpainted),
            )
        ):
            panel = ImageOps.contain(
                image,
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 38), fill=(0, 0, 0))
            draw.text(
                (8, 7),
                (
                    f"{label} | {Path(record['source']).stem} | "
                    f"{record['excluded_fraction']:.2%}"
                ),
                fill=(255, 230, 80),
            )
            x = column * cell_width + (cell_width - panel.width) // 2
            y = row * cell_height + (cell_height - panel.height) // 2
            sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("telea", "navier-stokes"), default="telea"
    )
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--camera")
    parser.add_argument(
        "--limit",
        type=int,
        help="evenly sample this many matched images for a bounded preview",
    )
    args = parser.parse_args()
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        parser.error("--radius must be positive and finite")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be within [1, 100]")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    data_root = args.data_root.expanduser().absolute()
    mask_root = args.mask_root.expanduser().absolute()
    output_root = args.output_root.expanduser().absolute()
    if not data_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("data root and mask root must exist")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_root}")

    images = sorted(data_root.glob("*/sensors/cameras/*/*.jpg"))
    if args.camera is not None:
        images = [
            path for path in images if path.parent.name == args.camera
        ]
    missing_masks = [
        mask_path_for_image(
            image,
            data_root=data_root,
            mask_root=mask_root,
        )
        for image in images
        if not mask_path_for_image(
            image,
            data_root=data_root,
            mask_root=mask_root,
        ).is_file()
    ]
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} masks are missing; first: "
            f"{missing_masks[0]}"
        )
    if args.limit is not None:
        images = [
            images[index]
            for index in evenly_spaced_indices(len(images), args.limit)
        ]
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _write_inpainted,
                image,
                data_root=data_root,
                mask_root=mask_root,
                output_root=output_root,
                method=args.method,
                radius=args.radius,
                jpeg_quality=args.jpeg_quality,
            ): image
            for image in images
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record["source"])
    contact_sheet = output_root / "inpaint_contact.jpg"
    build_contact_sheet(contact_sheet, records)
    report = {
        "format": "driving_scene_reconstruction.tbv_inpainted_rgb.v0",
        "scope": (
            "Image-space traffic-hole fill for a reconstruction pilot; "
            "not recovered ground truth behind occluders."
        ),
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "output_root": str(output_root),
        "method": args.method,
        "radius": args.radius,
        "jpeg_quality": args.jpeg_quality,
        "camera_filter": args.camera,
        "image_count": len(records),
        "excluded_fraction": distribution(
            [record["excluded_fraction"] for record in records]
        ),
        "inpaint_seconds": distribution(
            [record["inpaint_seconds"] for record in records]
        ),
        "total_bytes": sum(record["bytes"] for record in records),
        "contact_sheet": str(contact_sheet),
        "records": records,
    }
    manifest = output_root / "inpaint_manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "records"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
