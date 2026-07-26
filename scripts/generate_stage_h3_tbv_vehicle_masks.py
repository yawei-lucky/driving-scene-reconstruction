#!/usr/bin/env python3
"""Generate conservative COCO traffic masks for the bounded TbV tile data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Sequence


DYNAMIC_CATEGORY_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
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
    image_path: Path, data_root: Path, output_dir: Path
) -> Path:
    relative = image_path.resolve().relative_to(data_root.resolve())
    return output_dir.resolve() / relative.with_suffix(".png")


def selected_indices(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    dynamic_label_ids: set[int],
    score_threshold: float,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (label, score) in enumerate(zip(labels, scores))
        if label in dynamic_label_ids and score >= score_threshold
    )


def build_contact_sheet(
    data_root: Path,
    output_dir: Path,
    image_paths: Sequence[Path],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    front_paths = [
        path for path in image_paths if path.parent.name == "ring_front_center"
    ]
    if not front_paths:
        return
    sample_count = min(6, len(front_paths))
    sample_indices = tuple(
        round(index * (len(front_paths) - 1) / max(sample_count - 1, 1))
        for index in range(sample_count)
    )
    cell_width, cell_height = 520, 360
    sheet = Image.new(
        "RGB", (cell_width * 2, cell_height * sample_count), "black"
    )
    for row, source_index in enumerate(sample_indices):
        image_path = front_paths[source_index]
        mask_path = mask_path_for_image(image_path, data_root, output_dir)
        source = Image.open(image_path).convert("RGB")
        valid_mask = Image.open(mask_path).convert("L")
        excluded = valid_mask.point(lambda value: 0 if value else 180)
        overlay = Image.new("RGB", source.size, (255, 40, 40))
        reviewed = Image.composite(overlay, source, excluded)
        for column, (label, image) in enumerate(
            (("source", source), ("masked traffic in red", reviewed))
        ):
            panel = ImageOps.contain(
                image,
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 34), fill=(0, 0, 0))
            draw.text((8, 9), label, fill=(255, 230, 80))
            x = column * cell_width + (cell_width - panel.width) // 2
            y = row * cell_height + (cell_height - panel.height) // 2
            sheet.paste(panel, (x, y))
    sheet.save(output_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--dilation-pixels", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0.0 < args.score_threshold < 1.0:
        parser.error("--score-threshold must be within (0, 1)")
    if not 0.0 < args.mask_threshold < 1.0:
        parser.error("--mask-threshold must be within (0, 1)")
    if args.dilation_pixels < 0:
        parser.error("--dilation-pixels must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    manifest_path = output_dir / "vehicle_mask_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"completed manifest already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = tuple(
        sorted(
            path
            for path in data_root.rglob("*.jpg")
            if "sensors/cameras" in path.as_posix()
        )
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"no camera JPEGs under {data_root}")

    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    from PIL import Image
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    categories = tuple(weights.meta["categories"])
    dynamic_label_ids = {
        categories.index(name) for name in DYNAMIC_CATEGORY_NAMES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
    preprocess = weights.transforms()
    category_counts: Counter[str] = Counter()
    masked_fractions: list[float] = []
    reused_count = 0
    started = time.perf_counter()

    for batch_start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[
            batch_start : batch_start + args.batch_size
        ]
        pending_paths = [
            path
            for path in batch_paths
            if not mask_path_for_image(path, data_root, output_dir).is_file()
        ]
        reused_count += len(batch_paths) - len(pending_paths)
        if not pending_paths:
            continue
        tensors = [
            preprocess(Image.open(path).convert("RGB")).to(device)
            for path in pending_paths
        ]
        with torch.inference_mode():
            predictions = model(tensors)
        for image_path, source_tensor, prediction in zip(
            pending_paths, tensors, predictions
        ):
            labels = prediction["labels"].detach().cpu().tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            keep = selected_indices(
                labels,
                scores,
                dynamic_label_ids=dynamic_label_ids,
                score_threshold=args.score_threshold,
            )
            height, width = source_tensor.shape[-2:]
            excluded = torch.zeros(
                (1, 1, height, width), dtype=torch.float32, device=device
            )
            if keep:
                masks = prediction["masks"][list(keep)]
                excluded = masks.amax(dim=0, keepdim=True)
                if args.dilation_pixels:
                    kernel = args.dilation_pixels * 2 + 1
                    excluded = torch_functional.max_pool2d(
                        excluded, kernel_size=kernel, stride=1,
                        padding=args.dilation_pixels
                    )
                for index in keep:
                    category_counts[categories[labels[index]]] += 1
            excluded_array = (
                excluded[0, 0]
                .ge(args.mask_threshold)
                .detach()
                .cpu()
                .numpy()
            )
            valid_mask = np.where(excluded_array, 0, 255).astype(np.uint8)
            masked_fractions.append(float(np.mean(excluded_array)))
            destination = mask_path_for_image(
                image_path, data_root, output_dir
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(valid_mask, mode="L").save(
                destination, optimize=True
            )
        completed = min(batch_start + len(batch_paths), len(image_paths))
        if completed % 100 == 0 or completed == len(image_paths):
            print(f"masked {completed}/{len(image_paths)} images")

    if reused_count:
        # Reused masks do not have detection statistics in this invocation.
        masked_fractions = []
        for image_path in image_paths:
            mask = np.asarray(
                Image.open(
                    mask_path_for_image(image_path, data_root, output_dir)
                ).convert("L")
            )
            masked_fractions.append(float(np.mean(mask == 0)))
    contact_sheet = output_dir / "front_center_vehicle_mask_contact.jpg"
    build_contact_sheet(
        data_root, output_dir, image_paths, contact_sheet
    )
    weights_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    )
    report = {
        "format": "driving_scene_reconstruction.tbv_vehicle_masks.v0",
        "scope": (
            "COCO image-space traffic exclusion masks for a bounded "
            "static-background training comparison; not actor truth."
        ),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "image_count": len(image_paths),
        "reused_mask_count": reused_count,
        "model": "maskrcnn_resnet50_fpn_v2",
        "weights": "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "weights_url": weights.url,
        "weights_sha256": (
            hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if weights_path.is_file()
            else None
        ),
        "dynamic_categories": list(DYNAMIC_CATEGORY_NAMES),
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "dilation_pixels": args.dilation_pixels,
        "batch_size": args.batch_size,
        "device": str(device),
        "detected_instances_by_category": dict(
            sorted(category_counts.items())
        ),
        "masked_pixel_fraction": distribution(masked_fractions),
        "elapsed_seconds": time.perf_counter() - started,
        "contact_sheet": str(contact_sheet),
        "limitations": [
            "COCO detections can miss or misclassify traffic.",
            "Only RGB loss pixels are excluded; TbV LiDAR is not masked.",
            "Masks are not actor tracks and do not provide dynamic rendering.",
        ],
    }
    manifest_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
