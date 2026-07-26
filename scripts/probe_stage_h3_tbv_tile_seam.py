#!/usr/bin/env python3
"""Render identical observed world poses through two adjacent TbV tile models."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence


REFERENCE_LOG = "V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020"
CAMERA_NAME = "ring_front_center"
OVERLAP_START_SECONDS = 315970604.760172
OVERLAP_END_SECONDS = 315970606.9574283


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
        blend = position - low
        return ordered[low] * (1.0 - blend) + ordered[high] * blend

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _source_records(datamanager: Any) -> dict[int, tuple[Any, Path]]:
    records: dict[int, tuple[Any, Path]] = {}
    for dataset in (datamanager.train_dataset, datamanager.eval_dataset):
        for index, raw_filename in enumerate(dataset.image_filenames):
            path = Path(raw_filename)
            if (
                REFERENCE_LOG not in path.parts
                or path.parent.name != CAMERA_NAME
            ):
                continue
            timestamp_ns = int(path.stem)
            timestamp_seconds = timestamp_ns / 1e9
            if not (
                OVERLAP_START_SECONDS
                <= timestamp_seconds
                <= OVERLAP_END_SECONDS
            ):
                continue
            records[timestamp_ns] = (
                deepcopy(dataset.cameras[index : index + 1]),
                path,
            )
    return records


def _ground_truth(path: Path, camera: Any, image_module: Any) -> Any:
    image = image_module.open(path).convert("RGB")
    width = int(camera.width.item())
    height = int(camera.height.item())
    source_crop_height = round(height * image.width / width)
    image = image.crop(
        (0, 0, image.width, min(image.height, source_crop_height))
    )
    if image.size != (width, height):
        image = image.resize((width, height), image_module.Resampling.LANCZOS)
    return image


def render_tile(
    config_path: Path,
    output_dir: Path,
    *,
    label: str,
    target_timestamps_ns: tuple[int, ...] | None,
    sample_count: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from nerfstudio.scripts.render import streamline_ad_config
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image

    config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=streamline_ad_config,
    )
    del config
    pipeline.model.eval()
    datamanager = pipeline.datamanager
    records = _source_records(datamanager)
    available = tuple(sorted(records))
    if target_timestamps_ns is None:
        selected = tuple(
            available[index]
            for index in evenly_spaced_indices(len(available), sample_count)
        )
    else:
        missing = set(target_timestamps_ns) - set(available)
        if missing:
            raise RuntimeError(
                f"{label} lacks overlap frames {sorted(missing)}"
            )
        selected = target_timestamps_ns

    tile_dir = output_dir / label
    tile_dir.mkdir(parents=True, exist_ok=True)
    images: dict[int, np.ndarray] = {}
    gt_images: dict[int, np.ndarray] = {}
    render_seconds: list[float] = []
    psnr_values: list[float] = []
    near_black_fractions: list[float] = []

    for timestamp_ns in selected:
        source_camera, path = records[timestamp_ns]
        camera = source_camera.to(datamanager.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_seconds.append(time.perf_counter() - started)
        rgb = outputs["rgb"]
        if not bool(torch.isfinite(rgb).all()):
            raise RuntimeError(f"{label} produced non-finite RGB")
        rendered = (
            rgb.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        ground_truth = np.asarray(
            _ground_truth(path, camera, Image), dtype=np.uint8
        )
        if rendered.shape != ground_truth.shape:
            raise RuntimeError(
                f"{label} render/GT shape mismatch "
                f"{rendered.shape} != {ground_truth.shape}"
            )
        error = rendered.astype(np.float32) - ground_truth.astype(np.float32)
        mse = float(np.mean(error * error))
        psnr_values.append(
            float("inf") if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)
        )
        near_black_fractions.append(
            float(np.mean(np.max(rendered, axis=-1) < 8))
        )
        images[timestamp_ns] = rendered
        gt_images[timestamp_ns] = ground_truth
        Image.fromarray(rendered).save(
            tile_dir / f"{timestamp_ns}.jpg", quality=95
        )

    result = {
        "label": label,
        "config": str(config_path.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(checkpoint_step),
        "available_overlap_frame_count": len(available),
        "target_timestamps_ns": selected,
        "render_seconds": distribution(render_seconds),
        "psnr_db": distribution(psnr_values),
        "near_black_fraction": distribution(near_black_fractions),
        "images": images,
        "ground_truth": gt_images,
        "dataparser_transform": (
            datamanager.train_dataparser_outputs.dataparser_transform.tolist()
        ),
        "dataparser_scale": float(
            datamanager.train_dataparser_outputs.dataparser_scale
        ),
        "training_mask_count": sum(
            len(dataset._dataparser_outputs.mask_filenames or ())
            for dataset in (
                datamanager.train_dataset,
                datamanager.eval_dataset,
            )
        ),
    }
    del pipeline, datamanager, records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_contact_sheet(
    output_path: Path,
    tile_2: dict[str, Any],
    tile_3: dict[str, Any],
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    timestamps = tuple(tile_2["target_timestamps_ns"])
    cell_width, cell_height = 512, 350
    labels = ("source GT", "tile 2", "tile 3", "|tile 2 - tile 3|")
    sheet = Image.new(
        "RGB",
        (cell_width * len(labels), cell_height * len(timestamps)),
        "black",
    )
    for row, timestamp_ns in enumerate(timestamps):
        first = tile_2["images"][timestamp_ns]
        second = tile_3["images"][timestamp_ns]
        difference = np.abs(
            first.astype(np.int16) - second.astype(np.int16)
        ).astype(np.uint8)
        frames = (
            tile_2["ground_truth"][timestamp_ns],
            first,
            second,
            difference,
        )
        for column, (label, frame) in enumerate(zip(labels, frames)):
            image = Image.fromarray(frame).convert("RGB")
            image.thumbnail(
                (cell_width, cell_height), Image.Resampling.LANCZOS
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
            draw.text(
                (8, 9),
                f"{label} | {timestamp_ns}",
                fill=(255, 230, 80),
            )
            x = column * cell_width + (cell_width - image.width) // 2
            y = row * cell_height + (cell_height - image.height) // 2
            sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-2-config", type=Path, required=True)
    parser.add_argument("--tile-3-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--expected-checkpoint-step", type=int, default=99
    )
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_2 = render_tile(
        args.tile_2_config,
        output_dir,
        label="tile_2",
        target_timestamps_ns=None,
        sample_count=args.sample_count,
    )
    tile_3 = render_tile(
        args.tile_3_config,
        output_dir,
        label="tile_3",
        target_timestamps_ns=tuple(tile_2["target_timestamps_ns"]),
        sample_count=args.sample_count,
    )

    import numpy as np

    pairwise_mae = []
    pairwise_p95 = []
    for timestamp_ns in tile_2["target_timestamps_ns"]:
        difference = np.abs(
            tile_2["images"][timestamp_ns].astype(np.float32)
            - tile_3["images"][timestamp_ns].astype(np.float32)
        )
        pairwise_mae.append(float(np.mean(difference)))
        pairwise_p95.append(float(np.percentile(difference, 95)))

    contact_sheet = output_dir / "tile_2_3_overlap_contact_sheet.jpg"
    build_contact_sheet(contact_sheet, tile_2, tile_3)
    report = {
        "format": "driving_scene_reconstruction.tbv_tile_seam.v0",
        "scope": (
            f"{args.expected_checkpoint_step + 1}-step adjacent-tile "
            "observed-pose seam smoke; "
            "not a drivable-quality result"
        ),
        "reference_log": REFERENCE_LOG,
        "camera": CAMERA_NAME,
        "overlap_absolute_seconds": [
            OVERLAP_START_SECONDS,
            OVERLAP_END_SECONDS,
        ],
        "target_timestamps_ns": list(tile_2["target_timestamps_ns"]),
        "tile_2": {
            key: value
            for key, value in tile_2.items()
            if key not in ("images", "ground_truth")
        },
        "tile_3": {
            key: value
            for key, value in tile_3.items()
            if key not in ("images", "ground_truth")
        },
        "pairwise_rgb_absolute_difference": {
            "mean_per_frame": distribution(pairwise_mae),
            "p95_per_frame": distribution(pairwise_p95),
        },
        "expected_checkpoint_step": args.expected_checkpoint_step,
        "technical_gates": {
            "both_checkpoint_steps_match_expected": (
                tile_2["checkpoint_step"] == args.expected_checkpoint_step
                and tile_3["checkpoint_step"]
                == args.expected_checkpoint_step
            ),
            "five_or_requested_matched_world_poses": (
                len(tile_2["target_timestamps_ns"])
                == min(args.sample_count, tile_2["available_overlap_frame_count"])
            ),
            "both_models_use_metre_scale": (
                tile_2["dataparser_scale"] == 1.0
                and tile_3["dataparser_scale"] == 1.0
            ),
            "all_front_views_not_mostly_black": (
                tile_2["near_black_fraction"]["max"] < 0.25
                and tile_3["near_black_fraction"]["max"] < 0.25
            ),
        },
        "visual_seam_status": "requires_manual_review",
        "contact_sheet": str(contact_sheet),
        "limitations": [
            (
                f"{args.expected_checkpoint_step + 1} steps test a bounded "
                "adjacent-tile gate, not final quality."
            ),
            "Only the shared reference traversal front camera is rendered.",
            "No counterfactual lateral pose or checkpoint switching is tested.",
            (
                "Image-space training masks are present, but actors and "
                "LiDAR traffic points are not decomposed."
                if (
                    tile_2["training_mask_count"]
                    and tile_3["training_mask_count"]
                )
                else "Vehicles are not masked or decomposed."
            ),
        ],
    }
    report["technical_status"] = (
        "pass" if all(report["technical_gates"].values()) else "fail"
    )
    report_path = output_dir / "tbv_tile_seam.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["technical_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
