#!/usr/bin/env python3
"""Render a continuous TbV adjacent-tile seam and +/-1 m pose evidence."""

from __future__ import annotations

import argparse
import bisect
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Sequence


REFERENCE_LOG = "V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020"
CAMERA_NAME = "ring_front_center"
OVERLAP_START_SECONDS = 315970604.760172
OVERLAP_END_SECONDS = 315970606.9574283
LATERAL_OFFSETS_METERS = (-1.0, 0.0, 1.0)
LATERAL_PROGRESS_FRACTIONS = (0.0, 0.5, 1.0)


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


def interpolation_bracket(
    timestamps_ns: Sequence[int], target_ns: float
) -> tuple[int, int, float]:
    if len(timestamps_ns) < 2:
        raise ValueError("at least two timestamps are required")
    # IEEE-754 cannot exactly represent these 3e17 ns AV2 timestamps. Permit
    # only the sub-microsecond endpoint rounding introduced by interpolation.
    endpoint_tolerance_ns = 128.0
    if (
        target_ns < timestamps_ns[0] - endpoint_tolerance_ns
        or target_ns > timestamps_ns[-1] + endpoint_tolerance_ns
    ):
        raise ValueError("target lies outside the timestamp interval")
    target_ns = min(max(target_ns, timestamps_ns[0]), timestamps_ns[-1])
    upper = min(
        max(1, bisect.bisect_right(timestamps_ns, target_ns)),
        len(timestamps_ns) - 1,
    )
    lower = upper - 1
    span = timestamps_ns[upper] - timestamps_ns[lower]
    fraction = 0.0 if span == 0 else (
        target_ns - timestamps_ns[lower]
    ) / span
    return lower, upper, float(fraction)


def smoothstep_blend(
    route_fraction: float, start_fraction: float = 0.35,
    end_fraction: float = 0.65
) -> float:
    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError("route fraction must be within [0, 1]")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("blend interval must be ordered within [0, 1]")
    normalized = min(
        max((route_fraction - start_fraction) / (
            end_fraction - start_fraction
        ), 0.0),
        1.0,
    )
    return normalized * normalized * (3.0 - 2.0 * normalized)


def frame_count_for_speed(
    path_length_meters: float, speed_mps: float, fps: int
) -> int:
    if not math.isfinite(path_length_meters) or path_length_meters <= 0.0:
        raise ValueError("path length must be finite and positive")
    if not math.isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError("speed must be finite and positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(round(path_length_meters / speed_mps * fps) + 1, 2)


def streamline_tbv_probe_config(
    config: Any,
    *,
    data_root: Path | None = None,
) -> Any:
    """Avoid repeating training-only point filtering during RGB evaluation."""

    from nerfstudio.scripts.render import streamline_ad_config

    config = streamline_ad_config(config)
    dataparser = config.pipeline.datamanager.dataparser
    if data_root is not None:
        dataparser.data = data_root.expanduser().resolve()
    if getattr(dataparser, "mask_lidar_points", False):
        dataparser._checkpoint_trained_with_lidar_mask_filter = True
        dataparser.mask_lidar_points = False
    return config


def _source_records(datamanager: Any) -> dict[int, Any]:
    records: dict[int, Any] = {}
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
            records[timestamp_ns] = deepcopy(
                dataset.cameras[index : index + 1]
            )
    return records


def _interpolated_camera(
    records: dict[int, Any],
    timestamps_ns: tuple[int, ...],
    target_ns: float,
    *,
    lateral_left_meters: float,
    np: Any,
    torch: Any,
) -> Any:
    lower, upper, fraction = interpolation_bracket(
        timestamps_ns, target_ns
    )
    left = records[timestamps_ns[lower]]
    right = records[timestamps_ns[upper]]
    camera = deepcopy(left)
    left_pose = left.camera_to_worlds[0].detach().cpu().numpy()
    right_pose = right.camera_to_worlds[0].detach().cpu().numpy()
    translation = (
        left_pose[:3, 3] * (1.0 - fraction)
        + right_pose[:3, 3] * fraction
    )
    blended_rotation = (
        left_pose[:3, :3] * (1.0 - fraction)
        + right_pose[:3, :3] * fraction
    )
    u, _, vh = np.linalg.svd(blended_rotation)
    rotation = u @ vh
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    pose = np.concatenate((rotation, translation[:, None]), axis=1)
    # Nerfstudio camera x points right, so positive route-left is -x.
    pose[:3, 3] -= lateral_left_meters * pose[:3, 0]
    camera.camera_to_worlds = torch.from_numpy(pose)[None].to(
        dtype=left.camera_to_worlds.dtype
    )
    if left.times is not None and right.times is not None:
        camera.times = (
            left.times * (1.0 - fraction) + right.times * fraction
        )
    return camera


def _route_length(records: dict[int, Any], timestamps_ns: tuple[int, ...]) -> float:
    positions = [
        records[timestamp].camera_to_worlds[0, :3, 3].detach().cpu()
        for timestamp in timestamps_ns
    ]
    return sum(
        float((right - left).norm())
        for left, right in zip(positions, positions[1:])
    )


def _render_rgb(pipeline: Any, camera: Any, torch: Any) -> tuple[Any, float]:
    camera = camera.to(pipeline.datamanager.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = pipeline.model.get_outputs_for_camera(camera)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rgb = outputs["rgb"]
    if not bool(torch.isfinite(rgb).all()):
        raise RuntimeError("model produced non-finite RGB")
    image = (
        rgb.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return image, elapsed


def render_tile(
    config_path: Path,
    output_dir: Path,
    *,
    data_root: Path | None,
    label: str,
    target_source_timestamps_ns: tuple[int, ...] | None,
    query_timestamps_ns: tuple[float, ...] | None,
    speed_mps: float,
    fps: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image

    def update_config(config: Any) -> Any:
        return streamline_tbv_probe_config(config, data_root=data_root)

    _, pipeline, checkpoint_path, checkpoint_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    pipeline.model.eval()
    records = _source_records(pipeline.datamanager)
    available = tuple(sorted(records))
    if target_source_timestamps_ns is None:
        source_timestamps = available
    else:
        missing = set(target_source_timestamps_ns) - set(available)
        if missing:
            raise RuntimeError(f"{label} lacks source frames {sorted(missing)}")
        source_timestamps = target_source_timestamps_ns
    if len(source_timestamps) < 2:
        raise RuntimeError(f"{label} needs at least two overlap source frames")

    path_length_meters = _route_length(records, source_timestamps)
    if query_timestamps_ns is None:
        frame_count = frame_count_for_speed(path_length_meters, speed_mps, fps)
        query_timestamps = tuple(
            float(value)
            for value in np.linspace(
                source_timestamps[0],
                source_timestamps[-1],
                frame_count,
            )
        )
    else:
        query_timestamps = query_timestamps_ns

    tile_dir = output_dir / label
    frames_dir = tile_dir / "centerline"
    lateral_dir = tile_dir / "lateral"
    frames_dir.mkdir(parents=True, exist_ok=True)
    lateral_dir.mkdir(parents=True, exist_ok=True)
    render_seconds: list[float] = []
    near_black_fractions: list[float] = []
    frame_paths: list[Path] = []
    frame_hashes: list[str] = []

    import hashlib

    for index, target_ns in enumerate(query_timestamps):
        camera = _interpolated_camera(
            records,
            source_timestamps,
            target_ns,
            lateral_left_meters=0.0,
            np=np,
            torch=torch,
        )
        image, elapsed = _render_rgb(pipeline, camera, torch)
        near_black_fractions.append(
            float(np.mean(np.max(image, axis=-1) < 8))
        )
        render_seconds.append(elapsed)
        frame_path = frames_dir / f"frame_{index:06d}.jpg"
        Image.fromarray(image).save(frame_path, quality=95)
        frame_paths.append(frame_path)
        frame_hashes.append(
            hashlib.sha256(frame_path.read_bytes()).hexdigest()
        )

    lateral_paths: dict[str, str] = {}
    lateral_near_black: list[float] = []
    for progress_index, route_fraction in enumerate(
        LATERAL_PROGRESS_FRACTIONS
    ):
        target_ns = (
            query_timestamps[0] * (1.0 - route_fraction)
            + query_timestamps[-1] * route_fraction
        )
        for lateral_meters in LATERAL_OFFSETS_METERS:
            camera = _interpolated_camera(
                records,
                source_timestamps,
                target_ns,
                lateral_left_meters=lateral_meters,
                np=np,
                torch=torch,
            )
            image, elapsed = _render_rgb(pipeline, camera, torch)
            render_seconds.append(elapsed)
            lateral_near_black.append(
                float(np.mean(np.max(image, axis=-1) < 8))
            )
            name = (
                f"progress_{progress_index}_"
                f"left_{lateral_meters:+.1f}m.jpg"
            )
            path = lateral_dir / name
            Image.fromarray(image).save(path, quality=95)
            lateral_paths[
                f"{route_fraction:.1f},{lateral_meters:+.1f}"
            ] = str(path)

    result = {
        "label": label,
        "config": str(config_path.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(checkpoint_step),
        "source_timestamps_ns": source_timestamps,
        "query_timestamps_ns": query_timestamps,
        "source_frame_count": len(source_timestamps),
        "continuous_frame_count": len(query_timestamps),
        "path_length_meters": path_length_meters,
        "render_seconds": distribution(render_seconds),
        "centerline_near_black_fraction": distribution(
            near_black_fractions
        ),
        "lateral_near_black_fraction": distribution(lateral_near_black),
        "frame_paths": tuple(str(path) for path in frame_paths),
        "frame_hashes": frame_hashes,
        "lateral_paths": lateral_paths,
        "dataparser_transform": (
            pipeline.datamanager.train_dataparser_outputs
            .dataparser_transform.tolist()
        ),
        "dataparser_scale": float(
            pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        ),
        "training_mask_count": sum(
            len(dataset._dataparser_outputs.mask_filenames or ())
            for dataset in (
                pipeline.datamanager.train_dataset,
                pipeline.datamanager.eval_dataset,
            )
        ),
        "lidar_mask_filters": {
            split: dataset._dataparser_outputs.metadata.get(
                "lidar_mask_filter", {"enabled": False}
            )
            for split, dataset in (
                ("train", pipeline.datamanager.train_dataset),
                ("eval", pipeline.datamanager.eval_dataset),
            )
        },
    }
    del pipeline, records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _annotated_panel(
    image: Any,
    label: str,
    detail: str,
    *,
    image_module: Any,
    draw_module: Any,
) -> Any:
    result = image.copy().convert("RGB")
    draw = draw_module.Draw(result)
    draw.rectangle((0, 0, result.width, 48), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 230, 80))
    draw.text((8, 27), detail, fill=(235, 240, 245))
    return result


def _encode_video(
    frames_dir: Path, video_path: Path, fps: int
) -> dict[str, Any]:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )["streams"][0]
    return {
        "path": str(video_path),
        "bytes": video_path.stat().st_size,
        "width": int(probe["width"]),
        "height": int(probe["height"]),
        "decoded_frame_count": int(probe["nb_read_frames"]),
    }


def build_videos_and_metrics(
    output_dir: Path,
    tile_2: dict[str, Any],
    tile_3: dict[str, Any],
    *,
    speed_mps: float,
    fps: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    hard_dir = output_dir / "hard_switch_frames"
    blend_dir = output_dir / "blend_frames"
    hard_dir.mkdir()
    blend_dir.mkdir()
    pairwise_mae: list[float] = []
    pairwise_p95: list[float] = []
    hard_outputs: list[Any] = []
    blend_outputs: list[Any] = []
    frame_count = tile_2["continuous_frame_count"]
    path_length = tile_2["path_length_meters"]

    for index, (first_path, second_path) in enumerate(
        zip(tile_2["frame_paths"], tile_3["frame_paths"])
    ):
        first = np.asarray(Image.open(first_path).convert("RGB"))
        second = np.asarray(Image.open(second_path).convert("RGB"))
        difference = np.abs(
            first.astype(np.float32) - second.astype(np.float32)
        )
        pairwise_mae.append(float(np.mean(difference)))
        pairwise_p95.append(float(np.percentile(difference, 95)))
        fraction = index / (frame_count - 1)
        hard_weight = 0.0 if fraction < 0.5 else 1.0
        blend_weight = smoothstep_blend(fraction)
        hard = (
            first if hard_weight == 0.0 else second
        ).astype(np.uint8)
        blended = np.clip(
            first.astype(np.float32) * (1.0 - blend_weight)
            + second.astype(np.float32) * blend_weight,
            0.0,
            255.0,
        ).astype(np.uint8)
        hard_outputs.append(hard)
        blend_outputs.append(blended)
        progress = fraction * path_length
        detail = (
            f"{speed_mps:.1f} m/s | {progress:.1f}/{path_length:.1f} m"
        )
        first_panel = _annotated_panel(
            Image.fromarray(first),
            "tile 2",
            detail,
            image_module=Image,
            draw_module=ImageDraw,
        )
        second_panel = _annotated_panel(
            Image.fromarray(second),
            "tile 3",
            detail,
            image_module=Image,
            draw_module=ImageDraw,
        )
        for directory, output, label, weight in (
            (
                hard_dir,
                hard,
                "hard output",
                hard_weight,
            ),
            (
                blend_dir,
                blended,
                "short smooth blend output",
                blend_weight,
            ),
        ):
            output_panel = _annotated_panel(
                Image.fromarray(output),
                label,
                f"tile 3 weight {weight:.2f}",
                image_module=Image,
                draw_module=ImageDraw,
            )
            panel_width = first_panel.width * 3
            panel_height = first_panel.height
            canvas = Image.new(
                "RGB",
                (
                    panel_width + panel_width % 2,
                    panel_height + panel_height % 2,
                ),
                "black",
            )
            canvas.paste(first_panel, (0, 0))
            canvas.paste(second_panel, (first_panel.width, 0))
            canvas.paste(output_panel, (first_panel.width * 2, 0))
            canvas.save(directory / f"frame_{index:06d}.jpg", quality=94)

    def temporal_differences(frames: list[Any]) -> list[float]:
        return [
            float(
                np.mean(
                    np.abs(
                        right.astype(np.float32)
                        - left.astype(np.float32)
                    )
                )
            )
            for left, right in zip(frames, frames[1:])
        ]

    hard_temporal = temporal_differences(hard_outputs)
    blend_temporal = temporal_differences(blend_outputs)
    switch_frame_index = math.ceil((frame_count - 1) * 0.5)
    switch_delta_index = max(switch_frame_index - 1, 0)
    hard_video = _encode_video(
        hard_dir, output_dir / "hard_switch.mp4", fps
    )
    blend_video = _encode_video(
        blend_dir, output_dir / "smooth_blend.mp4", fps
    )
    return {
        "pairwise_rgb_mae": distribution(pairwise_mae),
        "pairwise_rgb_p95": distribution(pairwise_p95),
        "hard_temporal_rgb_mae": distribution(hard_temporal),
        "blend_temporal_rgb_mae": distribution(blend_temporal),
        "hard_switch_delta_rgb_mae": hard_temporal[switch_delta_index],
        "blend_same_delta_rgb_mae": blend_temporal[switch_delta_index],
        "hard_switch_frame_index": switch_frame_index,
        "blend_fraction_interval": [0.35, 0.65],
        "blend_length_meters": path_length * 0.30,
        "hard_video": hard_video,
        "blend_video": blend_video,
    }


def build_lateral_contact_sheet(
    output_path: Path,
    tile_2: dict[str, Any],
    tile_3: dict[str, Any],
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    cell_width, half_height = 420, 290
    sheet = Image.new(
        "RGB",
        (
            cell_width * len(LATERAL_OFFSETS_METERS),
            half_height * 2 * len(LATERAL_PROGRESS_FRACTIONS),
        ),
        "black",
    )
    for row, route_fraction in enumerate(LATERAL_PROGRESS_FRACTIONS):
        for column, lateral_meters in enumerate(LATERAL_OFFSETS_METERS):
            key = f"{route_fraction:.1f},{lateral_meters:+.1f}"
            for model_row, (label, paths) in enumerate(
                (
                    ("tile 2", tile_2["lateral_paths"]),
                    ("tile 3", tile_3["lateral_paths"]),
                )
            ):
                image = Image.open(paths[key]).convert("RGB")
                image = ImageOps.contain(
                    image,
                    (cell_width, half_height),
                    Image.Resampling.LANCZOS,
                )
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0))
                draw.text(
                    (8, 7),
                    (
                        f"{label} | progress {route_fraction:.1f} | "
                        f"left {lateral_meters:+.1f} m"
                    ),
                    fill=(255, 230, 80),
                )
                x = column * cell_width + (cell_width - image.width) // 2
                y = (
                    (row * 2 + model_row) * half_height
                    + (half_height - image.height) // 2
                )
                sheet.paste(image, (x, y))
    sheet.save(output_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-2-config", type=Path, required=True)
    parser.add_argument("--tile-3-config", type=Path, required=True)
    parser.add_argument("--tile-2-data-root", type=Path)
    parser.add_argument("--tile-3-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed-mps", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--expected-checkpoint-step", type=int, default=1999)
    args = parser.parse_args()
    if not math.isfinite(args.speed_mps) or args.speed_mps <= 0.0:
        parser.error("--speed-mps must be finite and positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    for path in (args.tile_2_config, args.tile_3_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_2 = render_tile(
        args.tile_2_config,
        output_dir,
        data_root=args.tile_2_data_root,
        label="tile_2",
        target_source_timestamps_ns=None,
        query_timestamps_ns=None,
        speed_mps=args.speed_mps,
        fps=args.fps,
    )
    tile_3 = render_tile(
        args.tile_3_config,
        output_dir,
        data_root=args.tile_3_data_root,
        label="tile_3",
        target_source_timestamps_ns=tuple(tile_2["source_timestamps_ns"]),
        query_timestamps_ns=tuple(tile_2["query_timestamps_ns"]),
        speed_mps=args.speed_mps,
        fps=args.fps,
    )
    video_metrics = build_videos_and_metrics(
        output_dir,
        tile_2,
        tile_3,
        speed_mps=args.speed_mps,
        fps=args.fps,
    )
    lateral_contact = output_dir / "lateral_minus1_0_plus1_contact.jpg"
    build_lateral_contact_sheet(lateral_contact, tile_2, tile_3)

    path_length_difference = abs(
        tile_2["path_length_meters"] - tile_3["path_length_meters"]
    )
    report = {
        "format": "driving_scene_reconstruction.tbv_tile_continuous.v0",
        "scope": (
            "Two independently loaded checkpoints rendered at corresponding "
            "interpolated city-frame poses; offline image switch/blend only, "
            "not checkpoint streaming or driving acceptance."
        ),
        "reference_log": REFERENCE_LOG,
        "camera": CAMERA_NAME,
        "speed_mps": args.speed_mps,
        "fps": args.fps,
        "lateral_convention": "positive is route-left",
        "lateral_offsets_meters": list(LATERAL_OFFSETS_METERS),
        "lateral_progress_fractions": list(
            LATERAL_PROGRESS_FRACTIONS
        ),
        "tile_2": {
            key: value
            for key, value in tile_2.items()
            if key not in ("frame_paths", "lateral_paths")
        },
        "tile_3": {
            key: value
            for key, value in tile_3.items()
            if key not in ("frame_paths", "lateral_paths")
        },
        "path_length_difference_meters": path_length_difference,
        "comparison": video_metrics,
        "lateral_contact_sheet": str(lateral_contact),
        "technical_gates": {
            "both_checkpoint_steps_match_expected": (
                tile_2["checkpoint_step"] == args.expected_checkpoint_step
                and tile_3["checkpoint_step"]
                == args.expected_checkpoint_step
            ),
            "at_least_five_shared_source_poses": (
                tile_2["source_frame_count"] >= 5
            ),
            "both_models_use_metre_scale": (
                tile_2["dataparser_scale"] == 1.0
                and tile_3["dataparser_scale"] == 1.0
            ),
            "continuous_path_is_at_least_15m": (
                min(
                    tile_2["path_length_meters"],
                    tile_3["path_length_meters"],
                )
                >= 15.0
            ),
            "model_path_lengths_agree_within_5cm": (
                path_length_difference <= 0.05
            ),
            "all_center_and_lateral_views_not_mostly_black": (
                tile_2["centerline_near_black_fraction"]["max"] < 0.25
                and tile_3["centerline_near_black_fraction"]["max"] < 0.25
                and tile_2["lateral_near_black_fraction"]["max"] < 0.25
                and tile_3["lateral_near_black_fraction"]["max"] < 0.25
            ),
            "both_videos_decode_every_frame": (
                video_metrics["hard_video"]["decoded_frame_count"]
                == tile_2["continuous_frame_count"]
                and video_metrics["blend_video"]["decoded_frame_count"]
                == tile_2["continuous_frame_count"]
            ),
        },
        "visual_status": "requires_manual_review",
        "limitations": [
            "The scene time follows the reference traversal only.",
            "The two checkpoints are loaded sequentially, not streamed live.",
            "Pixel blending can hide a cut but can also create double images.",
            (
                "Image-space masks and synchronized projected LiDAR return "
                "filtering are present, but these remain detector exclusions "
                "rather than actor tracks."
                if all(
                    tile["lidar_mask_filters"]["train"].get(
                        "checkpoint_trained_with_filter", False
                    )
                    for tile in (tile_2, tile_3)
                )
                else
                "Image-space training masks are present, but actors and "
                "LiDAR traffic points are not decomposed."
                if (
                    tile_2["training_mask_count"]
                    and tile_3["training_mask_count"]
                )
                else (
                    "TbV has no actor masks; traffic ghosts remain a "
                    "rejection risk."
                )
            ),
            "The +/-1 m poses have no counterfactual ground-truth images.",
        ],
    }
    report["technical_status"] = (
        "pass" if all(report["technical_gates"].values()) else "fail"
    )
    report_path = output_dir / "tbv_tile_continuous.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if report["technical_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
