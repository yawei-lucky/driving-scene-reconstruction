#!/usr/bin/env python3
"""Render and audit a complete H3 logged camera trajectory.

The evaluator merges the fixed train and held-out camera splits back into the
original 80-frame, six-camera sequence. It renders the exact logged cameras,
computes native image metrics, writes comparison videos, and measures temporal
residual after compensating ego/image motion with optical flow estimated only
from the ground-truth images.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image, ImageDraw, ImageFont
from pandaset import DataSet
from nerfstudio.scripts.render import streamline_ad_config
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.poses import (
    interpolate_trajectories,
    multiply as pose_multiply,
)

from driving_scene_reconstruction.h3.vehicle_object_layer import (
    PandaSetVehicleObjectParser,
)


CAMERA_NAMES = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "back",
    4: "left",
    5: "right",
}
CAMERA_ORDER = (1, 0, 2, 4, 3, 5)
FLOW_WIDTH = 480
COMPARISON_CELL = (640, 360)
RIG_CELL = (640, 360)
VEHICLE_CROP_SIZE = 128


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": float(np.mean(values)),
    }


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )


def gt_rgb(
    model: Any,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> torch.Tensor:
    image = model.composite_with_background(
        model.get_gt_img(batch["image"]),
        outputs["background"],
    )
    predicted = outputs["rgb"]
    return image[: predicted.shape[0], : predicted.shape[1], :]


def letterbox(
    image: np.ndarray,
    size: tuple[int, int],
    label: str,
) -> Image.Image:
    source = Image.fromarray(image).convert("RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(
        source,
        ((size[0] - source.width) // 2, (size[1] - source.height) // 2),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(22)
    bounds = draw.textbbox((0, 0), label, font=label_font)
    draw.rectangle((8, 8, bounds[2] + 20, bounds[3] + 18), fill="black")
    draw.text((14, 11), label, fill=(255, 230, 60), font=label_font)
    return canvas


def comparison_frame(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    camera_name: str,
    frame: str,
    split: str,
    metrics: dict[str, float],
    run_label: str,
) -> Image.Image:
    gt_tile = letterbox(ground_truth, COMPARISON_CELL, "GT")
    pred_tile = letterbox(prediction, COMPARISON_CELL, run_label)
    canvas = Image.new(
        "RGB",
        (COMPARISON_CELL[0] * 2, COMPARISON_CELL[1]),
        "black",
    )
    canvas.paste(gt_tile, (0, 0))
    canvas.paste(pred_tile, (COMPARISON_CELL[0], 0))
    draw = ImageDraw.Draw(canvas)
    label = (
        f"{camera_name}  frame {frame}  {split}  "
        f"PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.3f}  "
        f"LPIPS {metrics['lpips']:.3f}"
    )
    label_font = font(20)
    bounds = draw.textbbox((0, 0), label, font=label_font)
    left = max(0, (canvas.width - bounds[2]) // 2)
    draw.rectangle(
        (left - 8, canvas.height - 33, left + bounds[2] + 8, canvas.height),
        fill="black",
    )
    draw.text((left, canvas.height - 29), label, fill="white", font=label_font)
    return canvas


def resize_for_flow(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scaled_height = round(height * FLOW_WIDTH / width)
    return cv2.resize(
        image,
        (FLOW_WIDTH, scaled_height),
        interpolation=cv2.INTER_AREA,
    )


def sample_flow(
    flow: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    channels = [
        cv2.remap(
            flow[..., channel],
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for channel in range(2)
    ]
    return np.stack(channels, axis=-1)


def temporal_flow_metric(
    previous_gt: np.ndarray,
    current_gt: np.ndarray,
    previous_prediction: np.ndarray,
    current_prediction: np.ndarray,
) -> dict[str, float | int]:
    """Compute excess temporal residual after GT-derived motion compensation.

    Backward flow is defined at the current frame, so remapping the previous
    prediction with ``grid + backward`` aligns it to the current frame.
    Forward/backward consistency removes occlusions and unreliable flow.
    """

    previous_gt = resize_for_flow(previous_gt)
    current_gt = resize_for_flow(current_gt)
    previous_prediction = resize_for_flow(previous_prediction)
    current_prediction = resize_for_flow(current_prediction)

    previous_gray = cv2.cvtColor(previous_gt, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_gt, cv2.COLOR_RGB2GRAY)
    forward = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        4,
        21,
        4,
        7,
        1.5,
        0,
    )
    backward = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        4,
        21,
        4,
        7,
        1.5,
        0,
    )

    height, width = previous_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_x = grid_x + backward[..., 0]
    source_y = grid_y + backward[..., 1]
    in_bounds = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    sampled_forward = sample_flow(forward, source_x, source_y)
    consistency = np.linalg.norm(backward + sampled_forward, axis=-1)
    valid = in_bounds & (consistency <= 1.5)

    def warp(image: np.ndarray) -> np.ndarray:
        return cv2.remap(
            image,
            source_x,
            source_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    warped_gt = warp(previous_gt).astype(np.float32) / 255.0
    warped_prediction = warp(previous_prediction).astype(np.float32) / 255.0
    current_gt_float = current_gt.astype(np.float32) / 255.0
    current_prediction_float = current_prediction.astype(np.float32) / 255.0
    gt_residual = np.mean(np.abs(warped_gt - current_gt_float), axis=-1)
    prediction_residual = np.mean(
        np.abs(warped_prediction - current_prediction_float),
        axis=-1,
    )
    if not np.any(valid):
        raise RuntimeError("optical-flow consistency mask is empty")
    excess = np.maximum(prediction_residual - gt_residual, 0.0)
    return {
        "excess_warp_mae": float(excess[valid].mean()),
        "prediction_warp_mae": float(prediction_residual[valid].mean()),
        "gt_warp_mae": float(gt_residual[valid].mean()),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "flow_consistency_p95_pixels": float(
            np.percentile(consistency[valid], 95)
        ),
    }


def load_camera_records(
    datamanager: Any,
) -> dict[str, dict[int, dict[str, Any]]]:
    # Accessing these caches applies the exact upstream undistortion/crop and
    # updates each dataset camera's intrinsics before cameras are copied.
    split_sources = (
        ("train", datamanager.train_dataset, datamanager.cached_train),
        ("heldout", datamanager.eval_dataset, datamanager.cached_eval),
    )
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for split, dataset, cache in split_sources:
        for index, filename in enumerate(dataset.image_filenames):
            frame = Path(filename).stem.zfill(2)
            camera = deepcopy(dataset.cameras[index : index + 1]).to(
                datamanager.device
            )
            sensor_index = int(camera.metadata["sensor_idxs"].item())
            if sensor_index in records.setdefault(frame, {}):
                raise RuntimeError(f"duplicate camera record: {frame}/{sensor_index}")
            batch = cache[index].copy()
            batch["image"] = batch["image"].to(datamanager.device)
            records[frame][sensor_index] = {
                "camera": camera,
                "batch": batch,
                "split": split,
                "filename": str(filename),
            }

    if len(records) != 80:
        raise RuntimeError(f"expected 80 frames, found {len(records)}")
    for frame, cameras in records.items():
        if set(cameras) != set(CAMERA_NAMES):
            raise RuntimeError(f"frame {frame} does not contain all six cameras")
    return records


def load_vehicle_trajectories(
    config: Any,
    pipeline: Any,
) -> list[dict[str, Any]]:
    """Load all rigid vehicle cuboids without rebuilding the datamanager.

    The checkpoint pipeline supplies the exact time offset and world-to-model
    transform. Actor annotations are read independently so the dynamic-only
    baseline and stationary+moving vehicle pilot use identical crop regions.
    """

    dataparser_config = deepcopy(config.pipeline.datamanager.dataparser)
    dataparser_config._target = PandaSetVehicleObjectParser
    dataparser_config.include_stationary_rigid_actors = True
    dataparser_config.include_deformable_actors = False
    parser = dataparser_config.setup()
    dataset = DataSet(str(dataparser_config.data.absolute()))
    parser.sequence = dataset[dataparser_config.sequence]
    parser.sequence.load()
    trajectories = parser._get_actor_trajectories()

    outputs = pipeline.datamanager.train_dataparser_outputs
    time_offset = float(outputs.time_offset)
    world_to_model = outputs.dataparser_transform
    for trajectory in trajectories:
        trajectory["timestamps"] = (
            trajectory["timestamps"] - time_offset
        ).float()
        trajectory["poses"][:, :3] = pose_multiply(
            world_to_model, trajectory["poses"][:, :3]
        )
    return trajectories


def trajectory_pose_at_time(
    trajectory: dict[str, Any],
    time_seconds: float,
    extrapolation_seconds: float,
) -> torch.Tensor | None:
    timestamps = trajectory["timestamps"]
    if (
        time_seconds < float(timestamps[0]) - extrapolation_seconds
        or time_seconds > float(timestamps[-1]) + extrapolation_seconds
    ):
        return None
    query = torch.tensor([time_seconds], dtype=timestamps.dtype)
    poses, _, _ = interpolate_trajectories(
        trajectory["poses"].unsqueeze(1),
        timestamps,
        query,
        clamp_frac=False,
    )
    return poses[0]


def cuboid_corners(dims: torch.Tensor) -> torch.Tensor:
    signs = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=dims.dtype,
    )
    return signs * dims.reshape(1, 3) / 2


def project_vehicle_crops(
    trajectories: list[dict[str, Any]],
    camera: Any,
    extrapolation_seconds: float,
    minimum_side_pixels: int,
) -> list[dict[str, Any]]:
    """Project actor cuboids to conservative 2D crop rectangles."""

    camera_to_world = camera.camera_to_worlds[0].detach().cpu()
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    fx = float(camera.fx.item())
    fy = float(camera.fy.item())
    cx = float(camera.cx.item())
    cy = float(camera.cy.item())
    width = int(camera.width.item())
    height = int(camera.height.item())
    time_seconds = float(camera.times.item())

    crops = []
    for actor_id, trajectory in enumerate(trajectories):
        pose = trajectory_pose_at_time(
            trajectory,
            time_seconds,
            extrapolation_seconds,
        )
        if pose is None:
            continue
        local_corners = cuboid_corners(trajectory["dims"])
        world_corners = (
            local_corners @ pose[:3, :3].T + pose[:3, 3]
        )
        camera_corners = (world_corners - translation) @ rotation
        in_front = camera_corners[:, 2] < -0.1
        if int(in_front.sum()) < 4:
            continue
        visible = camera_corners[in_front]
        depth = -visible[:, 2]
        u = fx * visible[:, 0] / depth + cx
        v = cy - fy * visible[:, 1] / depth
        left = float(u.min())
        right = float(u.max())
        top = float(v.min())
        bottom = float(v.max())
        padding_x = 0.08 * (right - left)
        padding_y = 0.08 * (bottom - top)
        left = max(0, int(np.floor(left - padding_x)))
        right = min(width, int(np.ceil(right + padding_x)))
        top = max(0, int(np.floor(top - padding_y)))
        bottom = min(height, int(np.ceil(bottom + padding_y)))
        if (
            right - left < minimum_side_pixels
            or bottom - top < minimum_side_pixels
        ):
            continue
        crops.append(
            {
                "actor_id": actor_id,
                "label": str(trajectory["label"]),
                "stationary": bool(trajectory["stationary"]),
                "bbox_xyxy": [left, top, right, bottom],
                "area_pixels": (right - left) * (bottom - top),
            }
        )
    return crops


def crop_metrics(
    model: Any,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    bbox_xyxy: list[int],
) -> dict[str, float]:
    left, top, right, bottom = bbox_xyxy
    gt_crop = ground_truth[top:bottom, left:right]
    prediction_crop = prediction[top:bottom, left:right]
    mse = torch.mean((gt_crop - prediction_crop) ** 2).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    gt_batch = torch_functional.interpolate(
        gt_crop.permute(2, 0, 1).unsqueeze(0),
        size=(VEHICLE_CROP_SIZE, VEHICLE_CROP_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    prediction_batch = torch_functional.interpolate(
        prediction_crop.permute(2, 0, 1).unsqueeze(0),
        size=(VEHICLE_CROP_SIZE, VEHICLE_CROP_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    lpips = model.lpips(prediction_batch, gt_batch)
    return {"vehicle_crop_psnr": float(psnr), "vehicle_crop_lpips": float(lpips)}


def vehicle_crop_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    crop: dict[str, Any],
    metrics: dict[str, float],
    camera_name: str,
    frame: str,
) -> Image.Image:
    left, top, right, bottom = crop["bbox_xyxy"]
    gt = letterbox(
        ground_truth[top:bottom, left:right],
        (256, 256),
        "GT",
    )
    rendered = letterbox(
        prediction[top:bottom, left:right],
        (256, 256),
        "prediction",
    )
    canvas = Image.new("RGB", (512, 292), "black")
    canvas.paste(gt, (0, 0))
    canvas.paste(rendered, (256, 0))
    description = (
        f"{camera_name} f{frame} actor {crop['actor_id']} "
        f"{'stationary' if crop['stationary'] else 'moving'} "
        f"PSNR {metrics['vehicle_crop_psnr']:.2f} "
        f"LPIPS {metrics['vehicle_crop_lpips']:.3f}"
    )
    ImageDraw.Draw(canvas).text(
        (8, 263),
        description,
        fill="white",
        font=font(16),
    )
    return canvas


def encode_video(frame_pattern: Path, output_path: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--vehicle-crop-metrics", action="store_true")
    parser.add_argument("--vehicle-min-side-pixels", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "comparison_frames"
    rig_dir = args.output_dir / "rig_frames"

    load_start = time.perf_counter()
    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        test_mode="test",
        update_config_callback=streamline_ad_config,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    model = pipeline.model
    run_label = f"SplatAD step {step:,}"
    run_tag = f"step_{step:09d}"
    records = load_camera_records(pipeline.datamanager)
    vehicle_trajectories = (
        load_vehicle_trajectories(config, pipeline)
        if args.vehicle_crop_metrics
        else []
    )

    ordered_frames = sorted(records, key=int)
    # Warm all camera heads before measuring the sequence.
    with torch.no_grad():
        for sensor_index in CAMERA_ORDER:
            model.get_outputs_for_camera(
                records[ordered_frames[0]][sensor_index]["camera"]
            )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_frame: list[dict[str, Any]] = []
    flow_pairs: list[dict[str, Any]] = []
    vehicle_crops: list[dict[str, Any]] = []
    finite_failures = 0
    previous: dict[int, dict[str, Any]] = {}
    for frame in ordered_frames:
        rig_canvas = Image.new(
            "RGB",
            (RIG_CELL[0] * 3, RIG_CELL[1] * 2),
            "black",
        )
        for rig_position, sensor_index in enumerate(CAMERA_ORDER):
            record = records[frame][sensor_index]
            camera = record["camera"]
            batch = record["batch"]
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0

            prediction_tensor = outputs["rgb"]
            ground_truth_tensor = gt_rgb(model, outputs, batch)
            finite = bool(
                torch.isfinite(prediction_tensor).all()
                and torch.isfinite(outputs["depth"]).all()
                and torch.isfinite(outputs["accumulation"]).all()
            )
            finite_failures += int(not finite)
            with torch.no_grad():
                metrics, _ = model.get_image_metrics_and_images(outputs, batch)
            image_metrics = {
                name: float(metrics[name]) for name in ("psnr", "ssim", "lpips")
            }
            prediction = rgb_uint8(prediction_tensor)
            ground_truth = rgb_uint8(ground_truth_tensor)

            if args.vehicle_crop_metrics and record["split"] == "heldout":
                projected_crops = project_vehicle_crops(
                    vehicle_trajectories,
                    camera,
                    config.pipeline.datamanager.dataparser.trajectory_extrapolation_length,
                    args.vehicle_min_side_pixels,
                )
                for crop in projected_crops:
                    metrics = crop_metrics(
                        model,
                        ground_truth_tensor,
                        prediction_tensor,
                        crop["bbox_xyxy"],
                    )
                    crop_record = {
                        "frame": frame,
                        "camera": CAMERA_NAMES[sensor_index],
                        "sensor_index": sensor_index,
                        "split": record["split"],
                        **crop,
                        **metrics,
                    }
                    vehicle_crops.append(crop_record)
                    crop_image = vehicle_crop_comparison(
                        ground_truth,
                        prediction,
                        crop,
                        metrics,
                        CAMERA_NAMES[sensor_index],
                        frame,
                    )
                    crop_path = (
                        args.output_dir
                        / "vehicle_crops"
                        / CAMERA_NAMES[sensor_index]
                        / (
                            f"{int(frame):03d}_actor_"
                            f"{crop['actor_id']:03d}.jpg"
                        )
                    )
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop_image.save(crop_path, quality=92)
                    crop_record["comparison_image"] = str(crop_path)

            comparison = comparison_frame(
                ground_truth,
                prediction,
                CAMERA_NAMES[sensor_index],
                frame,
                record["split"],
                image_metrics,
                run_label,
            )
            comparison_path = (
                frames_dir / CAMERA_NAMES[sensor_index] / f"{int(frame):03d}.jpg"
            )
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.save(comparison_path, quality=92)

            rig_tile = letterbox(
                prediction,
                RIG_CELL,
                f"{CAMERA_NAMES[sensor_index]}  frame {frame}",
            )
            rig_canvas.paste(
                rig_tile,
                (
                    (rig_position % 3) * RIG_CELL[0],
                    (rig_position // 3) * RIG_CELL[1],
                ),
            )

            current = {
                "ground_truth": ground_truth,
                "prediction": prediction,
                "frame": frame,
            }
            if sensor_index in previous:
                temporal = temporal_flow_metric(
                    previous[sensor_index]["ground_truth"],
                    ground_truth,
                    previous[sensor_index]["prediction"],
                    prediction,
                )
                flow_pairs.append(
                    {
                        "camera": CAMERA_NAMES[sensor_index],
                        "sensor_index": sensor_index,
                        "from_frame": previous[sensor_index]["frame"],
                        "to_frame": frame,
                        **temporal,
                    }
                )
            previous[sensor_index] = current
            per_frame.append(
                {
                    "frame": frame,
                    "camera": CAMERA_NAMES[sensor_index],
                    "sensor_index": sensor_index,
                    "split": record["split"],
                    "time_seconds": float(camera.times.item()),
                    "finite": finite,
                    "latency_ms": latency_ms,
                    **image_metrics,
                }
            )
        rig_dir.mkdir(parents=True, exist_ok=True)
        rig_canvas.save(rig_dir / f"{int(frame):03d}.jpg", quality=92)

    video_paths: dict[str, str] = {}
    for sensor_index in CAMERA_ORDER:
        camera_name = CAMERA_NAMES[sensor_index]
        output_path = (
            args.output_dir
            / f"scene_040_{camera_name}_gt_vs_{run_tag}_{args.fps}fps.mp4"
        )
        encode_video(
            frames_dir / camera_name / "%03d.jpg",
            output_path,
            args.fps,
        )
        video_paths[camera_name] = str(output_path)
    rig_video = (
        args.output_dir
        / f"scene_040_six_camera_{run_tag}_{args.fps}fps.mp4"
    )
    encode_video(rig_dir / "%03d.jpg", rig_video, args.fps)

    by_camera: dict[str, Any] = {}
    image_gate_pass = True
    for sensor_index in CAMERA_ORDER:
        camera_name = CAMERA_NAMES[sensor_index]
        camera_frames = [
            item for item in per_frame if item["sensor_index"] == sensor_index
        ]
        camera_pairs = [
            item for item in flow_pairs if item["sensor_index"] == sensor_index
        ]
        excess = [item["excess_warp_mae"] for item in camera_pairs]
        valid_fractions = [item["valid_fraction"] for item in camera_pairs]
        psnr_drops = []
        for previous_frame, current_frame in zip(
            camera_frames,
            camera_frames[1:],
        ):
            drop = previous_frame["psnr"] - current_frame["psnr"]
            if drop > 3.0:
                psnr_drops.append(
                    {
                        "from_frame": previous_frame["frame"],
                        "to_frame": current_frame["frame"],
                        "drop_db": drop,
                        "from_psnr": previous_frame["psnr"],
                        "to_psnr": current_frame["psnr"],
                    }
                )
        temporal_pass = (
            percentile(excess, 95) <= 0.03 and max(excess) <= 0.05
        )
        flow_coverage_pass = min(valid_fractions) >= 0.50
        image_gate_pass = (
            image_gate_pass and temporal_pass and flow_coverage_pass
        )
        split_metrics = {}
        for split in ("train", "heldout"):
            split_frames = [
                item for item in camera_frames if item["split"] == split
            ]
            split_metrics[split] = {
                "frames": len(split_frames),
                "psnr": distribution(
                    [item["psnr"] for item in split_frames]
                ),
                "ssim": distribution(
                    [item["ssim"] for item in split_frames]
                ),
                "lpips": distribution(
                    [item["lpips"] for item in split_frames]
                ),
            }
        by_camera[camera_name] = {
            "frames": len(camera_frames),
            "heldout_frames": sum(
                item["split"] == "heldout" for item in camera_frames
            ),
            "metric_scope_note": (
                "The all-frame distributions mix 72 optimized train views "
                "with 8 held-out views and are temporal diagnostics, not "
                "held-out generalization metrics. Use by_split for that "
                "distinction."
            ),
            "psnr": distribution([item["psnr"] for item in camera_frames]),
            "ssim": distribution([item["ssim"] for item in camera_frames]),
            "lpips": distribution([item["lpips"] for item in camera_frames]),
            "by_split": split_metrics,
            "latency_ms": distribution(
                [item["latency_ms"] for item in camera_frames]
            ),
            "excess_warp_mae": distribution(excess),
            "flow_valid_fraction": distribution(valid_fractions),
            "temporal_thresholds": {
                "p95_at_most": 0.03,
                "every_pair_at_most": 0.05,
                "every_pair_valid_fraction_at_least": 0.50,
            },
            "automated_temporal_pass": temporal_pass,
            "flow_coverage_pass": flow_coverage_pass,
            "psnr_drops_over_3db": psnr_drops,
        }

    vehicle_crop_summary: dict[str, Any] | None = None
    if args.vehicle_crop_metrics:
        by_motion = {}
        for motion, stationary in (("stationary", True), ("moving", False)):
            selected = [
                item
                for item in vehicle_crops
                if item["stationary"] is stationary
            ]
            by_motion[motion] = {
                "crops": len(selected),
                "psnr": distribution(
                    [item["vehicle_crop_psnr"] for item in selected]
                ),
                "lpips": distribution(
                    [item["vehicle_crop_lpips"] for item in selected]
                ),
            }
        crop_by_camera = {}
        for sensor_index in CAMERA_ORDER:
            selected = [
                item
                for item in vehicle_crops
                if item["sensor_index"] == sensor_index
            ]
            crop_by_camera[CAMERA_NAMES[sensor_index]] = {
                "crops": len(selected),
                "psnr": distribution(
                    [item["vehicle_crop_psnr"] for item in selected]
                ),
                "lpips": distribution(
                    [item["vehicle_crop_lpips"] for item in selected]
                ),
            }
        vehicle_crop_summary = {
            "scope": (
                "Held-out camera views only. The same stationary+moving rigid "
                "PandaSet cuboids are projected for every checkpoint. Each "
                "visible cuboid crop is padded 8%, resized to 128x128 only for "
                "LPIPS, and weighted equally in the reported distributions. "
                "Projection uses each camera's center timestamp and updated "
                "undistorted intrinsics; it does not apply per-row rolling-"
                "shutter actor correction."
            ),
            "trajectory_count": len(vehicle_trajectories),
            "crop_count": len(vehicle_crops),
            "minimum_side_pixels": args.vehicle_min_side_pixels,
            "psnr": distribution(
                [item["vehicle_crop_psnr"] for item in vehicle_crops]
            ),
            "lpips": distribution(
                [item["vehicle_crop_lpips"] for item in vehicle_crops]
            ),
            "by_motion": by_motion,
            "by_camera": crop_by_camera,
        }

    report = {
        "config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "load_seconds": load_seconds,
        "scope": (
            "all 80 logged timestamps and all six cameras, reconstructed by "
            "merging the original fixed train and held-out splits"
        ),
        "frames": 80,
        "views": len(per_frame),
        "finite_failures": finite_failures,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "render_latency_ms": distribution(
            [item["latency_ms"] for item in per_frame]
        ),
        "by_camera": by_camera,
        "vehicle_crop_metrics": vehicle_crop_summary,
        "automated_gate": {
            "finite_480_views": finite_failures == 0 and len(per_frame) == 480,
            "all_camera_temporal_and_flow_coverage_thresholds": image_gate_pass,
            "manual_video_review": "pending",
            "overall": (
                "pending_manual_review"
                if image_gate_pass
                else "inconclusive_due_to_automated_gate"
            ),
            "note": (
                "Excess warp measures additional flicker beyond the GT flow "
                "residual; a blurry prediction can still score well. Low "
                "forward/backward-flow coverage makes that camera "
                "inconclusive. Passing does not certify nearby-pose image "
                "quality or dynamic-object generalization."
            ),
        },
        "videos": {
            "six_camera_prediction": str(rig_video),
            "per_camera_ground_truth_vs_prediction": video_paths,
        },
        "per_frame": per_frame,
        "flow_pairs": flow_pairs,
        "vehicle_crops": vehicle_crops,
    }
    report_path = args.output_dir / "scene_040_temporal_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"per_frame", "flow_pairs", "vehicle_crops"}
    }
    print(json.dumps(summary, indent=2))
    print(f"WROTE: {report_path}")


if __name__ == "__main__":
    main()
