#!/usr/bin/env python3
"""Load one released MTGS checkpoint and render observed or nearby poses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-block-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-renders", type=int, default=10)
    parser.add_argument(
        "--corridor-probe",
        action="store_true",
        help="also render a fixed-time world-pose grid out to +/-5 lateral meters",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def main() -> None:
    args = parse_args()
    if args.repeat_renders < 1:
        raise ValueError("--repeat-renders must be positive")
    for path in (args.config, args.checkpoint, args.road_block_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "NERFSTUDIO_DATAPARSER_CONFIGS",
        "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser",
    )
    os.environ.setdefault(
        "NERFSTUDIO_METHOD_CONFIGS",
        "mtgs=mtgs.config.MTGS:method",
    )

    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible in the MTGS environment")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    config = yaml.load(args.config.read_text(), Loader=yaml.Loader)
    config.load_dir = args.checkpoint.parent.resolve()
    config.pipeline.datamanager.dataparser.road_block_config = str(
        args.road_block_config.resolve()
    )
    config.pipeline.datamanager.dataparser.eval_2hz = True
    config.pipeline.datamanager.eval_cache_strategy = "on_demand"
    config.pipeline.datamanager.load_mask = False
    config.pipeline.datamanager.load_custom_masks = ()
    config.pipeline.datamanager.load_instance_masks = False
    config.pipeline.datamanager.load_semantic_masks_from = False
    config.pipeline.datamanager.load_lidar_depth = False
    config.pipeline.datamanager.load_pseudo_depth = False
    config.pipeline.model.output_depth_during_training = False
    config.pipeline.model.predict_normals = False
    config.pipeline.model.color_corrected_metrics = False
    config.pipeline.model.lpips_metric = False
    config.pipeline.model.dinov2_metric = False

    pipeline = config.pipeline.setup(device=device, test_mode="test")
    pipeline.eval()
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)
    load_seconds = time.perf_counter() - started

    camera, _ = next(
        iter(pipeline.datamanager.fixed_indices_eval_dataloader)
    )
    camera = camera.to(device)
    render_seconds = []
    output = None
    with torch.inference_mode():
        for _ in range(args.repeat_renders):
            torch.cuda.synchronize()
            render_started = time.perf_counter()
            output = pipeline.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - render_started)
    assert output is not None
    if "rgb" not in output:
        raise RuntimeError(
            f"MTGS output has no rgb tensor; keys={sorted(output)}"
        )
    rgb = output["rgb"].detach().float().cpu().numpy()
    finite = bool(np.isfinite(rgb).all())
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"unexpected MTGS RGB shape {rgb.shape}")
    image = Image.fromarray(
        np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    )
    image_path = output_dir / "observed_front_probe.jpg"
    image.save(image_path, format="JPEG", quality=95)

    corridor = None
    if args.corridor_probe:
        from PIL import ImageDraw

        lateral_offsets = (-5.0, -3.0, 0.0, 3.0, 5.0)
        forward_offsets = (0.0, 15.0, 30.0)
        base_pose = camera.camera_to_worlds.detach().clone()
        right_axis = base_pose[..., :3, 0]
        forward_axis = -base_pose[..., :3, 2]
        probe_records = []
        probe_images = []
        corridor_dir = output_dir / "corridor_frames"
        corridor_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for forward_m in forward_offsets:
                row = []
                for lateral_m in lateral_offsets:
                    probe_camera = camera
                    probe_camera.camera_to_worlds = base_pose.clone()
                    probe_camera.camera_to_worlds[..., :3, 3] = (
                        base_pose[..., :3, 3]
                        + lateral_m * right_axis
                        + forward_m * forward_axis
                    )
                    torch.cuda.synchronize()
                    probe_started = time.perf_counter()
                    probe_output = pipeline.model.get_outputs_for_camera(
                        camera=probe_camera
                    )
                    torch.cuda.synchronize()
                    probe_seconds = time.perf_counter() - probe_started
                    probe_rgb = (
                        probe_output["rgb"].detach().float().cpu().numpy()
                    )
                    probe_finite = bool(np.isfinite(probe_rgb).all())
                    probe_image = Image.fromarray(
                        np.clip(probe_rgb * 255.0, 0.0, 255.0).astype(
                            np.uint8
                        )
                    )
                    filename = (
                        f"forward_{forward_m:+05.1f}m_"
                        f"lateral_{lateral_m:+04.1f}m.jpg"
                    )
                    frame_path = corridor_dir / filename
                    probe_image.save(frame_path, format="JPEG", quality=95)
                    row.append(probe_image)
                    probe_records.append(
                        {
                            "forward_m": forward_m,
                            "lateral_m": lateral_m,
                            "finite": probe_finite,
                            "render_seconds": probe_seconds,
                            "image_path": str(frame_path),
                        }
                    )
                probe_images.append(row)

        tile_width, tile_height = image.size
        label_height = 38
        contact_sheet = Image.new(
            "RGB",
            (
                tile_width * len(lateral_offsets),
                (tile_height + label_height) * len(forward_offsets),
            ),
            color=(20, 20, 20),
        )
        draw = ImageDraw.Draw(contact_sheet)
        for row_index, (forward_m, row) in enumerate(
            zip(forward_offsets, probe_images)
        ):
            for column_index, (lateral_m, probe_image) in enumerate(
                zip(lateral_offsets, row)
            ):
                x = column_index * tile_width
                y = row_index * (tile_height + label_height)
                contact_sheet.paste(probe_image, (x, y + label_height))
                draw.text(
                    (x + 12, y + 10),
                    f"forward {forward_m:+.0f} m | lateral {lateral_m:+.0f} m",
                    fill=(245, 245, 245),
                )
        contact_sheet_path = output_dir / "corridor_contact_sheet.jpg"
        contact_sheet.save(contact_sheet_path, format="JPEG", quality=92)
        probe_times = [record["render_seconds"] for record in probe_records]
        corridor = {
            "fixed_time_and_heading": True,
            "lateral_offsets_m": list(lateral_offsets),
            "forward_offsets_m": list(forward_offsets),
            "pose_count": len(probe_records),
            "all_finite": all(record["finite"] for record in probe_records),
            "render_seconds_p50": statistics.median(probe_times),
            "render_seconds_p95": percentile(probe_times, 0.95),
            "contact_sheet_path": str(contact_sheet_path),
            "poses": probe_records,
        }

    warm = render_seconds[1:] or render_seconds
    p50 = statistics.median(warm)
    p95 = percentile(warm, 0.95)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    metadata = {
        key: (
            value.detach().cpu().tolist()
            if hasattr(value, "detach")
            else value
        )
        for key, value in (camera.metadata or {}).items()
        if key in {"cam_name", "travel_id", "cam_token"}
    }
    report = {
        "format": "driving_scene_reconstruction.mtgs_checkpoint_gate.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source_commit": "7ab67a3e386e5a4830017819324922a7bb9f26f7",
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(step),
        "road_block_config": str(args.road_block_config.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "load_seconds": load_seconds,
        "peak_memory": {
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
            "reserved_gib": peak_reserved / 1024**3,
            "passes_21_5_gib_gate": peak_reserved <= 21.5 * 1024**3,
        },
        "observed_front_render": {
            "camera_metadata": metadata,
            "shape": list(rgb.shape),
            "finite": finite,
            "repeat_count": args.repeat_renders,
            "warm_seconds_p50": p50,
            "warm_seconds_p95": p95,
            "warm_fps_p50": 1.0 / p50,
            "warm_fps_p95": 1.0 / p95,
            "image_path": str(image_path),
        },
        "wide_corridor_probe": corridor,
        "summary": {
            "verdict": (
                "pass"
                if finite
                and peak_reserved <= 21.5 * 1024**3
                and math.isfinite(p95)
                and (
                    not args.corridor_probe
                    or (
                        corridor is not None
                        and corridor["all_finite"]
                        and math.isfinite(corridor["render_seconds_p95"])
                    )
                )
                else "fail"
            ),
            "training_performed": False,
            "wide_corridor_visual_probe_performed": args.corridor_probe,
        },
    }
    report_path = output_dir / "mtgs_checkpoint_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"image: {image_path}")
    print(f"report: {report_path}")
    print(f"verdict: {report['summary']['verdict']}")


if __name__ == "__main__":
    main()
