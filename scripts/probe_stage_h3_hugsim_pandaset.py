#!/usr/bin/env python3
"""Probe an exported HUGSIM PandaSet scene at logged and lateral poses.

The probe is intentionally inference-only.  It renders the front camera with
native dynamic objects, repeats the logged pose without native dynamics, and
renders a small lateral-pose grid.  When a SplatAD temporal comparison
directory is supplied, the output video places both reconstructions beside the
same PandaSet source frame.

Run this script with HUGSIM's own Python environment, not the Stage H3
SplatAD environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_HUGSIM_ROOT = Path("/home/yawei/HUGSIM")
DEFAULT_MODEL_DIR = Path(
    "/home/yawei/HUGSIM_assets/scenes/pandaset/040/040"
)
DEFAULT_SOURCE_DIR = Path(
    "/home/yawei/stage3_external/data/pandaset/040"
)
DEFAULT_SPLATAD_COMPARISON_DIR = Path(
    "/home/yawei/stage3_external/artifacts/"
    "scene_040_temporal_gate_8000/comparison_frames/front"
)
DEFAULT_SPLATAD_REPORT = Path(
    "/home/yawei/stage3_external/artifacts/"
    "scene_040_temporal_gate_8000/scene_040_temporal_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/yawei/stage3_external/artifacts/"
    "scene_040_hugsim_official_probe_20260728"
)
DEFAULT_FRAME_INDICES = (19, 39, 59)
DEFAULT_LATERAL_LEFT_METERS = (-3.0, -1.0, 1.0, 3.0)
VIDEO_FPS = 10
TILE_SIZE = (480, 270)
LABEL_HEIGHT = 44


@dataclass(frozen=True)
class FrameRecord:
    index: int
    metadata: dict[str, Any]
    source_path: Path


def parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("frame indices must be unique")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def resize_rgb(image: np.ndarray, size: tuple[int, int] = TILE_SIZE) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return pil_image.resize(size, Image.Resampling.LANCZOS)


def labeled_tile(
    image: np.ndarray,
    label: str,
    *,
    size: tuple[int, int] = TILE_SIZE,
) -> Image.Image:
    font = load_font()
    tile = Image.new("RGB", (size[0], size[1] + LABEL_HEIGHT), (18, 18, 18))
    tile.paste(resize_rgb(image, size), (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(tile)
    draw.text((10, 8), label, fill=(246, 222, 82), font=font)
    return tile


def compose_row(items: Iterable[tuple[str, np.ndarray]]) -> Image.Image:
    tiles = [labeled_tile(image, label) for label, image in items]
    row = Image.new(
        "RGB",
        (TILE_SIZE[0] * len(tiles), TILE_SIZE[1] + LABEL_HEIGHT),
        (0, 0, 0),
    )
    for index, tile in enumerate(tiles):
        row.paste(tile, (index * TILE_SIZE[0], 0))
    return row


def compose_grid(rows: Iterable[Image.Image]) -> Image.Image:
    row_images = list(rows)
    if not row_images:
        raise ValueError("cannot compose an empty grid")
    width = max(row.width for row in row_images)
    height = sum(row.height for row in row_images)
    grid = Image.new("RGB", (width, height), (0, 0, 0))
    offset = 0
    for row in row_images:
        grid.paste(row, (0, offset))
        offset += row.height
    return grid


def source_path_for_frame(
    source_dir: Path,
    camera_name: str,
    metadata_path: str,
) -> Path:
    filename = PurePosixPath(metadata_path).name
    return source_dir / "camera" / camera_name / filename


def load_frame_records(
    metadata_path: Path,
    source_dir: Path,
    camera_name: str,
) -> list[FrameRecord]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records: list[FrameRecord] = []
    for frame in metadata.get("frames", []):
        rgb_path = PurePosixPath(frame["rgb_path"])
        if rgb_path.parent.name != camera_name:
            continue
        index = int(rgb_path.stem)
        source_path = source_path_for_frame(
            source_dir,
            camera_name,
            frame["rgb_path"],
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        records.append(
            FrameRecord(
                index=index,
                metadata=frame,
                source_path=source_path,
            )
        )
    records.sort(key=lambda record: record.index)
    if not records:
        raise ValueError(f"no {camera_name!r} records in {metadata_path}")
    if len({record.index for record in records}) != len(records):
        raise ValueError(f"duplicate {camera_name!r} frame indices")
    return records


def load_source_rgb(record: FrameRecord) -> np.ndarray:
    frame = record.metadata
    target_size = (int(frame["width"]), int(frame["height"]))
    image = Image.open(record.source_path).convert("RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def load_splatad_rgb(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Extract the rendered right half from an existing GT/render comparison."""

    image = Image.open(path).convert("RGB")
    width, height = image.size
    render = image.crop((width // 2, 0, width, height))
    target_height, target_width = target_shape
    render = render.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(render)


def load_splatad_metrics(
    report_path: Path,
    camera_name: str,
) -> dict[int, dict[str, float | str | bool]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_camera = camera_name.removesuffix("_camera")
    metrics = {}
    for record in report.get("per_frame", []):
        if record.get("camera") != report_camera:
            continue
        frame_index = int(record["frame"])
        metrics[frame_index] = {
            key: record[key]
            for key in (
                "finite",
                "latency_ms",
                "lpips",
                "psnr",
                "split",
                "ssim",
            )
        }
    if not metrics:
        raise ValueError(
            f"no camera {report_camera!r} records in {report_path}"
        )
    return metrics


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_float = np.asarray(reference, dtype=np.float64) / 255.0
    candidate_float = np.asarray(candidate, dtype=np.float64) / 255.0
    delta = candidate_float - reference_float
    mse = float(np.mean(delta * delta))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "mse": mse,
        "psnr_db": math.inf if mse == 0 else float(-10.0 * math.log10(mse)),
        "near_black_fraction": float(np.mean(np.max(candidate, axis=2) < 8)),
    }


def dynamic_difference_metrics(
    factual: np.ndarray,
    static_only: np.ndarray,
) -> dict[str, float]:
    delta = np.abs(
        factual.astype(np.float32) - static_only.astype(np.float32)
    )
    pixel_delta = np.max(delta, axis=2)
    return {
        "mean_abs_rgb_255": float(np.mean(delta)),
        "changed_pixel_fraction_gt_8": float(np.mean(pixel_delta > 8.0)),
        "changed_pixel_fraction_gt_24": float(np.mean(pixel_delta > 24.0)),
    }


def lateral_pose(c2w: np.ndarray, left_meters: float) -> np.ndarray:
    """Translate a camera left in its logged horizontal coordinate frame."""

    pose = np.asarray(c2w, dtype=np.float64).copy()
    camera_right = pose[:3, 0]
    camera_right /= np.linalg.norm(camera_right)
    pose[:3, 3] -= left_meters * camera_right
    return pose


def render_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(args.hugsim_root))
    from gaussian_renderer import GaussianModel, render
    from scene.cameras import Camera
    from scene.obj_model import ObjModel

    model_dir = args.model_dir.resolve()
    metadata_path = model_dir / "meta_data.json"
    records = load_frame_records(
        metadata_path,
        args.source_dir.resolve(),
        args.camera_name,
    )
    records_by_index = {record.index: record for record in records}
    missing_indices = sorted(set(args.frame_indices) - set(records_by_index))
    if missing_indices:
        raise ValueError(f"selected frames not present: {missing_indices}")
    splatad_metrics_by_index = (
        load_splatad_metrics(
            args.splatad_report.resolve(),
            args.camera_name,
        )
        if args.splatad_report is not None
        else {}
    )
    missing_splatad_metrics = sorted(
        set(records_by_index) - set(splatad_metrics_by_index)
    )
    if args.splatad_report is not None and missing_splatad_metrics:
        raise ValueError(
            "SplatAD report is missing frame metrics: "
            f"{missing_splatad_metrics}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    selected_dir = output_dir / "selected_frames"
    selected_dir.mkdir()
    video_frames_dir = output_dir / "video_frames"
    if args.video:
        video_frames_dir.mkdir()

    cfg = OmegaConf.load(model_dir / "cfg.yaml")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    load_started = time.perf_counter()
    gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
    model_params, model_iteration = torch.load(
        model_dir / "scene.pth",
        map_location="cuda",
        weights_only=False,
    )
    gaussians.restore(model_params, None)

    dynamic_ids = sorted(
        {
            dynamic_id
            for record in records
            for dynamic_id in record.metadata.get("dynamics", {})
        }
    )
    dynamic_gaussians = {}
    dynamic_checkpoint_paths = {}
    for dynamic_id in dynamic_ids:
        checkpoint_path = model_dir / f"dynamic_{dynamic_id}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        dynamic = ObjModel(cfg.model.sh_degree, feat_mutable=False)
        dynamic_params, _ = torch.load(
            checkpoint_path,
            map_location="cuda",
            weights_only=False,
        )
        dynamic.restore(dynamic_params, None)
        dynamic_gaussians[dynamic_id] = dynamic
        dynamic_checkpoint_paths[dynamic_id] = checkpoint_path
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    background = torch.tensor(
        [1.0, 1.0, 1.0] if cfg.model.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    render_latency_records: list[dict[str, Any]] = []

    def render_record(
        record: FrameRecord,
        source_rgb: np.ndarray,
        *,
        pose: np.ndarray | None = None,
        include_dynamics: bool = True,
        variant: str,
    ) -> np.ndarray:
        frame = record.metadata
        dynamics = (
            {
                dynamic_id: torch.tensor(
                    matrix,
                    dtype=torch.float32,
                    device="cuda",
                )
                for dynamic_id, matrix in frame.get("dynamics", {}).items()
            }
            if include_dynamics
            else {}
        )
        camera = Camera(
            width=int(frame["width"]),
            height=int(frame["height"]),
            image=source_rgb.astype(np.float32) / 255.0,
            K=np.asarray(frame["intrinsics"], dtype=np.float64),
            c2w=(
                np.asarray(frame["camtoworld"], dtype=np.float64)
                if pose is None
                else pose
            ),
            image_name=f"{args.camera_name}_{record.index:02d}",
            data_device="cuda",
            timestamp=float(frame["timestamp"]),
            dynamics=dynamics,
        )
        torch.cuda.synchronize()
        render_started = time.perf_counter()
        with torch.no_grad():
            package = render(
                viewpoint=camera,
                prev_viewpoint=None,
                pc=gaussians,
                dynamic_gaussians=dynamic_gaussians,
                unicycles=None,
                bg_color=background,
                render_optical=False,
            )
        rendered_tensor = package["render"].detach()
        if not bool(torch.isfinite(rendered_tensor).all()):
            raise ValueError(f"non-finite render at frame {record.index}")
        rendered = (
            rendered_tensor.clamp(0, 1)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .__mul__(255.0)
            .round()
            .astype(np.uint8)
        )
        render_latency_records.append(
            {
                "frame_index": record.index,
                "variant": variant,
                "latency_ms": (time.perf_counter() - render_started) * 1000.0,
            }
        )
        return rendered

    selected = set(args.frame_indices)
    selected_arrays: dict[int, dict[str, np.ndarray]] = {}
    per_frame_results: dict[str, Any] = {}
    render_records = records if args.video else [
        records_by_index[index] for index in args.frame_indices
    ]
    for record in render_records:
        source_rgb = load_source_rgb(record)
        factual = render_record(
            record,
            source_rgb,
            include_dynamics=True,
            variant="logged_factual",
        )
        static_only = render_record(
            record,
            source_rgb,
            include_dynamics=False,
            variant="logged_static_only",
        )

        if args.splatad_comparison_dir is not None:
            splatad_path = (
                args.splatad_comparison_dir.resolve()
                / f"{record.index:03d}.jpg"
            )
            if not splatad_path.is_file():
                raise FileNotFoundError(splatad_path)
            splatad = load_splatad_rgb(splatad_path, source_rgb.shape[:2])
        else:
            splatad_path = None
            splatad = np.zeros_like(source_rgb)

        per_frame_results[str(record.index)] = {
            "source_path": str(record.source_path),
            "timestamp_s": float(record.metadata["timestamp"]),
            "native_dynamic_ids": sorted(record.metadata.get("dynamics", {})),
            "hugsim_factual_metrics": image_metrics(source_rgb, factual),
            "hugsim_static_only_metrics": image_metrics(source_rgb, static_only),
            "splatad_static8k_evaluator_metrics": (
                splatad_metrics_by_index.get(record.index)
                if args.splatad_report is not None
                else None
            ),
            "dynamic_difference": dynamic_difference_metrics(
                factual,
                static_only,
            ),
        }

        if record.index in selected:
            selected_arrays[record.index] = {
                "source": source_rgb,
                "splatad": splatad,
                "factual": factual,
                "static_only": static_only,
            }
            frame_dir = selected_dir / f"{record.index:03d}"
            frame_dir.mkdir()
            Image.fromarray(factual).save(frame_dir / "hugsim_factual.png")
            Image.fromarray(static_only).save(
                frame_dir / "hugsim_static_only.png"
            )

        if args.video:
            comparison_row = compose_row(
                (
                    ("PandaSet GT", source_rgb),
                    ("SplatAD static-8k", splatad),
                    ("HUGSIM factual", factual),
                    ("HUGSIM static-only", static_only),
                )
            )
            comparison_row.save(
                video_frames_dir / f"{record.index:03d}.jpg",
                quality=92,
            )

    contact_rows = []
    for frame_index in args.frame_indices:
        record = records_by_index[frame_index]
        arrays = selected_arrays[frame_index]
        source_rgb = arrays["source"]
        frame_result = per_frame_results[str(frame_index)]
        offset_items: list[tuple[str, np.ndarray]] = []
        frame_dir = selected_dir / f"{frame_index:03d}"
        for left_meters in args.lateral_left_meters:
            pose = lateral_pose(
                np.asarray(record.metadata["camtoworld"], dtype=np.float64),
                left_meters,
            )
            rendered = render_record(
                record,
                source_rgb,
                pose=pose,
                include_dynamics=True,
                variant=f"lateral_left_{left_meters:+.1f}m",
            )
            label = f"HUGSIM left {left_meters:+.0f} m"
            offset_items.append((label, rendered))
            offset_path = frame_dir / f"left_{left_meters:+.1f}m.png"
            Image.fromarray(rendered).save(offset_path)
            frame_result.setdefault("lateral_views", {})[
                f"{left_meters:+.1f}"
            ] = {
                "path": str(offset_path),
                "near_black_fraction": float(
                    np.mean(np.max(rendered, axis=2) < 8)
                ),
            }

        contact_rows.append(
            compose_row(
                (
                    ("PandaSet GT", arrays["source"]),
                    ("SplatAD static-8k", arrays["splatad"]),
                    ("HUGSIM factual", arrays["factual"]),
                    ("HUGSIM static-only", arrays["static_only"]),
                    *offset_items,
                )
            )
        )

    contact_sheet = output_dir / "hugsim_scene040_probe_contact_sheet.jpg"
    compose_grid(contact_rows).save(contact_sheet, quality=94)

    video_path = None
    if args.video:
        video_path = output_dir / "hugsim_scene040_vs_splatad_10fps.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(VIDEO_FPS),
                "-i",
                str(video_frames_dir / "%03d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            check=True,
        )

    def aggregate_metric(
        metric_group: str,
        metric_name: str,
    ) -> dict[str, float]:
        values = np.asarray(
            [
                frame[metric_group][metric_name]
                for frame in per_frame_results.values()
                if frame[metric_group] is not None
            ],
            dtype=np.float64,
        )
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    def aggregate_render_latency(variant: str) -> dict[str, float | int]:
        values = np.asarray(
            [
                record["latency_ms"]
                for record in render_latency_records
                if record["variant"] == variant
            ],
            dtype=np.float64,
        )
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        }

    result = {
        "audit_id": "stage_h3_hugsim_pandaset_official_checkpoint_probe",
        "date": date.today().isoformat(),
        "mode": "inference_only",
        "scene": "PandaSet 040",
        "camera": args.camera_name,
        "hugsim": {
            "repo": str(args.hugsim_root.resolve()),
            "commit": git_commit(args.hugsim_root.resolve()),
            "model_dir": str(model_dir),
            "scene_checkpoint_bytes": (model_dir / "scene.pth").stat().st_size,
            "scene_checkpoint_sha256": sha256_file(model_dir / "scene.pth"),
            "model_iteration": int(model_iteration),
            "dynamic_checkpoint_count": len(dynamic_checkpoint_paths),
            "model_load_seconds": model_load_seconds,
        },
        "source_dir": str(args.source_dir.resolve()),
        "splatad_comparison_dir": (
            str(args.splatad_comparison_dir.resolve())
            if args.splatad_comparison_dir is not None
            else None
        ),
        "splatad_report": (
            str(args.splatad_report.resolve())
            if args.splatad_report is not None
            else None
        ),
        "rendered_frame_count": len(render_records),
        "selected_frame_indices": list(args.frame_indices),
        "lateral_left_meters": list(args.lateral_left_meters),
        "aggregate": {
            "hugsim_factual_psnr_db": aggregate_metric(
                "hugsim_factual_metrics",
                "psnr_db",
            ),
            "hugsim_static_only_psnr_db": aggregate_metric(
                "hugsim_static_only_metrics",
                "psnr_db",
            ),
            "splatad_static8k_psnr_db": (
                aggregate_metric(
                    "splatad_static8k_evaluator_metrics",
                    "psnr",
                )
                if args.splatad_report is not None
                else None
            ),
            "dynamic_changed_pixel_fraction_gt_8": aggregate_metric(
                "dynamic_difference",
                "changed_pixel_fraction_gt_8",
            ),
            "render_latency_ms": {
                "logged_factual": aggregate_render_latency(
                    "logged_factual"
                ),
                "logged_static_only": aggregate_render_latency(
                    "logged_static_only"
                ),
            },
        },
        "per_frame": per_frame_results,
        "artifacts": {
            "contact_sheet": str(contact_sheet),
            "video": str(video_path) if video_path is not None else None,
        },
        "claim_boundary": (
            "This probe compares an official exported HUGSIM checkpoint and the "
            "existing project SplatAD static-8k checkpoint on PandaSet scene "
            "040. The renderers use their own preprocessing and camera models; "
            "the comparison is a practical visual gate, not a controlled "
            "architecture benchmark. Lateral views have no real RGB target."
        ),
    }
    manifest_path = output_dir / "hugsim_scene040_probe.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"contact_sheet={contact_sheet}")
    if video_path is not None:
        print(f"video={video_path}")
    print(f"manifest={manifest_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hugsim-root", type=Path, default=DEFAULT_HUGSIM_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--camera-name", default="front_camera")
    parser.add_argument(
        "--splatad-comparison-dir",
        type=Path,
        default=DEFAULT_SPLATAD_COMPARISON_DIR,
    )
    parser.add_argument(
        "--splatad-report",
        type=Path,
        default=DEFAULT_SPLATAD_REPORT,
        help=(
            "existing SplatAD temporal report used for numeric comparison; "
            "the labeled comparison JPEGs are used only for visualization"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frame-indices",
        type=parse_int_csv,
        default=DEFAULT_FRAME_INDICES,
        help="comma-separated frame indices for the lateral contact sheet",
    )
    parser.add_argument(
        "--lateral-left-meters",
        type=parse_float_csv,
        default=DEFAULT_LATERAL_LEFT_METERS,
        help="comma-separated lateral translations; positive is camera-left",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="render all logged front frames and encode a four-column video",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hugsim_root.is_dir():
        raise FileNotFoundError(args.hugsim_root)
    if not (args.model_dir / "scene.pth").is_file():
        raise FileNotFoundError(args.model_dir / "scene.pth")
    if not args.source_dir.is_dir():
        raise FileNotFoundError(args.source_dir)
    if args.splatad_comparison_dir is not None:
        if not args.splatad_comparison_dir.is_dir():
            raise FileNotFoundError(args.splatad_comparison_dir)
    if args.splatad_report is not None and not args.splatad_report.is_file():
        raise FileNotFoundError(args.splatad_report)
    render_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
