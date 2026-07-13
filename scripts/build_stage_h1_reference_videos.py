#!/usr/bin/env python3
"""Build and validate per-camera videos from Nerfstudio dataset renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_CAMERAS = (
    "front-forward",
    "left-forward",
    "right-forward",
    "left-backward",
    "right-backward",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--comparison-camera", default="front-forward")
    parser.add_argument("--video-width", type=int, default=960)
    return parser.parse_args()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable is not on PATH: {name}")
    return path


def timestamp_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def image_map(root: Path, split: str, output_name: str, camera: str) -> dict[str, Path]:
    directory = root / split / output_name / camera
    if not directory.is_dir():
        return {}
    images: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise RuntimeError(f"Duplicate image stem for {camera}: {path.stem}")
        images[path.stem] = path
    return images


def merged_camera_frames(root: Path, camera: str) -> list[Path]:
    frames: dict[str, Path] = {}
    for split in ("train", "test"):
        for stem, path in image_map(root, split, "rgb", camera).items():
            if stem in frames:
                raise RuntimeError(f"Duplicate rendered timestamp for {camera}: {stem}")
            frames[stem] = path
    return sorted(frames.values(), key=timestamp_key)


def link_sequence(paths: Iterable[Path], directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, source in enumerate(paths, start=1):
        destination = directory / f"{count - 1:06d}.jpg"
        destination.symlink_to(source.resolve())
    return count


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def encode_video(
    ffmpeg: str,
    frames: list[Path],
    output: Path,
    fps: float,
    video_width: int,
) -> None:
    if not frames:
        raise RuntimeError(f"No frames available for {output.name}")
    with tempfile.TemporaryDirectory(prefix="stage_h1_frames_") as temp_dir:
        sequence = Path(temp_dir)
        link_sequence(frames, sequence)
        run(
            [
                ffmpeg,
                "-y",
                "-v",
                "warning",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                str(sequence / "%06d.jpg"),
                "-vf",
                f"scale={video_width}:-2",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )


def encode_comparison(
    ffmpeg: str,
    predictions: list[Path],
    targets: list[Path],
    output: Path,
    fps: float,
    video_width: int,
) -> None:
    if not predictions or len(predictions) != len(targets):
        raise RuntimeError("Comparison inputs must be non-empty and aligned")
    with tempfile.TemporaryDirectory(prefix="stage_h1_compare_") as temp_dir:
        root = Path(temp_dir)
        link_sequence(targets, root / "target")
        link_sequence(predictions, root / "prediction")
        run(
            [
                ffmpeg,
                "-y",
                "-v",
                "warning",
                "-framerate",
                str(fps),
                "-i",
                str(root / "target" / "%06d.jpg"),
                "-framerate",
                str(fps),
                "-i",
                str(root / "prediction" / "%06d.jpg"),
                "-filter_complex",
                f"[0:v]scale={video_width}:-2[gt];[1:v]scale={video_width}:-2[pred];[gt][pred]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )


def probe_video(ffprobe: str, path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def validate_video(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    expected_frames: int,
    expected_width: int,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video is missing or empty: {path}")
    run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])
    probe = probe_video(ffprobe, path)
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream: {path}")
    frame_count = int(streams[0].get("nb_read_frames", 0))
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch for {path.name}: expected {expected_frames}, got {frame_count}"
        )
    width = int(streams[0].get("width", 0))
    if width != expected_width:
        raise RuntimeError(
            f"Width mismatch for {path.name}: expected {expected_width}, got {width}"
        )
    return probe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_frames(root: Path, camera: str) -> tuple[list[Path], list[Path]]:
    prediction_map = image_map(root, "test", "rgb", camera)
    target_map = image_map(root, "test", "gt-rgb", camera)
    common = sorted(set(prediction_map) & set(target_map), key=lambda value: (int(value), value))
    if set(prediction_map) != set(target_map):
        raise RuntimeError(f"Prediction/target frame mismatch for {camera}")
    return [prediction_map[key] for key in common], [target_map[key] for key in common]


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.video_width <= 0:
        raise ValueError("--video-width must be positive")
    render_root = args.render_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("--output-dir must be outside the Git repository")
    if output_dir == Path("/data") or Path("/data") in output_dir.parents:
        raise ValueError("--output-dir must be outside /data")
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    artifacts: list[dict[str, object]] = []

    reference_stems: set[str] | None = None
    for camera in args.cameras:
        frames = merged_camera_frames(render_root, camera)
        stems = {frame.stem for frame in frames}
        if len(frames) != 200 or len(stems) != 200:
            raise RuntimeError(f"Expected exactly 200 unique frames for {camera}, got {len(stems)}")
        if reference_stems is None:
            reference_stems = stems
        elif stems != reference_stems:
            raise RuntimeError(f"Timestamp set mismatch for {camera}")
        output = output_dir / f"scene_094_{camera}_reconstruction.mp4"
        encode_video(ffmpeg, frames, output, args.fps, args.video_width)
        probe = validate_video(ffmpeg, ffprobe, output, len(frames), args.video_width)
        artifacts.append(
            {
                "role": "reconstruction",
                "camera": camera,
                "path": str(output),
                "source_frames": len(frames),
                "sha256": sha256(output),
                "probe": probe,
            }
        )

    predictions, targets = comparison_frames(render_root, args.comparison_camera)
    if len(predictions) != 200:
        raise RuntimeError(
            f"Expected exactly 200 comparison frames, got {len(predictions)}"
        )
    comparison_output = output_dir / f"scene_094_{args.comparison_camera}_gt_vs_reconstruction.mp4"
    encode_comparison(
        ffmpeg,
        predictions,
        targets,
        comparison_output,
        args.fps,
        args.video_width,
    )
    probe = validate_video(
        ffmpeg,
        ffprobe,
        comparison_output,
        len(predictions),
        args.video_width * 2,
    )
    artifacts.append(
        {
            "role": "heldout_comparison_gt_left_prediction_right",
            "camera": args.comparison_camera,
            "path": str(comparison_output),
            "source_frames": len(predictions),
            "sha256": sha256(comparison_output),
            "probe": probe,
        }
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "render_root": str(render_root),
        "fps": args.fps,
        "artifacts": artifacts,
        "video_width": args.video_width,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")
    for artifact in artifacts:
        print(f"video={artifact['path']} frames={artifact['source_frames']}")


if __name__ == "__main__":
    main()
