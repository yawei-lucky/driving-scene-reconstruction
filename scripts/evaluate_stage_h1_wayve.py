#!/usr/bin/env python3
"""Arrange Nerfstudio renders and run the official WayveScenes101 evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import wayve_scenes.evaluation as wayve_evaluation

TRAIN_CAMERAS = (
    "left-forward",
    "right-forward",
    "left-backward",
    "right-backward",
)
TEST_CAMERA = "front-forward"
CAMERAS = TRAIN_CAMERAS + (TEST_CAMERA,)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--scene", default="scene_094")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def rendered_images(render_root: Path, camera: str, expected_stems: set[str]) -> list[Path]:
    split = "test" if camera == TEST_CAMERA else "train"
    directory = render_root / split / "rgb" / camera
    if not directory.is_dir():
        directory = render_root / split / "rgb"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem in expected_stems
    )
    if len(images) != 200:
        raise RuntimeError(f"Expected 200 renders for {camera}, got {len(images)}")
    return images


def indexed_images(directory: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise RuntimeError(f"Duplicate image stem in {directory}: {path.stem}")
        images[path.stem] = path
    return images


def validate_decodable_pair(prediction: Path, target: Path) -> None:
    with Image.open(prediction) as image:
        image.verify()
    with Image.open(prediction) as image:
        prediction_size = image.size
    with Image.open(target) as image:
        image.verify()
    with Image.open(target) as image:
        target_size = image.size
    if prediction_size != target_size:
        raise RuntimeError(
            f"Image size mismatch for {prediction.name}: "
            f"prediction={prediction_size}, target={target_size}"
        )


def streaming_fid(predictions: list[str], targets: list[str]) -> float:
    """Equivalent official FID computation without stacking full-resolution inputs."""
    metric = FrechetInceptionDistance(feature=64)
    for start in range(0, len(predictions), 8):
        prediction_batch = torch.from_numpy(
            np.stack([np.asarray(Image.open(path).convert("RGB")) for path in predictions[start : start + 8]])
        ).permute(0, 3, 1, 2)
        target_batch = torch.from_numpy(
            np.stack([np.asarray(Image.open(path).convert("RGB")) for path in targets[start : start + 8]])
        ).permute(0, 3, 1, 2)
        metric.update(target_batch, real=True)
        metric.update(prediction_batch, real=False)
    return torch.clamp(metric.compute(), min=0.0).item()


def arrange_predictions(
    render_root: Path,
    prediction_root: Path,
    target_root: Path,
    scene: str,
    stems_by_camera: dict[str, set[str]],
) -> None:
    for camera in CAMERAS:
        sources = rendered_images(render_root, camera, stems_by_camera[camera])
        source_by_stem = {source.stem: source for source in sources}
        targets = indexed_images(target_root / scene / "images" / camera)
        if set(source_by_stem) != set(targets):
            missing = sorted(set(targets) - set(source_by_stem))[:5]
            extra = sorted(set(source_by_stem) - set(targets))[:5]
            raise RuntimeError(
                f"Filename mismatch for {camera}: missing={missing}, extra={extra}"
            )
        output_dir = prediction_root / scene / "images" / camera
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            validate_decodable_pair(source, targets[source.stem])
            destination = output_dir / f"{source.stem}.jpeg"
            if destination.is_symlink():
                if destination.resolve() == source.resolve():
                    continue
                destination.unlink()
            elif destination.exists():
                raise RuntimeError(f"Refusing to replace non-symlink: {destination}")
            destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    render_root = args.render_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    transforms = json.loads(args.transforms.expanduser().resolve().read_text(encoding="utf-8"))
    stems_by_camera: dict[str, set[str]] = {camera: set() for camera in CAMERAS}
    for frame in transforms["frames"]:
        path = Path(frame["file_path"])
        if path.parent.name in stems_by_camera:
            stems_by_camera[path.parent.name].add(path.stem)
    repo_root = Path(__file__).resolve().parents[1]
    for label, path in (("prediction root", prediction_root), ("output path", output_path)):
        if path == repo_root or repo_root in path.parents:
            raise ValueError(f"{label} must be outside the Git repository")
        if path == Path("/data") or Path("/data") in path.parents:
            raise ValueError(f"{label} must be outside /data")
    target_scene = target_root / args.scene
    if not (target_scene / "images").is_dir() or not (target_scene / "masks").is_dir():
        raise FileNotFoundError(f"Invalid target scene: {target_scene}")

    if prediction_root.exists() and any(prediction_root.iterdir()):
        raise RuntimeError(f"Prediction root must be fresh: {prediction_root}")
    arrange_predictions(render_root, prediction_root, target_root, args.scene, stems_by_camera)
    wayve_evaluation.get_fid_metric = streaming_fid
    metrics_all, metrics_train, metrics_test = wayve_evaluation.evaluate_submission(
        dir_pred=str(prediction_root),
        dir_target=str(target_root),
        scene_list=[args.scene],
        use_masks=True,
    )
    result = {
        "scene": args.scene,
        "prediction_root": str(prediction_root),
        "target_root": str(target_root),
        "split_policy": {
            "train_cameras": list(TRAIN_CAMERAS),
            "test_cameras": [TEST_CAMERA],
        },
        "fid_implementation": "official feature=64 metric with memory-safe batches of 8",
        "all": metrics_all,
        "train": metrics_train,
        "test": metrics_test,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    print(f"output={output_path}")
    print(
        json.dumps(
            {
                "test_psnr": metrics_test["psnr"],
                "test_ssim": metrics_test["ssim"],
                "test_lpips": metrics_test["lpips"],
                "test_fid": metrics_test["fid"],
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
