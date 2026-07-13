#!/usr/bin/env python3
"""Create a non-mutating WayveScenes101 official camera-split dataset view."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

TRAIN_CAMERAS = (
    "left-forward",
    "right-forward",
    "left-backward",
    "right-backward",
)
TEST_CAMERA = "front-forward"
EXPECTED_PER_CAMERA = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--output-scene", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def camera_name(frame: dict[str, object]) -> str:
    parts = Path(str(frame["file_path"])).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot infer camera from {frame['file_path']}")
    return parts[-2]


def resolve_scene_path(source: Path, value: object, label: str) -> Path:
    candidate = (source / str(value)).resolve()
    if candidate == source or source not in candidate.parents:
        raise ValueError(f"{label} escapes source scene: {value}")
    return candidate


def main() -> None:
    args = parse_args()
    source = args.source_scene.expanduser().resolve()
    output = args.output_scene.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output == source:
        raise ValueError("--output-scene must differ from --source-scene")
    if output.exists():
        raise FileExistsError(f"Output scene already exists: {output}")
    if output == repo_root or repo_root in output.parents:
        raise ValueError("--output-scene must be outside the Git repository")
    if output == Path("/data") or Path("/data") in output.parents:
        raise ValueError("--output-scene must be outside /data")
    transforms_path = source / "transforms.json"
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    expected_cameras = set(TRAIN_CAMERAS) | {TEST_CAMERA}
    counts = Counter(camera_name(frame) for frame in frames)

    if data.get("camera_model") != "OPENCV_FISHEYE":
        raise ValueError("Expected top-level camera_model=OPENCV_FISHEYE")
    if set(counts) != expected_cameras:
        raise ValueError(f"Unexpected camera set: {sorted(counts)}")
    if any(counts[camera] != EXPECTED_PER_CAMERA for camera in expected_cameras):
        raise ValueError(f"Expected 200 frames per camera, got {dict(counts)}")

    missing_images = []
    missing_masks = []
    seen_images: set[Path] = set()
    seen_masks: set[Path] = set()
    for frame in frames:
        image_path = resolve_scene_path(source, frame["file_path"], "file_path")
        mask_value = frame.get("mask_path")
        if image_path in seen_images:
            raise ValueError(f"Duplicate frame file_path: {frame['file_path']}")
        seen_images.add(image_path)
        if not image_path.is_file():
            missing_images.append(str(image_path))
        if mask_value is None:
            missing_masks.append(str(mask_value))
            continue
        mask_path = resolve_scene_path(source, mask_value, "mask_path")
        if mask_path in seen_masks:
            raise ValueError(f"Duplicate frame mask_path: {mask_value}")
        seen_masks.add(mask_path)
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f"Missing images={len(missing_images)} masks={len(missing_masks)}"
        )

    train_filenames = sorted(
        str(frame["file_path"])
        for frame in frames
        if camera_name(frame) in TRAIN_CAMERAS
    )
    test_filenames = sorted(
        str(frame["file_path"])
        for frame in frames
        if camera_name(frame) == TEST_CAMERA
    )
    if len(train_filenames) != 800 or len(test_filenames) != 200:
        raise AssertionError("Official split must contain 800 train and 200 test images")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.prepare.", dir=output.parent)
    )
    published = False
    try:
        for name in ("images", "masks", "sparse_pc.ply", "colmap_sparse"):
            source_path = source / name
            if source_path.exists():
                ensure_symlink(source_path, temporary / name)

        output_data = dict(data)
        output_data["train_filenames"] = train_filenames
        output_data["val_filenames"] = test_filenames
        output_data["test_filenames"] = test_filenames
        output_transforms = temporary / "transforms.json"
        output_transforms.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        manifest = {
            "source_scene": str(source),
            "source_transforms_sha256": sha256(transforms_path),
            "output_transforms_sha256": sha256(output_transforms),
            "camera_counts": dict(sorted(counts.items())),
            "train_cameras": list(TRAIN_CAMERAS),
            "test_cameras": [TEST_CAMERA],
            "train_images": len(train_filenames),
            "test_images": len(test_filenames),
            "dataset_license": "WayveScenes101 non-commercial research use",
        }
        manifest_path = temporary / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(output)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(manifest, indent=2))
    print(f"output_scene={output}")


if __name__ == "__main__":
    main()
