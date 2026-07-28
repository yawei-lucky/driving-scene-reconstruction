#!/usr/bin/env python3
"""Freeze selected TbV download windows as a hard-linked data view."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path


def parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "window indices must be comma-separated integers"
        ) from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "window indices must contain non-negative integers"
        )
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("window indices must be unique")
    return indices


def _reuse_existing(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset_prefix: str,
    window_indices: tuple[int, ...],
    unique_objects: dict[str, dict[str, object]],
) -> dict[str, object]:
    destination_manifest = output_dir / "selection_manifest.json"
    if not destination_manifest.is_file():
        raise RuntimeError(
            f"refusing non-empty output without a frozen manifest: {output_dir}"
        )
    try:
        frozen = json.loads(destination_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot validate existing frozen manifest: {destination_manifest}"
        ) from error
    try:
        recorded_source = Path(str(frozen["materialized_from"])).resolve()
    except KeyError as error:
        raise RuntimeError(
            f"existing frozen manifest has no source: {destination_manifest}"
        ) from error
    if (
        recorded_source != manifest_path
        or frozen.get("materialization") != "hardlink"
        or frozen.get("selected_window_indices") != list(window_indices)
    ):
        raise RuntimeError(
            f"existing frozen view does not match this request: {output_dir}"
        )

    recorded_objects: dict[str, int] = {}
    for log in frozen.get("logs", []):
        for item in log.get("objects", []):
            recorded_objects[str(item.get("key"))] = int(item.get("size", -1))
    expected_objects = {
        key: int(item.get("size", -1))
        for key, item in unique_objects.items()
    }
    if recorded_objects != expected_objects:
        raise RuntimeError(
            f"existing frozen view has different manifest objects: {output_dir}"
        )
    for key, size in expected_objects.items():
        destination = output_dir / Path(key).relative_to(dataset_prefix)
        if not destination.is_file() or destination.stat().st_size != size:
            raise RuntimeError(
                f"existing frozen object is missing or has the wrong size: "
                f"{destination}"
            )
    return {
        "manifest": str(destination_manifest),
        "linked_objects": len(expected_objects),
        "logical_bytes": sum(expected_objects.values()),
        "selected_window_indices": list(window_indices),
        "reused_existing": True,
    }


def materialize(
    manifest_path: Path,
    output_dir: Path,
    window_indices: tuple[int, ...],
) -> dict[str, object]:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    logs = source.get("logs")
    if not isinstance(logs, list):
        raise ValueError("source manifest has no logs list")
    if max(window_indices) >= len(logs):
        raise ValueError(
            f"window index exceeds manifest range [0, {len(logs) - 1}]"
        )

    selected_logs = [deepcopy(logs[index]) for index in window_indices]
    unique_objects: dict[str, dict[str, object]] = {}
    for log in selected_logs:
        objects = log.get("objects")
        if not isinstance(objects, list):
            raise ValueError("source manifest log has no objects list")
        for item in objects:
            key = item.get("key")
            raw_path = item.get("path")
            if not isinstance(key, str) or not isinstance(raw_path, str):
                raise ValueError("manifest object needs string key and path")
            prior = unique_objects.setdefault(key, item)
            if prior.get("size") != item.get("size"):
                raise ValueError(f"inconsistent size for duplicate object {key}")

    dataset_prefix = source.get("dataset_prefix")
    if not isinstance(dataset_prefix, str) or not dataset_prefix:
        raise ValueError("source manifest has no dataset_prefix")
    if output_dir.exists() and any(output_dir.iterdir()):
        return _reuse_existing(
            manifest_path=manifest_path,
            output_dir=output_dir,
            dataset_prefix=dataset_prefix,
            window_indices=window_indices,
            unique_objects=unique_objects,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for key, item in unique_objects.items():
        if not key.startswith(dataset_prefix):
            raise ValueError(f"object lies outside dataset prefix: {key}")
        source_path = Path(str(item["path"])).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        relative_path = Path(key).relative_to(dataset_prefix)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, destination)
        linked += 1

    for log in selected_logs:
        for item in log["objects"]:
            relative_path = Path(item["key"]).relative_to(dataset_prefix)
            item["path"] = str(output_dir / relative_path)

    frozen_manifest = {
        key: deepcopy(value)
        for key, value in source.items()
        if key not in ("logs", "output_dir", "window_object_references",
                       "total_objects", "total_bytes")
    }
    frozen_manifest.update(
        {
            "output_dir": str(output_dir),
            "logs": selected_logs,
            "window_object_references": sum(
                len(log["objects"]) for log in selected_logs
            ),
            "total_objects": len(unique_objects),
            "total_bytes": sum(
                int(item["size"]) for item in unique_objects.values()
            ),
            "materialized_from": str(manifest_path),
            "materialization": "hardlink",
            "selected_window_indices": list(window_indices),
        }
    )
    destination_manifest = output_dir / "selection_manifest.json"
    destination_manifest.write_text(
        json.dumps(frozen_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(destination_manifest),
        "linked_objects": linked,
        "logical_bytes": frozen_manifest["total_bytes"],
        "selected_window_indices": list(window_indices),
        "reused_existing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window-indices",
        type=parse_indices,
        required=True,
        help="comma-separated zero-based window indices",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.manifest,
                args.output_dir,
                args.window_indices,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
