#!/usr/bin/env python3
"""Download only the selected Stage H3 TbV sensor windows.

The public Argoverse S3 bucket is granular down to individual images and
LiDAR sweeps.  This tool preserves the official per-log directory layout while
selecting a bounded timestamp interval, so an interrupted run can be resumed
without fetching a complete log.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


S3_ENDPOINT = "https://argoverse.s3.amazonaws.com/"
TBV_PREFIX = "datasets/av2/tbv/"
XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)
DEFAULT_OUTPUT = Path("/home/yawei/stage3_external/data/tbv_branch_pilot")


@dataclass(frozen=True)
class Window:
    log_id: str
    start_seconds: float
    end_seconds: float

    @property
    def start_ns(self) -> int:
        return round(self.start_seconds * 1e9)

    @property
    def end_ns(self) -> int:
        return round(self.end_seconds * 1e9)


WINDOWS = (
    Window(
        "OCaNX1bQSmlP3jEQH80C0TZYzZhKLV81__Spring_2020",
        315972566.15,
        315972576.15,
    ),
    Window(
        "QMnNKZiFaxnuGQmxpGkZFdM2EE7uWqDQ__Spring_2020",
        315968138.15,
        315968148.15,
    ),
)


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str


def _fetch_bytes(url: str, *, attempts: int = 3, timeout: float = 60.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "stage-h3-tbv-window/1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # network errors vary by Python release
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"failed to read {url}") from last_error


def _listing_url(prefix: str, continuation: str | None = None) -> str:
    parameters = {"list-type": "2", "prefix": prefix}
    if continuation:
        parameters["continuation-token"] = continuation
    return S3_ENDPOINT + "?" + urllib.parse.urlencode(parameters)


def parse_object_page(xml_bytes: bytes) -> tuple[list[S3Object], str | None]:
    root = ET.fromstring(xml_bytes)
    objects = []
    for content in root.findall("s3:Contents", XML_NAMESPACE):
        key = content.findtext("s3:Key", default="", namespaces=XML_NAMESPACE)
        size = int(content.findtext("s3:Size", default="0", namespaces=XML_NAMESPACE))
        etag = content.findtext("s3:ETag", default="", namespaces=XML_NAMESPACE)
        objects.append(S3Object(key=key, size=size, etag=etag.strip('"')))
    continuation = root.findtext(
        "s3:NextContinuationToken", namespaces=XML_NAMESPACE
    )
    return objects, continuation


def list_objects(prefix: str) -> list[S3Object]:
    objects: list[S3Object] = []
    continuation: str | None = None
    while True:
        page, continuation = parse_object_page(
            _fetch_bytes(_listing_url(prefix, continuation))
        )
        objects.extend(page)
        if not continuation:
            return objects


def timestamp_ns(obj: S3Object) -> int | None:
    try:
        return int(Path(obj.key).stem)
    except ValueError:
        return None


def select_sensor_objects(
    objects: Iterable[S3Object], window: Window, *, stride: int = 1
) -> list[S3Object]:
    selected = [
        obj
        for obj in objects
        if (stamp := timestamp_ns(obj)) is not None
        and window.start_ns <= stamp <= window.end_ns
    ]
    selected.sort(key=lambda obj: int(Path(obj.key).stem))
    return selected[::stride]


def plan_window(window: Window, *, camera_stride: int) -> list[S3Object]:
    root = f"{TBV_PREFIX}{window.log_id}/"
    selected = list_objects(root + "calibration/")
    selected.extend(list_objects(root + "map/"))
    selected.extend(list_objects(root + "city_SE3_egovehicle.feather"))
    selected.extend(
        select_sensor_objects(list_objects(root + "sensors/lidar/"), window)
    )
    for camera in CAMERAS:
        selected.extend(
            select_sensor_objects(
                list_objects(root + f"sensors/cameras/{camera}/"),
                window,
                stride=camera_stride,
            )
        )
    return sorted({obj.key: obj for obj in selected}.values(), key=lambda obj: obj.key)


def local_path(output_dir: Path, obj: S3Object) -> Path:
    relative = Path(obj.key).relative_to(TBV_PREFIX)
    return output_dir / relative


def download_object(output_dir: Path, obj: S3Object) -> str:
    destination = local_path(output_dir, obj)
    if destination.is_file() and destination.stat().st_size == obj.size:
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = S3_ENDPOINT + urllib.parse.quote(obj.key, safe="/")
    payload = _fetch_bytes(url, timeout=120.0)
    if len(payload) != obj.size:
        raise RuntimeError(
            f"size mismatch for {obj.key}: expected {obj.size}, got {len(payload)}"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return "downloaded"


def build_manifest(
    plans: dict[Window, list[S3Object]], output_dir: Path, camera_stride: int
) -> dict[str, object]:
    logs = []
    for window, objects in plans.items():
        counts = {camera: 0 for camera in CAMERAS}
        lidar_count = 0
        for obj in objects:
            for camera in CAMERAS:
                if f"/sensors/cameras/{camera}/" in obj.key:
                    counts[camera] += 1
            if "/sensors/lidar/" in obj.key:
                lidar_count += 1
        logs.append(
            {
                "log_id": window.log_id,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "camera_stride": camera_stride,
                "camera_counts": counts,
                "lidar_count": lidar_count,
                "object_count": len(objects),
                "bytes": sum(obj.size for obj in objects),
                "objects": [
                    {
                        "key": obj.key,
                        "size": obj.size,
                        "etag": obj.etag,
                        "path": str(local_path(output_dir, obj)),
                    }
                    for obj in objects
                ],
            }
        )
    return {
        "schema_version": 1,
        "source": S3_ENDPOINT,
        "dataset_prefix": TBV_PREFIX,
        "output_dir": str(output_dir),
        "logs": logs,
        "total_objects": sum(len(objects) for objects in plans.values()),
        "total_bytes": sum(
            obj.size for objects in plans.values() for obj in objects
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--camera-stride",
        type=int,
        default=2,
        help="keep every Nth camera frame independently for each camera",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.camera_stride < 1:
        parser.error("--camera-stride must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    plans = {
        window: plan_window(window, camera_stride=args.camera_stride)
        for window in WINDOWS
    }
    manifest = build_manifest(plans, args.output_dir, args.camera_stride)
    print(
        json.dumps(
            {
                "total_objects": manifest["total_objects"],
                "total_bytes": manifest["total_bytes"],
                "logs": [
                    {
                        key: log[key]
                        for key in (
                            "log_id",
                            "camera_counts",
                            "lidar_count",
                            "object_count",
                            "bytes",
                        )
                    }
                    for log in manifest["logs"]
                ],
            },
            indent=2,
        )
    )
    if args.plan_only:
        return

    results = {"downloaded": 0, "reused": 0}
    objects = [obj for plan in plans.values() for obj in plan]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_object, args.output_dir, obj): obj
            for obj in objects
        }
        for future in as_completed(futures):
            results[future.result()] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **results}, indent=2))


if __name__ == "__main__":
    main()
