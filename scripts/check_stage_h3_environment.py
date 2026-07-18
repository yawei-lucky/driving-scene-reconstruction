#!/usr/bin/env python3
"""Validate the pinned Stage H3 environment without requiring PandaSet data."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


EXPECTED_VERSIONS = {
    "dataclass-wizard": "0.35.1",
    "gsplat": "1.0.0",
    "neurad-studio": "0.1.0",
    "numpy": "1.24.4",
    "pandaset": "0.3.dev0",
    "tinycudann": "1.6",
    "torch": "2.0.1+cu118",
    "torchvision": "0.15.2+cu118",
    "viser": "0.1.28",
}

EXPECTED_COMMITS = {
    "neurad-studio": "e6f7e4e509b828a952d8584b7165f7844711ecb2",
    "pandaset-devkit": "59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0",
    "splatad-gsplat": "6e31ad766d39e0c33f9034a2ed772d51364b2343",
    "viser-neurad": "57142e42df8edd4de33fd60a08d6bb6c35970aa1",
}

EXPECTED_PACKAGE_SOURCES = {
    "neurad-studio": ("neurad-studio", EXPECTED_COMMITS["neurad-studio"]),
    "pandaset": ("pandaset-devkit", EXPECTED_COMMITS["pandaset-devkit"]),
    "gsplat": ("splatad-gsplat", EXPECTED_COMMITS["splatad-gsplat"]),
    "viser": ("viser-neurad", EXPECTED_COMMITS["viser-neurad"]),
}

EXPECTED_TCNN_COMMIT = "8e6e242f36dd197134c9b9275a8e5108a8e3af78"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def check_versions() -> None:
    print("Python packages:")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        print(f"  {package}: {actual}")
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")


def check_commits(code_root: Path) -> None:
    print("Pinned source commits:")
    for directory, expected in EXPECTED_COMMITS.items():
        path = code_root / directory
        if not path.is_dir():
            raise RuntimeError(f"missing source checkout: {path}")
        actual = git_head(path)
        print(f"  {directory}: {actual}")
        if actual != expected:
            raise RuntimeError(f"{directory}: expected {expected}, found {actual}")


def direct_url(package: str) -> dict[str, object]:
    value = metadata.distribution(package).read_text("direct_url.json")
    if value is None:
        raise RuntimeError(f"{package} has no direct_url.json source record")
    return json.loads(value)


def check_package_sources(code_root: Path) -> None:
    print("Installed package sources:")
    for package, (directory, expected_commit) in EXPECTED_PACKAGE_SOURCES.items():
        source = direct_url(package)
        vcs_info = source.get("vcs_info")
        if isinstance(vcs_info, dict):
            actual_commit = vcs_info.get("commit_id")
            print(f"  {package}: VCS commit {actual_commit}")
            if actual_commit != expected_commit:
                raise RuntimeError(
                    f"{package}: expected source commit {expected_commit}, "
                    f"found {actual_commit}"
                )
            continue

        parsed = urlparse(str(source.get("url")))
        actual_path = Path(unquote(parsed.path)).resolve()
        expected_path = (code_root / directory).resolve()
        if package == "pandaset":
            expected_path /= "python"
        print(f"  {package}: local source {actual_path}")
        if actual_path != expected_path:
            raise RuntimeError(
                f"{package}: expected local source {expected_path}, "
                f"found {actual_path}"
            )

    tcnn_source = direct_url("tinycudann")
    tcnn_vcs = tcnn_source.get("vcs_info")
    tcnn_commit = tcnn_vcs.get("commit_id") if isinstance(tcnn_vcs, dict) else None
    print(f"  tinycudann: VCS commit {tcnn_commit}")
    if tcnn_commit != EXPECTED_TCNN_COMMIT:
        raise RuntimeError(
            f"tinycudann: expected source commit {EXPECTED_TCNN_COMMIT}, "
            f"found {tcnn_commit}"
        )


def check_entrypoints() -> None:
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.pandaset_dataparser import (
        AVAILABLE_CAMERAS,
        PANDASET_SEQ_LEN,
        PandaSetDataParserConfig,
    )
    from pandaset import DataSet

    del DataSet
    if not {"splatad", "neurad"}.issubset(method_configs):
        raise RuntimeError("SplatAD or NeuRAD is not registered")

    config = PandaSetDataParserConfig()
    expected_cameras = {
        "front",
        "front_left",
        "front_right",
        "back",
        "left",
        "right",
    }
    if set(AVAILABLE_CAMERAS) != expected_cameras:
        raise RuntimeError(f"unexpected PandaSet cameras: {AVAILABLE_CAMERAS}")
    if config.lidars != ("Pandar64",):
        raise RuntimeError(f"unexpected default PandaSet LiDAR: {config.lidars}")
    if PANDASET_SEQ_LEN != 80:
        raise RuntimeError(f"unexpected PandaSet sequence length: {PANDASET_SEQ_LEN}")

    print("Entrypoints:")
    print("  methods: splatad, neurad")
    print("  dataparser: pandaset-data")
    print(f"  cameras: {', '.join(AVAILABLE_CAMERAS)}")
    print(f"  default LiDAR: {config.lidars[0]}")
    print(f"  frames per sequence: {PANDASET_SEQ_LEN}")


def check_tinycudann(torch: object) -> None:
    import tinycudann as tcnn

    network = tcnn.Network(
        n_input_dims=3,
        n_output_dims=4,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 1,
        },
    )
    output = network(torch.rand(128, 3, device="cuda"))
    if output.shape != (128, 4) or not torch.isfinite(output).all():
        raise RuntimeError("tiny-cuda-nn produced an invalid result")
    print(f"  tiny-cuda-nn forward: {tuple(output.shape)}, finite")


def check_camera_rasterization(torch: object) -> None:
    from gsplat.rendering import rasterization

    count = 256
    means = torch.rand(count, 3, device="cuda")
    means[:, 2] += 1.0
    quats = torch.randn(count, 4, device="cuda")
    scales = torch.rand(count, 3, device="cuda") * 0.08 + 0.02
    opacities = torch.rand(count, device="cuda") * 0.5 + 0.5
    velocities = torch.zeros(count, 3, device="cuda")
    colors = torch.rand(count, 3, device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    intrinsics = torch.tensor(
        [[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    render, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        velocities=velocities,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=64,
    )
    if render.shape != (1, 64, 64, 3) or alpha.shape != (1, 64, 64, 1):
        raise RuntimeError("camera rasterizer returned an unexpected shape")
    if not torch.isfinite(render).all() or not torch.isfinite(alpha).all():
        raise RuntimeError("camera rasterizer produced a non-finite result")
    print(
        "  camera rasterizer: "
        f"{tuple(render.shape)}, alpha_max={alpha.max().item():.4f}"
    )


def check_lidar_rasterization(torch: object) -> None:
    from gsplat.rendering import lidar_rasterization

    count = 256
    min_azimuth, max_azimuth = -16.0, 16.0
    min_elevation, max_elevation = -4.0, 4.0
    elevation_channels = 8
    azimuth_resolution = 1.0
    width = math.ceil((max_azimuth - min_azimuth) / azimuth_resolution)
    tile_width, tile_height = 8, 4

    means = torch.rand(count, 3, device="cuda")
    quats = torch.randn(count, 4, device="cuda")
    scales = torch.rand(count, 3, device="cuda") * 0.08 + 0.02
    opacities = torch.rand(count, device="cuda") * 0.5 + 0.5
    velocities = torch.zeros(count, 3, device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    features = torch.rand(1, count, 4, device="cuda")

    azimuth = torch.linspace(
        min_azimuth + azimuth_resolution / 2,
        max_azimuth - azimuth_resolution / 2,
        width,
        device="cuda",
    )
    elevation_step = (max_elevation - min_elevation) / elevation_channels
    elevation = torch.linspace(
        min_elevation + elevation_step / 2,
        max_elevation - elevation_step / 2,
        elevation_channels,
        device="cuda",
    )
    raster_points = torch.stack(
        torch.meshgrid(elevation, azimuth, indexing="ij"),
        dim=-1,
    )[..., [1, 0]]
    ranges = torch.full((elevation_channels, width, 1), 4.0, device="cuda")
    times = torch.zeros(elevation_channels, width, 1, device="cuda")
    raster_points = torch.cat((raster_points, ranges, times), dim=-1)[None]
    tile_boundaries = torch.linspace(
        min_elevation,
        max_elevation,
        math.ceil(elevation_channels / tile_height) + 1,
        device="cuda",
    )

    render, alpha, alpha_until, _ = lidar_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        lidar_features=features,
        velocities=velocities,
        viewmats=viewmats,
        raster_pts=raster_points,
        tile_elevation_boundaries=tile_boundaries,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        n_elevation_channels=elevation_channels,
        azimuth_resolution=azimuth_resolution,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    expected_render_shape = (1, elevation_channels, width, 5)
    if render.shape != expected_render_shape:
        raise RuntimeError(f"LiDAR rasterizer shape: {render.shape}")
    if alpha.shape != (1, elevation_channels, width, 1):
        raise RuntimeError(f"LiDAR alpha shape: {alpha.shape}")
    if alpha_until is None or alpha_until.shape != alpha.shape:
        raise RuntimeError("LiDAR alpha-until output is missing or malformed")
    if not all(torch.isfinite(value).all() for value in (render, alpha, alpha_until)):
        raise RuntimeError("LiDAR rasterizer produced a non-finite result")
    print(
        "  LiDAR rasterizer: "
        f"{tuple(render.shape)}, alpha_max={alpha.max().item():.4f}"
    )


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA device")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device_name = torch.cuda.get_device_name(0)
    print("GPU execution:")
    print(f"  device: {device_name}")
    print(f"  PyTorch CUDA runtime: {torch.version.cuda}")
    check_tinycudann(torch)
    check_camera_rasterization(torch)
    check_lidar_rasterization(torch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h3-root",
        type=Path,
        default=Path(os.environ.get("H3_ROOT", "/home/yawei/stage3_external")),
    )
    args = parser.parse_args()

    check_versions()
    check_commits(args.h3_root / "code")
    check_package_sources(args.h3_root / "code")
    check_entrypoints()
    check_gpu()
    print("PASS: Stage H3 environment and synthetic GPU path are ready.")


if __name__ == "__main__":
    main()
