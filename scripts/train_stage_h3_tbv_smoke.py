#!/usr/bin/env python3
"""Run the bounded two-traversal TbV SplatAD smoke programmatically."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import warnings

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.scripts.train import main as train_main

from stage_h3_tbv_dataparser import TbVDataParserConfig


def build_config(args: argparse.Namespace):
    config = deepcopy(method_configs["splatad"])
    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.timestamp = args.timestamp
    config.vis = "tensorboard"
    config.max_num_iterations = args.iterations
    config.steps_per_save = args.iterations
    config.steps_per_eval_image = max(args.iterations // 2, 1)
    config.steps_per_eval_all_images = 100_000
    config.pipeline.calc_fid_steps = (999_999,)
    config.pipeline.datamanager.max_thread_workers = args.workers
    config.pipeline.datamanager.downsample_factor = args.downsample_factor
    dataparser_kwargs = dict(
        data=args.data,
        train_split_fraction=args.train_split_fraction,
    )
    if args.sequences:
        dataparser_kwargs.update(
            sequences=tuple(args.sequences),
            window_start_seconds=tuple(args.window_start_seconds),
            window_end_seconds=tuple(args.window_end_seconds),
        )
    if args.mask_root is not None:
        dataparser_kwargs["mask_root"] = args.mask_root
    if args.image_mask_root is not None:
        dataparser_kwargs["image_mask_root"] = args.image_mask_root
    if args.rgb_root is not None:
        dataparser_kwargs["rgb_root"] = args.rgb_root
    if args.disable_image_masks:
        dataparser_kwargs["load_image_masks"] = False
    if args.mask_lidar_points:
        dataparser_kwargs["mask_lidar_points"] = True
    if args.lidar_persistence_root is not None:
        dataparser_kwargs["lidar_persistence_root"] = (
            args.lidar_persistence_root
        )
    config.pipeline.datamanager.dataparser = TbVDataParserConfig(
        **dataparser_kwargs
    )
    if (
        args.mask_root is not None or args.image_mask_root is not None
    ) and not args.disable_image_masks:
        from stage_h3_mask_aligned_splatad import MaskAlignedSplatADModel

        config.pipeline.model._target = MaskAlignedSplatADModel
    config.pipeline.model.max_steps = args.iterations
    config.pipeline.model.max_num_seed_points = args.max_num_seed_points
    if args.num_downscales is not None:
        config.pipeline.model.num_downscales = args.num_downscales
    if args.resolution_schedule is not None:
        config.pipeline.model.resolution_schedule = args.resolution_schedule
    if args.steps_per_eval_image is not None:
        config.steps_per_eval_image = args.steps_per_eval_image
    return config


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"av2\.utils\.io",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/yawei/stage3_external/data/tbv_branch_pilot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/yawei/stage3_external/outputs/tbv_h3"),
    )
    parser.add_argument(
        "--experiment-name", default="tbv_branch_pair_splatad_smoke_100"
    )
    parser.add_argument("--timestamp", default="2026-07-22_100step")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--downsample-factor", type=float, default=0.25)
    parser.add_argument("--train-split-fraction", type=float, default=0.9)
    parser.add_argument("--max-num-seed-points", type=int, default=250_000)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--image-mask-root", type=Path)
    parser.add_argument("--disable-image-masks", action="store_true")
    parser.add_argument("--mask-lidar-points", action="store_true")
    parser.add_argument("--lidar-persistence-root", type=Path)
    parser.add_argument("--num-downscales", type=int)
    parser.add_argument("--resolution-schedule", type=int)
    parser.add_argument("--steps-per-eval-image", type=int)
    parser.add_argument("--sequence", dest="sequences", action="append")
    parser.add_argument(
        "--window-start-seconds", action="append", type=float
    )
    parser.add_argument(
        "--window-end-seconds", action="append", type=float
    )
    args = parser.parse_args()
    custom_lengths = tuple(
        len(value or ())
        for value in (
            args.sequences,
            args.window_start_seconds,
            args.window_end_seconds,
        )
    )
    if any(custom_lengths) and (
        not all(custom_lengths) or len(set(custom_lengths)) != 1
    ):
        parser.error(
            "--sequence, --window-start-seconds, and "
            "--window-end-seconds must be repeated equally"
        )
    if args.mask_lidar_points and args.mask_root is None:
        parser.error("--mask-lidar-points requires --mask-root")
    if (
        args.disable_image_masks
        and args.mask_root is None
        and args.image_mask_root is None
    ):
        parser.error(
            "--disable-image-masks requires a mask root to disable"
        )
    if (
        args.lidar_persistence_root is not None
        and not args.mask_lidar_points
    ):
        parser.error(
            "--lidar-persistence-root requires --mask-lidar-points"
        )
    if args.num_downscales is not None and args.num_downscales < 0:
        parser.error("--num-downscales cannot be negative")
    if (
        args.resolution_schedule is not None
        and args.resolution_schedule <= 0
    ):
        parser.error("--resolution-schedule must be positive")
    if (
        args.steps_per_eval_image is not None
        and args.steps_per_eval_image <= 0
    ):
        parser.error("--steps-per-eval-image must be positive")
    train_main(build_config(args))


if __name__ == "__main__":
    main()
