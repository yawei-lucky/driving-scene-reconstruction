#!/usr/bin/env python3
"""Run the bounded two-traversal TbV SplatAD smoke programmatically."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

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
    config.pipeline.datamanager.dataparser = TbVDataParserConfig(
        data=args.data,
        train_split_fraction=args.train_split_fraction,
    )
    config.pipeline.model.max_steps = args.iterations
    config.pipeline.model.max_num_seed_points = args.max_num_seed_points
    return config


def main() -> None:
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
    args = parser.parse_args()
    train_main(build_config(args))


if __name__ == "__main__":
    main()
