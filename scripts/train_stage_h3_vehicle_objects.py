#!/usr/bin/env python3
"""Train a scene 040 actor-aware rigid-vehicle object-layer pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts import train as train_script

from driving_scene_reconstruction.h3.vehicle_object_layer import (
    ActorAwareMCMCStrategy,
    PandaSetVehicleObjectParser,
    VehicleObjectSplatADModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--max-num-iterations", type=int, default=8000)
    parser.add_argument("--model-max-steps", type=int, default=8000)
    parser.add_argument("--steps-per-save", type=int, default=2000)
    parser.add_argument("--steps-per-eval-image", type=int, default=1000)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument(
        "--moving-only",
        action="store_true",
        help="Keep stationary rigid cuboids in the static background.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainerConfig:
    if not args.source_config.is_file():
        raise FileNotFoundError(args.source_config)
    config = yaml.load(
        args.source_config.read_text(encoding="utf-8"),
        Loader=yaml.Loader,
    )
    if not isinstance(config, TrainerConfig):
        raise TypeError(f"unexpected config type: {type(config)}")

    dataparser = config.pipeline.datamanager.dataparser
    dataparser._target = PandaSetVehicleObjectParser
    dataparser.include_stationary_rigid_actors = not args.moving_only
    dataparser.include_deformable_actors = False

    model = config.pipeline.model
    model._target = VehicleObjectSplatADModel
    model.strategy = "mcmc"

    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.timestamp = args.timestamp
    config.max_num_iterations = args.max_num_iterations
    model.max_steps = args.model_max_steps
    config.steps_per_save = args.steps_per_save
    config.steps_per_eval_image = args.steps_per_eval_image
    config.steps_per_eval_all_images = 100000
    config.save_only_latest_checkpoint = not args.keep_all_checkpoints
    config.load_checkpoint = None
    config.load_dir = None
    config.load_step = None
    config.load_optimizer = True
    config.load_scheduler = True
    return config


def pipeline_model(pipeline: Any) -> Any:
    if hasattr(pipeline, "module"):
        pipeline = pipeline.module
    model = pipeline.model
    return model.module if hasattr(model, "module") else model


def actor_audit(trainer: Any, phase: str) -> dict[str, Any]:
    pipeline = (
        trainer.pipeline.module
        if hasattr(trainer.pipeline, "module")
        else trainer.pipeline
    )
    model = pipeline_model(pipeline)
    ids = model.gauss_params["id"].detach().flatten().round().to(torch.long)
    actor_count = int(model.dynamic_actors.n_actors)
    gaussian_counts = {
        str(actor_id): int((ids == actor_id).sum())
        for actor_id in range(actor_count)
    }
    trajectories = pipeline.datamanager.train_dataparser_outputs.metadata[
        "trajectories"
    ]
    label_counts = Counter(str(item["label"]) for item in trajectories)
    stationary_counts = Counter(
        str(item["label"])
        for item in trajectories
        if bool(item["stationary"])
    )
    moving_counts = Counter(
        str(item["label"])
        for item in trajectories
        if not bool(item["stationary"])
    )
    return {
        "phase": phase,
        "actor_count": actor_count,
        "trajectory_count": len(trajectories),
        "stationary_trajectory_count": sum(stationary_counts.values()),
        "moving_trajectory_count": sum(moving_counts.values()),
        "trajectory_labels": dict(sorted(label_counts.items())),
        "stationary_trajectory_labels": dict(sorted(stationary_counts.items())),
        "moving_trajectory_labels": dict(sorted(moving_counts.items())),
        "actor_gaussians": gaussian_counts,
        "actors_with_zero_gaussians": [
            actor_id
            for actor_id, count in gaussian_counts.items()
            if count == 0
        ],
        "actor_gaussians_min": min(gaussian_counts.values(), default=0),
        "actor_gaussians_total": sum(gaussian_counts.values()),
        "background_gaussians": int((ids == actor_count).sum()),
        "total_gaussians": int(ids.numel()),
        "strategy": type(model.strategy).__name__,
        "actor_aware_strategy_active": isinstance(
            model.strategy, ActorAwareMCMCStrategy
        ),
    }


def vehicle_train_loop(
    local_rank: int,
    world_size: int,
    config: TrainerConfig,
    global_rank: int = 0,
) -> None:
    train_script._set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()

    audit_path = config.get_base_dir() / "vehicle_object_layer_audit.json"
    audit = {
        "scope": (
            "PandaSet scene 040; "
            + (
                "only moving rigid cuboids use SplatAD actor IDs. "
                if not config.pipeline.datamanager.dataparser.include_stationary_rigid_actors
                else "stationary and moving rigid cuboids use SplatAD actor IDs. "
            )
            + "MCMC relocation is constrained to donors with the same actor ID."
        ),
        "initial": actor_audit(trainer, "initial"),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"VEHICLE_OBJECT_LAYER_INITIALIZED: {audit_path}")

    trainer.train()

    audit["final"] = actor_audit(trainer, "final")
    initial_counts = audit["initial"]["actor_gaussians"]
    final_counts = audit["final"]["actor_gaussians"]
    audit["per_actor_id_preserved"] = (
        set(initial_counts) == set(final_counts)
        and all(final_counts[key] >= initial_counts[key] for key in initial_counts)
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["final"], indent=2, sort_keys=True))
    print(f"VEHICLE_OBJECT_LAYER_AUDIT: {audit_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args)
    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()
    train_script.launch(
        main_func=vehicle_train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


if __name__ == "__main__":
    main()
