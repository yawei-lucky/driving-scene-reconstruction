#!/usr/bin/env python3
"""Resume a pinned run with exact optimizer and scheduler state.

The pinned Trainer loads a checkpoint before it constructs optimizers, but its
checkpoint loader also tries to restore optimizer state at that point. This
thin entrypoint preserves the intended ordering without modifying the pinned
checkout:

1. construct the pipeline and restore its model/global-step state;
2. construct optimizers against the checkpoint-shaped parameters;
3. restore optimizer and scheduler state from the same checkpoint;
4. verify the restored learning rates and scheduler epochs; then train.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts import train as train_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--additional-iterations", type=int, required=True)
    parser.add_argument("--model-max-steps", type=int, required=True)
    parser.add_argument("--steps-per-save", type=int, default=2000)
    parser.add_argument("--steps-per-eval-image", type=int, default=1000)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(config: TrainerConfig) -> Path:
    if config.load_checkpoint is not None:
        return config.load_checkpoint
    if config.load_dir is None:
        raise RuntimeError("exact resume requires load_checkpoint or load_dir")
    if config.load_step is None:
        steps = sorted(
            int(path.stem.split("-")[-1])
            for path in config.load_dir.glob("step-*.ckpt")
        )
        if not steps:
            raise FileNotFoundError(
                f"no checkpoints found under {config.load_dir}"
            )
        step = steps[-1]
    else:
        step = config.load_step
    return config.load_dir / f"step-{step:09d}.ckpt"


def optimizer_learning_rates(
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, list[float]]:
    return {
        name: [float(group["lr"]) for group in optimizer.param_groups]
        for name, optimizer in optimizers.items()
    }


def expected_optimizer_learning_rates(
    states: dict[str, dict[str, Any]],
) -> dict[str, list[float]]:
    return {
        name: [float(group["lr"]) for group in state["param_groups"]]
        for name, state in states.items()
    }


def assert_learning_rates_equal(
    actual: dict[str, list[float]],
    expected: dict[str, list[float]],
) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"optimizer groups differ: {sorted(actual)} != {sorted(expected)}"
        )
    for name in actual:
        if len(actual[name]) != len(expected[name]):
            raise RuntimeError(f"optimizer param-group count differs for {name}")
        for observed, wanted in zip(actual[name], expected[name]):
            if not math.isclose(observed, wanted, rel_tol=1e-12, abs_tol=1e-15):
                raise RuntimeError(
                    f"optimizer learning rate differs for {name}: "
                    f"{observed} != {wanted}"
                )


def exact_resume_train_loop(
    local_rank: int,
    world_size: int,
    config: TrainerConfig,
    global_rank: int = 0,
) -> None:
    train_script._set_random_seed(config.machine.seed + global_rank)
    checkpoint_path = resolve_checkpoint(config)
    restore_optimizer = config.load_optimizer
    restore_scheduler = config.load_scheduler
    if not restore_optimizer or not restore_scheduler:
        raise RuntimeError(
            "exact resume requires load_optimizer=True and load_scheduler=True"
        )

    # The pinned Trainer restores pipeline/global-step/scaler successfully when
    # these flags are temporarily disabled, then constructs checkpoint-shaped
    # optimizers. Restore the two flags before training for an honest config.
    config.load_optimizer = False
    config.load_scheduler = False
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    config.load_optimizer = restore_optimizer
    config.load_scheduler = restore_scheduler

    # Trainer.setup() has released its temporary CPU checkpoint after restoring
    # the pipeline. Load it once more only now, avoiding two simultaneous
    # checkpoint copies during the memory-heavy pipeline construction.
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_step = int(loaded_state["step"])
    if trainer._start_step != checkpoint_step + 1:
        raise RuntimeError(
            f"global step mismatch: start {trainer._start_step}, "
            f"checkpoint {checkpoint_step}"
        )
    trainer.optimizers.load_optimizers(loaded_state["optimizers"])
    if "schedulers" not in loaded_state:
        raise RuntimeError("checkpoint does not contain scheduler state")
    trainer.optimizers.load_schedulers(loaded_state["schedulers"])

    actual_lrs = optimizer_learning_rates(trainer.optimizers.optimizers)
    expected_lrs = expected_optimizer_learning_rates(loaded_state["optimizers"])
    assert_learning_rates_equal(actual_lrs, expected_lrs)
    scheduler_epochs = {
        name: int(scheduler.state_dict()["last_epoch"])
        for name, scheduler in trainer.optimizers.schedulers.items()
    }
    expected_scheduler_epochs = {
        name: int(state["last_epoch"])
        for name, state in loaded_state["schedulers"].items()
    }
    if scheduler_epochs != expected_scheduler_epochs:
        raise RuntimeError(
            "scheduler last_epoch differs after restore: "
            f"{scheduler_epochs} != {expected_scheduler_epochs}"
        )

    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "training_start_step": trainer._start_step,
        "optimizer_learning_rates": actual_lrs,
        "scheduler_last_epoch": scheduler_epochs,
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "scope_note": (
            "Optimizer/scheduler/model/global-step state is exact. The source "
            "checkpoint does not preserve RNG or dataloader state, so resumed "
            "sampling is not bitwise identical to uninterrupted training."
        ),
    }
    audit_path = config.get_base_dir() / "exact_resume_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"EXACT_RESUME_VERIFIED: {audit_path}")
    trainer.train()


def load_config(args: argparse.Namespace) -> TrainerConfig:
    if not args.source_config.is_file():
        raise FileNotFoundError(args.source_config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    config = yaml.load(
        args.source_config.read_text(encoding="utf-8"),
        Loader=yaml.Loader,
    )
    if not isinstance(config, TrainerConfig):
        raise TypeError(f"unexpected config type: {type(config)}")

    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.timestamp = args.timestamp
    config.max_num_iterations = args.additional_iterations
    config.pipeline.model.max_steps = args.model_max_steps
    config.steps_per_save = args.steps_per_save
    config.steps_per_eval_image = args.steps_per_eval_image
    config.steps_per_eval_all_images = 100000
    config.save_only_latest_checkpoint = not args.keep_all_checkpoints
    config.load_checkpoint = args.checkpoint
    config.load_dir = None
    config.load_step = None
    config.load_optimizer = True
    config.load_scheduler = True
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args)
    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()
    train_script.launch(
        main_func=exact_resume_train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


if __name__ == "__main__":
    main()
