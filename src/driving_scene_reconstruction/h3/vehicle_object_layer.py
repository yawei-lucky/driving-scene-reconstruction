"""PandaSet vehicle-object extensions for the Stage H3 SplatAD pilot.

The pinned PandaSet parser assigns only moving cuboids to SplatAD actor layers.
That leaves parked vehicles in the static background even though they dominate
several of scene 040's near-camera failures. This module keeps the upstream
parser and model interfaces while changing two narrowly scoped behaviors:

* stationary rigid cuboids are also emitted as actor trajectories;
* MCMC relocation samples within each actor ID, so refinement cannot recycle
  every Gaussian of one actor into another actor or the background.

The module is loaded only by the H3 environment and does not add dependencies
to the repository's lightweight simulator package.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import (
    _multinomial_sample,
    _update_param_with_optimizer,
)
from torch import Tensor

from nerfstudio.data.dataparsers import pandaset_dataparser as pandaset_parser
from nerfstudio.model_components.strategy import ADMCMCStrategy
from nerfstudio.models.splatad import SplatADModel


class PandaSetVehicleObjectParser(pandaset_parser.PandaSet):
    """PandaSet parser that includes stationary rigid actors when requested."""

    def _get_actor_trajectories(self) -> list[Dict]:
        allowed_classes = tuple(pandaset_parser.ALLOWED_RIGID_CLASSES)
        if self.config.include_deformable_actors:
            allowed_classes += tuple(pandaset_parser.ALLOWED_DEFORMABLE_CLASSES)

        include_stationary = bool(
            getattr(self.config, "include_stationary_rigid_actors", False)
        )
        cuboids = []
        for frame_index in range(pandaset_parser.PANDASET_SEQ_LEN):
            current = self.sequence.cuboids[frame_index]
            labels = np.asarray(current["label"])
            stationary_mask = np.asarray(current["stationary"], dtype=np.bool_)
            allowed_mask = np.asarray(
                [label in allowed_classes for label in labels]
            )
            stationary_rigid_mask = stationary_mask & np.asarray(
                [
                    label in pandaset_parser.ALLOWED_RIGID_CLASSES
                    for label in labels
                ]
            )
            valid_mask = ((~stationary_mask) & allowed_mask) | (
                include_stationary & stationary_rigid_mask
            )
            current = current[valid_mask]
            if not len(current):
                continue

            uuid = np.asarray(current["uuid"])
            label = np.asarray(current["label"])
            yaw = current["yaw"].astype(np.float32)
            rotation = pandaset_parser._yaw_to_rotation_matrix(yaw)
            stationary = np.asarray(current["stationary"], dtype=np.bool_)
            position = np.vstack(
                [
                    current["position.x"].astype(np.float32),
                    current["position.y"].astype(np.float32),
                    current["position.z"].astype(np.float32),
                ]
            ).T
            poses = np.eye(4)[None].repeat(len(uuid), axis=0)
            poses[:, :3, :3] = rotation
            poses[:, :3, 3] = position
            dims = np.vstack(
                [
                    current["dimensions.x"].astype(np.float32),
                    current["dimensions.y"].astype(np.float32),
                    current["dimensions.z"].astype(np.float32),
                ]
            ).T
            sensor_id = np.asarray(
                current["cuboids.sensor_id"], dtype=np.int32
            )
            sibling_id = np.asarray(current["cuboids.sibling_id"])

            if self.config.correct_cuboid_time:
                lidar_pose = pandaset_parser._pandaset_pose_to_matrix(
                    self.sequence.lidar.poses[frame_index]
                )
                position_in_lidar = (
                    position @ lidar_pose[:3, :3].T + lidar_pose[:3, 3]
                )
                angle = (
                    np.arctan2(
                        position_in_lidar[:, 0], position_in_lidar[:, 1]
                    )
                    - np.pi / 2
                )
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                time_difference = (
                    angle
                    / (2 * np.pi)
                    * np.diff(self.sequence.lidar.timestamps).mean()
                )
                cuboid_times = (
                    self.sequence.camera["front_camera"].timestamps[frame_index]
                    + time_difference
                )
            else:
                cuboid_times = np.repeat(
                    self.sequence.camera["front_camera"].timestamps[frame_index],
                    len(uuid),
                )

            for cuboid_index in range(len(uuid)):
                cuboids.append(
                    {
                        "uuid": uuid[cuboid_index],
                        "label": label[cuboid_index],
                        "poses": poses[cuboid_index],
                        "stationary": stationary[cuboid_index],
                        "dims": dims[cuboid_index],
                        "sensor_ids": sensor_id[cuboid_index],
                        "sibling_id": (
                            sibling_id[cuboid_index]
                            if sensor_id[cuboid_index] != -1
                            else None
                        ),
                        "timestamps": np.asarray(
                            cuboid_times[cuboid_index]
                        ),
                    }
                )

        return pandaset_parser._cuboids_to_trajectories(cuboids)


class ActorAwareMCMCStrategy(ADMCMCStrategy):
    """MCMC relocation that preserves the Gaussian count of every actor ID."""

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[
            Dict[str, torch.nn.Parameter], torch.nn.ParameterDict
        ],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        optimizers_prefix: str = "",
    ) -> int:
        if "id" not in params:
            return super()._relocate_gs(
                params, optimizers, binoms, optimizers_prefix
            )

        opacities = torch.sigmoid(params["opacities"]).flatten()
        original_dead_mask = opacities <= self.min_opacity
        ids = params["id"].detach().flatten().round().to(torch.long)
        dead_indices_by_id: list[Tensor] = []
        sampled_indices_by_id: list[Tensor] = []

        for actor_id in ids.unique(sorted=True):
            group_indices = torch.where(ids == actor_id)[0]
            group_dead = group_indices[original_dead_mask[group_indices]]
            group_alive = group_indices[~original_dead_mask[group_indices]]

            # If an entire group is below threshold, retain its strongest point
            # as an anchor. This makes even a completely faded actor recoverable.
            if group_alive.numel() == 0:
                anchor_offset = torch.argmax(opacities[group_indices])
                anchor = group_indices[anchor_offset].reshape(1)
                group_alive = anchor
                group_dead = group_indices[group_indices != anchor]

            if group_dead.numel() == 0:
                continue
            weights = opacities[group_alive].clamp_min(
                torch.finfo(torch.float32).eps
            )
            sampled_offsets = _multinomial_sample(
                weights, int(group_dead.numel()), replacement=True
            )
            dead_indices_by_id.append(group_dead)
            sampled_indices_by_id.append(group_alive[sampled_offsets])

        if not dead_indices_by_id:
            return 0

        dead_indices = torch.cat(dead_indices_by_id)
        sampled_indices = torch.cat(sampled_indices_by_id)
        eps = torch.finfo(torch.float32).eps
        sample_ratios = (
            torch.bincount(
                sampled_indices, minlength=len(params["means"])
            )[sampled_indices]
            + 1
        )
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(params["opacities"])[sampled_indices],
            scales=torch.exp(params["scales"])[sampled_indices],
            ratios=sample_ratios,
            binoms=binoms,
        )
        new_opacities = torch.clamp(
            new_opacities, max=1.0 - eps, min=self.min_opacity
        )

        def parameter_update(name: str, parameter: Tensor) -> Tensor:
            requires_grad = parameter.requires_grad
            if name == "opacities":
                parameter[sampled_indices] = torch.logit(new_opacities)
            elif name == "scales":
                parameter[sampled_indices] = torch.log(new_scales)
            parameter[dead_indices] = parameter[sampled_indices]
            return torch.nn.Parameter(
                parameter, requires_grad=requires_grad
            )

        def optimizer_update(_key: str, value: Tensor) -> Tensor:
            value[sampled_indices] = 0
            return value

        unique_before, counts_before = torch.unique(
            ids, sorted=True, return_counts=True
        )
        _update_param_with_optimizer(
            parameter_update,
            optimizer_update,
            params,
            optimizers,
            optimizers_prefix=optimizers_prefix,
        )
        unique_after, counts_after = torch.unique(
            params["id"].detach().flatten().round().to(torch.long),
            sorted=True,
            return_counts=True,
        )
        if not (
            torch.equal(unique_after, unique_before)
            and torch.equal(counts_after, counts_before)
        ):
            raise RuntimeError(
                "actor-aware MCMC relocation changed per-ID Gaussian counts"
            )
        return int(dead_indices.numel())


class VehicleObjectSplatADModel(SplatADModel):
    """SplatAD model using actor-aware MCMC refinement."""

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.strategy != "mcmc":
            return
        self.strategy = ActorAwareMCMCStrategy(
            cap_max=self.config.mcmc_cap_max,
            noise_lr=self.config.mcmc_noise_lr,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            refine_every=self.config.refine_every,
            min_opacity=self.config.mcmc_min_opacity,
            verbose=self.config.verbose,
        )
        self.strategy_state = self.strategy.initialize_state()
