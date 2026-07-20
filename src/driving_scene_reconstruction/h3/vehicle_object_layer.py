"""PandaSet vehicle-object extensions for the Stage H3 SplatAD pilot.

The pinned PandaSet parser assigns only moving cuboids to SplatAD actor layers.
That leaves parked vehicles in the static background even though they dominate
several of scene 040's near-camera failures. This module keeps the upstream
parser and model interfaces while changing two narrowly scoped behaviors:

* stationary rigid cuboids are also emitted as actor trajectories;
* MCMC relocation samples within each actor ID, so refinement cannot recycle
  every Gaussian of one actor into another actor or the background;
* actor-local MCMC noise is bounded by each padded cuboid;
* calibrated LiDAR-frame cuboid timing is an explicit, serialized option.

The module is loaded only by the H3 environment and does not add dependencies
to the repository's lightweight simulator package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type, Union

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
        use_calibrated_lidar_frame = bool(
            getattr(
                self.config,
                "use_calibrated_lidar_frame_for_cuboid_time",
                False,
            )
        )
        if (
            self.config.correct_cuboid_time
            and use_calibrated_lidar_frame
            and not hasattr(self, "extrinsics")
        ):
            with open(
                pandaset_parser.EXTRINSICS_FILE_PATH,
                encoding="utf-8",
            ) as stream:
                self.extrinsics = pandaset_parser.yaml.load(
                    stream,
                    Loader=pandaset_parser.yaml.FullLoader,
                )
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
                if use_calibrated_lidar_frame:
                    # PandaSet cuboid centers are in world coordinates. The
                    # upstream parser already avoids sequence.lidar.poses when
                    # it loads points because those poses are unreliable.
                    # Rebuild the same lidar-to-world transform from the
                    # synchronized front camera, then invert it before
                    # computing lidar azimuth.
                    front_camera = self.sequence.camera["front_camera"]
                    front_camera_to_world = (
                        pandaset_parser._pandaset_pose_to_matrix(
                            front_camera.poses[frame_index]
                        )
                    )
                    front_transform = self.extrinsics["front_camera"][
                        "extrinsic"
                    ]["transform"]
                    lidar_to_front_camera = (
                        pandaset_parser._pandaset_pose_to_matrix(
                            {
                                "position": front_transform["translation"],
                                "heading": front_transform["rotation"],
                            }
                        )
                    )
                    lidar_to_world = (
                        front_camera_to_world @ lidar_to_front_camera
                    )
                    world_to_lidar = np.linalg.inv(lidar_to_world)
                    homogeneous_positions = np.concatenate(
                        [
                            position,
                            np.ones((len(position), 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    position_in_lidar = (
                        homogeneous_positions @ world_to_lidar.T
                    )[:, :3]
                else:
                    # Preserve the pinned upstream behavior for checkpoints
                    # trained before the calibrated-frame fix was introduced.
                    lidar_pose = pandaset_parser._pandaset_pose_to_matrix(
                        self.sequence.lidar.poses[frame_index]
                    )
                    position_in_lidar = (
                        position @ lidar_pose[:3, :3].T
                        + lidar_pose[:3, 3]
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


@dataclass
class PandaSetVehicleObjectParserConfig(
    pandaset_parser.PandaSetDataParserConfig
):
    """Explicit, checkpoint-serialized options for vehicle actor parsing."""

    _target: Type = field(
        default_factory=lambda: PandaSetVehicleObjectParser
    )
    include_stationary_rigid_actors: bool = False
    use_calibrated_lidar_frame_for_cuboid_time: bool = False


class ActorAwareMCMCStrategy(ADMCMCStrategy):
    """MCMC refinement that preserves actor IDs and actor-local geometry."""

    actor_bounds: Tensor | None = None
    last_constraint_stats: dict[str, int] | None = None

    def configure_actor_bounds(self, actor_bounds: Tensor) -> None:
        self.actor_bounds = actor_bounds.detach().clone()
        self.last_constraint_stats = {
            "actor_gaussians": 0,
            "projected_back_inside": 0,
        }

    @torch.no_grad()
    def _constrain_actor_means(
        self,
        params: Union[
            Dict[str, torch.nn.Parameter], torch.nn.ParameterDict
        ],
        optimizers: Dict[str, torch.optim.Optimizer],
        optimizers_prefix: str = "",
    ) -> None:
        if self.actor_bounds is None or "id" not in params:
            return
        ids = params["id"].detach().flatten().round().to(torch.long)
        actor_mask = (ids >= 0) & (ids < len(self.actor_bounds))
        actor_indices = torch.where(actor_mask)[0]
        if actor_indices.numel() == 0:
            return

        bounds = self.actor_bounds.to(
            device=params["means"].device,
            dtype=params["means"].dtype,
        )
        point_bounds = bounds[ids[actor_indices]] * 0.999
        current = params["means"][actor_indices]
        constrained = torch.maximum(
            torch.minimum(current, point_bounds),
            -point_bounds,
        )
        projected = (constrained != current).any(dim=-1)
        projected_indices = actor_indices[projected]
        params["means"][actor_indices] = constrained

        optimizer = optimizers.get(f"{optimizers_prefix}means")
        if optimizer is not None and projected_indices.numel():
            parameter = optimizer.param_groups[0]["params"][0]
            for value in optimizer.state.get(parameter, {}).values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == len(params["means"])
                ):
                    value[projected_indices] = 0

        self.last_constraint_stats = {
            "actor_gaussians": int(actor_indices.numel()),
            "projected_back_inside": int(projected_indices.numel()),
        }

    def step_post_backward(
        self,
        params: Union[
            Dict[str, torch.nn.Parameter], torch.nn.ParameterDict
        ],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
        optimizers_prefix: str = "",
    ) -> None:
        super().step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            lr=lr,
            optimizers_prefix=optimizers_prefix,
        )
        self._constrain_actor_means(
            params,
            optimizers,
            optimizers_prefix,
        )

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
        self.strategy.configure_actor_bounds(
            self.dynamic_actors.actor_bounds()
        )
        self.strategy_state = self.strategy.initialize_state()
