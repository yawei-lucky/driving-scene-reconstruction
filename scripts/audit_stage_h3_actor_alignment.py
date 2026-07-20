#!/usr/bin/env python3
"""Audit whether SplatAD actor Gaussians remain inside actor-local cuboids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from nerfstudio.cameras.camera_utils import rotation_6d_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint to audit. Repeat to inspect training progression.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    return parser.parse_args()


def state_key(state: dict[str, Any], suffix: str) -> str:
    matches = [key for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one state key ending in {suffix!r}: {matches}")
    return matches[0]


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q))


def trajectory_drift(
    state: dict[str, Any],
    actor_index: int,
) -> dict[str, float]:
    positions = state[state_key(state, "dynamic_actors.actor_positions")]
    initial_positions = state[
        state_key(state, "dynamic_actors.initial_positions")
    ]
    present = state[
        state_key(state, "dynamic_actors.actor_present_at_time")
    ][:, actor_index]
    displacement = (
        positions[:, actor_index] - initial_positions[:, actor_index]
    ).norm(dim=-1)[present]

    rotations = rotation_6d_to_matrix(
        state[state_key(state, "dynamic_actors.actor_rotations_6d")]
    )
    initial_rotations = rotation_6d_to_matrix(
        state[state_key(state, "dynamic_actors.initial_rotations_6d")]
    )
    relative = (
        rotations[:, actor_index]
        @ initial_rotations[:, actor_index].transpose(-1, -2)
    )
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1) / 2).clamp(-1, 1))
    angle = torch.rad2deg(angle[present])
    return {
        "translation_mean_metres": float(displacement.mean()),
        "translation_p95_metres": quantile(displacement, 0.95),
        "translation_max_metres": float(displacement.max()),
        "rotation_mean_degrees": float(angle.mean()),
        "rotation_p95_degrees": quantile(angle, 0.95),
        "rotation_max_degrees": float(angle.max()),
    }


def audit_checkpoint(
    checkpoint_path: Path,
    min_opacity: float,
) -> dict[str, Any]:
    loaded = torch.load(checkpoint_path, map_location="cpu")
    state = loaded["pipeline"]
    means = state[state_key(state, "gauss_params.means")]
    ids = (
        state[state_key(state, "gauss_params.id")]
        .flatten()
        .round()
        .to(torch.long)
    )
    opacities = torch.sigmoid(
        state[state_key(state, "gauss_params.opacities")].flatten()
    )
    actor_sizes = state[state_key(state, "dynamic_actors.actor_sizes")]
    actor_padding = state[state_key(state, "dynamic_actors.actor_padding")]
    actor_bounds = actor_sizes / 2 + actor_padding

    actors = []
    for actor_index, bounds in enumerate(actor_bounds):
        actor_mask = ids == actor_index
        actor_means = means[actor_mask]
        actor_opacities = opacities[actor_mask]
        inside = (actor_means.abs() <= bounds).all(dim=-1)
        active = actor_opacities > min_opacity
        active_inside = inside[active]
        actor_record = {
            "actor_index": actor_index,
            "gaussians": int(actor_mask.sum()),
            "actor_size_metres": actor_sizes[actor_index].tolist(),
            "padded_half_extent_metres": bounds.tolist(),
            "inside_fraction": float(inside.float().mean()),
            "active_fraction": float(active.float().mean()),
            "active_inside_fraction": (
                float(active_inside.float().mean())
                if active_inside.numel()
                else None
            ),
            "local_abs_p50_metres": torch.quantile(
                actor_means.abs(), 0.50, dim=0
            ).tolist(),
            "local_abs_p95_metres": torch.quantile(
                actor_means.abs(), 0.95, dim=0
            ).tolist(),
            "local_abs_max_metres": actor_means.abs().max(dim=0).values.tolist(),
            "opacity_p50": quantile(actor_opacities, 0.50),
            "opacity_p95": quantile(actor_opacities, 0.95),
            "trajectory_drift": trajectory_drift(state, actor_index),
        }
        actors.append(actor_record)

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(loaded["step"]),
        "minimum_active_opacity": min_opacity,
        "actor_count": len(actors),
        "actors": actors,
        "actors_with_any_active_gaussians": sum(
            actor["active_fraction"] > 0 for actor in actors
        ),
        "actors_with_majority_active_gaussians_inside": sum(
            actor["active_inside_fraction"] is not None
            and actor["active_inside_fraction"] >= 0.5
            for actor in actors
        ),
    }


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_report(records: list[dict[str, Any]], output_path: Path) -> None:
    row_height = 52
    header_height = 108
    width = 1800
    rows = sum(len(record["actors"]) for record in records)
    image = Image.new(
        "RGB",
        (width, header_height + row_height * rows),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (20, 14),
        "Stage H3 moving-actor local-geometry audit",
        fill="white",
        font=font(30),
    )
    columns = (
        (20, "step"),
        (130, "actor"),
        (230, "GS"),
        (390, "inside"),
        (560, "active"),
        (730, "active inside"),
        (950, "local |xyz| p95 (m)"),
        (1320, "trajectory drift p95"),
    )
    for x, label in columns:
        draw.text((x, 70), label, fill=(255, 226, 70), font=font(19))

    y = header_height
    for record_index, record in enumerate(records):
        for actor in record["actors"]:
            shade = (30, 30, 30) if actor["actor_index"] % 2 == 0 else (24, 24, 24)
            draw.rectangle((0, y, width, y + row_height), fill=shade)
            active_inside = actor["active_inside_fraction"]
            health = (
                (70, 210, 100)
                if active_inside is not None and active_inside >= 0.5
                else (235, 80, 70)
            )
            p95 = actor["local_abs_p95_metres"]
            drift = actor["trajectory_drift"]
            values = (
                (20, str(record["checkpoint_step"]), "white"),
                (130, str(actor["actor_index"]), "white"),
                (230, f"{actor['gaussians']:,}", "white"),
                (390, f"{actor['inside_fraction']:.3f}", health),
                (560, f"{actor['active_fraction']:.3f}", "white"),
                (
                    730,
                    "none" if active_inside is None else f"{active_inside:.3f}",
                    health,
                ),
                (
                    950,
                    f"{p95[0]:.2f}, {p95[1]:.2f}, {p95[2]:.2f}",
                    "white",
                ),
                (
                    1320,
                    (
                        f"{drift['translation_p95_metres']:.3f} m / "
                        f"{drift['rotation_p95_degrees']:.2f} deg"
                    ),
                    "white",
                ),
            )
            for x, value, color in values:
                draw.text((x, y + 13), value, fill=color, font=font(19))
            y += row_height
        if record_index < len(records) - 1:
            draw.line((0, y - 1, width, y - 1), fill=(220, 220, 220), width=2)
    image.save(output_path, quality=94)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        audit_checkpoint(checkpoint, args.min_opacity)
        for checkpoint in args.checkpoint
    ]
    report = {
        "scope": (
            "Checkpoint-local audit. Actor means are stored in cuboid-local "
            "coordinates and should remain within padded actor half-extents."
        ),
        "source_observation": (
            "The pinned ADDefaultStrategy prunes actor-local Gaussians outside "
            "actor_bounds, while the pinned ADMCMCStrategy adds positional "
            "noise without an actor-bound constraint."
        ),
        "checkpoints": records,
        "decision": {
            "spatial_constraint_required": any(
                actor["active_inside_fraction"] is None
                or actor["active_inside_fraction"] < 0.5
                for record in records
                for actor in record["actors"]
            )
        },
    }
    report_path = args.output_dir / "actor_alignment_audit.json"
    visual_path = args.output_dir / "actor_alignment_audit.jpg"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    draw_report(records, visual_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE: {report_path}")
    print(f"WROTE: {visual_path}")


if __name__ == "__main__":
    main()
