#!/usr/bin/env python3
"""Build bounded visual and temporal evidence for a video-inpainting smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("distribution requires at least one value")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        blend = position - low
        return ordered[low] * (1.0 - blend) + ordered[high] * blend

    return {
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def sorted_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def label_frame(image: Any, label: str, cv2: Any) -> Any:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (80, 240, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-frames", type=Path, required=True)
    parser.add_argument("--ground-truth-frames", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--mask-dilation", type=int, default=4)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from PIL import Image

    input_paths = sorted_images(args.input_frames)
    ground_truth_paths = sorted_images(args.ground_truth_frames)
    mask_paths = sorted_images(args.masks)
    output_paths = sorted_images(args.output_frames)
    counts = {
        "input": len(input_paths),
        "ground_truth": len(ground_truth_paths),
        "mask": len(mask_paths),
        "output": len(output_paths),
    }
    if len(set(counts.values())) != 1 or not input_paths:
        raise RuntimeError(f"frame-count mismatch: {counts}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    changed_outside_fraction: list[float] = []
    outside_mae: list[float] = []
    inside_change_mae: list[float] = []
    input_temporal_mae: list[float] = []
    output_temporal_mae: list[float] = []
    panels: list[Any] = []
    previous_input = None
    previous_output = None

    for index, paths in enumerate(
        zip(input_paths, ground_truth_paths, mask_paths, output_paths)
    ):
        input_path, ground_truth_path, mask_path, output_path = paths
        input_rgb = np.asarray(Image.open(input_path).convert("RGB"))
        ground_truth = np.asarray(
            Image.open(ground_truth_path).convert("RGB")
        )
        mask = np.asarray(Image.open(mask_path).convert("L"))
        output_rgb = np.asarray(Image.open(output_path).convert("RGB"))
        height, width = output_rgb.shape[:2]
        input_rgb = np.asarray(
            Image.fromarray(input_rgb).resize((width, height))
        )
        ground_truth = np.asarray(
            Image.fromarray(ground_truth).resize((width, height))
        )
        mask = np.asarray(
            Image.fromarray(mask).resize(
                (width, height), Image.Resampling.NEAREST
            )
        )
        masked = mask > 0
        kernel_size = args.mask_dilation * 2 + 1
        dilated = cv2.dilate(
            masked.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)
        difference = np.abs(
            output_rgb.astype(np.float32) - input_rgb.astype(np.float32)
        ).mean(axis=2)
        outside = ~dilated
        changed_outside_fraction.append(
            float(np.mean(difference[outside] > 2.0))
        )
        outside_mae.append(float(np.mean(difference[outside])))
        inside_change_mae.append(float(np.mean(difference[masked])))
        if previous_input is not None and previous_output is not None:
            temporal_region = dilated
            input_temporal_mae.append(
                float(
                    np.mean(
                        np.abs(
                            input_rgb.astype(np.float32)
                            - previous_input.astype(np.float32)
                        )[temporal_region]
                    )
                )
            )
            output_temporal_mae.append(
                float(
                    np.mean(
                        np.abs(
                            output_rgb.astype(np.float32)
                            - previous_output.astype(np.float32)
                        )[temporal_region]
                    )
                )
            )
        previous_input = input_rgb
        previous_output = output_rgb

        overlay = input_rgb.copy()
        overlay[masked] = (
            0.4 * overlay[masked] + 0.6 * np.array([0, 255, 0])
        ).astype(np.uint8)
        panel = np.hstack(
            [
                label_frame(
                    cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR),
                    "source GT (contains traffic)",
                    cv2,
                ),
                label_frame(
                    cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR),
                    "static-8k",
                    cv2,
                ),
                label_frame(
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    "removal mask",
                    cv2,
                ),
                label_frame(
                    cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR),
                    "ProPainter",
                    cv2,
                ),
            ]
        )
        panels.append(panel)

    video_path = output_dir / "static8k_vs_propainter.mp4"
    video = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panels[0].shape[1], panels[0].shape[0]),
    )
    if not video.isOpened():
        raise RuntimeError("failed to open MP4 writer")
    for panel in panels:
        video.write(panel)
    video.release()

    contact_indices = sorted(
        {
            round(index * (len(panels) - 1) / 5)
            for index in range(6)
        }
    )
    contact = np.vstack(
        [
            cv2.resize(
                panels[index],
                (
                    panels[index].shape[1] // 2,
                    panels[index].shape[0] // 2,
                ),
                interpolation=cv2.INTER_AREA,
            )
            for index in contact_indices
        ]
    )
    contact_path = output_dir / "static8k_vs_propainter_contact.jpg"
    cv2.imwrite(str(contact_path), contact, [cv2.IMWRITE_JPEG_QUALITY, 94])

    report = {
        "format": "driving_scene_reconstruction.generative_repair_audit.v0",
        "scope": (
            "Offline observed-pose ProPainter smoke. This measures technical "
            "continuity and preservation, not hidden-background truth or "
            "driving acceptance."
        ),
        "frame_counts": counts,
        "output_frame_resolution_height_width": list(
            output_rgb.shape[:2]
        ),
        "comparison_panel_resolution_height_width": list(
            panels[0].shape[:2]
        ),
        "mask_dilation_pixels_at_inference_resolution": args.mask_dilation,
        "changed_outside_dilated_mask_fraction": distribution(
            changed_outside_fraction
        ),
        "outside_dilated_mask_mae": distribution(outside_mae),
        "inside_original_mask_change_mae": distribution(inside_change_mae),
        "naive_masked_temporal_mae": {
            "static_8k": distribution(input_temporal_mae),
            "propainter": distribution(output_temporal_mae),
            "warning": (
                "Unwarped consecutive-frame differences include ego motion; "
                "they are a flicker screen, not a temporal truth metric."
            ),
        },
        "video": str(video_path),
        "contact_sheet": str(contact_path),
        "technical_status": (
            "pass"
            if max(changed_outside_fraction) <= 0.01
            else "preservation_failure"
        ),
        "visual_status": "requires_manual_review",
        "limitations": [
            "No hidden-background ground truth exists behind source traffic.",
            "Source detection masks are imperfect and intermittent.",
            "This image-space result is view-specific and cannot be reused as "
            "geometry for a different free-driving pose.",
        ],
    }
    (output_dir / "generative_repair_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
