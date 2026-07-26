"""Small dependency-light helpers for TbV image-guided LiDAR filtering."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any


def closest_timestamp_index(
    timestamps_ns: tuple[int, ...],
    target_timestamp_ns: int,
    max_delta_ns: int,
) -> int | None:
    """Return the deterministic nearest timestamp index within a bound."""

    if max_delta_ns < 0:
        raise ValueError("max_delta_ns must be non-negative")
    if any(
        current >= following
        for current, following in zip(timestamps_ns, timestamps_ns[1:])
    ):
        raise ValueError("timestamps_ns must be strictly increasing")
    insertion = bisect_left(timestamps_ns, target_timestamp_ns)
    candidates = tuple(
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(timestamps_ns)
    )
    if not candidates:
        return None
    nearest = min(
        candidates,
        key=lambda index: (
            abs(timestamps_ns[index] - target_timestamp_ns),
            timestamps_ns[index],
        ),
    )
    if abs(timestamps_ns[nearest] - target_timestamp_ns) > max_delta_ns:
        return None
    return nearest


def projected_mask_exclusion(
    image_points: Any,
    projection_valid: Any,
    valid_pixel_mask: Any,
) -> Any:
    """Return points landing on excluded pixels in a raw camera mask.

    ``valid_pixel_mask`` follows Nerfstudio convention: zero is excluded and a
    non-zero value is valid. Projection coordinates are rounded exactly as in
    the AV2 colored-sweep helper.
    """

    import numpy as np

    image_points = np.asarray(image_points)
    projection_valid = np.asarray(projection_valid, dtype=bool)
    valid_pixel_mask = np.asarray(valid_pixel_mask)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("image_points must have shape (N, 2)")
    if projection_valid.shape != (len(image_points),):
        raise ValueError("projection_valid must have shape (N,)")
    if valid_pixel_mask.ndim != 2:
        raise ValueError("valid_pixel_mask must be a grayscale image")

    excluded = np.zeros(len(image_points), dtype=bool)
    candidates = projection_valid & np.isfinite(image_points).all(axis=1)
    candidate_indices = np.flatnonzero(candidates)
    if not len(candidate_indices):
        return excluded

    rounded = np.rint(image_points[candidate_indices]).astype(np.int64)
    height, width = valid_pixel_mask.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    inside_indices = candidate_indices[inside]
    inside_pixels = rounded[inside]
    excluded[inside_indices] = (
        valid_pixel_mask[inside_pixels[:, 1], inside_pixels[:, 0]] == 0
    )
    return excluded
