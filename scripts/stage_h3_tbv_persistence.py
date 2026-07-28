"""Helpers for conservative cross-visit TbV LiDAR persistence masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


VOXEL_AXIS_BITS = 21
VOXEL_AXIS_SIZE = 1 << VOXEL_AXIS_BITS
VOXEL_AXIS_BIAS = VOXEL_AXIS_SIZE // 2
VOXEL_Y_SHIFT = VOXEL_AXIS_BITS
VOXEL_X_SHIFT = 2 * VOXEL_AXIS_BITS


def encode_voxel_triplet(x: int, y: int, z: int) -> int:
    """Pack one local signed voxel coordinate into a sortable 63-bit key."""

    coordinates = (x, y, z)
    if any(
        value < -VOXEL_AXIS_BIAS
        or value >= VOXEL_AXIS_BIAS
        for value in coordinates
    ):
        raise ValueError("voxel coordinate exceeds the signed 21-bit range")
    return (
        ((x + VOXEL_AXIS_BIAS) << VOXEL_X_SHIFT)
        | ((y + VOXEL_AXIS_BIAS) << VOXEL_Y_SHIFT)
        | (z + VOXEL_AXIS_BIAS)
    )


def neighbor_key_offsets(radius_voxels: int) -> tuple[int, ...]:
    """Return packed-key offsets for a Chebyshev voxel neighborhood."""

    if radius_voxels < 0 or radius_voxels > 2:
        raise ValueError("neighbor radius must be between zero and two voxels")
    return tuple(
        dx * (1 << VOXEL_X_SHIFT)
        + dy * (1 << VOXEL_Y_SHIFT)
        + dz
        for dx in range(-radius_voxels, radius_voxels + 1)
        for dy in range(-radius_voxels, radius_voxels + 1)
        for dz in range(-radius_voxels, radius_voxels + 1)
    )


def encode_voxel_array(voxels: Any) -> Any:
    """Vectorized form of :func:`encode_voxel_triplet` for an ``(N, 3)`` array."""

    import numpy as np

    values = np.asarray(voxels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("voxels must have shape (N, 3)")
    if values.size and (
        values.min() < -VOXEL_AXIS_BIAS
        or values.max() >= VOXEL_AXIS_BIAS
    ):
        raise ValueError("voxel coordinate exceeds the signed 21-bit range")
    shifted = values + VOXEL_AXIS_BIAS
    return (
        (shifted[:, 0] << VOXEL_X_SHIFT)
        | (shifted[:, 1] << VOXEL_Y_SHIFT)
        | shifted[:, 2]
    )


def supported_unique_keys(
    source_keys: Any,
    other_visit_keys: Any,
    *,
    radius_voxels: int,
) -> Any:
    """Mark source voxels supported by the other visit's neighborhood."""

    import numpy as np

    source = np.asarray(source_keys, dtype=np.int64)
    other = np.asarray(other_visit_keys, dtype=np.int64)
    if source.ndim != 1 or other.ndim != 1:
        raise ValueError("voxel keys must be one-dimensional")
    if source.size > 1 and np.any(source[1:] <= source[:-1]):
        raise ValueError("source voxel keys must be unique and sorted")
    if other.size > 1 and np.any(other[1:] <= other[:-1]):
        raise ValueError("other-visit voxel keys must be unique and sorted")
    supported = np.zeros(len(source), dtype=bool)
    for offset in neighbor_key_offsets(radius_voxels):
        pending = ~supported
        if not pending.any():
            break
        candidates = source[pending] + offset
        positions = np.searchsorted(other, candidates)
        in_range = positions < len(other)
        matched = np.zeros(len(candidates), dtype=bool)
        matched[in_range] = (
            other[positions[in_range]] == candidates[in_range]
        )
        supported[np.flatnonzero(pending)[matched]] = True
    return supported


def persistence_mask_path(
    *,
    data_root: Path,
    persistence_root: Path,
    lidar_path: Path,
) -> Path:
    """Map one source LiDAR feather to its persistence bit-mask path."""

    relative = lidar_path.expanduser().resolve().relative_to(
        data_root.expanduser().resolve()
    )
    return (
        persistence_root.expanduser().resolve()
        / relative
    ).with_suffix(".npz")


def save_persistence_mask(path: Path, persistent: Any) -> None:
    """Save one exact-length boolean mask in a compact deterministic form."""

    import numpy as np

    values = np.asarray(persistent, dtype=bool)
    if values.ndim != 1:
        raise ValueError("persistence mask must be one-dimensional")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray([0], dtype=np.int16),
        point_count=np.asarray([len(values)], dtype=np.int64),
        persistent_bits=np.packbits(values, bitorder="little"),
    )


def load_persistence_mask(path: Path, expected_points: int) -> Any:
    """Load and validate one exact-length persistence mask."""

    import numpy as np

    if expected_points < 0:
        raise ValueError("expected point count cannot be negative")
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR persistence mask is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        point_count = int(payload["point_count"][0])
        bits = payload["persistent_bits"]
    if version != 0:
        raise ValueError(f"unsupported persistence-mask version {version}")
    if point_count != expected_points:
        raise ValueError(
            f"persistence-mask point count {point_count} != {expected_points}: "
            f"{path}"
        )
    return np.unpackbits(
        bits,
        count=point_count,
        bitorder="little",
    ).astype(bool)


def distribution(values: Iterable[float]) -> dict[str, float]:
    """Return a compact deterministic distribution without SciPy."""

    import numpy as np

    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }
