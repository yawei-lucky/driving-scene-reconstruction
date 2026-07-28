"""Small route-tile contract for streaming reconstructed driving scenes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SceneTile:
    """One independently reconstructed asset on a shared route coordinate."""

    tile_id: str
    route_start_meters: float
    route_end_meters: float
    config_path: str
    checkpoint_path: str
    checkpoint_step: int
    data_root: str
    source_log: str
    source_time_window_seconds: tuple[float, float]
    dataparser_transform: tuple[tuple[float, float, float, float], ...]
    support_half_width_meters: float
    renderer_profile: str
    evidence_path: str

    def __post_init__(self) -> None:
        if not self.tile_id.strip():
            raise ValueError("tile_id must be non-empty")
        numeric = (
            self.route_start_meters,
            self.route_end_meters,
            self.support_half_width_meters,
            *self.source_time_window_seconds,
            *(
                value
                for row in self.dataparser_transform
                for value in row
            ),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("scene-tile distances must be finite")
        if self.route_end_meters <= self.route_start_meters:
            raise ValueError("scene tile must have positive route length")
        if self.support_half_width_meters <= 0.0:
            raise ValueError("support half-width must be positive")
        if self.checkpoint_step < 0:
            raise ValueError("checkpoint step cannot be negative")
        if (
            len(self.source_time_window_seconds) != 2
            or self.source_time_window_seconds[1]
            <= self.source_time_window_seconds[0]
        ):
            raise ValueError("source time window must be ordered")
        if (
            len(self.dataparser_transform) != 3
            or any(len(row) != 4 for row in self.dataparser_transform)
        ):
            raise ValueError("dataparser transform must be 3x4")
        for label, value in (
            ("config_path", self.config_path),
            ("checkpoint_path", self.checkpoint_path),
            ("data_root", self.data_root),
            ("source_log", self.source_log),
            ("renderer_profile", self.renderer_profile),
            ("evidence_path", self.evidence_path),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")

    @property
    def route_length_meters(self) -> float:
        return self.route_end_meters - self.route_start_meters

    def contains(self, progress_meters: float) -> bool:
        if not math.isfinite(progress_meters):
            raise ValueError("route progress must be finite")
        return (
            self.route_start_meters
            <= progress_meters
            <= self.route_end_meters
        )

    def overlap(self, other: "SceneTile") -> tuple[float, float] | None:
        start = max(self.route_start_meters, other.route_start_meters)
        end = min(self.route_end_meters, other.route_end_meters)
        return (start, end) if end > start else None


@dataclass(frozen=True)
class SceneTileWeight:
    tile_id: str
    weight: float


@dataclass(frozen=True)
class SceneTileRoute:
    """Ordered, gap-free tiles plus one deterministic overlap blend policy."""

    tiles: tuple[SceneTile, ...]
    blend_start_fraction: float = 0.35
    blend_end_fraction: float = 0.65

    def __post_init__(self) -> None:
        if not self.tiles:
            raise ValueError("scene-tile route needs at least one tile")
        if not 0.0 <= self.blend_start_fraction < self.blend_end_fraction <= 1.0:
            raise ValueError("blend fractions must be ordered within [0, 1]")
        if tuple(sorted(self.tiles, key=lambda tile: tile.route_start_meters)) != (
            self.tiles
        ):
            raise ValueError("scene tiles must be ordered by route start")
        if len({tile.tile_id for tile in self.tiles}) != len(self.tiles):
            raise ValueError("scene tile IDs must be unique")
        for left, right in zip(self.tiles, self.tiles[1:]):
            if left.overlap(right) is None:
                raise ValueError(
                    f"scene tiles {left.tile_id} and {right.tile_id} "
                    "need a positive overlap"
                )
        for first, third in zip(self.tiles, self.tiles[2:]):
            if third.route_start_meters <= first.route_end_meters:
                raise ValueError("triple-overlap routes are not supported")

    @property
    def route_start_meters(self) -> float:
        return self.tiles[0].route_start_meters

    @property
    def route_end_meters(self) -> float:
        return self.tiles[-1].route_end_meters

    @property
    def route_length_meters(self) -> float:
        return self.route_end_meters - self.route_start_meters

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def weights_at(self, progress_meters: float) -> tuple[SceneTileWeight, ...]:
        if (
            not math.isfinite(progress_meters)
            or progress_meters < self.route_start_meters
            or progress_meters > self.route_end_meters
        ):
            raise ValueError(
                f"route progress must be within "
                f"[{self.route_start_meters:.3f}, "
                f"{self.route_end_meters:.3f}]m"
            )
        active = tuple(
            tile for tile in self.tiles if tile.contains(progress_meters)
        )
        if len(active) == 1:
            return (SceneTileWeight(active[0].tile_id, 1.0),)
        if len(active) != 2:
            raise RuntimeError("route progress must resolve to one or two tiles")
        first, second = active
        overlap = first.overlap(second)
        assert overlap is not None
        start, end = overlap
        fraction = (progress_meters - start) / (end - start)
        normalized = min(
            max(
                (fraction - self.blend_start_fraction)
                / (self.blend_end_fraction - self.blend_start_fraction),
                0.0,
            ),
            1.0,
        )
        second_weight = self._smoothstep(normalized)
        if second_weight <= 0.0:
            return (SceneTileWeight(first.tile_id, 1.0),)
        if second_weight >= 1.0:
            return (SceneTileWeight(second.tile_id, 1.0),)
        return (
            SceneTileWeight(first.tile_id, 1.0 - second_weight),
            SceneTileWeight(second.tile_id, second_weight),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "format": "driving_scene_reconstruction.scene_tile_route.v0",
            "route_start_meters": self.route_start_meters,
            "route_end_meters": self.route_end_meters,
            "route_length_meters": self.route_length_meters,
            "blend_fraction_interval": [
                self.blend_start_fraction,
                self.blend_end_fraction,
            ],
            "tiles": [asdict(tile) for tile in self.tiles],
            "overlaps": [
                {
                    "from_tile": left.tile_id,
                    "to_tile": right.tile_id,
                    "start_meters": left.overlap(right)[0],
                    "end_meters": left.overlap(right)[1],
                    "length_meters": (
                        left.overlap(right)[1] - left.overlap(right)[0]
                    ),
                }
                for left, right in zip(self.tiles, self.tiles[1:])
            ],
        }
