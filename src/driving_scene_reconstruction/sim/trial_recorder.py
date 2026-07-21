"""Browser trial recording for the H3 drivable reconstruction loop."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping


DRIVING_KEYS = frozenset("wasd")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _finite_float(
    value: object,
    name: str,
    *,
    non_negative: bool = False,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: object, name: str, *, non_negative: bool = False) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    try:
        numeric_value = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not math.isfinite(numeric_value) or numeric_value != result:
        raise ValueError(f"{name} must be an integer")
    if non_negative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _clean_keys(value: object) -> str:
    keys = "".join(sorted(set(str(value).lower())))
    if not set(keys) <= DRIVING_KEYS:
        raise ValueError("control keys must contain only W/S/A/D")
    return keys


class BrowserTrialRecorder:
    """Persist operator-trial timing and state evidence as JSON.

    The browser measures timings the Python renderer cannot see, especially a
    physical input edge to the next loaded image. The server owns the JSON file
    so a completed trial remains available even if the browser tab closes.
    """

    def __init__(
        self,
        *,
        scene: str,
        movement_profile: str,
        output_scale: float,
        dt_seconds: float,
        checkpoint_step: int,
        logged_duration_seconds: float,
        output_path: str | Path | None = None,
    ) -> None:
        self.scene = scene
        self.movement_profile = movement_profile
        self.output_scale = output_scale
        self.dt_seconds = dt_seconds
        self.checkpoint_step = checkpoint_step
        self.logged_duration_seconds = logged_duration_seconds
        self.started_at_utc = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        self.output_path = (
            Path(output_path).expanduser().resolve() if output_path else None
        )
        self.samples: list[dict[str, Any]] = []
        self.reset_events: list[dict[str, Any]] = []
        self.write()

    def _from_client_or_server(
        self,
        payload: Mapping[str, Any],
        server: Mapping[str, Any],
        key: str,
    ) -> object:
        if key in payload:
            return payload[key]
        return server.get(key)

    def record_sample(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        server = payload.get("server", {})
        if not isinstance(server, Mapping):
            raise ValueError("server sample must be an object")
        profile = self._from_client_or_server(
            payload,
            server,
            "movement_profile",
        )
        if profile is not None and str(profile) != self.movement_profile:
            raise ValueError(
                f"sample profile {profile!r} does not match "
                f"{self.movement_profile!r}"
            )
        sample = {
            "event": "tick",
            "sample_index": len(self.samples),
            "sequence": _integer(payload.get("sequence"), "sequence", non_negative=True),
            "control_keys": _clean_keys(payload.get("keys", "")),
            "client_unix_ms": _finite_float(
                payload.get("client_unix_ms"),
                "client_unix_ms",
                non_negative=True,
                optional=True,
            ),
            "browser_request_to_image_ms": _finite_float(
                payload.get("browser_request_to_image_ms"),
                "browser_request_to_image_ms",
                non_negative=True,
            ),
            "browser_input_to_image_ms": _finite_float(
                payload.get("browser_input_to_image_ms"),
                "browser_input_to_image_ms",
                non_negative=True,
                optional=True,
            ),
            "time_seconds": _finite_float(
                self._from_client_or_server(payload, server, "time"),
                "time",
                non_negative=True,
            ),
            "duration_seconds": _finite_float(
                self._from_client_or_server(payload, server, "duration"),
                "duration",
                non_negative=True,
            ),
            "x_forward_meters": _finite_float(
                self._from_client_or_server(payload, server, "x"),
                "x",
            ),
            "y_left_meters": _finite_float(
                self._from_client_or_server(payload, server, "y"),
                "y",
            ),
            "yaw_degrees": _finite_float(
                self._from_client_or_server(payload, server, "yaw_degrees"),
                "yaw_degrees",
            ),
            "relative_speed_mps": _finite_float(
                self._from_client_or_server(payload, server, "speed"),
                "speed",
                non_negative=True,
            ),
            "logical_frame": _integer(
                self._from_client_or_server(payload, server, "logical_frame"),
                "logical_frame",
                non_negative=True,
            ),
            "renderer_ms": _finite_float(
                self._from_client_or_server(payload, server, "renderer_ms"),
                "renderer_ms",
                non_negative=True,
            ),
            "server_control_to_jpeg_ms": _finite_float(
                self._from_client_or_server(
                    payload,
                    server,
                    "server_control_to_jpeg_ms",
                ),
                "server_control_to_jpeg_ms",
                non_negative=True,
            ),
            "frame_selection_error_ms": _finite_float(
                self._from_client_or_server(
                    payload,
                    server,
                    "frame_selection_error_ms",
                ),
                "frame_selection_error_ms",
                non_negative=True,
            ),
            "camera_time_spread_ms": _finite_float(
                self._from_client_or_server(
                    payload,
                    server,
                    "camera_time_spread_ms",
                ),
                "camera_time_spread_ms",
                non_negative=True,
            ),
        }
        self.samples.append(sample)
        self.write()
        return self.summary()

    def record_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        reset_event = {
            "event": "reset",
            "reset_index": len(self.reset_events),
            "time_seconds": _finite_float(
                payload.get("time"),
                "time",
                non_negative=True,
            ),
            "x_forward_meters": _finite_float(payload.get("x"), "x"),
            "y_left_meters": _finite_float(payload.get("y"), "y"),
            "yaw_degrees": _finite_float(payload.get("yaw_degrees"), "yaw_degrees"),
            "relative_speed_mps": _finite_float(
                payload.get("speed"),
                "speed",
                non_negative=True,
            ),
            "logical_frame": _integer(
                payload.get("logical_frame"),
                "logical_frame",
                non_negative=True,
            ),
            "server_control_to_jpeg_ms": _finite_float(
                payload.get("server_control_to_jpeg_ms"),
                "server_control_to_jpeg_ms",
                non_negative=True,
            ),
        }
        self.reset_events.append(reset_event)
        self.write()
        return self.summary()

    def _series(self, key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in self.samples
            if sample.get(key) is not None
        ]

    def summary(self) -> dict[str, Any]:
        frames = [int(sample["logical_frame"]) for sample in self.samples]
        times = [float(sample["time_seconds"]) for sample in self.samples]
        max_time = max(times) if times else 0.0
        return {
            "scene": self.scene,
            "movement_profile": self.movement_profile,
            "sample_count": len(self.samples),
            "reset_count": len(self.reset_events),
            "checkpoint_step": self.checkpoint_step,
            "output_scale": self.output_scale,
            "dt_seconds": self.dt_seconds,
            "logged_duration_seconds": self.logged_duration_seconds,
            "max_log_time_seconds": max_time,
            "last_log_time_seconds": times[-1] if times else 0.0,
            "completed_log": bool(
                times
                and max_time >= self.logged_duration_seconds - self.dt_seconds * 0.5
            ),
            "logical_frames_monotonic": all(
                right >= left for left, right in zip(frames, frames[1:])
            ),
            "first_logical_frame": frames[0] if frames else None,
            "last_logical_frame": frames[-1] if frames else None,
            "observed_key_sets": sorted(
                {str(sample["control_keys"]) for sample in self.samples}
            ),
            "max_abs_forward_meters": max(
                (abs(float(sample["x_forward_meters"])) for sample in self.samples),
                default=0.0,
            ),
            "max_abs_left_meters": max(
                (abs(float(sample["y_left_meters"])) for sample in self.samples),
                default=0.0,
            ),
            "max_abs_yaw_degrees": max(
                (abs(float(sample["yaw_degrees"])) for sample in self.samples),
                default=0.0,
            ),
            "browser_request_to_image_ms": _distribution(
                self._series("browser_request_to_image_ms")
            ),
            "browser_input_to_image_ms": _distribution(
                self._series("browser_input_to_image_ms")
            ),
            "server_control_to_jpeg_ms": _distribution(
                self._series("server_control_to_jpeg_ms")
            ),
            "renderer_ms": _distribution(self._series("renderer_ms")),
            "frame_selection_error_ms": _distribution(
                self._series("frame_selection_error_ms")
            ),
            "camera_time_spread_ms": _distribution(
                self._series("camera_time_spread_ms")
            ),
        }

    def report(self) -> dict[str, Any]:
        return {
            "trial": {
                "started_at_utc": self.started_at_utc,
                "scene": self.scene,
                "movement_profile": self.movement_profile,
                "checkpoint_step": self.checkpoint_step,
                "output_scale": self.output_scale,
                "dt_seconds": self.dt_seconds,
                "logged_duration_seconds": self.logged_duration_seconds,
                "output_path": str(self.output_path) if self.output_path else None,
                "measurement_scope": {
                    "browser_request_to_image_ms": (
                        "browser tick request start to image load; includes "
                        "server render/JPEG and browser image load event, but "
                        "not monitor scan-out"
                    ),
                    "browser_input_to_image_ms": (
                        "physical key/mouse/touch edge to the first image load "
                        "that reflects a sampled control state"
                    ),
                },
            },
            "summary": self.summary(),
            "reset_events": self.reset_events,
            "samples": self.samples,
        }

    def report_bytes(self) -> bytes:
        return (
            json.dumps(self.report(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def write(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(self.report_bytes())
