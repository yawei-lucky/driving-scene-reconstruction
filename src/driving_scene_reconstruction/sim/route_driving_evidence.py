"""Machine-readable evidence for route-constrained browser driving trials."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


def _finite(value: object, name: str, *, non_negative: bool = False) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (non_negative and result < 0.0):
        qualifier = " non-negative" if non_negative else ""
        raise ValueError(f"{name} must be a finite{qualifier} number")
    return result


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": ordered[low] * (1.0 - fraction) + ordered[high] * fraction,
        "maximum": max(values),
    }


class RouteDrivingEvidenceRecorder:
    """Persist state→render→browser evidence without certifying image truth."""

    FORMAT = "driving_scene_reconstruction.route_driving_evidence.v0"

    def __init__(
        self,
        *,
        scene: str,
        config_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_step: int,
        output_scale: float,
        dt_seconds: float,
        camera_names: Sequence[str],
        route_contract: Mapping[str, Any],
        limitations: Sequence[str],
        output_path: str | Path | None = None,
    ) -> None:
        if not scene.strip():
            raise ValueError("scene cannot be empty")
        if checkpoint_step < 0:
            raise ValueError("checkpoint step must be non-negative")
        self.scene = scene
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.checkpoint_step = int(checkpoint_step)
        self.output_scale = _finite(output_scale, "output scale", non_negative=True)
        self.dt_seconds = _finite(dt_seconds, "dt", non_negative=True)
        if self.output_scale <= 0.0 or self.dt_seconds <= 0.0:
            raise ValueError("output scale and dt must be positive")
        self.camera_names = tuple(str(name) for name in camera_names)
        if not self.camera_names or any(not name for name in self.camera_names):
            raise ValueError("camera names cannot be empty")
        self.route_contract = dict(route_contract)
        self.limitations = tuple(str(item) for item in limitations)
        self.output_path = (
            Path(output_path).expanduser().resolve() if output_path else None
        )
        self.started_at_utc = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        self.samples: list[dict[str, Any]] = []
        self.route_events: list[dict[str, Any]] = []
        self.reset_events: list[dict[str, Any]] = []
        self.write()

    @staticmethod
    def _clean_support(value: object) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("route support must be an object")
        result = dict(value)
        for name in (
            "progress_from_anchor_meters",
            "lateral_offset_meters",
            "distance_to_centerline_meters",
            "heading_error_degrees",
            "distance_margin_meters",
            "heading_margin_degrees",
        ):
            result[name] = _finite(result.get(name), f"support.{name}")
        result["within_declared_support"] = bool(
            result.get("within_declared_support")
        )
        result["selection_required"] = bool(result.get("selection_required"))
        return result

    def record_server_sample(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        sequence = int(payload.get("sequence", -1))
        if sequence < 0 or any(
            item["sequence"] == sequence for item in self.samples
        ):
            raise ValueError("sample sequence must be unique and non-negative")
        keys = "".join(sorted(set(str(payload.get("control_keys", "")).lower())))
        if not set(keys) <= set("wasd"):
            raise ValueError("control keys must contain only W/S/A/D")
        sample = {
            "sequence": sequence,
            "server_recorded_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "control_keys": keys,
            "simulation_time_seconds": _finite(
                payload.get("simulation_time_seconds"),
                "simulation time",
                non_negative=True,
            ),
            "ego_pose": {
                "x_meters": _finite(payload.get("x_meters"), "x"),
                "y_meters": _finite(payload.get("y_meters"), "y"),
                "yaw_degrees": _finite(payload.get("yaw_degrees"), "yaw"),
                "speed_mps": _finite(
                    payload.get("speed_mps"), "speed", non_negative=True
                ),
            },
            "route_support": self._clean_support(payload.get("route_support")),
            "renderer_profile": str(payload.get("renderer_profile", "")),
            "frozen_scene_time_seconds": _finite(
                payload.get("frozen_scene_time_seconds"), "frozen scene time"
            ),
            "camera_count": int(payload.get("camera_count", 0)),
            "all_camera_frames_finite": bool(
                payload.get("all_camera_frames_finite")
            ),
            "renderer_ms": _finite(
                payload.get("renderer_ms"), "renderer ms", non_negative=True
            ),
            "server_control_to_jpeg_ms": _finite(
                payload.get("server_control_to_jpeg_ms"),
                "server control to jpeg ms",
                non_negative=True,
            ),
            "frame_sha256": str(payload.get("frame_sha256", "")),
            "boundary_hit": bool(payload.get("boundary_hit")),
            "boundary_reason": payload.get("boundary_reason"),
            "browser_timing": None,
        }
        if sample["camera_count"] != len(self.camera_names):
            raise ValueError("sample camera count does not match trial camera names")
        if len(sample["frame_sha256"]) != 64:
            raise ValueError("frame sha256 must contain 64 hexadecimal characters")
        try:
            int(sample["frame_sha256"], 16)
        except ValueError as error:
            raise ValueError("frame sha256 must be hexadecimal") from error
        self.samples.append(sample)
        self.write()
        return sample

    def record_browser_timing(
        self,
        *,
        sequence: int,
        browser_request_to_image_ms: object,
        browser_input_to_image_ms: object | None,
        client_unix_ms: object,
    ) -> dict[str, Any]:
        sample = next(
            (item for item in self.samples if item["sequence"] == sequence), None
        )
        if sample is None:
            raise ValueError(f"unknown sample sequence {sequence}")
        if sample["browser_timing"] is not None:
            raise ValueError(
                f"browser timing already recorded for sequence {sequence}"
            )
        sample["browser_timing"] = {
            "client_unix_ms": _finite(
                client_unix_ms, "client unix ms", non_negative=True
            ),
            "browser_request_to_image_ms": _finite(
                browser_request_to_image_ms,
                "browser request to image ms",
                non_negative=True,
            ),
            "browser_input_to_image_ms": (
                None
                if browser_input_to_image_ms is None
                else _finite(
                    browser_input_to_image_ms,
                    "browser input to image ms",
                    non_negative=True,
                )
            ),
        }
        self.write()
        return sample

    def record_route_event(
        self, event: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if event not in {"branch_selected", "branch_selection_required"}:
            raise ValueError("unknown route event")
        record = {
            "event": event,
            "event_index": len(self.route_events),
            "server_recorded_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "browser_timing": None,
            **dict(payload),
        }
        self.route_events.append(record)
        self.write()
        return record

    def record_route_browser_timing(
        self,
        *,
        event_index: int,
        browser_selection_to_image_ms: object,
        client_unix_ms: object,
    ) -> dict[str, Any]:
        event = next(
            (
                item
                for item in self.route_events
                if item["event_index"] == event_index
            ),
            None,
        )
        if event is None or event["event"] != "branch_selected":
            raise ValueError(f"unknown branch-selection event {event_index}")
        if event["browser_timing"] is not None:
            raise ValueError(
                f"browser timing already recorded for route event {event_index}"
            )
        event["browser_timing"] = {
            "client_unix_ms": _finite(
                client_unix_ms, "client unix ms", non_negative=True
            ),
            "browser_selection_to_image_ms": _finite(
                browser_selection_to_image_ms,
                "browser selection to image ms",
                non_negative=True,
            ),
        }
        self.write()
        return event

    def record_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "event": "reset",
            "reset_index": len(self.reset_events),
            "server_recorded_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            **dict(payload),
        }
        self.reset_events.append(record)
        self.write()
        return record

    def _series(self, key: str) -> list[float]:
        return [float(sample[key]) for sample in self.samples]

    def summary(self) -> dict[str, Any]:
        browser = [
            sample["browser_timing"]
            for sample in self.samples
            if sample["browser_timing"] is not None
        ]
        supports = [sample["route_support"] for sample in self.samples]
        branch_browser_timings = [
            float(event["browser_timing"]["browser_selection_to_image_ms"])
            for event in self.route_events
            if event["event"] == "branch_selected"
            and event["browser_timing"] is not None
        ]
        return {
            "sample_count": len(self.samples),
            "reset_count": len(self.reset_events),
            "route_event_count": len(self.route_events),
            "selected_branches": sorted(
                {
                    str(event.get("branch"))
                    for event in self.route_events
                    if event["event"] == "branch_selected"
                }
            ),
            "browser_timing_coverage": (
                len(browser) / len(self.samples) if self.samples else 0.0
            ),
            "support_violation_count": sum(
                not bool(item["within_declared_support"]) for item in supports
            ),
            "boundary_hit_count": sum(
                bool(item["boundary_hit"]) for item in self.samples
            ),
            "max_abs_lateral_offset_meters": max(
                (abs(float(item["lateral_offset_meters"])) for item in supports),
                default=0.0,
            ),
            "minimum_distance_margin_meters": min(
                (float(item["distance_margin_meters"]) for item in supports),
                default=None,
            ),
            "renderer_ms": _distribution(self._series("renderer_ms")),
            "server_control_to_jpeg_ms": _distribution(
                self._series("server_control_to_jpeg_ms")
            ),
            "browser_request_to_image_ms": _distribution(
                [float(item["browser_request_to_image_ms"]) for item in browser]
            ),
            "browser_input_to_image_ms": _distribution(
                [
                    float(item["browser_input_to_image_ms"])
                    for item in browser
                    if item["browser_input_to_image_ms"] is not None
                ]
            ),
            "browser_branch_selection_to_image_ms": _distribution(
                branch_browser_timings
            ),
            "all_samples_have_finite_camera_frames": all(
                bool(sample["all_camera_frames_finite"])
                for sample in self.samples
            ),
            "claim_status": "evidence_only_not_certified",
        }

    def report(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "trial": {
                "started_at_utc": self.started_at_utc,
                "scene": self.scene,
                "config_path": self.config_path,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_step": self.checkpoint_step,
                "output_scale": self.output_scale,
                "dt_seconds": self.dt_seconds,
                "camera_names": list(self.camera_names),
                "route_contract": self.route_contract,
                "limitations": list(self.limitations),
                "output_path": str(self.output_path) if self.output_path else None,
                "measurement_scope": {
                    "server_control_to_jpeg_ms": (
                        "server request receipt through state update, "
                        "seven-camera render, and mosaic encode; evidence-file "
                        "write is excluded"
                    ),
                    "browser_request_to_image_ms": (
                        "browser request start through the corresponding JPEG "
                        "load event; monitor scan-out is excluded"
                    ),
                    "browser_input_to_image_ms": (
                        "physical keyboard/pointer input edge through the first "
                        "corresponding JPEG load event; monitor scan-out is excluded"
                    ),
                    "browser_branch_selection_to_image_ms": (
                        "physical 1/2 key or branch-button input through the "
                        "first selected-profile JPEG load event; monitor "
                        "scan-out is excluded"
                    ),
                },
            },
            "summary": self.summary(),
            "route_events": self.route_events,
            "reset_events": self.reset_events,
            "samples": self.samples,
        }

    def report_bytes(self) -> bytes:
        return (json.dumps(self.report(), indent=2, sort_keys=True) + "\n").encode()

    def write(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(self.report_bytes())
