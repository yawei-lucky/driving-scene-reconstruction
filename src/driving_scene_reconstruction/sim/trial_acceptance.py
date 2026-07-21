"""Acceptance checks for recorded H3 browser driving trials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .trial_recorder import MANUAL_REVIEW_GATES


@dataclass(frozen=True)
class TrialAcceptanceConfig:
    """Thresholds for treating one browser trial JSON as acceptance evidence."""

    expected_scene: str = "040"
    expected_checkpoint_step: int = 7999
    expected_movement_profile: str | None = None
    min_sample_count: int = 70
    min_reset_count: int = 1
    min_browser_input_samples: int = 1
    max_browser_request_to_image_p95_ms: float = 100.0
    max_browser_input_to_image_p95_ms: float = 100.0
    max_server_control_to_jpeg_p95_ms: float = 100.0
    max_camera_time_spread_p95_ms: float = 100.0
    require_control_input: bool = True


@dataclass(frozen=True)
class AcceptanceGate:
    """One pass/fail decision in the browser trial acceptance check."""

    name: str
    passed: bool
    detail: str


def load_trial_report(path: str | Path) -> dict[str, Any]:
    """Load one BrowserTrialRecorder JSON report."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("trial JSON root must be an object")
    return data


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    if number is None:
        return None
    integer = int(number)
    return integer if number == integer else None


def _bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _distribution_count(distribution: object) -> int:
    if not isinstance(distribution, Mapping):
        return 0
    value = _integer(distribution.get("count"))
    return value if value is not None and value >= 0 else 0


def _distribution_p95(distribution: object) -> float | None:
    if not isinstance(distribution, Mapping):
        return None
    return _number(distribution.get("p95"))


def _pass_if(name: str, passed: bool, detail: str) -> AcceptanceGate:
    return AcceptanceGate(name=name, passed=passed, detail=detail)


def _latest_manual_review(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reviews = report.get("manual_reviews")
    if not isinstance(reviews, list) or not reviews:
        return None
    latest = reviews[-1]
    return latest if isinstance(latest, Mapping) else None


def evaluate_trial_report(
    report: Mapping[str, Any],
    config: TrialAcceptanceConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether a recorded browser driving trial proves acceptance."""

    cfg = config or TrialAcceptanceConfig()
    trial = _mapping(report.get("trial"), "trial")
    summary = _mapping(report.get("summary"), "summary")
    latest_review = _latest_manual_review(report)

    gates: list[AcceptanceGate] = []

    scene = str(summary.get("scene", trial.get("scene", "")))
    gates.append(
        _pass_if(
            "expected_scene",
            scene == cfg.expected_scene,
            f"scene={scene!r}, expected={cfg.expected_scene!r}",
        )
    )

    checkpoint_step = _integer(summary.get("checkpoint_step"))
    gates.append(
        _pass_if(
            "accepted_static_checkpoint",
            checkpoint_step == cfg.expected_checkpoint_step,
            (
                f"checkpoint_step={checkpoint_step}, "
                f"expected={cfg.expected_checkpoint_step}"
            ),
        )
    )

    if cfg.expected_movement_profile is not None:
        movement_profile = str(
            summary.get("movement_profile", trial.get("movement_profile", ""))
        )
        gates.append(
            _pass_if(
                "expected_movement_profile",
                movement_profile == cfg.expected_movement_profile,
                (
                    f"movement_profile={movement_profile!r}, "
                    f"expected={cfg.expected_movement_profile!r}"
                ),
            )
        )

    sample_count = _integer(summary.get("sample_count"))
    gates.append(
        _pass_if(
            "enough_trial_samples",
            sample_count is not None and sample_count >= cfg.min_sample_count,
            f"sample_count={sample_count}, minimum={cfg.min_sample_count}",
        )
    )

    completed_log = _bool(summary.get("completed_log"))
    gates.append(
        _pass_if(
            "completed_logged_segment",
            completed_log is True,
            f"completed_log={completed_log}",
        )
    )

    logical_frames_monotonic = _bool(summary.get("logical_frames_monotonic"))
    gates.append(
        _pass_if(
            "logical_frames_monotonic",
            logical_frames_monotonic is True,
            f"logical_frames_monotonic={logical_frames_monotonic}",
        )
    )

    reset_count = _integer(summary.get("reset_count"))
    gates.append(
        _pass_if(
            "reset_recorded",
            reset_count is not None and reset_count >= cfg.min_reset_count,
            f"reset_count={reset_count}, minimum={cfg.min_reset_count}",
        )
    )

    observed_key_sets = summary.get("observed_key_sets")
    active_key_sets: list[str] = []
    if isinstance(observed_key_sets, list):
        active_key_sets = [
            str(keys)
            for keys in observed_key_sets
            if isinstance(keys, str) and keys
        ]
    if cfg.require_control_input:
        gates.append(
            _pass_if(
                "operator_control_input_observed",
                bool(active_key_sets),
                f"active observed_key_sets={active_key_sets}",
            )
        )

    request_p95 = _distribution_p95(summary.get("browser_request_to_image_ms"))
    gates.append(
        _pass_if(
            "browser_request_to_image_p95",
            request_p95 is not None
            and request_p95 <= cfg.max_browser_request_to_image_p95_ms,
            (
                f"p95={request_p95}, "
                f"maximum={cfg.max_browser_request_to_image_p95_ms}ms"
            ),
        )
    )

    input_distribution = summary.get("browser_input_to_image_ms")
    input_count = _distribution_count(input_distribution)
    input_p95 = _distribution_p95(input_distribution)
    gates.append(
        _pass_if(
            "browser_input_to_image_p95",
            input_count >= cfg.min_browser_input_samples
            and input_p95 is not None
            and input_p95 <= cfg.max_browser_input_to_image_p95_ms,
            (
                f"count={input_count}, p95={input_p95}, "
                f"minimum_count={cfg.min_browser_input_samples}, "
                f"maximum={cfg.max_browser_input_to_image_p95_ms}ms"
            ),
        )
    )

    server_p95 = _distribution_p95(summary.get("server_control_to_jpeg_ms"))
    gates.append(
        _pass_if(
            "server_control_to_jpeg_p95",
            server_p95 is not None
            and server_p95 <= cfg.max_server_control_to_jpeg_p95_ms,
            (
                f"p95={server_p95}, "
                f"maximum={cfg.max_server_control_to_jpeg_p95_ms}ms"
            ),
        )
    )

    camera_spread_p95 = _distribution_p95(summary.get("camera_time_spread_ms"))
    gates.append(
        _pass_if(
            "camera_time_spread_p95",
            camera_spread_p95 is not None
            and camera_spread_p95 <= cfg.max_camera_time_spread_p95_ms,
            (
                f"p95={camera_spread_p95}, "
                f"maximum={cfg.max_camera_time_spread_p95_ms}ms"
            ),
        )
    )

    manual_review_completed = _bool(summary.get("manual_review_completed"))
    manual_review_all_passed = _bool(summary.get("manual_review_all_passed"))
    gates.append(
        _pass_if(
            "manual_review_completed",
            manual_review_completed is True,
            f"manual_review_completed={manual_review_completed}",
        )
    )
    gates.append(
        _pass_if(
            "manual_review_all_passed",
            manual_review_all_passed is True,
            f"manual_review_all_passed={manual_review_all_passed}",
        )
    )

    status_by_gate = summary.get("manual_review_status_by_gate")
    manual_statuses_ok = isinstance(status_by_gate, Mapping) and all(
        status_by_gate.get(gate) == "pass" for gate in MANUAL_REVIEW_GATES
    )
    gates.append(
        _pass_if(
            "all_manual_gate_statuses_pass",
            manual_statuses_ok,
            f"manual_review_status_by_gate={status_by_gate}",
        )
    )

    if latest_review is None:
        review_sample_count = None
        review_last_log_time = None
    else:
        review_sample_count = _integer(latest_review.get("sample_count_at_review"))
        review_last_log_time = _number(latest_review.get("last_log_time_seconds"))
    gates.append(
        _pass_if(
            "manual_review_after_enough_samples",
            review_sample_count is not None
            and review_sample_count >= cfg.min_sample_count,
            (
                f"sample_count_at_review={review_sample_count}, "
                f"minimum={cfg.min_sample_count}"
            ),
        )
    )

    logged_duration = _number(summary.get("logged_duration_seconds"))
    dt_seconds = _number(summary.get("dt_seconds"))
    review_completed_after_log = False
    if (
        review_last_log_time is not None
        and logged_duration is not None
        and dt_seconds is not None
    ):
        review_completed_after_log = (
            review_last_log_time >= logged_duration - dt_seconds * 0.5
        )
    gates.append(
        _pass_if(
            "manual_review_after_completed_log",
            review_completed_after_log,
            (
                f"review_last_log_time_seconds={review_last_log_time}, "
                f"logged_duration_seconds={logged_duration}, dt_seconds={dt_seconds}"
            ),
        )
    )

    passed = all(gate.passed for gate in gates)
    failures = [gate.name for gate in gates if not gate.passed]
    return {
        "report_type": "h3_browser_trial_acceptance_check",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": passed,
        "config": asdict(cfg),
        "failures": failures,
        "gates": [asdict(gate) for gate in gates],
    }
