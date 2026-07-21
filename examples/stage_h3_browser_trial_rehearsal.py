#!/usr/bin/env python3
"""Run a scripted H3 browser trial rehearsal through the HTTP service."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any
from urllib import request
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim.trial_acceptance import (  # noqa: E402
    TrialAcceptanceConfig,
    evaluate_trial_report,
)
from driving_scene_reconstruction.sim.trial_recorder import (  # noqa: E402
    MANUAL_REVIEW_GATES,
)


ALLOWED_REHEARSAL_FAILURES = frozenset(
    {
        "manual_review_all_passed",
        "all_manual_gate_statuses_pass",
    }
)


def control_keys_for_step(step: int) -> str:
    """Deterministic W/S/A/D schedule that exercises visible offset controls."""

    phase = step % 40
    if phase < 8:
        return "w"
    if phase < 16:
        return "aw"
    if phase < 24:
        return "a"
    if phase < 32:
        return "d"
    return "s"


class HttpJsonClient:
    """Small stdlib HTTP client for the browser trial service."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("HTTP service path must start with /")
        return f"{self.base_url}{path}"

    def get_json(self, path: str) -> dict[str, Any]:
        with request.urlopen(
            self._url(path),
            timeout=self.timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} did not return a JSON object")
        return payload

    def get_bytes(self, path: str) -> bytes:
        with request.urlopen(
            self._url(path),
            timeout=self.timeout_seconds,
        ) as response:
            return response.read()

    def post_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        call = request.Request(
            self._url(path),
            data=data,
            headers=headers,
            method="POST",
        )
        with request.urlopen(call, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"{path} did not return a JSON object")
        return result


def run_rehearsal(
    client: Any,
    *,
    steps: int,
    expected_movement_profile: str | None,
    output_path: Path | None = None,
    acceptance_output_path: Path | None = None,
) -> dict[str, Any]:
    """Drive the live browser service and write a not-human acceptance report."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    started_at = time.perf_counter()
    initial_report = client.get_json("/trial.json")
    sequence = 0
    previous_keys = ""
    sample_summaries: list[dict[str, Any]] = []

    for index in range(steps):
        keys = control_keys_for_step(index)
        encoded_keys = quote(keys)
        tick_started = time.perf_counter()
        server = client.post_json(f"/tick?keys={encoded_keys}")
        sequence += 1
        frame_payload = client.get_bytes(f"/frame.jpg?n={sequence}")
        request_to_image_ms = (time.perf_counter() - tick_started) * 1000.0
        input_to_image_ms = request_to_image_ms if keys != previous_keys else None
        sample_result = client.post_json(
            "/trial-sample",
            {
                "sequence": sequence,
                "keys": keys,
                "client_unix_ms": time.time() * 1000.0,
                "browser_request_to_image_ms": request_to_image_ms,
                "browser_input_to_image_ms": input_to_image_ms,
                "server": server,
            },
        )
        sample_summaries.append(dict(sample_result.get("summary", {})))
        previous_keys = keys
        if not frame_payload:
            raise RuntimeError("empty frame payload")
        if float(server.get("time", 0.0)) >= float(server.get("duration", 1.0)):
            break

    client.post_json("/reset")
    client.post_json(
        "/trial-review",
        {
            "client_unix_ms": time.time() * 1000.0,
            "reviewer": "scripted_rehearsal_not_human",
            "gates": {gate: "unsure" for gate in MANUAL_REVIEW_GATES},
            "notes": (
                "Scripted HTTP rehearsal only. Manual visual gates intentionally "
                "left unsure so this cannot count as human acceptance."
            ),
        },
    )
    final_report = client.get_json("/trial.json")
    acceptance_result = evaluate_trial_report(
        final_report,
        TrialAcceptanceConfig(expected_movement_profile=expected_movement_profile),
    )
    failures = set(str(name) for name in acceptance_result["failures"])
    unexpected_failures = sorted(failures - ALLOWED_REHEARSAL_FAILURES)
    missing_expected_failures = sorted(ALLOWED_REHEARSAL_FAILURES - failures)
    rehearsal_passed = not unexpected_failures and not missing_expected_failures
    report = {
        "report_type": "h3_browser_trial_rehearsal_not_human_acceptance",
        "passed": rehearsal_passed,
        "duration_seconds": time.perf_counter() - started_at,
        "steps_requested": steps,
        "steps_recorded": len(sample_summaries),
        "allowed_acceptance_failures": sorted(ALLOWED_REHEARSAL_FAILURES),
        "unexpected_acceptance_failures": unexpected_failures,
        "missing_expected_acceptance_failures": missing_expected_failures,
        "initial_summary": initial_report.get("summary", {}),
        "final_summary": final_report.get("summary", {}),
        "acceptance_result": acceptance_result,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(payload, encoding="utf-8")
    if acceptance_output_path is not None:
        resolved = acceptance_output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(acceptance_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--acceptance-output", type=Path, default=None)
    parser.add_argument("--expected-movement-profile", default="visible")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = HttpJsonClient(args.base_url, timeout_seconds=args.timeout_seconds)
    report = run_rehearsal(
        client,
        steps=args.steps,
        expected_movement_profile=args.expected_movement_profile,
        output_path=args.output,
        acceptance_output_path=args.acceptance_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["passed"]:
        print("PASS: scripted browser trial rehearsal completed")
    else:
        print("FAIL: scripted browser trial rehearsal found unexpected gaps")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
