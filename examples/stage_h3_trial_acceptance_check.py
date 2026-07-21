#!/usr/bin/env python3
"""Check whether an H3 browser trial JSON is acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim.trial_acceptance import (  # noqa: E402
    TrialAcceptanceConfig,
    evaluate_trial_report,
    load_trial_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-scene", default="040")
    parser.add_argument("--expected-checkpoint-step", type=int, default=7999)
    parser.add_argument("--expected-movement-profile", default=None)
    parser.add_argument("--min-sample-count", type=int, default=70)
    parser.add_argument("--min-reset-count", type=int, default=1)
    parser.add_argument("--min-browser-input-samples", type=int, default=1)
    parser.add_argument(
        "--max-browser-request-to-image-p95-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--max-browser-input-to-image-p95-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--max-server-control-to-jpeg-p95-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--max-camera-time-spread-p95-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--allow-no-control-input",
        action="store_true",
        help="Do not require at least one non-empty W/S/A/D sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrialAcceptanceConfig(
        expected_scene=args.expected_scene,
        expected_checkpoint_step=args.expected_checkpoint_step,
        expected_movement_profile=args.expected_movement_profile,
        min_sample_count=args.min_sample_count,
        min_reset_count=args.min_reset_count,
        min_browser_input_samples=args.min_browser_input_samples,
        max_browser_request_to_image_p95_ms=(
            args.max_browser_request_to_image_p95_ms
        ),
        max_browser_input_to_image_p95_ms=args.max_browser_input_to_image_p95_ms,
        max_server_control_to_jpeg_p95_ms=args.max_server_control_to_jpeg_p95_ms,
        max_camera_time_spread_p95_ms=args.max_camera_time_spread_p95_ms,
        require_control_input=not args.allow_no_control_input,
    )
    report = load_trial_report(args.trial_json)
    result = evaluate_trial_report(report, config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["passed"]:
        print("PASS: H3 browser trial acceptance evidence is complete")
    else:
        print("FAIL: H3 browser trial acceptance evidence is incomplete")
        print("failed gates: " + ", ".join(result["failures"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
