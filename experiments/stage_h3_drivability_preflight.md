# Stage H3 Level 7B — Drivability Preflight And Trial Recording

Date: 2026-07-21

## Scope

This follow-up did not train a new model. It keeps the accepted PandaSet
scene-040 static-8k checkpoint fixed and turns the current drivability claims
into repeatable evidence:

```text
accepted static-8k SplatAD checkpoint
→ logged-time Renderer
→ profiled ego offsets
→ automated backend preflight
→ browser trial JSON recording
```

The result is not a completed human driving acceptance run. It is a preflight
that should pass before a human operator trial is treated as meaningful.

## Implementation

- Added `examples/stage_h3_drivability_preflight.py`.
- Added `drivability-preflight` to `scripts/run_stage_h3_pandaset_040.sh`.
- Added `BrowserTrialRecorder` for browser-side trial samples and reset events.
- The browser now exposes `/trial.json` and writes a server-owned JSON report
  by default.
- The browser now also saves the operator's five manual drivability review
  gates into the same trial JSON.
- Added dependency-light tests for the preflight control script and recorder.

## Exact Preflight Run

Command:

```bash
scripts/run_stage_h3_pandaset_040.sh drivability-preflight
```

Runtime:

```text
host: stf-precision-3680
checkpoint: scene_040_splatad_static_8000 step-000007999.ckpt
movement profile: visible
output scale: 0.5
observations: 80 at dt=0.1 s
```

Artifacts:

```text
/home/yawei/stage3_external/artifacts/scene_040_drivability_preflight/
├── counterfactual_front_preflight.jpg
├── sequence_contact_sheet.jpg
├── sequence_samples/
└── stage_h3_drivability_preflight.json
```

## Automated Results

All 17 automated preflight gates passed:

```text
accepted_static_8k_checkpoint: true
all_six_camera_outputs_valid: true
same_time_counterfactual_pixels_change: true
same_logical_frame_for_counterfactual_probes: true
positive_and_negative_yaw_are_distinct: true
logical_frames_never_go_backward: true
logical_frame_progressed: true
camera_times_and_sources_present: true
sensor_spread_reported_under_100ms: true
reset_exact_pixel_repeatability: true
scripted_states_repeatable: true
rendered_states_match_scripted_states: true
final_state_pixel_repeatability: true
script_exercises_forward_and_yaw_motion: true
inside_profile_limits: true
outside_profile_limits_rejected: true
renderer_observation_p95_at_most_100ms: true
```

Important numbers:

```text
logical frames: 0 through 79
six-camera Renderer latency at 0.5 scale:
  p50 69.07 ms
  p95 73.36 ms
  max 76.08 ms
sensor time spread:
  p50 81.32 ms
  p95 81.37 ms
  max 81.39 ms
front counterfactual mean absolute differences:
  forward +1.25 m: 13.530/255
  left +0.60 m: 16.300/255
  yaw +7.0 deg: 33.934/255
  yaw -7.0 deg: 33.537/255
  yaw +7 deg vs yaw -7 deg: 48.827/255
motion envelope reached:
  forward 2.0 m
  left 0.75 m
  yaw 8.0 deg
```

The preflight report type is
`h3_drivability_preflight_not_human_acceptance` to avoid confusing automated
backend gates with full driving acceptance.

## Browser Trial Recording Validation

Validation command:

```bash
H3_BROWSER_PORT=8778 \
H3_BROWSER_TRIAL_OUTPUT=/home/yawei/stage3_external/artifacts/scene_040_browser_trial_validation/browser_trial.json \
scripts/run_stage_h3_pandaset_040.sh logged-browser
```

Validated endpoints:

```text
GET /trial.json
POST /tick?keys=wa
POST /trial-sample
POST /reset
GET /trial.json
```

Evidence:

```text
initial report: checkpoint step 7999, movement profile visible, sample_count 0
W+A tick: logical_frame 1, yaw 2.4 deg, server_control_to_jpeg 89.88 ms
trial sample: sample_count 1, browser_input_to_image_ms 123.0 test value
reset: logical_frame 0, reset_count 1
trial output: /home/yawei/stage3_external/artifacts/scene_040_browser_trial_validation/browser_trial.json
```

The browser recorder distinguishes:

- `browser_request_to_image_ms`: browser request start to image load;
- `browser_input_to_image_ms`: key/mouse/touch edge to first loaded image that
  reflects a sampled control state.

Neither value includes monitor scan-out or an external hardware latency sensor.

## Manual Review Endpoint Validation

Validation command:

```bash
H3_BROWSER_PORT=8779 \
H3_BROWSER_TRIAL_OUTPUT=/home/yawei/stage3_external/artifacts/scene_040_browser_review_validation/browser_trial.json \
scripts/run_stage_h3_pandaset_040.sh logged-browser
```

Validated endpoints:

```text
GET /trial.json
POST /trial-review
POST /trial-review
GET /trial.json
```

Evidence:

```text
initial report: manual_review_count 0, all five gates marked missing
blocking review: manual_review_count 1, physical_input_display_latency unsure
  and dynamic_traffic_decision_impact fail blocked acceptance
all-pass review: manual_review_count 2, manual_review_all_passed true and
  manual_review_blocking_gates empty
trial output: /home/yawei/stage3_external/artifacts/scene_040_browser_review_validation/browser_trial.json
```

## Remaining Human Review

The preflight intentionally leaves these items open:

- road, lane, curb, and horizon continuity by visual review;
- steering direction by eye using the counterfactual front sheet;
- nearby-pose artifacts that could affect driving;
- physical key-to-display latency from the real Tailscale client;
- dynamic-traffic artifacts that could change obstacle or road decisions.

The next acceptance run should use the browser, drive the complete 7.899 s log,
then preserve `/trial.json`; it contains both browser-side latency samples and
the operator's manual review of the five driving-relevant gates.
