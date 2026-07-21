# Stage H3 Level 7 — Logged-Time Drivable Renderer MVP

Date: 2026-07-21

## Decision And Scope

The accepted PandaSet scene-040 static-8k checkpoint is frozen. This run did
not train another model. It tested the shortest geometry-grounded path toward
human driving:

```text
PandaSet logged pose(t)
+ bounded human forward / left / yaw offset
→ EgoState
→ one rigid six-camera transform
→ SplatAD static-8k GPU rendering
```

The operator is expected only to drive. Dynamic traffic is deferred for this
low-interference integration segment, not accepted as correct and not waived
for later traffic scenarios or autonomous-driving evaluation.

## Implementation

- `SplatADLoggedRenderer` implements the common repository Renderer protocol.
- It merges all 80 logical frames and all six PandaSet cameras from the train
  and held-out splits after the upstream crop/calibration cache is applied.
- `EgoState.time` selects the nearest logged rig pose over the real
  7.899239-second sequence; it never loops or silently extrapolates.
- Every requested camera retains its own native intrinsics, extrinsics,
  rolling-shutter metadata, and timestamp. A single bounded rigid ego offset
  is composed onto all six views.
- `LoggedEgoOffsetController` automatically advances log time while integrating
  normalized throttle, brake, and steering into a conservative nearby-pose
  envelope: +/−0.5 m forward, +/−0.25 m left, and +/−2 degrees yaw.
- The reusable entrypoint validates that the accepted checkpoint is exactly
  step 7,999 and exits nonzero when an automated smoke gate fails.

## Exact Run

Host and GPU:

```text
host: stf-precision-3680
GPU: NVIDIA GeForce RTX 4090 D, 24 GB
H3 environment: /home/yawei/stage3_external/envs/h3_splatad
checkpoint: scene_040_splatad_static_8000 step-000007999.ckpt
output scale: 0.5
observations: 80 at dt=0.1 s
```

Command:

```bash
scripts/run_stage_h3_pandaset_040.sh logged-renderer-smoke
```

Artifacts remain outside Git:

```text
/home/yawei/stage3_external/artifacts/scene_040_logged_renderer_mvp/
├── frames/000.jpg ... 079.jpg
├── pose_probe_centre.jpg
├── pose_probe_left_p0.10m.jpg
├── pose_probe_yaw_p1.0deg.jpg
├── scene_040_logged_renderer_mvp_7p9s.mp4
└── stage_h3_logged_renderer_smoke.json
```

## Results

Automated Renderer-level smoke:

| Check | Result |
|---|---:|
| Accepted checkpoint step | PASS — 7,999 |
| Six RGB outputs present, uint8 H×W×3 | PASS — 80/80 observations |
| Logged frame progression | PASS — logical frames 0 through 79 |
| Exact reset pixel repeatability | PASS — all six cameras |
| Front-view response to +0.10 m left | PASS — mean absolute change 9.069/255 |
| Front-view response to +1 degree yaw | PASS — mean absolute change 14.486/255 |
| Six-camera warm Renderer latency | PASS — p50 69.15 ms, p95 74.37 ms, max 75.71 ms |

The Renderer-only rate is about 13.4 complete observations/s at this scale, so
it clears the current 10 observations/s target. The latency excludes physical
input events, mosaic/JPEG encoding, browser transport, and display refresh and
must not be reported as full input-to-screen latency.

The maximum native exposure-time spread within one six-camera logical frame
was 81.392 ms. This is preserved PandaSet sensor asynchrony, not same-instant
capture. “Six-camera synchronization” here means correct logical grouping and
no stale/jumping camera, not identical exposure timestamps.

Manual visual sampling of frames 0, 20, 40, 60, and 79 found the central road,
double-yellow lines or intersection markings, curbs, and view directions
readable without a black hole or contradictory road layout. This is supporting
evidence, not a completed human-driving trial or exhaustive artifact mask.

## Independent Review And Corrections

An independent agent reviewed the design, code, tests, raw JSON, and rendered
images without relying on the implementation conclusion. It found two control
boundary bugs: oversized/NaN controls were not normalized at the controller
boundary, and brake from rest could create reverse motion. Both were fixed and
covered by tests before the full 80-frame rerun.

The final independent conclusion was: logged-time plus bounded human offset
plus six-camera Renderer-level MVP smoke passes. It explicitly did not accept
the full human interaction loop, full input-to-display latency, or dynamic
traffic correctness.

## Browser Driving Follow-Up

The same iteration added a minimal browser loop rather than a cockpit UI. It
automatically advances the real trajectory at 10 Hz, accepts only held
W/S/A/D and R reset, shows all six labeled cameras, and reports browser-side
frame-update time. The project host binds port 8766 for access at
`http://100.116.66.57:8766` over Tailscale.

This is a single-driver service. Only one tab/client may drive it at a time.
The log pose advances automatically; W/S change only the bounded relative
forward-offset speed and S does not pause the recorded trajectory. The page
states this permanently so the operator is not asked to infer the control
semantics.

The first 0.5-scale request exposed a real failure: rendering, resizing, and
JPEG preparation took about 116.8 ms from server receipt. The browser default
was therefore changed to render directly at 0.25 scale, which still produces a
1440×574 six-camera JPEG and avoids redundant downscaling. Ten sequential
warmed throttle requests then measured:

```text
Renderer: approximately 67.11-71.37 ms
server control receipt → six-camera JPEG ready:
  p50 75.89 ms
  p95 78.19 ms
  max 78.76 ms
state progression: 0.0 → 1.0 s, forward offset 0.0 → 0.275 m
```

The HTML page, JPEG endpoint, W-throttle state change, and exact R-reset
response all passed over local HTTP. This proves the service path and clears
the server-side 100 ms target. It still does not measure a physical key event,
network transfer to another computer, browser decoding, or monitor refresh;
the page displays that combined browser-side update time for the operator run.

## Acceptance Boundary And Next Action

This result passes the backend integration needed to build the driving loop.
It does not yet pass all six criteria in
`docs/drivability_acceptance_criteria.md`:

- full control-event-to-screen latency has not been measured;
- a human has not yet driven and reset the whole segment interactively;
- road/curb continuity has sampled visual evidence but no completed operator
  trial;
- nearby-pose probes are narrow and do not certify the entire envelope;
- static/baked vehicles remain unsuitable wherever they affect driving
  meaning.

Next, run the browser from the real Tailscale client through the complete
segment and record its displayed key/request-to-frame time plus the remaining
six drivability gates. Do not train a new dynamic checkpoint before that run
reveals a road- or obstacle-relevant failure.

## Visible Counterfactual-Movement Follow-Up

Date: 2026-07-21

The first operator impression was that the logged browser's human-controlled
offset was too small to see clearly. I therefore separated the movement
envelope into two explicit profiles without changing or retraining the
accepted static-8k checkpoint:

- `safe`: the original conservative envelope, +/−0.5 m forward, +/−0.25 m
  left, and +/−2 degrees yaw;
- `visible`: a demonstration envelope, +/−2.0 m forward, +/−0.75 m left, and
  +/−8 degrees yaw, with faster acceleration and yaw response.

The safe baseline was rerun first at 12 observations to confirm the original
counterfactual path was real but subtle:

```text
artifact: /home/yawei/stage3_external/artifacts/scene_040_counterfactual_baseline
front +0.10 m left mean absolute change: 9.069/255
front +1 degree yaw mean absolute change: 14.486/255
maximum offset after 1.1 s scripted control: x=0.109 m, y=0.0004 m
```

Then the visible profile rendered the complete 80-frame, six-camera sequence:

```text
artifact: /home/yawei/stage3_external/artifacts/scene_040_counterfactual_visible
movement profile: visible
front +1.25 m forward mean absolute change: 13.530/255
front +0.60 m left mean absolute change: 16.300/255
front +7 degree yaw mean absolute change: 33.934/255
logical frames: 0 through 79
six-camera Renderer p95: 73.62 ms at 0.5 output scale
all automated smoke gates: PASS
```

The new comparison image is:

```text
/home/yawei/stage3_external/artifacts/scene_040_counterfactual_visible/counterfactual_front_probe_comparison.jpg
```

The new 7.9-second visible-motion video is:

```text
/home/yawei/stage3_external/artifacts/scene_040_counterfactual_visible/scene_040_counterfactual_visible_7p9s.mp4
```

The browser entrypoint now defaults to `visible`, while
`logged-renderer-smoke` defaults to `safe` unless
`H3_LOGGED_MOVEMENT_PROFILE=visible` is set. A local browser-service check on
port 8777 confirmed that default visible browser controls are active: three
held `W+A` ticks reached x=0.239 m and yaw=7.2 degrees, with server
control-to-JPEG responses around 78-87 ms.

This follow-up improves demonstrability. It does not certify that the full
visible envelope is safe for formal driving acceptance; use
`H3_BROWSER_MOVEMENT_PROFILE=safe` for the conservative acceptance pass.
