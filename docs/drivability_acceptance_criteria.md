# Drivability Acceptance Criteria

Date: 2026-07-21

This document defines whether the reconstructed scene can support driving. It
is intentionally separate from PSNR, SSIM, LPIPS, and general visual-quality
evaluation.

## Required Gates

### 1. Road, Lane, And Curb Continuity

- The drivable corridor remains readable throughout the accepted log segment.
- Lane lines, curbs, road edges, and the horizon do not disappear, duplicate,
  or jump in a way that can change the steering decision.
- Any reconstruction artifact that can be interpreted as an obstacle is a
  failure even if aggregate image metrics remain high.

### 2. Correct Steering Response

- Positive/negative steering produces the expected left/right image motion.
- A small forward, lateral, or yaw offset changes every requested camera from
  the same ego transform; the calibrated rig must not shear apart.
- Control direction is checked independently from visual similarity.

### 3. Multi-Camera Time Consistency

- One simulator observation contains all requested cameras from one logical
  PandaSet frame.
- The original calibrated per-sensor timestamps are preserved and reported.
- No camera may silently reuse a stale frame, skip ahead, or jump to another
  logical timestamp.

### 4. Nearby-Pose Robustness

- Every query inside the declared forward/left/yaw envelope returns finite RGB
  frames for every requested camera.
- Small deviations from the logged trajectory must not create black holes,
  gross geometry tears, or a contradictory road layout.
- Queries outside the certified envelope fail explicitly instead of rendering
  an untrusted view.

### 5. Interaction Latency

- Warmed end-to-end latency is measured from accepted control state to the
  complete multi-camera observation.
- The current human-driving target is six-camera p95 at or below 100 ms
  (at least 10 observations/s) at the declared display resolution.
- Warm-up and steady-state latency are reported separately.

### 6. Reset And Repeatability

- Reset returns to the same log time, ego offset, camera frame, and speed.
- Re-rendering an identical state produces identical frame selection and
  deterministic pixels within the backend's declared tolerance.
- Repeated control scripts produce the same states, frame indices, and output
  metadata.

## MVP Acceptance Rule

The first log-local human-drivable MVP passes only when all six gates above
pass on one declared low-interference segment. Image metrics remain supporting
diagnostics, not substitutes for these gates. Dynamic-traffic correctness is a
separate mandatory gate before strict autonomous-driving evaluation, as
defined in `driver_attention_and_dynamic_traffic_requirements.md`.
