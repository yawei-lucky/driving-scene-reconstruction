# Stage H3 Level 7D — Scripted Browser Trial Rehearsal

Date: 2026-07-21

## Scope

This follow-up does not train a new model and does not replace a human driving
trial. It rehearses the live H3 browser HTTP loop before the operator run:

```text
logged-browser service
→ scripted W/S/A/D HTTP client
→ `/trial.json`
→ trial acceptance checker
→ rehearsal pass/fail
```

The key design choice is that the rehearsal submits all manual visual gates as
`unsure`. Therefore a successful rehearsal means the machine plumbing is ready;
it never means the reconstructed scene has passed human visual driving
acceptance.

## Implementation

- Added `examples/stage_h3_browser_trial_rehearsal.py`.
- Added `trial-rehearsal` to `scripts/run_stage_h3_pandaset_040.sh`.
- Added dependency-light tests with a fake browser service.

## What It Exercises

The scripted client drives these live endpoints:

```text
GET  /trial.json
POST /tick?keys=...
GET  /frame.jpg
POST /trial-sample
POST /reset
POST /trial-review
GET  /trial.json
```

The deterministic control schedule exercises:

```text
W  forward offset
A  left/yaw offset
D  right/yaw offset
S  brake / reverse relative-speed request
```

The rehearsal then runs the same trial-acceptance evaluator used by
`trial-check`. It passes only if the sole acceptance failures are the expected
manual visual gates:

```text
manual_review_all_passed
all_manual_gate_statuses_pass
```

Any machine-plumbing gap—too few samples, missing reset, no input latency
sample, over-budget latency, wrong movement profile, or incomplete log—makes
the rehearsal fail.

## Commands

Terminal 1 starts the live service, preferably with rehearsal-specific output:

```bash
H3_BROWSER_PORT=8780 \
H3_BROWSER_TRIAL_OUTPUT=/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial.json \
scripts/run_stage_h3_pandaset_040.sh logged-browser
```

Terminal 2 runs the rehearsal:

```bash
H3_BROWSER_PORT=8780 \
scripts/run_stage_h3_pandaset_040.sh trial-rehearsal
```

Default outputs:

```text
/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial_rehearsal.json
/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial_acceptance_check.json
```

## Validation

The lightweight tests passed:

```text
python3 -m unittest discover -s tests -v
58 tests passed
```

They verify that:

- the scripted control schedule exercises all four W/S/A/D controls;
- a fake complete browser service passes rehearsal when only manual visual
  gates are `unsure`;
- a short/incomplete fake run fails on non-manual machine gates.

The rehearsal was then run against the real H3 browser service on port 8781.
The service used the accepted static-8k checkpoint and wrote:

```text
/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial.json
/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial_rehearsal.json
/home/yawei/stage3_external/artifacts/scene_040_browser_trial_rehearsal/browser_trial_acceptance_check.json
```

Observed results:

```text
rehearsal result: pass
samples: 79
completed_log: true
reset_count: 1
observed key sets: a, aw, d, s, w
browser request-to-image p95: 77.22 ms
browser input-to-image p95: 82.61 ms over 10 input-change samples
server control-to-JPEG p95: 76.19 ms
camera time-spread p95: 81.37 ms
manual review: all five gates intentionally unsure
acceptance-check failures: manual_review_all_passed,
  all_manual_gate_statuses_pass
unexpected acceptance failures: none
```

The browser service was stopped after the run, and GPU process inspection
showed only the system remote-desktop process remained.

This stage prepares the operator trial. It does not decide whether the scene is
visually drivable.
