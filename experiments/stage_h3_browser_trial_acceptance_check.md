# Stage H3 Level 7C — Browser Trial Acceptance Check

Date: 2026-07-21

## Scope

This follow-up did not train or render a new model. It consumes the browser
trial JSON produced by the accepted static-8k H3 browser loop and decides
whether that saved run is complete enough to count as driving-acceptance
evidence:

```text
browser `/trial.json`
→ trial acceptance checker
→ pass/fail evidence JSON
```

The checker is deliberately not a visual model. It cannot decide whether a lane
looks good by itself; it enforces that the human operator drove the full
segment, saved the required manual verdicts after the run, and recorded timing
evidence inside the declared budget.

## Implementation

- Added `src/driving_scene_reconstruction/sim/trial_acceptance.py`.
- Added `examples/stage_h3_trial_acceptance_check.py`.
- Added `trial-check` to `scripts/run_stage_h3_pandaset_040.sh`.
- Added dependency-light tests covering passing and failing trial evidence.

## Default Gates

The default Stage-H3 trial check requires:

```text
scene: 040
checkpoint step: 7999
movement profile: visible
minimum samples: 70
minimum reset events: 1
minimum browser input-latency samples: 1
browser request-to-image p95: <= 100 ms
browser input-to-image p95: <= 100 ms
server control-to-JPEG p95: <= 100 ms
camera time-spread p95: <= 100 ms
manual review: all five gates pass
manual review timing: latest review after enough samples and completed log
```

The early-review guard matters because a user could otherwise click all manual
gates as `pass` before the browser actually traverses the segment.

## Command

After a real browser driving run:

```bash
scripts/run_stage_h3_pandaset_040.sh trial-check
```

Defaults:

```text
input:  /home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial.json
output: /home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial_acceptance_check.json
```

To check another saved trial:

```bash
H3_TRIAL_JSON=/path/to/browser_trial.json \
H3_TRIAL_CHECK_OUTPUT=/path/to/browser_trial_acceptance_check.json \
scripts/run_stage_h3_pandaset_040.sh trial-check
```

## Validation

The lightweight test suite now includes synthetic browser trial reports. The
full repository suite passed:

```text
python3 -m unittest discover -s tests -v
55 tests passed
```

The new synthetic reports prove:

- a complete 80-sample trial with reset, input latency, and all manual gates
  passing is accepted;
- an all-pass manual review saved before the completed run is rejected;
- missing reset is rejected;
- missing physical input latency is rejected;
- high browser/request latency is rejected;
- any `unsure` manual gate is rejected;
- a mismatched required movement profile is rejected.

The checker was also run on the earlier manual-review endpoint validation
artifact:

```bash
python3 examples/stage_h3_trial_acceptance_check.py \
  --trial-json /home/yawei/stage3_external/artifacts/scene_040_browser_review_validation/browser_trial.json \
  --output /tmp/stage_h3_trial_check_expected_fail.json
```

It failed as intended because that artifact has an all-pass manual review but
no complete driving samples, no reset, no operator control sample, and no
browser latency distributions. This proves the checker does not treat a manual
all-pass click alone as acceptance evidence.

This is still not a completed human trial. It is the reproducibility gate that
should be run immediately after the human browser trial.
