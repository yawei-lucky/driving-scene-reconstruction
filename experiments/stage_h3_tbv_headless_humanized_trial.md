# Stage H3 TbV No-Browser Humanized Trial

Date: 2026-07-25

## Question

Can the accepted TbV static-8k branch checkpoint complete one reproducible
straight/right driving trial without browser control, while deliberately
testing small A/D-like departures, recovery, auto-to-manual-like takeover,
reset, endpoint braking, and usable correction margin?

This is a deterministic human-like control test. It is not evidence that a
physical human operated the controls.

## Implementation

The new headless entry point calls the vehicle model, route adapter, and
three-front-camera renderer directly. It uses a 20 Hz simulation/render
interval and writes one H.264 video plus per-frame JSON evidence.

The common approach previously changed directly from the right-traversal
centreline to the registered straight-traversal centreline. Although both
routes observe the junction, their registered poses are not identical. The raw
handoff consumed almost the full +/-1 m corridor before a human could correct.

The route adapter now retains the observed common centreline through the
anchor and applies a smooth 12 m join to the observed straight centreline.
This join stays inside the segment observed by both traversals. The renderer
still uses the straight traversal's calibrated profile after branch selection.

The scripted sequence for each route is:

1. reset at common progress -20 m;
2. begin with the passive route follower;
3. take over near -19.5 m;
4. request a small A-like left offset and recover;
5. request a small D-like right offset and recover both lateral position and
   heading;
6. select straight or right;
7. follow the selected route at up to 4.0 m/s;
8. brake to rest at the supported endpoint.

## Acceptance Contract

- all three front cameras finite for every rendered sample;
- no support-boundary hit;
- both A and D departures reach at least 0.28 m;
- both recovery phases complete;
- at least 0.25 m emergency corridor margin throughout either route;
- at least 0.40 m correction margin through the straight handoff;
- no more than 8 degrees of heading error through the straight handoff;
- complete brake-to-rest and a distinct reset before each route.

The correction-margin gate is intentionally stricter than merely remaining
inside +/-1 m.

## Results

The accepted v4 run rendered 665 direct GPU samples into a 915-frame,
45.75-second, 1600x668 H.264 video at 20 fps.

| Measurement | Straight | Right |
| --- | ---: | ---: |
| Simulated route duration | 18.00 s | 15.25 s |
| Maximum left offset | +0.378 m | +0.378 m |
| Maximum right offset | -0.339 m | -0.339 m |
| Minimum margin anywhere | 0.622 m | 0.622 m |
| Minimum handoff margin | 0.921 m | 0.979 m |
| Maximum handoff heading error | 4.97 deg | 3.52 deg |
| Final speed | 0.000 m/s | 0.000 m/s |
| Three-camera renderer p50 | 38.81 ms | 40.65 ms |
| Three-camera renderer p95 | 43.44 ms | 43.54 ms |

Every automated gate passed. The first warmed render outlier reached about
435 ms; it is preserved in the JSON and not included in the warmed p95 claim.

Three earlier attempts remain outside Git as failed development evidence:

- v1 exposed a right-turn minimum margin of only 0.029 m and was rejected even
  though the original gate only checked boundary crossing;
- v2 restored right-turn reserve but the straight handoff reached 8.36 degrees;
- v3 increased the smooth join to 12 m and improved lateral margin, but did
  not change the residual heading error;
- v4 moved the A/D sequence earlier and required heading settle before cruise.

## Visual Driving Judgment

Manual review of the v4 contact sheet, a four-second timeline sheet, and full
frames at 15, 19, and 39 seconds found:

- the forward road, route direction, curbs, and usable lane corridor remain
  continuous and readable through both routes;
- the straight/right profile change does not create a blocking road jump;
- trees, parked vehicles, and the panorama edges still smear or stretch;
- a dark static smear remains near the right edge of the straight corridor and
  the right route contains a baked white van, but neither inspected segment
  creates a false obstacle that blocks the selected driving line.

Decision: pass for this restricted static-route driving test. Do not use this
result to claim dynamic-traffic truth, collision truth, unrestricted lateral
driving, or a completed real-human trial.

## Artifacts

All heavy outputs remain outside Git:

- video:
  `/home/yawei/stage3_external/artifacts/tbv_headless_humanized_20260725_v4/tbv_headless_humanized_straight_right.mp4`
- JSON:
  `/home/yawei/stage3_external/artifacts/tbv_headless_humanized_20260725_v4/tbv_headless_humanized_trial.json`
- contact sheet:
  `/home/yawei/stage3_external/artifacts/tbv_headless_humanized_20260725_v4/contact_sheet.jpg`
- 915 encoded source frames under the same artifact root.

No browser, HTTP service, or training run was used.

## Decision

Retain the new handoff and headless trial as the cheap TbV regression test.
The immediate next gate was the isolated released MTGS checkpoint on 24 GB,
followed by a multi-trajectory lateral corridor probe if it loaded. Both
subsequently passed and are recorded in
`stage_h3_mtgs_checkpoint_gate.md`.
