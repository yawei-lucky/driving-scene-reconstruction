# Stage H3 MTGS Simulated-Driver Closed Loop

Date: 2026-07-25

## Question

Can a deterministic simulated driver, rather than a directly prescribed camera
path, operate the vehicle model through the released MTGS route while the
resulting world state drives fixed-time front-camera rendering?

This is the control/render integration gate after
`stage_h3_mtgs_continuous_drive.md`. It does not use a browser, physical human,
training run, dynamic scene time, or collision model.

## Closed-Loop Contract

The loop runs at 20 Hz:

```text
pure-pursuit route follower plus speed controller
→ HumanControl(steer, throttle, brake)
→ SimpleVehicleModel
→ fail-closed +/-5 m route support
→ progress/lateral/heading render query
→ travel-3 CAM_F0 world pose
→ released MTGS step-30,000 checkpoint
→ RGB frame plus JSON evidence
```

The driver targets 12 m/s. A smooth route-relative programme asks for
centre/left/centre/right/centre motion with a requested 3 m amplitude. Steering
is rate-limited to 1.8 normalized units per second, and the final approach uses
bounded braking. The actual lateral motion is an output of the vehicle model;
the renderer does not receive the target path directly.

The MTGS camera-right axis is almost exactly opposite the control corridor's
route-left axis. The runner measures this alignment per frame
(-0.99994 to -0.99959) and applies the measured sign/scale rather than assuming
that control and camera lateral coordinates have the same sign.

## Final v2 Result

The accepted RTX 4090 D run produced:

| Measurement | Result |
| --- | ---: |
| Frames / encoded duration | 183 / 9.15 s |
| Resolution / frame rate | 960x540 / 20 FPS |
| Maximum vehicle speed | 11.994 m/s |
| Actual maximum left offset | +2.541 m |
| Actual maximum right offset | -2.671 m |
| Minimum support margin | 2.329 m |
| Maximum absolute heading error | 16.675 deg |
| Maximum absolute steer | 0.445 |
| Maximum per-tick steer change | 0.0636 |
| Maximum brake | 0.739 |
| Boundary hits | 0 |
| Endpoint stop | yes |
| Finite frames | 183 / 183 |
| Render p50 / p95 | 6.72 / 12.16 ms |
| Render/save wall time | 3.06 s |
| Peak CUDA reserved | 1.455 GiB |

The vehicle accelerated from rest, reached the cruise-speed region, followed
the gentle route bend, completed both lateral excursions and recoveries, then
braked to rest at progress 83.8 m. `direct_camera_path_used` is false in the
report.

The adjacent-frame image-body MAD measured 8.54 p50, 15.51 p95, and 17.90
maximum. The maximum transition is frame 112 to 113, at progress
51.2-51.9 m while the vehicle leaves the negative lateral extremum. The largest
values form a local recovery cluster rather than one isolated route-knot jump.
This is only a discontinuity diagnostic, not a realism metric.

## Visual Review

Manual review of the v2 contact sheet, both lateral extrema, the maximum-MAD
transition, and the endpoint found:

- the road, lane markings, curb, and forward corridor remain continuous;
- the vehicle-driven left/right motion and recovery agree with the overlays;
- no inspected frame creates a blocking false obstacle;
- the maximum-MAD pair changes smoothly during recovery;
- foliage, signs, fence edges, and some curb regions remain soft or stretched;
- fixed scene time leaves the road unusually empty and does not test actor
  motion.

Decision: pass the simulated-driver, vehicle-dynamics, support-boundary, and
single-front-camera render closed loop for this 84 m block. Do not use it to
claim a long route, physical-human driving, dynamic traffic truth, collision
truth, or surround-camera quality.

## Development Record

- v1 was the first complete GPU closed loop and passed. Its video is
  byte-identical to v2.
- v2 retains the same deterministic control and video while adding maximum
  steer-step and adjacent-frame image-body diagnostics to the machine report.

The rejected/earlier evidence is retained outside Git.

## Artifacts

- video:
  `/home/yawei/stage3_external/artifacts/mtgs_autodrive_20260725_v2/mtgs_autodrive_12mps_closed_loop.mp4`;
- video SHA-256:
  `c8deea497071d72341b0243a3feeb5a54b93cd481729caa56924c38191aa291a`;
- JSON:
  `/home/yawei/stage3_external/artifacts/mtgs_autodrive_20260725_v2/mtgs_autodrive.json`;
- contact sheet:
  `/home/yawei/stage3_external/artifacts/mtgs_autodrive_20260725_v2/contact_sheet.jpg`;
- 183 source frames under `frames/` in the same artifact directory.

Run the final gate with:

```bash
scripts/run_stage_h3_mtgs_gate.sh autodrive
```

## Next Gate

The six published MTGS blocks are geographically separate and only about
57-105 m long, so another released checkpoint does not create a continuous
long route. The next gate is data-first: identify one contiguous 300-500 m
nuPlan-compatible route with repeated traversal coverage, partition it into
overlapping 80-100 m reconstruction tiles, and verify one pair of genuinely
adjacent tiles before implementing checkpoint streaming. Do not loop this
84 m scene or concatenate unrelated blocks and call the result a long route.

Status on 2026-07-25: the data part of this gate passed on a 610.072 m repeated
TbV route. Seven overlapping 100 m tiles and one real adjacent pair are
recorded in `stage_h3_tbv_long_route_tile_audit.md`. The remaining gate is a
two-tile 100-step reconstruction and overlap-render seam smoke.
