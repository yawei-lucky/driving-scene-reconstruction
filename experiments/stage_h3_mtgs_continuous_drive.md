# Stage H3 MTGS Continuous High-Speed Drive Smoke

Date: 2026-07-25

## Question

Can the released MTGS Singapore checkpoint render a continuous, high-speed,
wide-lateral driving path rather than only isolated nearby poses, while
retaining road information useful to a driver?

This is the next gate after
`stage_h3_mtgs_checkpoint_gate.md`. It uses inference only and deliberately
keeps scene time fixed.

## Path And Evidence Contract

The probe selects travel 3 `CAM_F0` and extracts its 17 evaluation camera
poses. Their observed centreline is 84.252 m long. The tested path starts at
progress 5 m and ends at 77 m:

- simulated duration: 6.0 s;
- speed: 12.0 m/s;
- distance: 72.0 m;
- output: 121 endpoint-inclusive states at 20 fps, encoded as 6.05 s;
- lateral path: 0 -> +4 -> 0 -> -4 -> 0 m;
- each lateral leg uses cosine ease-in/ease-out, so the route-relative heading
  is zero at the start, both extrema, the centre crossing, and the end;
- the largest added route-relative heading is under 20 degrees;
- scene time, travel ID, frame token, and appearance ID remain fixed at the
  first travel-3 front-camera anchor.

The +/-5 m support boundary comes from the accepted Level-9M pose grid. It is a
reviewed reconstruction envelope, not a road-edge or collision certificate.

## Final v4 Result

The final RTX 4090 D / step-30,000 run produced:

| Measurement | Result |
| --- | ---: |
| MTGS setup and checkpoint load | 19.36 s |
| Frames | 121 |
| Resolution | 960x540 |
| Finite frames | 121 / 121 |
| Maximum absolute lateral offset | 4.000 m |
| Minimum remaining probe margin | 1.000 m |
| Render p50 | 6.43 ms / 155.54 FPS |
| Render p95 | 12.10 ms / 82.62 FPS |
| 121-frame render/save wall time | 1.82 s |
| Peak CUDA reserved | 1.455 GiB |

As a cheap interpolation-jump diagnostic, the mean absolute grayscale
difference between adjacent image bodies measured 13.72 p50, 17.02 p95, and
18.01 maximum. The maximum is close to the negative lateral extremum and is
not an isolated route-knot spike. This diagnostic does not measure realism; it
only supports the manual observation that no hard frame jump occurred.

## Visual Driving Review

Manual review of the final contact sheet and individual endpoint/extreme
frames found:

- road direction, lane markings, curb, and the open driving corridor remain
  readable throughout all 72 m;
- both +/-4 m excursions and both recoveries remain visually continuous;
- no sampled frame creates a blocking false obstacle on the driving surface;
- near foliage, signs, fences, and curb edges stretch or smear, especially
  during large lateral/yaw excursions;
- the road is unusually empty because time and dynamic state are frozen.

Decision: pass for a fixed-time, front-camera, continuous spatial driving
smoke. Do not use it to claim dynamic traffic truth, collision truth,
multi-camera surround quality, or completed human keyboard driving.

## Minimal Control Adapter

The same run constructed a small adapter from the actual 84.252 m camera
route:

- `SimpleVehicleModel` owns speed and steering;
- speed cap is 15 m/s;
- spawn progress is 5 m with the full 5 m support margin;
- the adapter exposes progress, lateral offset, heading error, and remaining
  support margin for each render query;
- leaving the +/-5 m or +/-30-degree envelope fails closed and stops;
- reaching the route endpoint stops;
- dependency-light tests cover acceleration beyond 10 m/s, the speed cap,
  boundary failure, endpoint stop, and spawn margin.

The adapter is not yet connected to a local keyboard/display loop. The video
uses the deterministic cosine path rather than `HumanControl`, so the two
pieces are separate evidence: renderer continuity and control-boundary
plumbing.

## Development Attempts

Heavy artifacts from all attempts remain outside Git:

- v1 was numerically and visually continuous but started with about 19 degrees
  of lane-change heading instead of stable straight driving;
- v2 fixed the heading endpoints but its 120 sampled states ended at 5.95 s
  while the target report said 6.0 s;
- v3 included both simulation endpoints and passed;
- v4 repeats the accepted v3 video while also constructing the actual-route
  control adapter and recording its contract.

The rejected v1/v2 artifacts are retained rather than rewritten.

## Artifacts

- video:
  `/home/yawei/stage3_external/artifacts/mtgs_continuous_drive_20260725_v4/mtgs_fixed_time_12mps_lane_change.mp4`;
- video SHA-256:
  `117bfcbc0bb24645b10172ca46bee53de9ec71ba7475ad11f0f23c2fc883b18f`;
- JSON:
  `/home/yawei/stage3_external/artifacts/mtgs_continuous_drive_20260725_v4/mtgs_continuous_drive.json`;
- contact sheet:
  `/home/yawei/stage3_external/artifacts/mtgs_continuous_drive_20260725_v4/contact_sheet.jpg`;
- 121 source frames under `frames/` in the same artifact directory.

Run the final gate with:

```bash
scripts/run_stage_h3_mtgs_gate.sh continuous-drive
```

## Next Gate

Connect `MtgsDrivingAdapter` and the existing camera-pose sampler inside one
local, no-browser front-camera loop. Run one short actual keyboard trial at
10-12 m/s, including a bounded lane change, recovery, brake, boundary
rejection, and reset. Keep fixed scene time until that control/render bridge
is stable.
