# Stage H3 World-Pose Free-Driving Probe

Date: 2026-07-22

## Purpose

Determine whether the accepted PandaSet scene-040 static-8k checkpoint can be
queried from a vehicle pose that is owned by the simulator instead of the
recorded trajectory. This is the first implementation step toward genuine
free driving. It does not certify a complete drivable corridor.

The test keeps the checkpoint fixed and does not train a model.

## Independent Review Correction

The initial 45-observation run reported zero effective camera scene-time spread
after clearing input rolling-shutter metadata. Independent code and checkpoint
review found that SplatAD still added its learned per-camera
`time_to_center_pixel_adjustment`, producing a real 7.036 ms spread. That claim
was therefore incorrect even though the coordinate transform itself was sound.

The renderer now disables SplatAD camera rolling-shutter compensation for this
frozen-world path, zeros the editing fallback, reports effective camera time,
and proves the result by rendering the same pose at simulation times 0 and 10
seconds with pixel-exact output. Anchor setup is also transactional and checks
the six-camera common interpolation interval. The 206-observation run recorded
below supersedes the initial automated result while preserving it as historical
evidence of the review finding.

## Implementation

Added `SplatADWorldRenderer` with different semantics from the existing logged
renderer:

```text
EgoState.time = simulation time
EgoState.x/y/yaw = absolute pose in an anchor-local metric world frame
source scene time = frozen at one explicit anchor
SimpleVehicleModel = owner of future x/y/yaw/speed
```

The six source-camera poses are interpolated onto the same anchor time before
they become the fixed virtual rig. This removes the approximately 81 ms source
exposure spread from the rig pose. All six rendered cameras then receive the
same planar rigid transform.

The accepted checkpoint is static. The probe therefore zeros the source-log
linear/angular velocity, rolling-shutter duration, and time-to-centre metadata
instead of reusing the recorded vehicle motion as if it described the simulated
vehicle. Physical rolling shutter from simulated vehicle velocity remains a
later implementation item.

The ego reference point is currently the synchronized six-camera centroid.
This is adequate for the first visual/control probe, but it is not yet a formal
rear-axle vehicle reference.

The original `SplatADLoggedRenderer` is unchanged and remains available for
logged-path regression tests.

## Reproduction

Default probe:

```bash
scripts/run_stage_h3_pandaset_040.sh world-pose-probe
```

Final recorded run:

```bash
H3_WORLD_POSE_RENDER_SCALE=0.5 \
H3_WORLD_POSE_PROBE_ROOT=/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050 \
scripts/run_stage_h3_pandaset_040.sh world-pose-probe
```

Fixed inputs:

- scene: PandaSet `040`;
- model: SplatAD static background;
- checkpoint: step `7,999`;
- anchor request: 4.0 s;
- selected anchor: logical frame 40 at 3.999463081 s;
- output scale: 0.5;
- lateral probes: -3, -2, -1, 0, +1, +2, +3 m;
- yaw probes: -10, -5, 0, +5, +10 degrees;
- continuous paths: 30-step straight, 30-step symmetric left/right turns,
  40-step symmetric left/right lane changes, and 15-step full braking from
  3 m/s, all at 0.1 s.

The first shell invocation failed before checkpoint loading because a
comma-separated value beginning with `-3` was parsed as a new command-line
option. The runner now uses `--argument=value`; no render or training occurred
in that failed attempt.

An initial 0.25-scale probe exposed that the anchor templates retained the
source cameras' asynchronous poses. That result was used only for diagnosis.
The renderer was then changed to interpolate every camera trajectory to the
front-camera anchor time, and the final 0.5-scale run below replaced it as the
accepted evidence.

## Final Automated Result

The machine-readable report is outside Git at:

```text
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/world_pose_probe.json
```

Result: **world-pose plumbing pass; motion semantics pass; visual corridor
requires review and is not certified**.

All automated plumbing and motion gates passed:

- accepted checkpoint is step 7,999;
- reset reproduced identical pixels;
- all 206 observations contained six finite `uint8` RGB images;
- every query used the same scene anchor and logical frame;
- simulation time advanced while scene time remained fixed;
- source-log motion metadata was not reused;
- all six camera poses and effective scene times were synchronized;
- the learned per-camera time adjustment was disabled;
- the same pose at simulation times 0 and 10 seconds was pixel-exact;
- left/right turns and lane changes were mirrored;
- straight motion had no lateral/yaw drift;
- both lane changes returned to within 0.626 degrees of the anchor heading;
- braking from 3 m/s stopped at x=1.069781 m, after which pose and pixels were
  exact across the tail frames;
- all requested +/-1, +/-2, and +/-3 m lateral offsets rendered.

The continuous left-turn path ended after 3.0 simulation seconds at:

```text
x = 5.653961 m
y = 1.097882 m
yaw = 21.977802 degrees
speed = 1.869850 m/s
```

This proves that steering now changes the future vehicle trajectory rather
than only rotating a view around the next logged pose.

Warmed six-camera Renderer latency at 0.5 output scale over 206 observations:

```text
p50 = 68.142 ms
p95 = 70.994 ms
max = 72.185 ms
```

The largest recorded front-view near-black-pixel fraction was approximately
0.0046%. This is only an obvious-hole proxy and is not a visual-quality gate.

## Visual Review

Reviewed artifacts:

```text
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/world_trajectory_xy.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/trajectory_left_turn_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/trajectory_right_turn_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/trajectory_left_lane_change_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/trajectory_right_lane_change_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_reviewed_scale_050/trajectory_full_brake_front.jpg
```

Observed:

- the road surface and main street layout remain recognizable across the full
  -3 m to +3 m lateral probe;
- no large black hole appears in the main front-driving area;
- lateral and yaw changes have the expected visual direction;
- the continuous path visibly moves and rotates through the reconstruction;
- near traffic-light poles, tree edges, curb/sidewalk regions, and other close
  geometry stretch or soften as the query moves farther from the source path;
- the final +21.98-degree turn is renderable, but its near-field geometry is
  not clean enough to certify as a trustworthy unrestricted turn.

Interpretation:

- +/-1 m is only a provisional containment boundary for a restricted browser;
- +/-2 m and +/-3 m are coverage/failure diagnostics, not accepted bounds;
- the current result does not yet establish LiDAR/road geometry correctness at
  those offsets and does not test dynamic actors.

Near-black fraction did not expose the visible stretching in close geometry,
so it is retained only as a diagnostic and never used as a corridor gate.

## Restricted Browser Result

Added a separate `world-browser` without changing the logged browser or
reusing its log-specific trial schema:

```bash
scripts/run_stage_h3_pandaset_040.sh world-browser
```

The browser starts stopped, maps W/S/A/D to throttle/brake/steering, caps speed
at 2 m/s and wheel angle at 15 degrees, and contains the vehicle to x +/-6 m,
y +/-1 m, yaw +/-15 degrees. It renders a candidate before committing state.
Crossing the boundary keeps the previous valid pose, stops, reports the reason,
and asks for reset.

GPU/HTTP rehearsal covered left acceleration, right recovery, coasting, full
braking, reset, and a deliberate yaw boundary. The deliberate hit stopped at
x=2.582 m, y=0.324 m, yaw=14.304 degrees when the rejected candidate would have
reached 15.400 degrees. Sampled warm server control-to-JPEG requests were about
68-74 ms at 0.25 scale. This is not a physical input-to-display measurement or
a human corridor acceptance.

## Decision And Next Action

Accept the new world-coordinate Renderer and vehicle-path plumbing as an
implemented backend milestone. Do not accept the full +/-3 m corridor yet.

Next:

1. retain the completed symmetric/straight/braking suite as a regression gate;
2. repeat y=-1/0/+1 m probes at multiple forward stations and measure
   road-region geometry/depth where static LiDAR support permits;
3. record a human driving review of the implemented restricted browser;
4. replace the anchor rectangle with a data-supported road-corridor model if
   multi-station evidence permits;
5. expand data coverage rather than training longer if the required road area
   exposes unobserved surfaces;
6. keep dynamic actors deferred until a residue affects the driving decision or
   closed-loop agent testing begins.

This result does not justify NeuRAD, MTGS, or new SplatAD training yet.

## Logged-Centreline Extension

The fixed anchor rectangle was replaced on 2026-07-22 by a 64.595 m polyline
derived from the synchronized six-camera rig trajectory in the same world
coordinates as `EgoState`. The polyline is only a reconstruction-support
boundary: it does not advance, steer, or pull the simulated vehicle.

A 100-step, 10.0 s GPU/HTTP run at 0.25 scale reached x=18.606 m, y=-0.163 m,
and yaw=-3.290 degrees at 2 m/s. It did not hit the boundary and stayed within
0.0134 m of the logged centreline. This demonstrates coherent rendering well
beyond the former 6 m rectangle; it does not certify the provisional +/-1 m
tube. The next test is multi-station visual/LiDAR support along the tube, then
an operator-driven trial.
