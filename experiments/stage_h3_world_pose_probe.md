# Stage H3 World-Pose Free-Driving Probe

Date: 2026-07-22

## Purpose

Determine whether the accepted PandaSet scene-040 static-8k checkpoint can be
queried from a vehicle pose that is owned by the simulator instead of the
recorded trajectory. This is the first implementation step toward genuine
free driving. It does not certify a complete drivable corridor.

The test keeps the checkpoint fixed and does not train a model.

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
H3_WORLD_POSE_PROBE_ROOT=/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050 \
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
- continuous path: 30 steps at 0.1 s, initial speed 2.0 m/s, normalized
  steering +0.35, normalized throttle 0.15.

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
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050/world_pose_probe.json
```

Result: **world-pose plumbing pass; visual corridor requires review**.

All automated plumbing gates passed:

- accepted checkpoint is step 7,999;
- reset reproduced identical pixels;
- all 45 observations contained six finite `uint8` RGB images;
- every query used the same scene anchor and logical frame;
- simulation time advanced while scene time remained fixed;
- source-log motion metadata was not reused;
- all six camera poses were synchronized to one scene time;
- the vehicle model changed both future yaw and future lateral position;
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

Warmed six-camera Renderer latency at 0.5 output scale over 45 observations:

```text
p50 = 67.413 ms
p95 = 68.455 ms
max = 69.526 ms
```

The largest recorded front-view near-black-pixel fraction was approximately
0.0029%. This is only an obvious-hole proxy and is not a visual-quality gate.

## Visual Review

Reviewed artifacts:

```text
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050/lateral_corridor_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050/yaw_probe_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050/continuous_turn_front.jpg
/home/yawei/stage3_external/artifacts/scene_040_world_pose_probe_scale_050/six_camera_turn_final.jpg
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

- +/-1 m is a reasonable first candidate for a restricted interactive
  free-driving prototype;
- +/-2 m and +/-3 m are promising visual probes, not accepted safe bounds;
- the current result does not yet establish LiDAR/road geometry correctness at
  those offsets and does not test dynamic actors.

## Decision And Next Action

Accept the new world-coordinate Renderer and vehicle-path plumbing as an
implemented backend milestone. Do not accept the full +/-3 m corridor yet.

Next:

1. add symmetric right-turn, lane-change, straight, and brake-to-fixed-position
   continuous paths;
2. measure road-region geometry/depth where static LiDAR support permits;
3. add a separate world-space browser entrypoint with conservative initial
   bounds around +/-1 m, without replacing the logged regression browser;
4. record a human driving review of the restricted path;
5. expand data coverage rather than training longer if the required road area
   exposes unobserved surfaces;
6. keep dynamic actors deferred until a residue affects the driving decision or
   closed-loop agent testing begins.

This result does not justify NeuRAD, MTGS, or new SplatAD training yet.
