# Stage H2 — Nearby-Pose Reconstruction Renderer

Date: 2026-07-17

## Goal

Connect the dependency-free simulator interfaces from Stage H0 to the trained
WayveScenes101 Splatfacto checkpoint from Stage H1.

The completed path is:

```text
HumanControl
→ SimpleVehicleModel.step
→ EgoState
→ nearby planar scene transform
→ Nerfstudio camera rig
→ Splatfacto RGB rendering
→ multi-camera display
```

## Coordinate Convention

The H2 renderer treats a selected dataset frame as the reference ego rig:

- Nerfstudio scene coordinates are z-up.
- An OpenGL-style front camera looks along local `-Z`.
- Projected front-camera `-Z` defines ego forward.
- Ego left is the perpendicular horizontal axis.
- `EgoState.x` is forward displacement in meters.
- `EgoState.y` is left displacement in meters.
- Positive yaw turns left.
- Nerfstudio `dataparser_scale` converts meters into model scene units.

The reference rig origin is the mean of the five camera centers. One planar
rigid transform is applied to all requested cameras, preserving their relative
positions and orientations.

This is a practical bridge for nearby-pose experiments. A future log adapter
should replace the inferred basis with an explicit calibrated ego pose.

## Camera Calibration

`NerfstudioRenderer` does not approximate the Wayve cameras with the lightweight
`CameraSpec` angles. It clones each reference camera from the loaded dataset,
preserving:

- focal lengths and principal point;
- image resolution;
- `OPENCV_FISHEYE` distortion;
- processed camera-to-world pose;
- camera metadata used by Nerfstudio.

The `CameraRig` names select dataset cameras such as `front-forward`.

## Safety Envelope

The default renderer rejects queries outside:

```text
forward: [-2.0m, +2.0m]
left:    [-0.5m, +0.5m]
yaw:     [-5deg, +5deg]
```

These are conservative experiment limits, not a measured guarantee of visual
or driving safety.

## Validation

Checkpoint:

```text
/home/yawei/stage1_external/outputs/wayvescenes101_h1/
scene_094_h1_big/splatfacto/run_v2/config.yml
```

Verified on the RTX 4090 D:

1. Loaded checkpoint step 7,999 through `nerfstudio.utils.eval_utils.eval_setup`.
2. Rendered reference frame 100, `front-forward`, at 240×135.
3. Rendered `+0.5m` forward, `+0.2m` left, `+2°` yaw.
4. Produced all five Wayve camera views at that displaced pose.
5. Ran two scripted control updates in the interactive program's headless mode.
6. Produced three five-view mosaics with changing ego-state overlays.
7. Started the display-less browser server, fetched its page and JPEG frame,
   and verified that a `W` HTTP control updated the ego state and re-rendered.

The first GPU call compiled the `gsplat` extension and took about 120 seconds.
With the extension cached, the displaced five-view call took about 1.15 seconds
including first-call warm-up. Subsequent low-resolution frames in the same
process were substantially faster. Performance needs a dedicated benchmark
before making a real-time claim.

Visual inspection showed consistent viewpoint changes and the expected static
scene layout. Moving vehicles remain severely blurred, matching the Stage H1
held-out failure analysis.

## Commands

One nearby pose:

```bash
scripts/run_stage_h2_scene_094.sh smoke \
  --forward 0.5 --left 0.2 --yaw-degrees 2 \
  --cameras front-forward left-forward right-forward left-backward right-backward
```

Keyboard display:

```bash
scripts/run_stage_h2_scene_094.sh interactive
```

Browser display for SSH or other sessions without `$DISPLAY`:

```bash
scripts/run_stage_h2_scene_094.sh interactive \
  --web \
  --output-scale 0.125
```

The default URL is `http://127.0.0.1:8765`. When connecting over SSH, forward
that port and open the URL on the client machine:

```bash
ssh -L 8765:127.0.0.1:8765 yawei@stf-precision-3680
```

Keys:

```text
W: throttle
S: brake
A: steer left
D: steer right
R: reset reference pose
Q / Escape: quit
```

Headless verification:

```bash
scripts/run_stage_h2_scene_094.sh interactive \
  --headless-steps 2 \
  --output-scale 0.125 \
  --output-dir /tmp/dsr_stage_h2_interactive
```

## Known Limitations

- The reference dataset frame is fixed; simulation time does not advance
  through the original log.
- Dynamic objects are baked into a static reconstruction.
- No collision, map, or road constraints are applied.
- No automatic uncertainty or rejection score is attached to rendered pixels.
- The Tk window path requires a graphical desktop and has only been validated
  through its Pillow-based headless counterpart in this execution environment.
