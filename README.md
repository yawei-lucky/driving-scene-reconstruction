# Driving Scene Reconstruction

This repository studies **log-driven driving scene reconstruction for a human-drivable panoramic simulator**.

The first target is not an autonomous-agent benchmark and not a fixed video generator. The first target is:

> Given a real driving log, reconstruct a scene representation that can render updated 360-degree / multi-view driving observations while a human controls the ego vehicle.

In other words:

```text
real driving log
→ scene reconstruction
→ human control input
→ ego state update
→ 360-degree / multi-view sensor rendering
→ human-drivable simulation loop
```

Autonomous-driving model integration is a later extension. For the initial system, the driver is a human using a keyboard, controller, or steering wheel.

## Scope

The repository covers:

- real-driving scene reconstruction from logs;
- 360-degree / panoramic / multi-camera driving view generation;
- human-in-the-loop ego control;
- lightweight ego vehicle state update;
- renderer interfaces for reconstructed scenes;
- evaluation of image, geometry, temporal, and driving-relevant consistency;
- credibility checks as one evaluation function, not as the whole project.

## Current Non-Goals

The current stage does **not** prioritize:

- connecting an autonomous-driving agent;
- training a new general world model;
- generating only a fixed video without closed-loop control;
- downloading or committing large datasets, checkpoints, or rendered videos;
- making photorealism the only success criterion.

## System Concept

The simulator should eventually support this loop:

```text
HumanControl(steer, throttle, brake)
→ EgoState(x, y, yaw, speed, time)
→ VehicleModel.step(...)
→ Renderer.render(scene, ego_state, camera_rig)
→ front / left / right / rear / panorama observations
```

Different rendering backends can be tested later:

```text
ReplayRenderer: original-log replay only
PanoramaRenderer: panorama / surround-view projection
ReconstructionRenderer: 3DGS / NeRF / NeuRAD / SplatAD scene rendering
HybridRenderer: geometry-based rendering plus repair / inpainting
```

For now, do **not** add a world-model renderer path. The immediate priority is
to make the reconstructed scene stable enough for a later human-drivable
cockpit-style simulator.

## Current Status

The repository now has three connected results:

```text
Stage H0
→ dependency-free human-control, ego-state, vehicle-model, and renderer interfaces

Stage H1
→ WayveScenes101 scene_094 Splatfacto baseline trained for 8,000 steps
→ official held-out front-camera metrics and validated reference videos

Stage H2
→ EgoState nearby-pose displacement mapped into Nerfstudio scene coordinates
→ real checkpoint rendering through the Renderer protocol
→ five-camera keyboard/display loop with a headless validation mode
```

The H2 renderer clones the dataset cameras' full intrinsics, fisheye distortion,
and rig extrinsics at a selected reference frame. It currently enforces a
conservative nearby-pose envelope of ±2 m forward, ±0.5 m left, and ±5° yaw.

The current baseline is still not production quality: dynamic vehicles remain
blurred, time is fixed to one reference frame, and no collision or responsive
traffic model exists.

## Run

The lightweight simulator and tests use the standard library:

```bash
python3 examples/sim_loop_smoke.py
python3 -m unittest discover -s tests -v
```

On the machine containing the Stage H1 checkpoint:

```bash
scripts/run_stage_h2_scene_094.sh smoke \
  --forward 0.5 --left 0.2 --yaw-degrees 2 \
  --cameras front-forward left-forward right-forward left-backward right-backward

scripts/run_stage_h2_scene_094.sh interactive
```

Interactive controls are `W/S/A/D`, `R` to reset, and `Q` or Escape to quit.
For an SSH or other display-less session, use the browser viewer:

```bash
scripts/run_stage_h2_scene_094.sh interactive --web --output-scale 0.25
```

For a larger detailed front view:

```bash
scripts/run_stage_h2_scene_094.sh interactive \
  --web \
  --output-scale 0.5 \
  --cameras front-forward
```

See `docs/stage_h2_reconstruction_renderer.md` for coordinate conventions,
validation evidence, and limitations.

## Current Next Step

Stage H3 should produce a stable drivable reconstruction baseline before UI or
input-device polish:

- run a small PandaSet multi-sensor pilot before another long camera-only
  training run;
- combine camera appearance with LiDAR metric geometry, fused ego poses/IMU,
  and dynamic-object annotations;
- verify synchronization and camera/LiDAR calibration on one short scene;
- compare image-only and LiDAR-assisted static-background reconstruction;
- keep WayveScenes101 `scene_094` as the camera-only hard baseline;
- reduce dynamic-object ghosts using 3D boxes or semantic labels;
- quantify nearby-pose stability, road-region artifacts, multi-camera
  consistency, depth/scale consistency, temporal flicker, and latency;
- add logged-trajectory time progression from fused ego poses once the static
  reconstruction is stable enough.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md`.

## Repository Structure

```text
.
├── README.md
├── PROJECT_STATE.md
├── pyproject.toml
├── docs/
│   ├── human_drivable_simulator_project.md
│   ├── codex_next_task_stage_h0.md
│   ├── stage_h3_stable_drivable_reconstruction_plan.md
│   ├── stage_h2_reconstruction_renderer.md
│   ├── problem_statement.md
│   └── mvp_leave_one_camera_out.md
├── src/driving_scene_reconstruction/sim/
├── examples/
├── tests/
├── experiments/
├── scripts/
└── data/
```
