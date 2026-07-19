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

The repository now has six connected results:

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

Stage H3-0A
→ isolated, pinned neurad-studio/SplatAD environment
→ synthetic camera and LiDAR CUDA kernels verified on RTX 4090 D

Stage H3-0B / Level 1
→ pinned PandaSet archive verified and only daylight scene 040 extracted
→ six cameras, Pandar64, fused poses, cuboids, and timestamps passed data/calibration gates
→ 100-step SplatAD checkpoint reloaded and rendered over 240 held-out images

Stage H3 Level 2
→ 90% temporal training split, six cameras, Pandar64, and actor tracks
→ 2,000-step checkpoint recovered recognizable static structure in all six views
→ 48-view means: PSNR 24.7109, SSIM 0.7392, LPIPS 0.4475

Stage H3 Level 3
→ exact-resume static 8,000-step checkpoint
→ 48-view means: PSNR 26.6605, SSIM 0.8145, LPIPS 0.2818
→ 126 finite nearby-pose views and measured LiDAR/temporal/latency evidence

Stage H3 Level 4
→ stationary+moving and moving-only actor-aware 8k ablations
→ both rejected; static 8k remains the accepted checkpoint
```

The H2 renderer clones the dataset cameras' full intrinsics, fisheye distortion,
and rig extrinsics at a selected reference frame. It currently enforces a
conservative nearby-pose envelope of ±2 m forward, ±0.5 m left, and ±5° yaw.

The current baseline is still not production quality. Static H3 8k is much
clearer and geometrically stronger than the 2k pilot, but close vehicles remain
blurred and sequential six-camera rendering is about 6.5 Hz. The tested actor
layers kept actor IDs alive but did not place moving vehicles correctly in the
images. H2 time is still fixed to one reference frame, and no collision or
responsive traffic model exists.

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
input-device polish. Static 8k is now the fixed accepted checkpoint. The next
gate does not start another long training run:

- project moving-actor LiDAR seeds and cuboids at exact camera timestamps;
- validate actor-local, world, and camera transforms including rolling shutter;
- render actor-only and background-only layers;
- identify where moving cars first leave their source image location;
- correct and test one actor/window before another full-scene run;
- keep PandarGT, cockpit UI, controller work, and unrestricted driving deferred.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md`.

Environment acceptance can be regenerated without PandaSet:

```bash
scripts/check_stage_h3_environment.sh
scripts/run_stage_h3_pandaset_040.sh data-gate
scripts/run_stage_h3_pandaset_040.sh static-8k
scripts/run_stage_h3_pandaset_040.sh paths
```

The static 8k run is reused when its checkpoint exists; it is not retrained by
default. Detailed results and rejected actor ablations are in
`experiments/stage_h3_static_8k_and_actor_ablations.md`.

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
