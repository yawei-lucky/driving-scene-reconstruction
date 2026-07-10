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

For now, do **not** add a world-model renderer path. The immediate priority is to define and implement the human-drivable simulator interface.

## Existing Stage-1 Baseline Status

A previous Codex run verified the WayveScenes101 + Nerfstudio / Splatfacto path as a reconstruction backend smoke test:

```text
WayveScenes101 scene_094
→ Nerfstudio preparation
→ Splatfacto 1-iteration smoke run
→ fisheye camera compatibility issue fixed
→ CUDA / gsplat issue resolved with CUDA 12.1
```

This is useful as a future renderer-backend experiment. It is **not** the main product objective by itself.

## Current Next Step

The next task is:

```text
Stage H0: define human-drivable log-based panoramic simulator MVP
```

See:

```text
docs/human_drivable_simulator_project.md
docs/codex_next_task_stage_h0.md
```

## Repository Structure

```text
.
├── README.md
├── PROJECT_STATE.md
├── docs/
│   ├── human_drivable_simulator_project.md
│   ├── codex_next_task_stage_h0.md
│   ├── problem_statement.md
│   └── mvp_leave_one_camera_out.md
├── experiments/
├── scripts/
└── data/
```
