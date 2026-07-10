# Project State — Driving Scene Reconstruction

## 1. Current Project Positioning

This repository focuses on **log-driven driving scene reconstruction for a human-drivable panoramic simulator**.

The core goal is:

> Given a real driving log, reconstruct a scene representation that can render updated 360-degree / multi-view driving observations while a human controls the ego vehicle.

The intended first-stage loop is:

```text
real driving log
→ scene reconstruction / scene representation
→ human control input
→ ego state update
→ 360-degree / multi-view rendering
→ human-drivable simulation loop
```

This project is related to credibility auditing, but credibility audit is only an evaluation function. The main object is a human-drivable simulator built on reconstructed real driving scenes.

## 2. What Changed

Earlier project notes emphasized novel-view synthesis and leave-one-camera-out evaluation. That remains useful, but it is now treated as a **renderer-backend baseline**, not the whole project.

The corrected interpretation is:

```text
not only: generate a missing camera view
not only: train NeRF / 3DGS
not only: generate a fixed driving video

instead: reconstruct a log-based scene that a human can drive through, with 360 / multi-view observations updating in closed loop
```

Autonomous-driving model integration is deferred. The first driver is a human.

World-model generation is not part of the immediate next step.

## 3. Current Motivation

The immediate motivation comes from a practical failure case:

- a human drives while viewing a reconstructed / generated 360-degree scene;
- the current 360 rotation or extrapolated views show artifacts;
- generated views may become blurry, distorted, or unstable;
- this is not only an image-quality problem, because wrong geometry, occlusion, lane structure, or object position can mislead the driver.

Therefore, the project should build toward:

```text
human-drivable log-based panoramic simulator
```

rather than only a visual demo.

## 4. System Layers

### Layer 1 — Log Scene Layer

Input may include:

- multi-camera video or panoramic video;
- ego pose / odometry;
- CAN signals such as speed, steering, throttle, brake;
- camera calibration;
- optional LiDAR, depth, 3D boxes, map, or lane annotations.

Output:

```text
scene representation that can be queried by ego pose and camera rig
```

### Layer 2 — Human Control Layer

Input:

- keyboard;
- gamepad;
- steering wheel;
- throttle / brake controls.

Initial implementation can use synthetic scripted controls for smoke testing.

### Layer 3 — Ego State Update Layer

Initial state fields:

```text
x, y, yaw, speed, timestamp
```

Initial vehicle model:

```text
simple kinematic bicycle model or yaw-rate approximation
```

The first version does not need full vehicle dynamics.

### Layer 4 — Rendering Layer

Given the scene, ego state, and camera rig, render:

- front view;
- left view;
- right view;
- rear view;
- panorama / 360-degree view;
- multi-screen driving display.

Candidate renderers:

```text
ReplayRenderer
PanoramaRenderer
ReconstructionRenderer
HybridRenderer
```

Do not add a world-model renderer in the immediate next step.

### Layer 5 — Evaluation / Credibility Layer

Checks include:

- image artifacts;
- geometry distortion;
- temporal instability;
- lane / road-boundary consistency;
- left / right / front / rear relation consistency;
- object disappearance or hallucination;
- occlusion errors;
- whether the generated view is safe enough for a human driver to use.

## 5. Role of NeRF / 3DGS / Splatfacto

NeRF / 3DGS / Splatfacto are **not** the final objective.

They are possible reconstruction renderer backends:

```text
real driving log images + camera poses
→ per-scene reconstruction / fitting
→ render observations for nearby ego poses
```

The existing WayveScenes101 + Splatfacto smoke run is useful because it verified that one reconstruction backend can be set up. However, a full Splatfacto training run should not be the next project step until the human-drivable simulator interface is defined.

## 6. Current Codex Result Summary

Codex has already verified more than the original Stage 1A resource check:

- WayveScenes101 and Nerfstudio code paths were reachable;
- the project uses `/home/yawei/stage1_external` instead of `/data` because `/data` has too little free space;
- a WayveScenes101 environment was created;
- `scene_094` was downloaded from an official WayveScenes101 source link;
- the scene was converted for Nerfstudio;
- a compatibility helper was added for top-level `camera_model=OPENCV_FISHEYE`;
- Splatfacto completed a 1-iteration smoke run using CUDA 12.1 and `TORCH_CUDA_ARCH_LIST=8.9`.

This proves that the reconstruction-backend path is feasible. It does not yet produce a useful simulator or a useful visual result.

## 7. Current Next Minimal Action

Next action:

```text
Stage H0: define human-drivable log-based panoramic simulator MVP
```

Do this before continuing heavy reconstruction training.

Expected deliverables:

```text
docs/human_drivable_simulator_project.md
docs/codex_next_task_stage_h0.md
src/driving_scene_reconstruction/sim/
examples/sim_loop_smoke.py
```

The next Codex task should implement only lightweight simulator interfaces:

- `EgoState` dataclass;
- `HumanControl` dataclass;
- simple vehicle model step function;
- renderer interface;
- camera rig / camera spec dataclasses if useful;
- dummy renderer smoke example.

No dataset download, model training, world-model integration, checkpoint generation, or large output generation should happen in the next task.
