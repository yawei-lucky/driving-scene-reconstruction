# Project State — Driving Scene Reconstruction

## 1. Project Positioning

This repository focuses on **real-driving scene reconstruction and view extrapolation**.

The core problem is:

> Given limited real sensor observations from a driving scene, reconstruct or synthesize additional views that are useful for 360-degree visualization, teleoperation, and downstream autonomous-driving evaluation.

This is related to credibility auditing, but it is broader than auditing. The main project object is the reconstruction / extrapolation system itself.

## 2. Current Motivation

The immediate motivation comes from a practical failure case:

- original-view 360-degree rendering has heavy artifacts;
- diffusion-based view extrapolation was attempted;
- generated extrapolated views remain blurry or unstable;
- overall effect is still not good enough for real application.

This suggests that the problem should be treated as a real scene reconstruction and novel-view extrapolation task, not merely as front-end visualization.

## 3. Current Research Questions

### RQ1 — Reconstruction

How can a real driving scene be reconstructed from multi-camera logs, ego poses, calibration, and optional LiDAR / depth / 3D boxes?

### RQ2 — View Extrapolation

How far can the system extrapolate views beyond observed camera viewpoints before visual, geometric, or task-level consistency breaks?

### RQ3 — 360-degree Driving Visualization

Can the reconstructed scene support stable 360-degree visualization for human teleoperation or monitoring?

### RQ4 — Driving-Relevant Consistency

Do reconstructed views preserve lane structure, object positions, occlusion relations, left-right relations, and temporal continuity?

### RQ5 — Credibility Function

When should a generated view be accepted, down-weighted, or rejected for downstream use?

## 4. Initial Technical Direction

The first stage should stay minimal and data-driven:

1. collect or identify a short multi-camera driving log;
2. prepare camera calibration and synchronized image streams;
3. define a leave-one-camera-out evaluation;
4. generate or approximate the held-out view using a baseline method;
5. analyze failures using image, geometry, temporal, and task-level criteria.

No large reconstruction system is required for the first version.

## 5. Initial Evaluation Layers

### Image Layer

- blur;
- smear;
- ghosting;
- color / exposure mismatch;
- local texture artifacts.

### Geometry Layer

- lane-line bending;
- road-boundary distortion;
- object-position drift;
- scale inconsistency;
- broken static structure.

### Temporal Layer

- flicker;
- object popping;
- inconsistent motion;
- unstable background;
- view-to-view discontinuity.

### Driving-Task Layer

- wrong same-lane / adjacent-lane relation;
- wrong front / rear / left / right relation;
- wrong approaching / receding relation;
- wrong occlusion relation;
- false obstacle or missing obstacle;
- misleading free-space judgment.

## 6. Minimal Repository Files

Current minimal files:

- `README.md`
- `PROJECT_STATE.md`
- `docs/problem_statement.md`
- `docs/mvp_leave_one_camera_out.md`
- `experiments/README.md`
- `scripts/README.md`
- `data/README.md`

## 7. Next Minimal Action

Write the first concrete MVP design:

> leave-one-camera-out view extrapolation evaluation on a short multi-camera driving clip.

The purpose is to make the problem measurable before choosing NeRF, 3DGS, diffusion, or any specific reconstruction model.
