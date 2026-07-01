# Driving Scene Reconstruction

This repository studies real-driving scene reconstruction and view extrapolation, with an initial focus on generating and evaluating 360-degree driving views from limited real sensor observations.

The goal is not only to produce visually plausible views, but to reconstruct driving-relevant scene structure in a way that can support human teleoperation, visualization, and downstream autonomy evaluation.

## Scope

The repository covers:

- real-driving scene reconstruction;
- multi-camera driving view synthesis;
- novel-view synthesis and view extrapolation;
- 360-degree driving visualization;
- geometry, temporal, and task-level consistency evaluation;
- credibility checks as one component of the reconstruction pipeline.

Credibility audit is treated as an evaluation function, not as the whole project. The main object is scene reconstruction itself.

## Initial Problem

A practical pain point is that large-angle view extrapolation often becomes blurry, distorted, or unstable. For driving applications, this is not just an image-quality issue. If generated views break lane geometry, object positions, occlusion relations, or left-right consistency, they may become unsafe or unusable for teleoperation and autonomous-driving evaluation.

## Initial MVP

The first minimal experiment is a leave-one-camera-out evaluation on multi-camera driving logs:

1. Use several real camera views as input.
2. Hold out one camera view as the target.
3. Generate the missing target view.
4. Compare the generated view with the real held-out camera image.
5. Analyze failures at image, geometry, temporal, and driving-task levels.

## Repository Structure

```text
.
├── README.md
├── PROJECT_STATE.md
├── docs/
│   ├── problem_statement.md
│   └── mvp_leave_one_camera_out.md
├── experiments/
│   └── README.md
├── scripts/
│   └── README.md
└── data/
    └── README.md
```

## Current Status

The project is at the problem-definition and MVP-design stage. No model implementation is assumed yet.
