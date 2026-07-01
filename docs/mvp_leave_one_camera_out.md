# MVP: Leave-One-Camera-Out View Extrapolation Evaluation

## 1. Purpose

The first MVP evaluates whether a scene reconstruction or view-extrapolation method can generate a missing driving camera view from other observed camera views.

This avoids vague judgments such as "the visual effect is not good" and turns the problem into a measurable experiment.

## 2. Basic Setup

Assume a synchronized multi-camera driving log, for example:

- front camera;
- front-left camera;
- front-right camera;
- rear-left camera;
- rear-right camera;
- rear camera.

For each experiment:

1. select one camera as the held-out target view;
2. use the remaining cameras as input observations;
3. generate the held-out target view;
4. compare the generated view with the real held-out image.

## 3. Input Requirements

Minimum required data:

- synchronized multi-camera images;
- camera intrinsics;
- camera extrinsics;
- timestamps.

Preferred additional data:

- ego pose / odometry;
- LiDAR point cloud;
- depth map;
- 3D boxes;
- lane annotations or HD map;
- vehicle CAN state.

## 4. Test Cases

The MVP should include several difficulty levels:

### Easy

- small viewpoint gap;
- static road structure;
- few dynamic objects.

### Medium

- moderate viewpoint gap;
- nearby lanes and road boundaries;
- several moving vehicles.

### Hard

- large-angle extrapolation;
- close vehicles;
- strong occlusion;
- turning or intersection scene;
- complex roadside structure.

## 5. Evaluation Layers

### 5.1 Image-Level Evaluation

Possible checks:

- visual inspection;
- PSNR / SSIM when appropriate;
- LPIPS or perceptual similarity;
- artifact annotation.

### 5.2 Geometry-Level Evaluation

Possible checks:

- lane-line alignment;
- road boundary consistency;
- object center displacement;
- projected LiDAR consistency;
- scale consistency.

### 5.3 Temporal Evaluation

Possible checks:

- frame-to-frame stability;
- object flicker;
- background warping;
- motion consistency.

### 5.4 Driving-Task Evaluation

Possible checks:

- same-lane / adjacent-lane relation;
- front / rear / left / right relation;
- approaching / receding relation;
- occlusion relation;
- false obstacle / missing obstacle;
- free-space consistency.

## 6. Output Format

Each evaluated clip should produce a short report:

```text
case_id:
held_out_camera:
input_cameras:
viewpoint_gap:
scene_type:
main_failures:
image_score:
geometry_score:
temporal_score:
task_consistency_score:
decision: accept / down-weight / reject
notes:
```

## 7. Success Criterion for the MVP

The MVP is successful if it can answer:

- which camera views are easiest or hardest to synthesize;
- how large-angle extrapolation fails;
- whether failures are merely visual or driving-relevant;
- which generated views should not be trusted for downstream use.
