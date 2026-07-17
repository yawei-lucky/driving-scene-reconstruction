# Research and Implementation Plan

Last updated: 2026-07-01

Status note: this is the initial research roadmap. H0, the first WayveScenes101
baseline, and the H2 renderer integration are now complete. See
`PROJECT_STATE.md` for current execution status.

## 1. Working Goal

Build a runnable research pipeline for **real-driving scene reconstruction and view extrapolation**.

The project should not stop at a literature survey. The intended loop is:

```text
public driving data
→ scene reconstruction / view synthesis baseline
→ held-out view evaluation
→ failure diagnosis
→ targeted improvement
→ repeat
```

The first concrete task is to synthesize missing or off-axis driving views from observed multi-camera images and measure where extrapolation fails.

## 2. Initial Strategy

Start with public resources that already match the project:

1. calibrated real driving images;
2. synchronized multi-camera views;
3. camera poses / intrinsics / extrinsics;
4. held-out view or leave-one-camera-out evaluation possibility;
5. existing model code that can be run without building the full system from scratch.

The first target should be **WayveScenes101**, because it is explicitly designed for novel-view synthesis in autonomous driving and already includes off-axis held-out evaluation.

The second target should be **nuScenes**, because it is widely used, has a 360-degree six-camera setup, and is supported by many autonomous-driving reconstruction methods.

After the pipeline is stable, expand to **Argoverse 2**, **KITTI-360**, **Waymo Open Dataset**, and **PandaSet**.

## 3. Public Dataset Survey

### 3.1 Tier 0 — Best First Dataset

#### WayveScenes101

Use first.

Why it is aligned:

- built for novel view synthesis and scene reconstruction in autonomous driving;
- 101 scenes, each 20 seconds;
- 5 synchronized cameras at 10 Hz;
- 101,000 images;
- COLMAP-format camera poses;
- held-out off-axis evaluation camera;
- metadata for weather, road type, time of day, dynamic agents, illumination;
- simple integration with Nerfstudio.

Why it matters:

This dataset directly matches our MVP: held-out / off-axis view extrapolation.

Project use:

```text
WayveScenes101
→ run existing Nerfstudio-compatible baseline
→ evaluate held-out camera
→ generate failure report
```

### 3.2 Tier 1 — General Autonomous-Driving Reconstruction Datasets

#### nuScenes

Use after WayveScenes101.

Relevant properties:

- 1,000 scenes;
- each scene is 20 seconds;
- 6 cameras, 5 radars, 1 lidar;
- full 360-degree field of view;
- 3D boxes for 23 classes.

Project use:

```text
nuScenes mini/full
→ leave-one-camera-out evaluation
→ compare static vs dynamic reconstruction behavior
→ test object and lane relation consistency
```

#### Argoverse 2 Sensor Dataset

Use for map-aware and relation-aware evaluation.

Relevant properties:

- 1,000 annotated sensor scenarios;
- 15 seconds per scenario;
- 7 ring cameras and 2 stereo cameras;
- synchronized camera and lidar;
- intrinsic and extrinsic calibration for all cameras;
- 3D cuboid annotations;
- local HD map with lane-level geometry and ground height.

Project use:

```text
Argoverse 2 Sensor
→ evaluate lane / road-boundary consistency
→ evaluate object-position and map-alignment consistency
```

#### KITTI-360

Use for long urban sequence reconstruction and 360-degree geometry.

Relevant properties:

- 73.7 km of driving;
- over 320k images and 100k laser scans;
- stereo front cameras plus side fisheye cameras;
- full 360-degree field of view;
- semantic and instance annotations;
- official novel-view-synthesis benchmark.

Project use:

```text
KITTI-360
→ long static-background reconstruction
→ lane / building / roadside structure consistency
→ compare with official NVS benchmark assumptions
```

#### Waymo Open Dataset

Use when we need stronger comparison with SOTA autonomous-driving reconstruction papers.

Relevant properties:

- 1,150 scenes;
- each scene is 20 seconds;
- synchronized and calibrated lidar and camera data;
- 2D and 3D bounding boxes with consistent frame IDs;
- widely used by Street Gaussians, S3Gaussian, SplatAD, and other methods.

Project use:

```text
Waymo Open Dataset
→ SOTA reproduction target
→ dynamic object reconstruction
→ camera + lidar rendering experiments
```

#### PandaSet

Use as an additional manageable multi-sensor dataset.

Relevant properties:

- more than 100 scenes;
- 6 cameras;
- one 360-degree mechanical spinning lidar and one forward-facing long-range lidar;
- object and semantic labels.

Project use:

```text
PandaSet
→ alternative small-to-medium reconstruction dataset
→ NeuRAD / SplatAD compatible experiments
```

## 4. Model and Codebase Survey

### 4.1 Classical / Geometric Baselines

These are important because they expose whether failures come from data, calibration, or model capacity.

Candidate baselines:

- depth-based image reprojection;
- lidar-projected image warping;
- plane-sweep / multi-plane image style baseline;
- COLMAP sparse reconstruction + rendering sanity check.

Expected role:

```text
not photorealistic
but useful for debugging calibration, pose, visibility, and occlusion
```

### 4.2 General Neural Reconstruction Baselines

#### Nerfstudio

Use as an implementation framework.

Candidate methods:

- Nerfacto as a NeRF-style baseline;
- Splatfacto as a 3D Gaussian Splatting-style baseline;
- WayveScenes101 integration as the first runnable path.

Expected role:

```text
fast first implementation
standardized training / rendering / metrics
common interface for dataset adapters
```

#### 3D Gaussian Splatting

Use as the main representation family for fast rendering and practical view synthesis.

Why:

- explicit scene representation;
- faster rendering than classic NeRF-style methods;
- practical for interactive or 360-degree visualization;
- many driving-scene variants exist.

Limitation:

- naive 3DGS is mostly static-scene oriented;
- dynamic objects and sparse camera overlap remain hard;
- large-angle extrapolation can create floaters, stretching, ghosting, and geometry artifacts.

### 4.3 Driving-Specific Reconstruction Models

#### NeuRAD / neurad-studio

Candidate for dynamic autonomous-driving neural rendering.

Relevant properties:

- designed for dynamic AD data;
- supports camera and lidar rendering;
- models rolling shutter, lidar ray dropping, beam divergence, and other sensor effects;
- neurad-studio supports multiple datasets including nuScenes, Argoverse 2, PandaSet, KITTIMOT, and Waymo v2.

Project role:

```text
first serious dynamic-scene baseline
camera + lidar rendering path
strong codebase for dataset support
```

#### SplatAD

Candidate for real-time Gaussian-splatting-based AD rendering.

Relevant properties:

- 3DGS-based rendering for both camera and lidar;
- designed for dynamic autonomous-driving scenes;
- explicitly targets real-time rendering;
- models rolling shutter, lidar intensity, and lidar ray dropouts.

Project role:

```text
main fast-rendering baseline after Nerfstudio / 3DGS
useful if the project moves toward teleoperation-style 360 visualization
```

#### Street Gaussians

Candidate for object-aware dynamic urban reconstruction.

Relevant properties:

- dynamic urban scenes represented with 3D Gaussians;
- separates foreground vehicles and background;
- uses tracked vehicle poses;
- supports scene editing and high-FPS rendering;
- evaluated on KITTI and Waymo.

Project role:

```text
object-aware dynamic reconstruction baseline
strong for vehicle/background decomposition
requires object tracks / 3D boxes or equivalent priors
```

#### EmerNeRF

Candidate for self-supervised dynamic neural field baseline.

Relevant properties:

- learns static, dynamic, and flow fields;
- targets dynamic driving scenes;
- uses self-supervision rather than relying on ground-truth dynamic masks;
- relevant for in-the-wild logs.

Project role:

```text
dynamic-scene decomposition baseline
good comparison against 3DGS approaches
likely slower than splatting-based methods
```

#### S3Gaussian / DeSiRe-GS

Candidate for self-supervised street Gaussian reconstruction.

Relevant properties:

- targets autonomous-driving street reconstruction;
- focuses on static-dynamic decomposition;
- reduces reliance on costly 3D object annotations.

Project role:

```text
second-stage improvement direction
useful when annotation-free dynamic decomposition becomes important
```

#### DrivingForward

Candidate for feed-forward reconstruction.

Relevant properties:

- feed-forward Gaussian Splatting for driving scene reconstruction;
- uses flexible surround-view input;
- jointly predicts pose, depth, and Gaussian primitives;
- evaluated on nuScenes.

Project role:

```text
future real-time / generalizable reconstruction direction
not the first implementation target unless code and checkpoints are easy to run
```

### 4.4 Generative / Diffusion-Based Direction

Use later, not first.

Reason:

- diffusion can improve perceptual quality;
- but without geometry control, it can hallucinate objects, change lanes, alter occlusion, or break temporal consistency;
- our first need is a measurable geometry-grounded baseline.

Potential later role:

```text
3D reconstruction provides geometry and visibility
video / image diffusion fills uncertain textures
consistency checker rejects unsafe hallucination
```

## 5. Evaluation Plan

### 5.1 Primary Evaluation: Held-Out View Synthesis

For each scene:

```text
input cameras: observed views
target camera: held-out view
output: synthesized target image sequence
reference: real target image sequence
```

Core metrics:

- PSNR;
- SSIM;
- LPIPS;
- temporal flicker score;
- object center displacement;
- lane / road-boundary alignment;
- occlusion correctness;
- front / rear / left / right consistency;
- same-lane / adjacent-lane consistency.

### 5.2 Failure Taxonomy

Each failure case should be labeled as one or more of:

- blur;
- smear;
- ghosting;
- geometry bending;
- object hallucination;
- object disappearance;
- wrong object scale;
- wrong object location;
- lane inconsistency;
- road-boundary inconsistency;
- temporal flicker;
- dynamic object popping;
- wrong left-right relation;
- wrong front-rear relation;
- wrong occlusion relation;
- false free-space;
- missing obstacle.

### 5.3 Decision Layer

Each generated view should receive one of three decisions:

```text
accept: usable for visualization / downstream evaluation
down-weight: visually useful but not reliable for task evidence
reject: unsafe or misleading for driving-relevant interpretation
```

This is the credibility-audit part of the project, but it is only one module.

## 6. Implementation Roadmap

### Stage 0 — Repository and Environment

Goal:

```text
make the repository runnable
```

Tasks:

- add Python project skeleton;
- add environment file;
- add data manifest format;
- add evaluation output format;
- add report template;
- define local data directory policy.

Expected output:

```text
src/driving_scene_reconstruction/
configs/
reports/
```

### Stage 1 — WayveScenes101 Baseline

Goal:

```text
run the first public held-out view evaluation
```

Tasks:

- download a small subset of WayveScenes101;
- inspect camera poses, camera names, masks, and metadata;
- run the official / Nerfstudio-compatible evaluation notebook;
- train or run a simple Nerfstudio baseline;
- render held-out camera views;
- save image metrics and visual comparisons.

Expected output:

```text
reports/wayvescenes101_baseline/
  metrics.json
  cases.csv
  visual_report.html
```

Pass condition:

```text
we can reproduce a held-out view synthesis result on at least one public scene
```

### Stage 2 — Our Own Evaluation Harness

Goal:

```text
own the evaluation rather than relying only on external notebooks
```

Tasks:

- implement a dataset-agnostic scene manifest;
- implement image metrics;
- implement temporal metrics;
- implement side-by-side video export;
- implement failure labeling schema;
- implement accept / down-weight / reject output.

Expected output:

```text
python -m dsr.eval --pred outputs/... --gt data/... --manifest scene.json
```

### Stage 3 — nuScenes Leave-One-Camera-Out

Goal:

```text
move from dedicated NVS dataset to mainstream autonomous-driving logs
```

Tasks:

- use nuScenes mini first;
- implement camera extraction and calibration loader;
- define train / target camera split;
- run a 3DGS or Nerfstudio baseline;
- evaluate held-out camera generation.

Expected output:

```text
reports/nuscenes_mini_leave_one_camera_out/
```

### Stage 4 — Dynamic-Scene Baseline

Goal:

```text
test whether dynamic-aware models reduce object and temporal failures
```

Tasks:

- run NeuRAD / neurad-studio on a small supported dataset;
- run SplatAD if installation and data compatibility are manageable;
- compare against static or naive 3DGS baseline;
- isolate dynamic-object failures.

Expected output:

```text
static_baseline vs dynamic_baseline comparison
```

### Stage 5 — Geometry and Task-Level Evaluation

Goal:

```text
measure whether generated views preserve driving-relevant structure
```

Tasks:

- add projected lidar consistency metric where lidar is available;
- add object-box reprojection consistency metric where 3D boxes are available;
- add lane / map consistency metric where HD maps are available;
- add relation checker for front / rear / left / right and same-lane / adjacent-lane relations.

Expected output:

```text
metrics beyond PSNR / SSIM / LPIPS
```

### Stage 6 — Improvement Loop

Goal:

```text
turn failure diagnosis into model improvement
```

Candidate improvements:

- better camera undistortion and rolling-shutter handling;
- exposure / color normalization between cameras;
- LiDAR-depth initialization;
- pose refinement;
- dynamic object decomposition;
- foreground-object tracks;
- map priors for lanes and road boundaries;
- temporal regularization;
- generative texture refinement constrained by geometry.

The improvement loop should always report whether gains are only visual or also task-relevant.

## 7. Initial Repository Structure to Add Next

Recommended next files:

```text
configs/
  wayvescenes101_baseline.yaml
  nuscenes_mini_leave_one_camera_out.yaml

src/driving_scene_reconstruction/
  __init__.py
  data/
    __init__.py
    manifest.py
  eval/
    __init__.py
    image_metrics.py
    temporal_metrics.py
    report.py
  utils/
    __init__.py

reports/
  README.md
```

Do not add heavy model code yet. First create adapters and wrappers around existing public implementations.

## 8. First Two-Week Execution Plan

### Days 1–2

- finalize environment strategy;
- add project skeleton;
- create scene manifest schema;
- document local data paths.

### Days 3–5

- download a small WayveScenes101 subset;
- run dataset inspection;
- verify camera poses and held-out camera protocol;
- export a small visual sanity report.

### Days 6–8

- run first Nerfstudio / 3DGS-compatible baseline;
- render held-out view;
- compute PSNR / SSIM / LPIPS;
- export side-by-side image and video comparison.

### Days 9–11

- add temporal flicker metric;
- add failure annotation CSV;
- mark cases as accept / down-weight / reject.

### Days 12–14

- summarize first findings;
- decide whether to continue with WayveScenes101, move to nuScenes mini, or fix data/model issues first.

## 9. Main Risks

### Data Risk

Public datasets are large and license-limited. Some require registration or non-commercial use.

Mitigation:

```text
start with small subsets and metadata-only scripts
```

### Compute Risk

NeRF / 3DGS training can be GPU-heavy.

Mitigation:

```text
start with one scene, low resolution, short sequence, fixed baseline
```

### Dynamic Object Risk

Naive static reconstruction will fail on moving vehicles, pedestrians, cyclists, and occlusions.

Mitigation:

```text
compare static baseline against dynamic-aware models
```

### Metric Risk

Image metrics may miss driving-critical errors.

Mitigation:

```text
add geometry, temporal, and relation metrics from the beginning
```

## 10. Current Recommendation

The first implementation target should be:

```text
WayveScenes101 subset
+ Nerfstudio / 3DGS-style baseline
+ held-out off-axis camera evaluation
+ failure report
```

This is the shortest route from research question to runnable evidence.

After that, expand to:

```text
nuScenes mini leave-one-camera-out
→ NeuRAD / SplatAD dynamic-scene baseline
→ Argoverse 2 map-aware relation evaluation
```

## References

- WayveScenes101 GitHub: https://github.com/wayveai/wayve_scenes
- WayveScenes101 paper: https://arxiv.org/abs/2407.08280
- nuScenes paper: https://arxiv.org/abs/1903.11027
- Argoverse 2 official site: https://www.argoverse.org/av2.html
- KITTI-360 official site: https://www.cvlibs.net/datasets/kitti-360/
- Waymo Open Dataset paper: https://arxiv.org/abs/1912.04838
- PandaSet paper: https://arxiv.org/abs/2112.12610
- NeuRAD paper: https://arxiv.org/abs/2311.15260
- neurad-studio GitHub: https://github.com/georghess/neurad-studio
- SplatAD paper: https://arxiv.org/abs/2411.16816
- Street Gaussians paper: https://arxiv.org/abs/2401.01339
- EmerNeRF paper: https://arxiv.org/abs/2311.02077
- S3Gaussian paper: https://arxiv.org/abs/2405.20323
- DeSiRe-GS paper: https://arxiv.org/abs/2411.11921
- DrivingForward paper: https://arxiv.org/abs/2409.12753
- DreamDrive paper: https://arxiv.org/abs/2501.00601
