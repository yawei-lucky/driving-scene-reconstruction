# Project State — Driving Scene Reconstruction

Last updated: 2026-07-18

## 1. Product Goal

Build a human-drivable simulator from reconstructed real driving logs:

```text
real driving log
→ scene reconstruction
→ human steering / throttle / brake
→ ego-state update
→ nearby-pose multi-camera rendering
→ human observes and continues driving
```

Autonomous-agent and world-model integration remain later extensions. The
current system is geometry-grounded and human-controlled.

## 2. Completed Stages

### Stage H0 — Simulator interfaces

Completed and independently reviewed:

- immutable `EgoState` and normalized `HumanControl`;
- deterministic kinematic bicycle `SimpleVehicleModel`;
- `CameraSpec`, `CameraRig`, `RenderedObservation`, and `Renderer`;
- dependency-free smoke loop;
- finite-value validation and 16 current unit tests.

### Stage H1 — Offline reconstruction baseline

Completed on WayveScenes101 `scene_094`:

- official split: 800 side/rear training images, 200 held-out front images;
- Nerfstudio 1.1.5 `splatfacto-big`;
- fresh 8,000-step run and 2.41 GB checkpoint;
- Nerfstudio held-out metrics: PSNR 15.5961, SSIM 0.5343, LPIPS 0.5630;
- official Wayve metrics: PSNR 13.2109, SSIM 0.3022, LPIPS 0.7058,
  FID 10.5488;
- 2,000 rendered/reference images and six validated 200-frame videos.

Large artifacts stay outside Git under `/home/yawei/stage1_external`. The
WayveScenes101 dataset is restricted to non-commercial research use.

### Stage H2 — Simulator/reconstruction connection

Implemented:

- `SceneReferenceFrame` derives a z-up ego basis from the processed front
  camera and the five-camera rig center;
- physical displacement in meters is converted with Nerfstudio's
  `dataparser_scale`;
- the same planar rigid transform is applied to every camera, preserving rig
  baselines and camera rotations;
- `NerfstudioRenderer` lazily loads a config/checkpoint and clones complete
  dataset camera calibration, including fisheye distortion;
- nearby-pose safety limits reject queries beyond ±2 m forward, ±0.5 m left,
  or ±5° yaw;
- one-shot render plus Tk and browser keyboard/display examples are available;
- a headless interaction mode supports automated validation.

GPU validation used the existing `run_v2` checkpoint:

```text
reference frame: 100
reference render: front-forward, 240x135
offset render: +0.5m forward, +0.2m left, +2deg yaw
offset cameras: all five Wayve cameras
result: all five RGB frames rendered successfully
warm headless loop: three five-view mosaics generated successfully
browser loop: page, JPEG frame, and W-control HTTP request verified
```

The first render triggered a one-time `gsplat` CUDA extension build. It required
CUDA 12.1 and `TORCH_CUDA_ARCH_LIST=8.9`, matching the H1 training setup.

## 3. What The System Can Do Now

```text
HumanControl
→ SimpleVehicleModel
→ EgoState near one logged reference pose
→ SceneReferenceFrame transform
→ NerfstudioRenderer
→ front / side / rear RGB arrays
→ Tk multi-view display
```

This is the first repository state where simulated ego motion changes pixels
produced by the trained reconstruction checkpoint.

## 4. Important Limitations

- `EgoState` is relative to one fixed logged reference rig, not yet composed
  with a time-varying logged ego trajectory.
- The nearby-pose limits are conservative engineering bounds, not empirically
  certified safe regions.
- Static Splatfacto blurs moving vehicles and pedestrians.
- There is no collision, road-boundary, traffic-agent, or map constraint.
- The camera basis is inferred from the reference front camera; a future log
  adapter should use an explicit calibrated ego pose.
- Full-resolution H1 evaluation was about 1.52 FPS. Low-resolution warmed
  rendering is interactive, but systematic latency benchmarking is still
  required.
- The current examples depend on the machine-specific H1 checkpoint and
  Nerfstudio environment.
- Geometry, temporal, and driving-task metrics have not yet been implemented.

## 5. Current Next Action — Stage H3

Stage H3 now prioritizes stable drivable scene reconstruction before cockpit UI
or steering-wheel polish. The goal is a clean, repeatable reconstruction that
can later support a video-like human-drivable simulator.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. first create and freeze a separate neurad-studio/SplatAD training and test
   environment; do not modify the verified Wayve environment;
2. verify GPU/CUDA extensions, method/dataparser entrypoints, checkpoint
   loading, artifact paths, and a no-data or tiny-fixture smoke test;
3. then start a narrow PandaSet pilot rather than another long camera-only
   training run;
4. synchronize multi-camera images, LiDAR, calibrated/fused ego poses, and
   dynamic annotations in one metric coordinate system;
5. verify camera/LiDAR calibration visually, then compare image-only and
   LiDAR-assisted static-background reconstruction;
6. use 3D boxes or semantic labels to keep moving vehicles and pedestrians from
   contaminating the static background;
7. retain WayveScenes101 `scene_094` and its existing checkpoint as the known
   camera-only hard baseline and regression comparison;
8. reuse a checkpoint whenever its scene, preprocessing, calibration, model
   configuration, code, and environment key match; use no-training, <=100-step,
   1k-2k, and 8k gates instead of repeatedly launching full training;
9. add logged-trajectory progression using fused ego poses, followed by small
   human-control offsets.

In this plan, camera images remain the source of visual appearance. LiDAR
anchors depth, metric scale, and ground geometry; ego pose/IMU anchors
time-varying sensor placement and gravity; annotations support dynamic-object
separation.

### H3 environment preflight observation

Observed on the project host on 2026-07-18:

- RTX 4090 D, 24 GB VRAM, driver `580.95.05`;
- `wayve_scenes_env`: Python 3.10.14, PyTorch 2.3.1+cu118, GPU accessible;
- approximately 292 GB free on the project filesystem at inspection time;
- PandaSet and neurad-studio are not present in the documented external roots;
- historical PandaSet scripts exist, but their `/data/external` default and
  full-archive download behavior do not match the current H3 environment-first
  workflow.

Therefore, the immediate implementation action is environment preparation and
script correction, not dataset training.
