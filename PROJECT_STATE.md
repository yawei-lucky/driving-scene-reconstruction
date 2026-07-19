# Project State — Driving Scene Reconstruction

Last updated: 2026-07-19

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

### Stage H3-0A — Multi-sensor environment

Completed on 2026-07-18 without modifying the H1/H2 environment:

- created `/home/yawei/stage3_external/envs/h3_splatad`;
- pinned neurad-studio, custom SplatAD gsplat, custom viser, PandaSet devkit,
  and tiny-cuda-nn source revisions;
- installed Python 3.10.20, CUDA toolkit 11.8, and PyTorch 2.0.1+cu118;
- verified `splatad`, `neurad`, and `pandaset-data` command registration;
- executed tiny-cuda-nn, camera Gaussian rasterization, and LiDAR Gaussian
  rasterization on the RTX 4090 D;
- added repeatable setup and acceptance commands that do not download data;
- reloaded and rendered the existing H1 checkpoint from the original
  `wayve_scenes_env` after H3 installation.

The environment record and exact evidence are in
`experiments/stage_h3_environment.md`.

### Stage H3-0B / Level 1 — PandaSet data and pipeline

Completed on scene `040` on 2026-07-19:

- user accepted the recorded PandaSet CC BY 4.0 plus additional terms;
- downloaded the pinned 44,520,528,731-byte archive, matched its exact
  SHA-256, and passed a complete ZIP integrity test;
- triaged semseg-capable scenes and selected scene 040 for daylight, stable
  exposure, visible road geometry, and usable camera overlap;
- selectively extracted only its 80 frames;
- loaded six 1920x1080 cameras, Pandar64/PandarGT, poses, timestamps, cuboids,
  GPS, and point-cloud semantics;
- accepted six cameras plus Pandar64 as the first baseline;
- preserved LiDAR-to-camera overlays at frames 0, 40, and 79;
- recorded sensor offsets of about 10.8-50.1 ms relative to the front camera;
- passed the exact SplatAD dataparser with 240 train cameras, 40 Pandar64
  sweeps, and 7 actor trajectories at the default 0.5 train split;
- completed a 100-step SplatAD smoke, saved a 297,232,596-byte checkpoint,
  reloaded it, and rendered 240 held-out views;
- measured smoke means of PSNR 15.9365, SSIM 0.6185, and LPIPS 0.8402.

The output is still visibly fuzzy and multicolored at 100 steps. This stage
proves the data, calibration, GPU, checkpoint, and rendering path; it does not
pass the stable-reconstruction visual gate. Exact evidence is in
`experiments/stage_h3_scene_040_smoke.md`.

### Stage H3 Level 2 — Scene 040 visual pilot

Completed on 2026-07-19:

- changed from the smoke's 0.5 split to a fixed 0.9 linspace temporal split;
- trained on 432 images and 72 Pandar64 sweeps; held out 48 images at 8
  timestamps across all six cameras;
- raised LiDAR downsampling from 0.25 to 0.5 and seed cap from 250,000 to
  750,000;
- completed 2,000 steps in 199.6 seconds and saved a 912,125,396-byte
  checkpoint;
- reloaded step 1,999 and rendered all 48 held-out views at 12.62 images/s;
- measured means of PSNR 24.7109, SSIM 0.7392, and LPIPS 0.4475;
- visually recovered recognizable road, buildings, sidewalks, trees, signals,
  vehicles, and rear/side layout in every camera.

The result remains soft around road texture, tree/sky edges, poles, windows,
and near vehicles. Metrics are not directly comparable with the 0.5-split
smoke or the Wayve held-out protocol. Nearby-pose geometry, quantitative depth,
temporal flicker, and dynamic-residue gates are still open. See
`experiments/stage_h3_scene_040_pilot.md`.

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

Separately, the H3 path now loads real PandaSet scene 040, uses six cameras,
Pandar64 geometry and cuboid actor tracks, trains SplatAD, saves/reloads a
checkpoint, and renders held-out RGB/depth. The 2,000-step checkpoint recovers
recognizable static structure at logged held-out poses, but has not yet passed
the nearby-pose and temporal stability gates required for driving.

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
- `NerfstudioRenderer` has not been adapted to a dynamic SplatAD checkpoint;
  direct checkpoint compatibility between the H2 Nerfstudio 1.1.5 path and the
  neurad-studio fork must not be assumed.
- PandaSet sequences are only 80 frames, so the first achievable simulator is
  log-local playback with small pose offsets, not unrestricted free roaming.
- The audited PandaSet parser robustly defaults to Pandar64. PandarGT exists in
  the dataset and command choices, but multi-LiDAR missing-point and raster
  handling requires a separate verification before it becomes a baseline.
- PandaSet semantic segmentation is point-cloud annotation for selected
  scenes, not a complete dense image-mask source.
- PandaSet sensors are asynchronously captured: scene 040 offsets span roughly
  10.8-50.1 ms relative to the front camera. Later timing logic must preserve
  sensor and per-point timestamps rather than assuming frame-index simultaneity.
- The H3 100-step smoke used a 0.5 train split, 0.25 data downsampling, and a
  250,000-point seed cap. Its blurred output is not suitable for driving.
- The H3 2,000-step pilot is much clearer but uses a denser 0.9 train split.
  Its fixed 48-view metrics measure interpolation near observed poses rather
  than wide extrapolation.
- Peak VRAM was not preserved during the 2,000-step run and must be measured
  before raising seed density or starting a longer run.
- The PandaSet back camera is intentionally cropped from 1920x1080 to 1920x820
  by the upstream parser; rear-view acceptance must account for that crop.
- Geometry, temporal, and driving-task metrics have not yet been implemented.

## 5. Current Next Action — Stage H3

Stage H3 prioritizes stable drivable scene reconstruction before cockpit UI or
steering-wheel polish. Environment, acquisition, calibration, Level 1, and the
Level 2 visual pilot have passed. The current action is a no-retraining
geometry/temporal acceptance of the 2,000-step checkpoint, not an 8k baseline.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. preserve and reuse the accepted 2,000-step config/checkpoint and fixed test
   timestamps;
2. render a fixed nearby-pose grid around several logged timestamps without
   changing training;
3. compare rendered depth with held-out Pandar64 on static road, curb, facade,
   pole, and vehicle regions;
4. measure trajectory flicker, camera-to-camera structure, actor residue,
   warmed rendering latency, and peak VRAM;
5. diagnose timestamp handling, seed geometry, loss scheduling, or
   actor/background separation if any geometry/temporal gate fails;
6. authorize an 8k scene-040 baseline only after those checks pass;
7. retain WayveScenes101 `scene_094` as the camera-only hard baseline;
8. add logged-trajectory progression and human-control offsets only after the
   accepted 8k or equivalent stable checkpoint exists.

In this plan, camera images remain the source of visual appearance. LiDAR
anchors depth, metric scale, and ground geometry; fused ego pose/IMU anchors
time-varying sensor placement and gravity; 3D cuboids drive the initial actor
trajectories. Point-cloud semantics are supplemental and do not substitute for
image masks.

### H3-0A environment result

Accepted on the project host on 2026-07-18:

- RTX 4090 D, 24 GB VRAM, driver `580.95.05`;
- H3: Python 3.10.20, PyTorch 2.0.1+cu118, CUDA toolkit 11.8;
- camera and LiDAR custom CUDA kernels passed with finite outputs;
- all audited upstream code is pinned under
  `/home/yawei/stage3_external/code`;
- approximately 264 GiB remained free at final acceptance;
- the inspected neurad-studio documentation explains checkpoint loading but
  does not publish an exact-sequence checkpoint catalog.

This environment remains the accepted H3 execution base. The subsequent
PandaSet acquisition and Level 1 smoke did not modify the H1/H2 environment.

### H3-0B acquisition and Level 1 result

The source/archive audit and accepted execution produced:

- the neurad-linked Hugging Face mirror contains one 44,520,528,731-byte ZIP
  at repository commit `e2e123aea3b3132c67f4b395ec6120f63e190271`;
- its recorded LFS SHA-256 oid is
  `6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e`;
- the archive has 103 scenes and 75,758 entries;
- full extracted payload is 44,732,715,419 bytes, so archive plus a full
  extraction would need 83.12 GiB;
- a single scene is at most 475,771,588 extracted bytes, so the pilot can keep
  the full archive plus one scene in about 41.91 GiB;
- 76 scenes contain point-cloud semantic annotations;
- the standard mirror exposes only the full archive, not per-scene packages;
- the archive license is CC BY 4.0 with additional dataset terms, and
  downloading or use constitutes acceptance;
- the exact archive now exists outside Git and passed size, SHA-256, and ZIP
  integrity checks;
- only daylight scene 040 was extracted, and its data/calibration gates passed;
- a reusable 100-step scene-040 SplatAD checkpoint and 240-view render exist
  outside Git;
- a reusable 2,000-step scene-040 SplatAD checkpoint and fixed 48-view render
  recover recognizable all-camera static structure;
- approximately 220 GiB remained free after the archive, scene, checkpoint,
  render, and caches.

The official PandaSet page's visible download link returned HTTP 404 during the
audit, while the neurad-linked mirror states that its uploader is not affiliated
with the dataset creators. Therefore provenance must be recorded with every
run. See `experiments/stage_h3_dataset_foundation.md` and
`experiments/stage_h3_scene_040_smoke.md`.
