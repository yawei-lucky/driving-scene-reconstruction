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

Separately, the H3 environment can now execute SplatAD's synthetic camera and
LiDAR rendering path and exposes the six-camera PandaSet parser. It has not yet
loaded PandaSet data or produced a PandaSet checkpoint, so this is an
environment milestone rather than a new reconstructed scene result.

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
- Geometry, temporal, and driving-task metrics have not yet been implemented.

## 5. Current Next Action — Stage H3

Stage H3 now prioritizes stable drivable scene reconstruction before cockpit UI
or steering-wheel polish. H3-0A environment preparation has passed. The current
action is the H3-0B PandaSet acquisition and calibration gate, not a long
training run.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. preserve the accepted H3 environment and continue using
   `scripts/check_stage_h3_environment.sh` as its acceptance test;
2. obtain explicit acceptance for the audited PandaSet terms and 44.5-GB
   download, validate the pinned hash, and extract only one scene;
3. select that 80-frame sequence and load one frame without training;
4. synchronize six-camera images, Pandar64, calibrated/fused ego poses, and
   dynamic annotations in one metric coordinate system;
5. verify camera/LiDAR calibration visually before running an optimization;
6. use SplatAD's cuboid-derived dynamic actor layer as the primary path and a
   masked static-background run as a diagnostic comparison;
7. retain WayveScenes101 `scene_094` and its existing checkpoint as the known
   camera-only hard baseline and regression comparison;
8. reuse a checkpoint whenever its scene, preprocessing, calibration, model
   configuration, code, and environment key match; use no-training, <=100-step,
   1k-2k, and 8k gates instead of repeatedly launching full training;
9. add logged-trajectory progression using fused ego poses, followed by small
   human-control offsets.

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
- no PandaSet data has been downloaded;
- the inspected neurad-studio documentation explains checkpoint loading but
  does not publish an exact-sequence checkpoint catalog.

The immediate action is now explicit dataset-term/download approval, followed
by archive integrity validation, one-scene inspection, and calibration
overlays. Only after those pass should the project run a <=100-step end-to-end
PandaSet smoke.

### H3-0B acquisition audit result

The read-only source and archive audit is complete:

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
  downloading or use constitutes acceptance.

The official PandaSet page's visible download link returned HTTP 404 during the
audit, while the neurad-linked mirror states that its uploader is not affiliated
with the dataset creators. Therefore provenance must be recorded with every
run. No data was downloaded because accepting the terms and acquiring 44.5 GB
requires explicit user approval. See
`experiments/stage_h3_dataset_foundation.md`.
