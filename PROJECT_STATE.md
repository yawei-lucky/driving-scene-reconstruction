# Project State — Driving Scene Reconstruction

Last updated: 2026-07-20

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

### Stage H3 Level 3 — Static 8k geometry/temporal baseline

Completed on 2026-07-19:

- exactly resumed the accepted step-1,999 pilot with optimizer and scheduler
  state and trained to step 7,999;
- improved the same 48 held-out views to PSNR 26.6605, SSIM 0.8145, and LPIPS
  0.2818;
- rendered all 126 fixed nearby-pose views without a finite-value failure;
- improved cuboid-excluded static LiDAR absolute error to 0.0614 m p50,
  0.6265 m p90, and 1.4925 m p95;
- rendered all 480 logged camera views without a finite-value failure;
- measured per-camera excess-warp p95 of 0.00457-0.00758;
- measured 28.01 ms p95 for one warmed camera and 153.19 ms p95 for a
  sequential six-camera rig.

Static 8k is the current best H3 checkpoint. It is not yet a stable-drivable
acceptance: close vehicles remain blurred, some optical-flow coverage is
inconclusive, and the six-camera path is about 6.5 Hz rather than 10 Hz.

### Stage H3 Level 4 — Vehicle actor ablations

Completed and rejected on 2026-07-19:

- implemented actor-aware MCMC relocation that preserves every actor ID;
- trained a stationary+moving candidate with 91 stationary and 7 moving
  actors;
- rejected it after actor layers consumed 42.3% of the final 5M Gaussians and
  degraded held-out appearance, vehicle crops, LiDAR geometry, temporal
  stability, and latency;
- trained the independent reviewer's requested moving-only 8k ablation with
  all 7 moving actors surviving and 4.959M background Gaussians retained;
- rejected it because moving crops fell from PSNR 21.3726 / LPIPS 0.2157 to
  19.7167 / 0.3649 and static LiDAR p90 worsened from 0.6265 m to 0.8529 m.

The second ablation separates two failures: stationary actor seed expansion
caused the first candidate's global capacity collapse, while the remaining
moving-actor failure is most consistent with actor-local geometry, cuboid
trajectory, timestamp, rolling-shutter, or world/camera transform
misalignment. See
`experiments/stage_h3_static_8k_and_actor_ablations.md`.

### Stage H3 Level 5 — Actor alignment and timing audit

Completed on 2026-07-20:

- proved actor 0/1 local Gaussian p95 coordinates reached 34-290 m while
  vehicle half-extents were about 1.2 x 2.6 x 1.2 m;
- traced the escape to per-step MCMC position noise without actor-bound
  handling, then added post-MCMC cuboid projection and optimizer-moment reset;
- kept all seven short-run actor geometries inside their padded cuboids, but
  rejected the step-2,000 candidate at moving-crop PSNR 18.2289 / LPIPS 0.4882;
- corrected cuboid azimuth timing in the calibrated LiDAR frame and serialized
  legacy/corrected semantics explicitly; the 253 observations that enter
  training change by 46.973 ms p50 and 55.852 ms p90;
- rejected the boundary-plus-time candidate: five of seven actors retained
  active Gaussians, but 39 calibrated-reference moving crops measured PSNR
  18.5608 / LPIPS 0.5331;
- retained static 8k as the only accepted H3 checkpoint.

The spatial and time corrections are correctness infrastructure, not accepted
visual checkpoints. See
`experiments/stage_h3_actor_alignment_and_timing.md`.

### Stage H3 Level 6 — Seed projection and painting audit

Completed and independently rejected on 2026-07-20:

- compared scan-centre and per-point actor seed assignment on real PandaSet
  point semantics and back-camera source images;
- used the actual 820-row rear crop, rolling-shutter row times, calibrated
  actor trajectories, and retained per-point LiDAR offsets;
- found a train-only semantic precision improvement from 87.76% to 90.28%,
  while held-out precision fell from 89.16% to 87.97% and usable points fell
  from 369 to 349;
- measured current-versus-corrected seed-paint projection displacement of
  0.375 px weighted median and 1.547 px weighted p95 on held-out points, with
  no consistent visible correction;
- rejected both a new actor training run and an unproven multi-camera painting
  rewrite;
- packaged the accepted static 8k as an exact 10-second, 80-frame six-camera
  result.

See `experiments/stage_h3_seed_projection_and_painting.md`.

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
Pandar64 geometry and cuboid actor tracks, trains SplatAD, saves/reloads
checkpoints, and renders held-out RGB/depth. The accepted static 8k checkpoint
recovers coherent static structure at logged and small nearby poses. It has
measured image, LiDAR, temporal, actor, VRAM, and warm-latency evidence, but
close dynamic vehicles and six-camera throughput still block a
stable-drivable acceptance.

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
- Peak VRAM is now recorded by the H3 geometry and temporal evaluators, but it
  was not preserved for the historical 2,000-step training process.
- The PandaSet back camera is intentionally cropped from 1920x1080 to 1920x820
  by the upstream parser; rear-view acceptance must account for that crop.
- Geometry and temporal evaluators are implemented. A final static-semantic
  LiDAR gate, cross-camera seam metric, and driving-task metric remain open.
- Both tested actor-aware candidates are rejected. Actor ID survival is not
  evidence that the actor appears at the correct image location.
- Actor-local MCMC geometry is now bounded, but opacity/supervision failure
  still leaves several moving actors visually absent.
- Cuboid time has an explicit calibrated mode. Per-point actor seed assignment
  is diagnosed but not enabled because the held-out semantic direction was
  negative for actor 0/1.
- Seed painting ignores point time, actor motion, and rolling shutter, but the
  audited held-out projection difference was mostly subpixel to about 2 px and
  did not prove a quality benefit.
- SplatAD discards trajectory `exists_at_time` during rendering, so an
  out-of-window actor can appear at its nearest pose.

## 5. Current Next Action — Stage H3

Stage H3 prioritizes stable drivable scene reconstruction before cockpit UI or
steering-wheel polish. Environment, acquisition, calibration, the 2k pilot,
static 8k baseline, actor ablations, and the alignment/timing audit are
complete. Static 8k remains the accepted checkpoint; all actor-aware visual
candidates are rejected.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. keep static 8k as the fixed comparison checkpoint;
2. retain actor-bound projection and calibrated cuboid-time semantics as
   correctness guards, but do not promote their rejected checkpoints;
3. repeat the point-semantic and source-projection audit on a larger actor
   visible in a front or side camera;
4. require held-out semantic precision and visible alignment to improve in the
   same direction;
5. correct and test one small actor/window only after that gate passes;
6. apply trajectory `exists_at_time` during rendering to prevent nearest-pose
   ghost actors outside their valid interval;
7. retain WayveScenes101 `scene_094` and static PandaSet 8k as fixed
   comparisons;
8. add logged-trajectory progression and human-control offsets only after the
   reconstruction gate is accepted.

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
