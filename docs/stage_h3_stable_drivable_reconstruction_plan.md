# Stage H3 — Stable Drivable Scene Reconstruction Plan

Date: 2026-07-19

Last strategy update: 2026-07-22

## TbV Execution Update — 2026-07-22

The user selected the lower-setup-cost TbV/SplatAD route before creating a
separate MTGS environment. This supersedes the MTGS-first next action in the
update below while preserving it as historical decision evidence.

The two selected ten-second Miami windows now pass bounded download, a
multi-traversal parser, shared-route LiDAR registration, 100-step save/reload,
and a 2,000-step held-out visual pilot in the accepted H3 environment. The
immediate next gate is a seven-camera TbV world-pose sweep over the common
approach, straight branch, right-turn branch, and bounded lateral offsets. Do
not train longer before checking that counterfactual corridor. Exact evidence
is in `../experiments/stage_h3_tbv_splatad_pilot.md`.

## Multi-Trajectory Pilot Update — 2026-07-22

A route branch is no longer required for the first expanded driving pilot. An
audit of all six published MTGS blocks found a smaller, lower-code-risk gate:
the 3.98 GB Singapore `365000_144000_365100_144080` release contains three
same-direction, eight-camera trajectories over an 84-87 m gentle curve, and an
official checkpoint is available.

Before implementing the TbV adapter, load that checkpoint in a separate MTGS
environment and test whether it fits the host's 24 GB GPU and renders a usable
corridor between the three observed tracks. Do not retrain first; the official
guide calls for at least 40 GB training VRAM. If checkpoint load or visual
gates fail, return to the verified TbV/SplatAD pilot. This update does not
replace the accepted scene-040 checkpoint or its regression/operator gates.
Evidence and exact stop conditions are in
`../experiments/stage_h3_mtgs_published_block_probe.md`.

## Free-Driving Architecture Update — 2026-07-22

This decision supersedes older next-action wording later in this document.
The logged-time bounded-offset browser remains useful regression evidence, but
it is not acceptance of genuine free driving: the recorded trajectory still
supplies the main road motion.

The first world-coordinate free-driving backend probe was implemented and
GPU-tested on 2026-07-22 using the accepted scene-040 static-8k checkpoint
without retraining:

1. separate simulation time, source-log time, absolute world ego pose, and the
   fixed six-camera rig extrinsics;
2. let the vehicle model own absolute position, yaw, and speed so steering
   changes the future path rather than only the rendered view;
3. render all six cameras from that one requested world pose;
4. measure isolated offsets and continuous vehicle paths before claiming a
   drivable envelope;
5. expand to multi-lane, multi-pass, or multi-branch data when the requested
   road area was not observed by the single scene-040 traversal;
6. return dynamic actors to the critical path when they hide the road, create
   false obstacles, or responsive traffic/closed-loop agent evaluation begins.

SplatAD remains the primary interactive renderer. NeuRAD is a matched
quality-oriented comparison, MTGS supplies the multi-traversal spatial-coverage
direction, and UniSim supplies a closed-loop compositional architecture
reference. None of those comparison/reference roles imply a completed
integration.

The complete decision, method boundaries, execution order, promotion gates,
and primary references are recorded in
`drivable_reconstruction_model_strategy.md`.

After independent review, the backend now passes plumbing and motion gates over
206 six-camera observations: straight motion, symmetric turns, symmetric lane
changes, and braking to a fixed pose. A learned 7.036 ms per-camera effective
time spread was found and removed before the final run. A separate restricted
world browser now uses a 64.595 m centreline derived from synchronized logged
poses instead of the initial fixed 6 m rectangle. A five-station x three-offset
visual sweep kept the front road readable across the route at -1/0/+1 m, so the
static-8k background is retained for the first restricted human-driving
prototype. The browser now starts at the recorded corridor beginning; a 30 s
GPU/HTTP run progressed 58.607 m through 300 observations without a boundary
hit. The next main-line gate is an operator drive. LiDAR diagnosis or a new
multi-traversal reconstruction follows a concrete driving-relevant failure,
not the assumption that every static checkpoint is unusable. Exact evidence is
in `../experiments/stage_h3_world_pose_probe.md` and
`../experiments/stage_h3_corridor_sweep.md`.

## Product-Priority Update — 2026-07-21

This decision supersedes older next-action wording later in this document.
The main line is now the shortest path to one stable, human-drivable,
log-local reconstructed scene. Do not resume open-ended actor training or UI
polish ahead of that outcome.

Priority order:

1. freeze the accepted PandaSet scene-040 static-8k checkpoint and connect it
   to the repository `Renderer` interface;
2. advance the ego vehicle along PandaSet's real logged trajectory;
3. compose bounded human forward, lateral, and yaw offsets on the logged pose,
   then render all six calibrated cameras from that one `EgoState`;
4. reach at least 10 complete six-camera observations/s and separately measure
   true control-to-display latency;
5. judge the MVP with the separate drivability gates in
   `drivability_acceptance_criteria.md`, not by LPIPS alone;
6. return dynamic traffic to the critical path as soon as it hides the road,
   creates a false obstacle, or closed-loop autonomous-driving evaluation
   begins.

Items 1-3 and the renderer-only part of item 4 were implemented and GPU-tested
on 2026-07-21. The complete 7.899-second log rendered logical frames 0-79 with
bounded scripted offsets. At 0.5 output scale, the warmed six-camera Renderer
latency was 69.15 ms p50, 74.37 ms p95, and 75.71 ms maximum. Exact reset,
nondecreasing frame selection, finite six-camera RGB, and nonzero nearby-pose
response passed. The subsequent 10 Hz browser entrypoint passed page, JPEG,
W-throttle, and reset HTTP checks. Ten warmed 0.25-scale requests measured
78.19 ms p95 from server receipt to six-camera JPEG readiness. Physical
keyboard-to-display acceptance remains an operator test.

The driver-attention rule and the non-waiver of dynamic correctness are
recorded separately in
`driver_attention_and_dynamic_traffic_requirements.md`.

## Execution Update

The no-retraining 2k gate, exact 2k-to-8k resume, two actor-layer ablations,
and the actor alignment/timing audit are now complete.

- Static 8k is the current accepted H3 checkpoint: held-out PSNR 26.6605,
  SSIM 0.8145, LPIPS 0.2818; static LiDAR error p50/p90 is
  0.0614/0.6265 m.
- It is not yet a stable-drivable acceptance: close vehicles remain blurred
  and six-camera sequential rendering is 153.19 ms p95 rather than <=100 ms.
- A 98-object stationary+moving actor pilot was rejected because it consumed
  42.3% of final Gaussian capacity, blurred vehicles, degraded geometry, and
  increased latency.
- A moving-only actor-aware 8k ablation preserved all 7 actors and restored
  background capacity, but still worsened moving crops, held-out SSIM, and
  LiDAR p90.
- The alignment audit proved MCMC noise moved actor-local Gaussians tens to
  hundreds of metres beyond vehicle cuboids. Actor-bound projection now
  prevents that escape.
- A calibrated LiDAR-frame cuboid-time correction changes scene 040 moving
  training observations by 46.97 ms p50 and 55.85 ms p90. Its semantics are
  serialized so legacy checkpoints remain reproducible.
- Boundary-only and boundary-plus-time short candidates still failed moving
  crop appearance and are rejected.

The next action is per-point seed-timing diagnosis before further long
training: assign actor 0/1 LiDAR points at their own scan timestamps, project
them into exact camera rows, and verify source-pixel alignment. Then apply
trajectory `exists_at_time` during rendering to prevent out-of-window ghost
actors. See `experiments/stage_h3_actor_alignment_and_timing.md`.

## Purpose

Stage H3 shifts the next milestone from a visually packaged cockpit demo to a
stable reconstructed driving scene that can later support a cockpit-style,
human-drivable simulator.

The target is:

```text
multi-sensor driving log
→ synchronized cameras + LiDAR + ego pose/IMU + dynamic annotations
→ geometry-constrained multi-camera scene reconstruction
→ clean static background with correct metric scale
→ nearby and logged-trajectory view changes
→ measured, repeatable visual stability
→ later cockpit display and steering-wheel control
```

The immediate priority is reconstruction quality and stability. A three-screen
UI, steering wheel support, traffic behavior, and autonomous-agent integration
remain downstream work until the reconstructed scene itself is trustworthy.

The rendered appearance still comes from camera images. LiDAR supplies sparse
metric geometry, ego pose/IMU supplies vehicle motion and gravity orientation,
and 3D cuboids provide the primary dynamic actor tracks. Available point-cloud
semantic labels are supplemental. These signals support visual reconstruction;
they do not replace it.

## Current Diagnosis

The H1/H2 baseline proves that the repository can train a WayveScenes101
Splatfacto model and render it from nearby ego poses, but it is still an early
research baseline rather than a stable drivable reconstruction:

- H1 used the official leave-front-camera split. This is useful for measuring
  view extrapolation, but it deliberately withholds the most important driving
  view from training.
- Static Splatfacto bakes moving vehicles and pedestrians into one fixed scene,
  producing gray or black ghost artifacts in exactly the regions that matter
  for driving.
- The H2 loop renders around one fixed reference frame. It does not yet advance
  through the original logged trajectory.
- The current safety envelope is limited to about +/-2 m forward, +/-0.5 m
  left, and +/-5 degrees yaw.
- The display is a five-camera mosaic for inspection, not a continuous
  cockpit-style surround view.
- WayveScenes101 is sufficient for the existing camera-only research baseline,
  but the current project inputs do not provide the LiDAR geometry and complete
  dynamic-object supervision needed for the next stability-focused baseline.

Therefore, simply extending the current 8k-step training to 30k steps is not
the right first move. Longer training may sharpen some static details, but it
will not by itself remove dynamic-object ghosts or make the scene drivable.

## Success Definition

Stage H3 succeeds when we can show a selected driving scene where:

- static background elements such as buildings, trees, road surface, rails, and
  lane or curb structure remain recognizable across the main camera views;
- dynamic-object residue no longer dominates the drivable area;
- front, left-forward, and right-forward views are spatially consistent enough
  for a human to understand the road layout;
- nearby pose changes do not collapse, smear, or introduce large floating
  artifacts in the main driving region;
- the reconstruction uses a verified metric coordinate system, stable ground
  geometry, and correctly aligned camera/LiDAR observations;
- logged-trajectory time progression can be inspected separately from human
  control offsets;
- visual quality, ghosting, temporal stability, and render latency are measured
  with repeatable scripts and preserved artifacts.

This is still not a full driving simulator acceptance test. It is the quality
gate before investing in cockpit UI and input-device polish.

## Workstreams

### H3-0A — Reproducible Training And Test Environment

Status: **completed and accepted on 2026-07-18**. Do not begin a long dataset
download or model run until the remaining H3-0B data and calibration gates
pass.

Initial host observations on 2026-07-18:

- NVIDIA GeForce RTX 4090 D with 24 GB VRAM is visible on the host;
- NVIDIA driver version is `580.95.05`;
- the existing `wayve_scenes_env` uses Python 3.10.14 and PyTorch
  2.3.1+cu118, and can access the GPU on the host;
- about 292 GB was free on the project filesystem at inspection time;
- PandaSet and neurad-studio were not present under the currently documented
  external artifact roots;
- the historical Stage-1 PandaSet scripts default to `/data/external`, while
  the working H1 artifacts are under `/home/yawei/stage1_external`;
- the historical fetch script downloads the full PandaSet archive, so it must
  not be run unchanged as an environment-only preparation command.

Implemented result:

- created `/home/yawei/stage3_external/envs/h3_splatad` without modifying
  `wayve_scenes_env`;
- pinned neurad-studio, custom SplatAD gsplat, custom viser, PandaSet devkit,
  and tiny-cuda-nn revisions;
- accepted Python 3.10.20, PyTorch 2.0.1+cu118, and CUDA toolkit 11.8;
- executed tiny-cuda-nn, camera rasterization, and LiDAR rasterization CUDA
  kernels on the RTX 4090 D;
- verified `splatad`, `neurad`, and `pandaset-data`;
- added `scripts/setup_stage_h3_environment.sh` and
  `scripts/check_stage_h3_environment.sh`;
- preserved an H1 checkpoint render as a post-install regression check;
- left PandaSet data and all new model training untouched.

See `experiments/stage_h3_environment.md` for exact revisions, failure
recovery, output shapes, and the current 264 GiB free-space observation.

Environment strategy:

- preserve `wayve_scenes_env` so the verified H1/H2 checkpoint remains
  reproducible;
- create a separate pinned H3 environment for neurad-studio, SplatAD, NeuRAD,
  the PandaSet devkit, and their CUDA extensions;
- pin the neurad-studio and custom gsplat commits instead of tracking moving
  upstream branches;
- use SplatAD as the first multi-sensor 3DGS method because neurad-studio
  already implements camera/LiDAR rendering and autonomous-driving
  dataparsers;
- keep NeuRAD as the second method only when SplatAD results require a
  quality-oriented comparison;
- use one configured external root for code, data, checkpoints, logs, and
  renders; do not put heavy artifacts in Git.

Environment tasks:

1. inspect upstream installation requirements and choose compatible pinned
   Python, PyTorch, CUDA toolkit, neurad-studio, and custom gsplat versions;
2. create the isolated Conda environment without modifying
   `wayve_scenes_env`;
3. clone and pin neurad-studio and PandaSet devkit revisions under the selected
   external code root;
4. provide one environment check command that reports host, GPU, driver,
   Python, PyTorch/CUDA, package versions, pinned commits, disk space, and
   artifact root;
5. verify imports and CUDA kernels without downloading PandaSet;
6. verify the PandaSet dataparser command and SplatAD/NeuRAD command-line
   entrypoints;
7. run a minimal synthetic or packaged-fixture check for camera transforms,
   LiDAR points, and rasterization;
8. document the exact environment creation and recovery commands.

Environment acceptance gate:

- CUDA is visible from the new environment;
- custom CUDA extensions compile and execute once;
- neurad-studio exposes `splatad`, `neurad`, and `pandaset-data`;
- an environment report can be regenerated with one command;
- a no-data or tiny-fixture smoke test passes;
- the existing H1/H2 environment and checkpoint still load unchanged;
- the external artifact root has a recorded storage budget before PandaSet is
  downloaded and extracted.

Expected lightweight artifacts:

```text
scripts/check_stage_h3_environment.sh
experiments/stage_h3_environment.md
```

### H3-0B — Multi-Sensor Data Foundation

After H3-0A passes, run one narrow pilot on a driving dataset that supplies
synchronized cameras, LiDAR, ego motion, and dynamic annotations.

Status: **data, calibration, Level 1, and the Level 2 logged-pose visual gate
passed on 2026-07-19 for PandaSet scene 040; nearby-pose geometry and temporal
stability have not yet passed.**

Execution result:

- the user accepted the recorded dataset terms and 44.5-GB acquisition;
- the pinned archive matched its exact size and SHA-256 and passed ZIP
  integrity validation;
- only daylight scene 040 was extracted after annotation and visual triage;
- six cameras, Pandar64, fused poses, timestamps, cuboids, GPS, and point-cloud
  semantics loaded for all 80 frames;
- representative Pandar64/camera overlays showed plausible alignment;
- asynchronous sensor offsets of about 10.8-50.1 ms were measured and retained
  as a timing requirement;
- the exact SplatAD parser produced 240 training cameras, 40 Pandar64 sweeps,
  and 7 actor trajectories under the default 0.5 split;
- a 100-step SplatAD checkpoint was saved, reloaded, and rendered over 240
  held-out images;
- the smoke output remains visibly under-trained, with broad colored splats and
  LPIPS 0.8402, so it proves integration rather than quality.
- a subsequent 2,000-step run used a fixed 0.9 temporal split, 0.5 LiDAR
  downsampling, and a 750,000-point seed cap;
- its 48-view fixed holdout reached PSNR 24.7109, SSIM 0.7392, and LPIPS
  0.4475, with recognizable static structure in all six cameras;
- the logged-pose visual gate now passes, but nearby-pose, depth, flicker,
  actor-residue, peak-VRAM, and warm-latency checks remain open.

See `experiments/stage_h3_dataset_foundation.md` and
`experiments/stage_h3_scene_040_smoke.md`. The Level 2 record is
`experiments/stage_h3_scene_040_pilot.md`.

Acquisition-audit evidence from 2026-07-18:

- the neurad-linked mirror contains one 44,520,528,731-byte ZIP at a recorded
  repository revision and LFS SHA-256 oid;
- archive plus full extraction needs about 83.12 GiB, while archive plus the
  largest single-scene extraction needs about 41.91 GiB;
- its central directory contains 103 scenes, 75,758 entries, and 76 scenes with
  point-cloud semantics;
- the standard mirror does not expose per-sequence packages;
- the official page's visible download link returned HTTP 404 during the audit;
- the archive uses CC BY 4.0 plus additional terms, so no data was downloaded
  without explicit acceptance.

See `experiments/stage_h3_dataset_foundation.md`.

The first pilot dataset is **PandaSet**. Its official documentation describes
camera images, two LiDAR sensors, GPS/IMU data, 3D cuboid annotations, and
point-cloud semantic-segmentation labels for selected scenes. This is a
manageable first integration target for testing geometry-constrained
reconstruction and dynamic-background separation. WayveScenes101 remains the
known camera-only baseline; it is not discarded.

Only one short PandaSet scene or time window should be integrated initially.
First check whether the selected distribution supports per-scene access. If the
source packages all scenes in one archive, download the archive only after the
access, license, checksum, extraction-size, and free-space checks pass; train
and preprocess only the selected pilot window.

Tasks:

- record the dataset version, access terms, selected scene, frame window,
  sensor list, timestamps, and source paths;
- load synchronized multi-camera frames, intrinsics, extrinsics, LiDAR points,
  ego poses or GPS/IMU-derived poses, and available 3D boxes or semantic labels;
- transform all observations into one documented metric ego/world coordinate
  system;
- verify calibration by projecting LiDAR points into each camera and preserving
  visual overlays as evidence;
- classify LiDAR points and image regions as static background or dynamic
  objects using annotations and temporal evidence;
- export a small, non-mutating Nerfstudio/3DGS-ready dataset with all large
  artifacts outside Git;
- train one short static-background reconstruction baseline with all available
  cameras;
- compare image-only initialization against LiDAR-assisted initialization
  and/or LiDAR depth supervision before choosing the permanent method.

The accepted first SplatAD sensor path is six cameras plus `Pandar64`. Although
PandaSet also contains the forward-facing `PandarGT`, the audited parser
defaults to Pandar64 and its elevation map, azimuth resolution, missing-point
logic, and raster assumptions are mature only for that sensor. The upstream
multi-LiDAR missing-point path still contains a TODO. PandarGT therefore needs
a separate timing, calibration, raster-layout, and ablation check; it is not
part of the first reliable baseline.

LiDAR should initially serve three concrete purposes:

1. initialize or anchor scene geometry in metric coordinates;
2. supervise rendered depth on reliable static points, especially road,
   building, curb, pole, and barrier surfaces;
3. reduce floating or incorrectly placed geometry when the virtual ego camera
   moves away from an observed pose.

Ego pose/IMU should initially serve three different purposes:

1. provide a time-ordered vehicle trajectory and gravity-aligned coordinate
   frame;
2. place every camera and LiDAR sweep consistently over time;
3. support later composition of small human-control offsets on top of the
   logged trajectory.

Raw IMU integration is not a first deliverable. Prefer the dataset's calibrated
or fused ego poses when available, because unaided acceleration and angular-rate
integration drifts over time.

Dynamic objects require a separate treatment. LiDAR alone does not remove
vehicle or pedestrian ghosts. SplatAD already supports actor Gaussians and
trajectories built from 3D cuboids, so the dynamic-aware object layer is the
primary pilot path. Use masking or down-weighting as a static-background
diagnostic and fallback, not as an assumed replacement for the implemented
actor model. Point-cloud semantic labels are supplemental; they are not dense
image masks.

Pilot gates:

- **Data gate:** synchronized cameras, LiDAR, poses, and annotations load for
  the selected window with no unexplained frame or timestamp mismatch.
- **Calibration gate:** projected static LiDAR points align visibly with road,
  building, curb, vehicle, and pole boundaries in representative cameras.
- **Geometry gate:** ground orientation and metric scale are correct, and a
  nearby-pose render shows fewer major floaters or depth failures than the
  corresponding image-only run.
- **Visual gate:** the static road and background are coherent enough to proceed
  to a longer 8k-step baseline; otherwise stop and diagnose calibration,
  synchronization, masks, or reconstruction method.

Expected artifact:

```text
experiments/stage_h3_dataset_foundation.md
```

Primary dataset reference, accessed 2026-07-18:

- PandaSet official site: <https://pandaset.org/>

### Dataset Priority And Roles

Use one dataset at a time. The order is based on the current goal of reaching a
stable drivable reconstruction quickly, not on which dataset is largest.

| Priority | Dataset | Relevant official contents | Role in this project |
| --- | --- | --- | --- |
| 1 | PandaSet | 6 cameras, a mechanical 360-degree LiDAR, a forward-facing LiDAR, GPS/IMU, 3D cuboids, and point-cloud semantic labels | First H3 pilot for multi-sensor reconstruction and dynamic-object separation |
| 2 | nuScenes | 6 cameras, 1 LiDAR, 5 radar sensors, ego localization/CAN-bus data, maps, and 3D boxes | Second adapter after PandaSet; start with the mini split for ecosystem and standard-dataset comparison |
| 3 | Argoverse 2 Sensor | 7 panoramic ring cameras, 2 stereo cameras, two 32-beam LiDARs, 6-DOF ego poses, 3D cuboids, and per-log maps; about 1 TB extracted for the full dataset | Later lane, curb, road-boundary, and map-consistency evaluation |
| 4 | Waymo Open Dataset | Perception data with 5 cameras, 5 LiDARs, calibrations, vehicle poses, and tracked 2D/3D labels; a separate End-to-End dataset has 8-camera 360-degree coverage | Later large-scale and higher-complexity validation, not the first migration target |

Primary references, accessed 2026-07-18:

- PandaSet: <https://pandaset.org/>
- nuScenes: <https://www.nuscenes.org/nuscenes>
- Argoverse 2 Sensor Dataset:
  <https://argoverse.github.io/user-guide/datasets/sensor.html>
- Waymo Open Dataset: <https://waymo.com/open/about/>
- neurad-studio/SplatAD/NeuRAD:
  <https://github.com/georghess/neurad-studio>

If the PandaSet pilot exposes a blocking dataset limitation, evaluate nuScenes
mini next. Do not integrate several new datasets in parallel during H3.

### H3-0C — Checkpoint Reuse And Training Budget

Avoid repeated full training. In this project, "pretrained weights" can refer to
two different things:

1. general-purpose perception weights, such as segmentation, detection,
   tracking, or monocular-depth networks, which can often be reused across
   scenes;
2. a reconstructed NeRF/3DGS checkpoint, which stores the geometry, appearance,
   dynamic actors, and coordinate system of one particular driving scene.

The second type is scene-specific. The current Wayve `scene_094` checkpoint can
be reused for rendering, evaluation, UI development, and regression testing on
that scene, but it cannot represent a new PandaSet street. A PandaSet checkpoint
can be reused without retraining only when the scene, sensor calibration,
coordinate convention, preprocessing, model configuration, and checkpoint
format match the intended run.

Before training, search the selected upstream release for a compatible
checkpoint of the exact PandaSet sequence. The H3-0A source audit found resume
instructions but no exact-sequence model-zoo catalog in the inspected
neurad-studio README or documentation. Continue the search when the sequence is
chosen; if a compatible checkpoint exists, load and evaluate it first. Treat it
as a baseline rather than assuming it matches this project's preprocessing or
stability target.

Every run should have a reproducibility key containing:

```text
dataset version
+ scene and frame window
+ calibration/preprocessing version
+ dynamic-mask version
+ method and configuration
+ code commit
+ environment version
```

If that key already has a valid checkpoint, reuse it. Retraining is justified
only when the scene data, calibration, masks, model/loss, or relevant
configuration changed, or when a checkpoint is missing or incompatible.

Use a staged budget:

- **Level 0 — no training:** environment, dataparser, synchronization,
  calibration overlay, one-batch loading, and checkpoint-loading checks;
- **Level 1 — smoke:** at most about 100 iterations to prove the complete GPU
  path and artifact writing;
- **Level 2 — pilot:** about 1k-2k iterations on one short window for fast
  visual failure diagnosis;
- **Level 3 — baseline:** 8k steps only after data, calibration, geometry, and
  visual gates pass;
- **Level 4 — longer run:** 15k/30k only when fixed metrics and visual evidence
  show that optimization time, rather than data or modeling, is the remaining
  limitation.

Save and reuse checkpoints, cached masks, LiDAR projections, processed sensor
manifests, and evaluation camera paths. UI, controller, and renderer development
must use a fixed accepted checkpoint instead of triggering training.

### H3a — Scene And Segment Triage

Find scenes and time ranges in both the new pilot data and existing
WayveScenes101 data that are suitable for a stable reconstruction baseline.

Tasks:

- select the PandaSet pilot window using dynamic-object clutter, LiDAR
  coverage, road visibility, lighting stability, and camera overlap;
- scan available WayveScenes101 scenes and summarize camera coverage, frame
  counts, image resolution, mask availability, and obvious data issues;
- rank candidate segments by dynamic-object clutter, near-field occlusion,
  road visibility, lighting stability, and camera overlap;
- keep `scene_094` as the known hard baseline, but add at least one cleaner
  segment for stability-first development;
- record selected scene IDs, frame windows, source paths, and reasons for
  selection in a small experiment note.

Expected artifact:

```text
experiments/stage_h3_scene_selection.md
```

### H3b — All-Camera Static Reconstruction Baseline

Train reconstructions intended for visual and geometric stability, not only
leave-one-camera-out stress testing.

Tasks:

- train the first geometry-constrained all-camera PandaSet SplatAD baseline
  only after H3-0A/H3-0B environment, calibration, and export gates pass;
- create a non-mutating split builder that trains with all five Wayve cameras;
- hold out frames by time or sparse frame index for evaluation rather than
  holding out the full front camera;
- train an 8k-step all-camera `splatfacto-big` baseline on the selected scene;
- render front, left-forward, right-forward, and full five-camera comparison
  videos;
- compare against the existing `scene_094` leave-front baseline.

Decision rule:

- do not move to 15k/30k training until the all-camera 8k baseline has been
  visually inspected and measured;
- if dynamic ghosts dominate the road, prioritize dynamic-object handling before
  longer training.

### H3c — Dynamic-Object Suppression

Build a cleaner static-background reconstruction by preventing moving objects
from corrupting the static scene.

Tasks:

- audit the existing Wayve masks and confirm which regions they actually cover;
- add or import dynamic-object masks for vehicles, pedestrians, cyclists, and
  other near-field moving objects;
- train a masked or down-weighted static-background baseline;
- compare raw all-camera training versus dynamic-suppressed training on the
  same frames and camera views;
- keep all generated masks and large artifacts outside Git, with only metadata
  and reproduction commands committed.

Acceptance signal:

- road, building, rail, tree, and curb structure should become cleaner even if
  removed dynamic objects leave holes or incomplete regions;
- the main driving area should not be blocked by large gray or black ghost
  blobs.

### H3d — Stability Evaluation

Turn "looks stable" into repeatable evidence.

Tasks:

- render a nearby-pose grid around selected reference frames, for example:

```text
forward: -1.0 m, 0.0 m, +1.0 m
left:    -0.25 m, 0.0 m, +0.25 m
yaw:     -3 deg, 0 deg, +3 deg
```

- record render latency after explicit warm-up;
- generate per-camera contact sheets and short videos for inspection;
- report image metrics where ground truth exists;
- compare rendered depth against held-out static LiDAR points where the
  calibration and visibility permit it;
- report ground-plane stability and metric-scale consistency;
- add driving-relevant checks for road-region ghosting, lane or curb stability,
  temporal flicker, and multi-camera consistency.

Expected artifact:

```text
experiments/stage_h3_stability_evaluation.md
```

### H3e — Logged-Trajectory Time Progression

After the static reconstruction is clean enough, make the renderer move along
the original recorded ego trajectory instead of one fixed reference frame.

Tasks:

- use calibrated/fused ego poses from the multi-sensor dataset as the primary
  trajectory source;
- retain the current camera-derived Wayve rig trajectory as a comparison path,
  with its weaker geometric provenance stated explicitly;
- render the original trajectory first, without human offsets;
- then compose small human-control deviations on top of the logged trajectory;
- keep the deviation envelope conservative until the nearby-pose grid shows
  stable results.

This work creates the foundation for the "can drive inside it" feeling, but it
should not hide poor reconstruction quality.

## Deferred Until The Scene Is Stable

The following are useful, but they should wait until Stage H3 produces a stable
reconstruction:

- three-screen cockpit UI;
- steering wheel or gamepad integration;
- collision and road-boundary constraints;
- responsive traffic agents;
- autonomous-driving model integration;
- highly polished browser presentation.

## First Concrete Task

H3-0A, H3-0B, Level 1, and the Level 2 logged-pose visual pilot are complete.
Continue by reusing the 2,000-step checkpoint without retraining:

1. keep `scripts/check_stage_h3_environment.sh` as the H3 environment
   acceptance command and preserve the verified data, calibration, config,
   checkpoint, fixed test split, metrics, and visual summaries;
2. render a fixed nearby-pose grid around several logged timestamps and inspect
   front/side/rear floaters, holes, and ground deformation;
3. compare rendered depth against held-out Pandar64 in static road, curb,
   facade, pole, and vehicle regions;
4. measure logged-trajectory flicker, camera-to-camera structure, dynamic actor
   residue, warmed render latency, and peak VRAM;
5. stop and diagnose timing, seed geometry, loss schedules, or actor/background
   separation if any geometry or temporal gate fails;
6. authorize the 8k all-camera baseline only after these no-retraining checks
   pass.

PandaSet SplatAD remains the geometry- and actor-aware candidate and Wayve
`scene_094` remains the known camera-only hard baseline. Cockpit UI,
logged-trajectory playback, and controller integration stay deferred until the
reconstruction gate passes.
