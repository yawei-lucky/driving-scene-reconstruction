# Stage H3 — Stable Drivable Scene Reconstruction Plan

Date: 2026-07-18

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
and 3D boxes or semantic labels identify dynamic regions. These signals support
visual reconstruction; they do not replace it.

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

### H3-0 — Multi-Sensor Data Foundation

Before spending more GPU time on the current camera-only baseline, run one
narrow pilot on a driving dataset that supplies synchronized cameras, LiDAR,
ego motion, and dynamic annotations.

The first pilot dataset is **PandaSet**. Its official documentation describes
camera images, two LiDAR sensors, GPS/IMU data, 3D cuboid annotations, and
semantic-segmentation labels. This is a manageable first integration target for
testing geometry-constrained reconstruction and dynamic-background separation.
WayveScenes101 remains the known camera-only baseline; it is not discarded.

Only one short PandaSet scene or time window should be integrated initially.
Do not download or convert the full dataset before the pilot passes its gates.

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
vehicle or pedestrian ghosts. Use 3D boxes, semantic labels, or instance tracks
to exclude or down-weight moving-object pixels and points when training the
static background. Object-layer reconstruction can follow after the background
is stable.

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

If the PandaSet pilot exposes a blocking limitation, evaluate nuScenes next.
Argoverse 2 is reserved for stronger map and lane-geometry work, and Waymo Open
Dataset for a later larger-scale experiment. Do not integrate several new
datasets in parallel during H3-0.

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

- train the first geometry-constrained all-camera PandaSet baseline only after
  H3-0 calibration and export gates pass;
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

Start H3 with a narrow, evidence-producing task:

1. audit PandaSet access, terms, sensor files, annotations, and local storage
   requirements without downloading the full dataset;
2. select one short pilot scene or frame window;
3. load and synchronize cameras, LiDAR, fused ego poses, and dynamic
   annotations;
4. produce camera/LiDAR calibration overlays and a documented common
   coordinate system;
5. export a minimal Nerfstudio/3DGS-ready static-background dataset;
6. run a short image-only versus LiDAR-assisted reconstruction comparison;
7. write the result, visual evidence, commands, artifact paths, and failure
   cases in `experiments/stage_h3_dataset_foundation.md`.

Only after the H3-0 gates pass should we spend GPU time on the longer all-camera
baseline. The next comparison then uses PandaSet as the geometry-grounded
candidate and Wayve `scene_094` as the known camera-only hard baseline.
