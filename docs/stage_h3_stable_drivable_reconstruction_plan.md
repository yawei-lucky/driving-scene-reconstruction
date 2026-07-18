# Stage H3 — Stable Drivable Scene Reconstruction Plan

Date: 2026-07-18

## Purpose

Stage H3 shifts the next milestone from a visually packaged cockpit demo to a
stable reconstructed driving scene that can later support a cockpit-style,
human-drivable simulator.

The target is:

```text
real driving log
→ stable multi-camera scene reconstruction
→ clean static background and reduced dynamic ghosts
→ nearby and logged-trajectory view changes
→ measured, repeatable visual stability
→ later cockpit display and steering-wheel control
```

The immediate priority is reconstruction quality and stability. A three-screen
UI, steering wheel support, traffic behavior, and autonomous-agent integration
remain downstream work until the reconstructed scene itself is trustworthy.

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
- logged-trajectory time progression can be inspected separately from human
  control offsets;
- visual quality, ghosting, temporal stability, and render latency are measured
  with repeatable scripts and preserved artifacts.

This is still not a full driving simulator acceptance test. It is the quality
gate before investing in cockpit UI and input-device polish.

## Workstreams

### H3a — Scene And Segment Triage

Find scenes and time ranges that are suitable for a stable reconstruction
baseline.

Tasks:

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

Train a reconstruction intended for visual quality, not leave-one-camera-out
stress testing.

Tasks:

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

- expose calibrated logged ego poses or derive a stable rig trajectory from the
  dataset transforms;
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

1. add an all-camera WayveScenes101 split builder;
2. select one cleaner candidate scene or segment, while keeping `scene_094` as
   the hard comparison;
3. train an 8k-step all-camera Splatfacto baseline;
4. render front and five-camera visual comparisons;
5. write a short result note with metrics, contact sheets, and failure cases.

Only after this baseline should we decide whether to spend GPU time on longer
training, dynamic masks, or a different reconstruction method.
