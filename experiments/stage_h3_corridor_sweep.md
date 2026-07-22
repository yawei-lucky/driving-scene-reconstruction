# Stage H3 Full-Corridor Keep-or-Rebuild Sweep

Date: 2026-07-22

## Question

Is the accepted PandaSet scene-040 static-8k checkpoint already too weak for
the first restricted human-driving prototype, or can development continue
without another training run?

This is deliberately a cheap decision test, not a final reconstruction audit.

## Method

The new sweep samples five longitudinal stations spanning the 64.595 m logged
centreline. At each station it renders the six synchronized cameras at lateral
offsets -1, 0, and +1 m using checkpoint step 7999 at 0.5 output scale.

```bash
H3_CORRIDOR_SWEEP_ROOT=/home/yawei/stage3_external/artifacts/scene_040_corridor_sweep_full_scale_050 \
H3_CORRIDOR_SWEEP_SCALE=0.5 \
H3_CORRIDOR_SWEEP_STATIONS=-34,-17,0,15,29 \
scripts/run_stage_h3_pandaset_040.sh corridor-sweep
```

The signed stations are relative to the anchor-local world origin. They
cover the route from approximately 0.60 m to 63.60 m of corridor progress.

## Results

- 15/15 observations contained valid RGB output from all six cameras.
- Renderer latency was 67.41 ms p50, 70.78 ms p95, and 73.22 ms maximum.
- Direct visual inspection found the front road continuous and readable at all
  five stations and all three offsets; no large hole or incorrect road surface
  blocked driving.
- Close parked cars, foliage, and some side-camera regions still smear or
  stretch, especially near the route ends. The result does not establish final
  360-degree quality or geometry-trustworthy closed-loop simulation.

Evidence is outside Git:

- `/home/yawei/stage3_external/artifacts/scene_040_corridor_sweep_full_scale_050/corridor_sweep.json`
- `/home/yawei/stage3_external/artifacts/scene_040_corridor_sweep_full_scale_050/corridor_5x3_front.jpg`
- the 15 `six_forward_*.jpg` mosaics in the same directory.

## Immediate Follow-Through

The manual world browser now resets to the beginning of the recorded corridor
instead of the anchor midpoint. A 30 s scripted GPU/HTTP drive exercised 300
six-camera observations and progressed 58.607 m of 64.595 m with no boundary
hit. Maximum distance from the centreline was 0.0182 m. Sampled mosaics showed
coherent forward road progression.

The run record and four sampled mosaics are under:

`/home/yawei/.codex/visualizations/2026/07/17/019f701a-4fa3-7dc3-843a-fa53969531b3/`

## Decision

Retain static-8k as the visual background for the first restricted
human-driving prototype. It is not the final model, but the cheap test does not
justify stopping for another training run. The next gate is an operator drive.
Use targeted LiDAR diagnosis, additional traversals, and MTGS/SplatAD rebuilding
only when a concrete segment harms driving or geometry-trustworthy closed-loop
evaluation begins.
