# Driving Scene Reconstruction

This repository studies **log-driven driving scene reconstruction for a human-drivable panoramic simulator**.

The first target is not an autonomous-agent benchmark and not a fixed video generator. The first target is:

> Given a real driving log, reconstruct a scene representation that can render updated 360-degree / multi-view driving observations while a human controls the ego vehicle.

In other words:

```text
real driving log
→ scene reconstruction
→ human control input
→ ego state update
→ 360-degree / multi-view sensor rendering
→ human-drivable simulation loop
```

Autonomous-driving model integration is a later extension. For the initial system, the driver is a human using a keyboard, controller, or steering wheel.

## Scope

The repository covers:

- real-driving scene reconstruction from logs;
- 360-degree / panoramic / multi-camera driving view generation;
- human-in-the-loop ego control;
- lightweight ego vehicle state update;
- renderer interfaces for reconstructed scenes;
- evaluation of image, geometry, temporal, and driving-relevant consistency;
- credibility checks as one evaluation function, not as the whole project.

## Current Non-Goals

The current stage does **not** prioritize:

- connecting an autonomous-driving agent;
- training a new general world model;
- generating only a fixed video without closed-loop control;
- downloading or committing large datasets, checkpoints, or rendered videos;
- making photorealism the only success criterion.

## System Concept

The simulator should eventually support this loop:

```text
HumanControl(steer, throttle, brake)
→ EgoState(x, y, yaw, speed, time)
→ VehicleModel.step(...)
→ Renderer.render(scene, ego_state, camera_rig)
→ front / left / right / rear / panorama observations
```

Different rendering backends can be tested later:

```text
ReplayRenderer: original-log replay only
PanoramaRenderer: panorama / surround-view projection
ReconstructionRenderer: 3DGS / NeRF / NeuRAD / SplatAD scene rendering
HybridRenderer: geometry-based rendering plus repair / inpainting
```

For now, do **not** add a world-model renderer path. The immediate priority is
to make the reconstructed scene stable enough for a later human-drivable
cockpit-style simulator.

## Current Status

The repository now has a connected H0-H3 path:

```text
Stage H0
→ dependency-free human-control, ego-state, vehicle-model, and renderer interfaces

Stage H1
→ WayveScenes101 scene_094 Splatfacto baseline trained for 8,000 steps
→ official held-out front-camera metrics and validated reference videos

Stage H2
→ EgoState nearby-pose displacement mapped into Nerfstudio scene coordinates
→ real checkpoint rendering through the Renderer protocol
→ five-camera keyboard/display loop with a headless validation mode

Stage H3-0A
→ isolated, pinned neurad-studio/SplatAD environment
→ synthetic camera and LiDAR CUDA kernels verified on RTX 4090 D

Stage H3-0B / Level 1
→ pinned PandaSet archive verified and only daylight scene 040 extracted
→ six cameras, Pandar64, fused poses, cuboids, and timestamps passed data/calibration gates
→ 100-step SplatAD checkpoint reloaded and rendered over 240 held-out images

Stage H3 Level 2
→ 90% temporal training split, six cameras, Pandar64, and actor tracks
→ 2,000-step checkpoint recovered recognizable static structure in all six views
→ 48-view means: PSNR 24.7109, SSIM 0.7392, LPIPS 0.4475

Stage H3 Level 3
→ exact-resume static 8,000-step checkpoint
→ 48-view means: PSNR 26.6605, SSIM 0.8145, LPIPS 0.2818
→ 126 finite nearby-pose views and measured LiDAR/temporal/latency evidence

Stage H3 Level 4
→ stationary+moving and moving-only actor-aware 8k ablations
→ both rejected; static 8k remains the accepted checkpoint

Stage H3 Level 5
→ proved MCMC actor-local points escaped 34-290 m from vehicle cuboids
→ added actor bounds and explicit calibrated cuboid timing
→ short candidates remained visually weak and were rejected

Stage H3 Level 6
→ audited scan-centre versus per-point actor seeds on real source images
→ held-out vehicle precision fell from 89.16% to 87.97%
→ rejected more training; static 8k remains the accepted result

Stage H3 Level 7
→ static-8k connected to the common Renderer over all 80×6 logged cameras
→ PandaSet trajectory time plus bounded human forward/left/yaw offsets
→ full 7.899 s sequence rendered as logical frames 0-79
→ six-camera Renderer p95 74.37 ms at 0.5 scale; exact reset passed
→ browser W/S/A/D/R loop validated; server control-to-JPEG p95 78.19 ms

Stage H3 Level 7A
→ explicit `safe` and `visible` logged-movement profiles
→ visible profile defaults for the browser: ±2.0 m forward, ±0.75 m left,
  and ±8° yaw
→ counterfactual probe confirmed stronger same-time view changes and produced
  a 7.9 s visible-movement demo

Stage H3 Level 7B
→ automated drivability preflight for the accepted static-8k renderer
→ visible-profile preflight passed 17 backend gates over all 80 logical frames
→ browser loop now records trial JSON with request→image and input→image timing
```

The H2 renderer clones the dataset cameras' full intrinsics, fisheye distortion,
and rig extrinsics at a selected reference frame. It currently enforces a
conservative nearby-pose envelope of ±2 m forward, ±0.5 m left, and ±5° yaw.

The current baseline is still not production quality. The accepted static H3
8k checkpoint now supports logged-time, six-camera rendering with small human
offsets at about 13.4 complete observations/s at 0.5 output scale. A browser
W/S/A/D/R loop now advances that trajectory automatically at 10 Hz. This is a
tested backend/browser-service result, not yet a real operator
keyboard-to-display driving trial. Close vehicles remain blurred or baked into
the background, and no collision or responsive traffic model exists. Such
traffic artifacts are a mandatory later blocker whenever they can change the
driving decision.

## Run

The lightweight simulator and tests use the standard library:

```bash
python3 examples/sim_loop_smoke.py
python3 -m unittest discover -s tests -v
```

On the machine containing the Stage H1 checkpoint:

```bash
scripts/run_stage_h2_scene_094.sh smoke \
  --forward 0.5 --left 0.2 --yaw-degrees 2 \
  --cameras front-forward left-forward right-forward left-backward right-backward

scripts/run_stage_h2_scene_094.sh interactive
```

Interactive controls are `W/S/A/D`, `R` to reset, and `Q` or Escape to quit.
For an SSH or other display-less session, use the browser viewer:

```bash
scripts/run_stage_h2_scene_094.sh interactive --web --output-scale 0.25
```

For a larger detailed front view:

```bash
scripts/run_stage_h2_scene_094.sh interactive \
  --web \
  --output-scale 0.5 \
  --cameras front-forward
```

See `docs/stage_h2_reconstruction_renderer.md` for coordinate conventions,
validation evidence, and limitations.

For the accepted PandaSet static-8k logged-time backend, with no retraining:

```bash
scripts/run_stage_h3_pandaset_040.sh logged-renderer-smoke
```

This produces the full 80-frame, six-camera sequence plus nearby-pose and reset
evidence under `/home/yawei/stage3_external/artifacts/`. It requires the pinned
H3 environment and the existing scene-040 static-8k checkpoint.

For a visibly larger counterfactual-motion check:

```bash
H3_LOGGED_MOVEMENT_PROFILE=visible \
scripts/run_stage_h3_pandaset_040.sh logged-renderer-smoke
```

For the current automated drivability preflight:

```bash
scripts/run_stage_h3_pandaset_040.sh drivability-preflight
```

This writes a JSON report plus counterfactual and sequence review images under
`/home/yawei/stage3_external/artifacts/scene_040_drivability_preflight/`. It is
a backend preflight, not a substitute for the human browser driving trial.

To drive the same reconstruction from another Tailscale-connected computer:

```bash
scripts/run_stage_h3_pandaset_040.sh logged-browser
```

Then open `http://100.116.66.57:8766` in one browser tab only. The logged car
advances automatically; `W/S` adjust its forward offset speed, `A/D` adjust
the heading offset, and `R` restarts the log. `S` does not pause the underlying
recorded trajectory. The browser now defaults to the visible movement profile
so counterfactual motion is easy to see. Use `H3_BROWSER_MOVEMENT_PROFILE=safe`
when running the conservative acceptance envelope. The default 0.25 browser
render scale produces a 1440-pixel-wide six-camera view; it is twice the
linear camera resolution of the earlier 0.125 viewer that was judged too small.
The browser also exposes `/trial.json` and writes the same trial report to
`/home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial.json`
by default.

## Current Next Step

Static 8k remains the fixed accepted checkpoint. The next main-line step is a
operator acceptance run of the new browser loop over the complete real
7.899-second trajectory. Record the page's browser-side frame-update latency,
road/curb continuity, steering direction, reset, and deterministic replay. Do
not start another dynamic training run before this driving run reveals an
artifact that affects the road or obstacle decision.

An automated preflight now exists before that operator run. It should be green
before a human trial is treated as meaningful, but it deliberately leaves
road/lane continuity, steering direction by eye, physical display latency, and
dynamic-traffic decision impact as review items.

The success criteria are deliberately separate from generic image metrics:

- `docs/drivability_acceptance_criteria.md` defines whether the scene can be
  driven;
- `docs/driver_attention_and_dynamic_traffic_requirements.md` records that the
  operator only drives and that dynamic correctness is deferred, not waived;
- `experiments/stage_h3_logged_renderer_mvp.md` records the Level-7 run and its
  acceptance boundary.

See also `docs/stage_h3_stable_drivable_reconstruction_plan.md`.

Environment acceptance can be regenerated without PandaSet:

```bash
scripts/check_stage_h3_environment.sh
scripts/run_stage_h3_pandaset_040.sh data-gate
scripts/run_stage_h3_pandaset_040.sh static-8k
scripts/run_stage_h3_pandaset_040.sh paths
```

The static 8k run is reused when its checkpoint exists; it is not retrained by
default. Detailed results and rejected actor ablations are in
`experiments/stage_h3_static_8k_and_actor_ablations.md`.

## Repository Structure

```text
.
├── README.md
├── PROJECT_STATE.md
├── pyproject.toml
├── docs/
│   ├── human_drivable_simulator_project.md
│   ├── codex_next_task_stage_h0.md
│   ├── stage_h3_stable_drivable_reconstruction_plan.md
│   ├── stage_h2_reconstruction_renderer.md
│   ├── problem_statement.md
│   └── mvp_leave_one_camera_out.md
├── src/driving_scene_reconstruction/sim/
├── examples/
├── tests/
├── experiments/
├── scripts/
└── data/
```
