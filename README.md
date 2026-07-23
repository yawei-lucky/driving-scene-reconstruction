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
  plus in-browser manual drivability review gates

Stage H3 Level 7C
→ browser trial acceptance checker consumes `/trial.json`
→ fails incomplete runs, missing reset/input latency, early manual-review clicks,
  over-budget latency, and any non-pass manual gate

Stage H3 Level 7D
→ scripted browser trial rehearsal drives the live HTTP loop
→ validates full-sample/reset/latency plumbing while leaving manual gates unsure

Stage H3 Level 8
→ new SplatADWorldRenderer freezes scene time and accepts simulated world poses
→ SimpleVehicleModel owns future x/y/yaw/speed instead of the logged trajectory
→ six camera poses synchronized to one anchor time and moved as one rigid rig
→ +/-1/2/3 m, +/-5/10 deg, and a continuous 22 deg left turn rendered
→ 45 six-camera observations passed; 0.5-scale Renderer p95 68.46 ms

Stage H3 Level 8A
→ independent review found and removed a learned 7.04 ms camera-time spread
→ 206 six-camera observations cover straight, symmetric turns, symmetric lane
  changes, and a complete brake-to-rest path
→ plumbing and vehicle-motion gates pass; 0.5-scale Renderer p95 70.99 ms
→ separate manual-default world browser uses real x/y/yaw dynamics and stops at
  an explicit provisional x +/-6 m, y +/-1 m, yaw +/-15 deg boundary

Stage H3 Level 8B
→ synchronized source poses provide a 64.6 m logged centreline in world space
→ the manual vehicle remains free, while the centreline supplies a provisional
  data-coverage tube instead of the old fixed 6 m rectangle
→ a 10 s GPU/HTTP run reached x=18.61 m with no boundary hit

Stage H3 Level 8C
→ a five-station x three-offset sweep sampled the complete 64.6 m corridor
→ all 15 six-camera observations were valid and the front road stayed usable
  at -1/0/+1 m, so static-8k is retained as the first simulator background
→ the manual browser now spawns at the recorded corridor start
→ a 30 s GPU/HTTP drive produced 300 observations and progressed 58.61 m

Stage H3 Level 9A
→ all 103 PandaSet GPS tracks scanned directly from the verified archive
→ scene 040 has no joinable second traversal; the nearest is 165.3 m away
→ one direction-change pair only repeats an approach and exposes no second branch
→ no verified same-intersection multi-direction set remains
→ selected daylight/night scenes 003+057 for a small shared-static coverage pilot
→ about 38 m overlaps for registration and the estimated union route is 127.9 m

Stage H3 Level 9B
→ listed all 1,043 public Argoverse TbV logs and downloaded only 531.6 MB of poses
→ found 301 branch-review and 168 opposite-direction metadata candidates
→ verified one Miami pair with about 115 m of shared approach, then straight/right branches
→ selected two ten-second windows centred on that branch for the first smoke

Stage H3 Level 9C
→ audited registered trajectories for all six released MTGS road blocks
→ promoted the smallest 3.98 GB Singapore block to a normal-road pilot candidate
→ three same-direction eight-camera traversals cover an 84-87 m gentle curve
→ official train paths are about 5.2 m apart and the held-out path lies between them
→ selected a checkpoint-only 24 GB VRAM gate before any new training or parser work

Stage H3 Level 9D
→ downloaded only 278 MB for the selected two ten-second TbV windows
→ added a two-traversal adapter with 14 camera and two aggregate-LiDAR IDs
→ shared-route LiDAR alignment passed at 0.109 m p50 and 0.241 m p90
→ 100-step save/reload smoke and all 140 held-out renders passed
→ 2,000-step result reached PSNR 20.2621, SSIM 0.6753, LPIPS 0.5193
→ both observed routes are recognizable; counterfactual driving remains untested

Stage H3 Level 9E
→ added an experiment-local seven-camera world-pose renderer in one branch frame
→ sampled 36 poses / 252 views over the shared entrance, straight, right turn,
  and -1/0/+1 m offsets
→ the 2,000-step model passed spatial coverage but failed the visual driving gate
→ exact-resumed to step 7,999; held-out quality reached 23.2130 / 0.7734 / 0.3805
→ the identical 8k probe removed most road floaters and retained both branches
→ accepted as a restricted front-corridor candidate; continuous driving is pending

Stage H3 Level 9F
→ connected the TbV 8k world-pose renderer to a route-constrained browser
→ spawns at common progress -20 m and stops at the shared anchor until the
  operator selects straight or right
→ enforces a fail-closed +/-1 m centreline tube and rejects further control
  after a support-boundary hit until reset
→ emits state, route support/margins, renderer profile, frame hash, seven-camera
  render time, server-to-JPEG time, and optional browser timing per control
→ a 241-sample GPU/HTTP machine rehearsal exercised both branches and one
  intentional boundary hit; physical keyboard-to-display review remains open

Stage H3 Level 9G
→ replaced the default seven-camera mosaic with a calibrated 150° cylindrical
  front-left/front-centre/front-right driving view
→ added an ego-up logged-trajectory support inset that shows the +/-1 m tube,
  branch paths, vehicle pose, lateral offset, and remaining margin
→ moved the complete aspect-ratio-preserving seven-camera view to `/diagnostic`
→ raised the adapter's configurable default speed cap from 2.0 to 4.0 m/s
→ both traversal calibrations cover 99.98% of the requested panorama
→ a 25-sample 4.0 m/s host smoke retained seven finite cameras and route support;
  server control-to-driving-JPEG measured 97.40/100.34 ms p50/p95
```

The H2 renderer clones the dataset cameras' full intrinsics, fisheye distortion,
and rig extrinsics at a selected reference frame. It currently enforces a
conservative nearby-pose envelope of ±2 m forward, ±0.5 m left, and ±5° yaw.

The current baseline is still not production quality. The accepted static H3
8k checkpoint now supports logged-time, six-camera rendering with small human
offsets at about 13.4 complete observations/s at 0.5 output scale. A browser
W/S/A/D/R loop now defaults to manual time progression: it opens stopped and
advances the logged trajectory only while a driving key is held. Auto playback
is still available with `H3_BROWSER_TIME_MODE=auto`. This is a tested
backend/browser-service result, not yet a real operator keyboard-to-display
driving trial. Close vehicles remain blurred or baked into the background, and
no collision or responsive traffic model exists. Such traffic artifacts are a
mandatory later blocker whenever they can change the driving decision.

The Level-8 world-pose backend is connected to a separate restricted browser.
The 64.6 m centreline comes from the synchronized recorded rig trajectory; it
does not replay or pull the simulated vehicle. A cheap full-length sweep kept
the road readable at five stations and -1/0/+1 m lateral offsets. This is enough
to retain static-8k for the first restricted human-driving prototype, but it is
not final 360-degree or geometry-trustworthy acceptance: close side objects
still deform. Wider +/-2 m and +/-3 m views remain failure/coverage diagnostics.

An archive-wide metadata pilot shows that the existing PandaSet release
cannot expand scene 040 directly and does not provide a verified straight/left/
right set at one intersection. It does contain same-direction repeats recorded
under daylight and night conditions. A follow-up metadata-only scan of
Argoverse TbV found and visually verified a better spatial-coverage candidate:
two daylight Miami logs share about 115 m of one approach, then split into a
straight route and a right turn. No left-turn traversal was found at the exact
junction. At Level 9B this was only a data-selection result; Level 9D below
records its subsequent SplatAD pilot. It is not an MTGS integration.

A follow-up audit removed the unnecessary branch requirement and identified a
published MTGS gentle-curve block, but the user selected the lower-setup-cost
TbV route first. The two bounded TbV windows now pass shared-static LiDAR,
save/reload, held-out, and common-world counterfactual gates. Exact resume from
2k to 8k raised held-out quality from 20.2621/0.6753/0.5193 to
23.2130/0.7734/0.3805, and the same 36-pose sweep kept the shared entrance,
straight branch, right turn, and -1/0/+1 m front views readable. This is a
restricted route-constrained driving candidate, not a dynamic or certified
360-degree simulator. MTGS remains a later fallback rather than the current
environment task.

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

For the minimal TbV shared-entrance driving adapter, using the fixed 8k
checkpoint without retraining:

```bash
scripts/run_stage_h3_tbv_pilot.sh driving-adapter
```

Open `http://localhost:8768` through the existing SSH/VS Code port-forwarding
path. Drive with W/S/A/D, reset with R, and choose `1` for straight or `2` for
right when the vehicle stops at the shared anchor. The default view is a
calibrated approximately 150-degree cylindrical projection of the three front
cameras with a logged-trajectory support inset. The full seven-camera mosaic
is available separately at `/diagnostic`; it is a reconstruction diagnostic,
not the driving view. The default speed cap is 4.0 m/s and can be changed with
`H3_TBV_MAX_SPEED_MPS`.

The live evidence report is available at `/evidence.json` and is also written
outside Git under
`/home/yawei/stage3_external/artifacts/tbv_branch_pair_driving_adapter/`. The
report is evidence-only; it does not certify the rendered scene.

For the true world-pose backend and current corridor probe, without retraining:

```bash
scripts/run_stage_h3_pandaset_040.sh world-pose-probe
```

This freezes one static scene time, synchronizes the six camera poses, renders
the requested +/-1/2/3 m and yaw probes, then executes straight, left/right
turn, left/right lane-change, and full-brake paths with `SimpleVehicleModel`.
The output JSON deliberately separates plumbing success, motion success, and
the still-open human/geometry corridor verdict.

For the cheap five-station keep-or-rebuild sweep:

```bash
scripts/run_stage_h3_pandaset_040.sh corridor-sweep
```

This reuses static-8k and writes 15 six-camera mosaics, a front-view contact
sheet, and a JSON timing report. It is a visual coverage decision, not a LiDAR
geometry certificate.

### Fastest visual try

For the first browser where steering changes the vehicle's future world path,
run this on the project host `shidi`:

```bash
cd /home/yawei/driving-scene-reconstruction
scripts/run_stage_h3_pandaset_040.sh world-browser
```

Then open this from a Tailscale-connected browser:

```text
http://100.116.66.57:8767
```

The page opens stopped at the beginning of the recorded 64.6 m corridor. Use
`W`/up arrow for throttle, release it to coast,
`S`/down arrow to brake, `A`/left arrow and `D`/right arrow to steer, and `R`
to reset. There is no implicit log playback: x/y/yaw come from the vehicle
model. A 64.6 m centreline derived from the real log now replaces the old fixed
x +/-6 m rectangle. The vehicle can move freely within a provisional +/-1 m
tube and +/-30-degree road-heading difference; reaching that support boundary
stops it and asks for reset. The centreline is a coverage reference, not auto
playback. Its +/-1 m width is accepted for this restricted first prototype,
not certified for closed-loop autonomous-driving evaluation. Use one
driving browser tab at a time because the service owns one shared vehicle.

For a larger 2880-pixel-wide six-camera image:

```bash
H3_WORLD_BROWSER_RENDER_SCALE=0.5 \
scripts/run_stage_h3_pandaset_040.sh world-browser
```

The launcher never kills an unknown old process. It reserves the port before
loading the checkpoint and fails fast if the port is occupied. Choose another
port when needed:

```bash
H3_WORLD_BROWSER_PORT=8781 scripts/run_stage_h3_pandaset_040.sh world-browser
```

Then open:

```text
http://100.116.66.57:8781
```

The earlier `logged-browser` remains available as a full-trajectory replay and
regression tool on port 8766; it is not genuine free driving.

Preferred when working through VS Code Remote SSH: forward the world browser
service through VS Code instead of visiting the Tailscale IP directly. Start
the service
from a VS Code terminal on `shidi`:

```bash
H3_WORLD_BROWSER_HOST=127.0.0.1 \
scripts/run_stage_h3_pandaset_040.sh world-browser
```

Then open VS Code's `Ports` panel, forward port `8767` if it was not detected
automatically, and open:

```text
http://localhost:8767
```

You can also run the command `Simple Browser: Show` inside VS Code and enter
that same URL. This keeps the connection inside VS Code's SSH/tunnel channel,
so the local proxy on port `25378` does not need to be disabled. If a local
proxy extension still captures `localhost`, add `localhost` and `127.0.0.1` to
its bypass list. If the VS Code webview does not focus keyboard input reliably,
click inside the page once or use the on-screen W/S/A/D buttons.

The world browser's six-camera mosaic preserves the selected Renderer output
resolution and fills the available browser width. Its on-screen shortcut pad
starts collapsed. Trial recording remains separate because the existing
recorder is intentionally specific to completed logged-path playback.

If your local browser traffic is routed through a proxy app on port `25378`,
keep the proxy on but add a direct/bypass rule for the Tailscale address. In
proxy tools such as Clash-like rule systems, the important direct rules are:

```text
IP-CIDR,100.64.0.0/10,DIRECT,no-resolve
IP-CIDR,100.116.66.57/32,DIRECT,no-resolve
```

If your proxy UI has a “bypass list”, “no proxy”, or “direct domains/IPs”
field, add:

```text
100.116.66.57
100.64.0.0/10
localhost
127.0.0.1
*.ts.net
```

Then test direct access with:

```text
http://100.116.66.57:8767/state.json
```

To run the earlier logged-path replay/regression instead:

```bash
scripts/run_stage_h3_pandaset_040.sh logged-browser
```

Then open `http://100.116.66.57:8766` in one browser tab only. The logged car
does not advance until the driver accelerates. `W`/up increases speed, releasing
it coasts, and `S`/down brakes to zero. `A`/left and `D`/right provide steering,
and `R` restarts the log. Auto-play can be started or paused from the page. The
browser defaults to the visible movement profile so counterfactual motion is
easy to see. Use
`H3_BROWSER_MOVEMENT_PROFILE=safe` when running the conservative acceptance
envelope. The default 0.25 browser render scale produces a
1440-pixel-wide six-camera view; it is twice the linear camera resolution of
the earlier 0.125 viewer that was judged too small. The browser also exposes
`/trial.json` and writes the same trial report to
`/home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial.json`
by default. After driving the segment, use the page's manual review panel to
save the road/lane/curb, steering-response, nearby-artifact, physical-latency,
and dynamic-traffic decision gates into the same JSON file.

Before a real logged-path operator run, the logged service can be rehearsed
from another terminal:

```bash
scripts/run_stage_h3_pandaset_040.sh trial-rehearsal
```

This scripted rehearsal exercises the live `/tick`, `/frame.jpg`,
`/trial-sample`, `/reset`, and `/trial-review` endpoints. It intentionally
saves all manual gates as `unsure`, so it is not human acceptance; it should
only pass when the machine plumbing is complete and `trial-check` fails solely
because human visual verdicts are still missing.

Then check whether that saved run is complete enough to count as acceptance
evidence:

```bash
scripts/run_stage_h3_pandaset_040.sh trial-check
```

The checker reads the browser trial JSON and writes
`/home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial_acceptance_check.json`
by default. It should fail until the operator has completed the segment,
recorded at least one reset and physical input latency sample, and saved all
manual drivability gates as `pass`.

## Current Next Step

The PandaSet and TbV static-8k checkpoints now remain fixed. The minimal
route-constrained TbV adapter and evidence outlet are implemented and have
passed a GPU/HTTP machine rehearsal. The next gate is the real operator run:
drive the common approach, select straight and right in separate reset runs,
capture physical keyboard-to-image timing through the browser evidence path,
and record whether road continuity, branch choice, traversal-profile switching,
and baked traffic remain decision-safe. Do not add more static TbV iterations
before this operator gate. The published MTGS checkpoint remains a
separate-environment fallback; PandaSet `003+057` remains the same-direction
parser/alignment control.

Do not join another scene to 040: the nearest available track is about 165.3 m
away. Do not claim intersection branching from the current archive: the scan
found same-direction repeats but no verified moving multi-direction pair.
The existing world browser, corridor sweep, logged-time tools, and eventual
operator trial remain valid regression evidence while this separate coverage
pilot is built.

The reconstruction-model decision is recorded in
`docs/drivable_reconstruction_model_strategy.md`: SplatAD remains the primary
interactive renderer; NeuRAD is a matched quality comparison; MTGS provides
the multi-traversal spatial-coverage direction; and UniSim is a closed-loop
architecture reference. The latter three roles do not imply completed
integrations.

The success criteria are deliberately separate from generic image metrics:

- `docs/drivability_acceptance_criteria.md` defines whether the scene can be
  driven;
- `docs/driver_attention_and_dynamic_traffic_requirements.md` records that the
  operator only drives and that dynamic correctness is deferred, not waived;
- `experiments/stage_h3_logged_renderer_mvp.md` records the Level-7 run and its
  acceptance boundary.
- `experiments/stage_h3_world_pose_probe.md` records the Level-8 world-pose run,
  visual findings, and remaining corridor boundary.
- `experiments/stage_h3_corridor_sweep.md` records the full-corridor
  keep-or-rebuild decision and its limits.
- `experiments/stage_h3_multi_trajectory_inventory.md` records the all-scene
  trajectory scan, selected 003+057 candidate, and the minimum pilot gates.
- `experiments/stage_h3_external_multi_trajectory_resources.md` records the
  external-resource comparison, complete TbV pose scan, verified Miami branch
  pair, and revised minimum pilot.
- `experiments/stage_h3_mtgs_published_block_probe.md` records the six released
  MTGS trajectories, selected Singapore gentle curve, official checkpoint
  evidence, environment conflict, and checkpoint-only gate.
- `experiments/stage_h3_tbv_splatad_pilot.md` records the bounded TbV download,
  multi-traversal parser, LiDAR alignment, and 100/2,000-step reload renders.
- `experiments/stage_h3_tbv_world_pose_corridor_probe.md` records the 2k/8k
  world-pose sweeps, exact-resume recovery, visual decision, and next gate.
- `experiments/stage_h3_tbv_driving_adapter.md` records the route adapter,
  evidence schema, GPU/HTTP rehearsal, display correction, and remaining human
  gate.
- `experiments/stage_h3_tbv_cockpit_presentation.md` records the calibrated
  front panorama, trajectory-support inset, diagnostic split, host latency,
  and remaining seam/human-driving gate.

See also `docs/stage_h3_stable_drivable_reconstruction_plan.md`.

Environment acceptance can be regenerated without PandaSet:

```bash
scripts/check_stage_h3_environment.sh
scripts/run_stage_h3_pandaset_040.sh data-gate
scripts/run_stage_h3_pandaset_040.sh static-8k
scripts/run_stage_h3_pandaset_040.sh paths
scripts/run_stage_h3_tbv_pilot.sh data-gate
scripts/run_stage_h3_tbv_pilot.sh pilot
scripts/run_stage_h3_tbv_pilot.sh render-pilot
scripts/run_stage_h3_tbv_pilot.sh world-pose-probe
scripts/run_stage_h3_tbv_pilot.sh static-8k
scripts/run_stage_h3_tbv_pilot.sh render-static-8k
scripts/run_stage_h3_tbv_pilot.sh world-pose-probe-8k
scripts/run_stage_h3_tbv_pilot.sh driving-adapter
scripts/run_stage_h3_tbv_pilot.sh paths
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
│   ├── drivable_reconstruction_model_strategy.md
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
