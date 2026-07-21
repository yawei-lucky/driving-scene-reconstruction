# Project State — Driving Scene Reconstruction

Last updated: 2026-07-21

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

### Stage H3 Level 7 — Logged-time drivable Renderer MVP

Completed and independently reviewed on 2026-07-21 without retraining:

- connected the accepted scene-040 static-8k step-7,999 checkpoint to the
  repository `Renderer` protocol;
- merged all 80 logical frames and all six PandaSet cameras while retaining
  calibrated poses, rolling shutter, original crops, and native timestamps;
- changed `EgoState.time` from one fixed reference pose to the real
  7.899239-second logged ego trajectory;
- added a bounded human-offset controller over that trajectory with a
  conservative +/−0.5 m forward, +/−0.25 m left, and +/−2 degree envelope;
- rendered the complete logical sequence 0-79 with six finite RGB views per
  observation and exact six-camera pixel repeatability after reset;
- measured warmed six-camera Renderer latency at 0.5 scale: 69.15 ms p50,
  74.37 ms p95, and 75.71 ms maximum, or about 13.4 observations/s;
- added a 10 Hz browser loop that accepts only W/S/A/D/R driving actions;
  local HTTP checks passed for the page, frame, throttle, and reset, and ten
  warmed 0.25-scale throttle requests measured 75.89 ms p50, 78.19 ms p95,
  and 78.76 ms maximum from server receipt to six-camera JPEG readiness;
- independently found and fixed controller normalization and brake-from-rest
  errors before the full rerun;
- preserved an 80-frame video, nearby-pose probes, and machine-readable report
  outside Git.

This passes the logged-time/offset/Renderer integration smoke. It does not yet
pass a complete human driving trial or physical-input-to-display latency, and
does not accept static/baked traffic as correct. Exact evidence is in
`experiments/stage_h3_logged_renderer_mvp.md`.

### Stage H3 Level 7A — Visible counterfactual movement profile

Completed on 2026-07-21 without retraining:

- kept the accepted static-8k step-7,999 checkpoint fixed;
- added two explicit logged-time movement profiles:
  - `safe`: the previous conservative envelope of +/−0.5 m forward,
    +/−0.25 m left, and +/−2 degrees yaw;
  - `visible`: +/−2.0 m forward, +/−0.75 m left, and +/−8 degrees yaw, with
    faster relative acceleration and yaw response for demonstrations;
- confirmed the earlier small counterfactual probe did change the trained
  reconstruction, but was visually subtle: +0.10 m left changed the front view
  by 9.069/255 mean absolute pixels and +1 degree yaw by 14.486/255;
- confirmed the visible profile has much stronger same-time counterfactual
  response: +1.25 m forward changed the front view by 13.530/255, +0.60 m left
  by 16.300/255, and +7 degrees yaw by 33.934/255;
- rendered the complete 80-frame, six-camera visible-motion sequence with all
  automated smoke gates passing and p95 Renderer latency of 73.62 ms at 0.5
  output scale;
- validated the browser default is now `visible`: three held `W+A` ticks reached
  x=0.239 m and yaw=7.2 degrees, with server control-to-JPEG times of roughly
  78-87 ms.

This is a demonstrability improvement, not a new certified safe pose envelope.
Use `H3_BROWSER_MOVEMENT_PROFILE=safe` for the conservative acceptance run.

### Stage H3 Level 7B — Drivability preflight and browser trial recording

Completed on 2026-07-21 without retraining:

- added `drivability-preflight`, a GPU preflight that uses the accepted
  static-8k step-7,999 checkpoint and emits a machine-readable report plus
  human-review images;
- verified the visible profile over all 80 logical frames with 17 automated
  backend gates passing, including accepted checkpoint, six finite camera
  outputs, same-time counterfactual pixel changes, same logical frame for
  counterfactual probes, monotonic logical frames, camera time/source metadata,
  reset repeatability, scripted-state repeatability, final-state pixel
  repeatability, inside/outside profile limit behavior, and p95 latency;
- measured visible-profile preflight latency at 0.5 output scale: 69.07 ms p50,
  73.36 ms p95, and 76.08 ms max for the six-camera Renderer observation;
- preserved review artifacts outside Git under
  `/home/yawei/stage3_external/artifacts/scene_040_drivability_preflight/`,
  including `counterfactual_front_preflight.jpg`,
  `sequence_contact_sheet.jpg`, and
  `stage_h3_drivability_preflight.json`;
- added browser trial recording so the real operator run exposes `/trial.json`
  and, by default, writes
  `/home/yawei/stage3_external/artifacts/scene_040_browser_trial/browser_trial.json`;
- added an in-browser manual review panel that writes the operator's
  road/lane/curb, steering-response, nearby-artifact, physical-latency, and
  dynamic-traffic decision gates into the same trial JSON;
- validated the browser recording endpoint locally: the page trial report
  initialized with checkpoint step 7,999 and visible profile, a `W+A` tick
  returned logical frame 1 and server control-to-JPEG time of 89.88 ms, a
  trial sample was recorded, reset was recorded, and `/trial.json` returned
  `sample_count=1` and `reset_count=1`.
- validated the manual review endpoint locally: initial `/trial.json` marked
  all five gates missing, a blocking review preserved `unsure` and `fail`
  verdicts, and a later all-pass review returned `manual_review_all_passed`
  true with empty `manual_review_blocking_gates`.

This is still not a completed human driving acceptance run. The preflight
explicitly leaves road/lane continuity, steering direction by eye, nearby
artifact judgment, physical key-to-display latency, and dynamic-traffic
decision impact as manual review items. Those items are now recordable from
the browser, but they are not accepted until a real operator completes and
saves the review.

### Stage H3 Level 7C — Browser trial acceptance checker

Completed on 2026-07-21 without retraining:

- added a dependency-light trial acceptance evaluator for saved browser
  `trial.json` files;
- added `examples/stage_h3_trial_acceptance_check.py` and the
  `scripts/run_stage_h3_pandaset_040.sh trial-check` mode;
- default checks require scene 040, the accepted step-7,999 checkpoint, visible
  movement profile, at least 70 browser samples, completed logged segment,
  monotonic logical frames, at least one reset, at least one non-empty W/S/A/D
  input sample, browser request-to-image p95 at or below 100 ms, browser
  input-to-image p95 at or below 100 ms, server control-to-JPEG p95 at or
  below 100 ms, camera time-spread p95 at or below 100 ms, and all five manual
  drivability gates marked `pass`;
- the checker also rejects a manual all-pass review if the latest review was
  saved before enough samples or before the completed log time, preventing an
  early click from becoming acceptance evidence.
- validation passed the full dependency-light suite with 55 tests, and the
  checker correctly rejected the earlier manual-review endpoint artifact
  because it had no complete driving samples, reset, operator input sample, or
  browser latency distributions.

This checker does not make visual judgments itself. It turns the recorded
operator verdicts and browser timing into a reproducible pass/fail evidence
package.

### Stage H3 Level 7D — Scripted browser trial rehearsal

Completed on 2026-07-21 without retraining:

- added `examples/stage_h3_browser_trial_rehearsal.py`;
- added `scripts/run_stage_h3_pandaset_040.sh trial-rehearsal`;
- the rehearsal drives the live browser HTTP service through `/tick`,
  `/frame.jpg`, `/trial-sample`, `/reset`, and `/trial-review`;
- its deterministic W/S/A/D schedule exercises forward, left-yaw, right-yaw,
  and brake inputs over the logged segment;
- it submits a manual review with all five gates set to `unsure` and reviewer
  `scripted_rehearsal_not_human`, so the artifact cannot be mistaken for human
  acceptance;
- the rehearsal evaluates the resulting trial JSON and only passes when all
  non-visual machine gates pass while the expected manual-gate failures remain.
- the real rehearsal run on port 8781 passed with 79 samples, completed log,
  reset_count 1, key sets `a`, `aw`, `d`, `s`, and `w`, browser
  request-to-image p95 77.22 ms, browser input-to-image p95 82.61 ms over 10
  input-change samples, server control-to-JPEG p95 76.19 ms, and camera
  time-spread p95 81.37 ms; the only acceptance-check failures were the
  expected manual visual gates.

This is a pre-operator service rehearsal. It validates the HTTP and recording
plumbing before a human trial, but it still cannot judge road/lane quality.

## 3. What The System Can Do Now

```text
HumanControl
→ LoggedEgoOffsetController
→ PandaSet logged pose(time) + profiled EgoState offset
→ one rigid transform of the calibrated six-camera rig
→ SplatADLoggedRenderer using accepted static-8k
→ six RGB arrays at about 13.4 Renderer observations/s
→ automated drivability preflight
→ manual-default browser speed/steering loop with arrows or W/S/A/D,
  an auto-play toggle, timing, and trial JSON
→ browser trial acceptance checker
→ scripted browser trial rehearsal
```

This is the first repository state where simulated ego motion changes pixels
produced by the trained reconstruction checkpoint. The default browser loop now
uses the `visible` profile so those counterfactual changes are easier for a
human to see, and it now opens stopped: logged time advances only while W/S/A/D
or the matching arrow key is used to begin driving. Releasing the accelerator
now coasts at the current relative speed, while braking reduces that speed to
zero before the render loop stops. The page can start/pause auto-play without
restarting the server, while `H3_BROWSER_TIME_MODE=auto` still starts in
auto-play mode. The browser now gives the six-camera mosaic the full available
width and keeps the shortcut pad and manual acceptance form collapsed until
requested. The `safe` profile preserves the previous conservative bounds.

The earlier H2 fixed-pose Wayve renderer remains available. The H3 path now
also loads real PandaSet scene 040, uses six cameras, Pandar64 geometry and
cuboid actor tracks, and follows every logged rig pose with bounded human
offsets. Static structure remains coherent over the complete short sequence
at the tested scale. The backend preflight and browser/server path now work;
the next unresolved integration gate is a real operator trial including
physical key-to-display timing and visual review, followed by the saved
`trial-check` result. The scripted rehearsal can catch service plumbing
failures before that human run. Dynamic traffic remains a later mandatory gate.

## 4. Important Limitations

- The nearby-pose limits are conservative engineering bounds, not empirically
  certified safe regions.
- The `visible` browser movement profile deliberately exceeds the earlier
  conservative envelope to make counterfactual motion obvious; it is not a
  certified driving-safe region.
- Static Splatfacto blurs moving vehicles and pedestrians.
- There is no collision, road-boundary, traffic-agent, or map constraint.
- H3 uses the logged rig-center trajectory tangent as its offset basis; the
  complete envelope still needs systematic visual certification.
- H3 Renderer latency passes 100 ms at 0.5 scale, but this excludes physical
  input, mosaic/JPEG encoding, browser transport, and display refresh.
- Browser trial recording measures browser request-to-image and input-to-image
  load events, but still does not include monitor scan-out or a calibrated
  external latency sensor.
- The current examples depend on the machine-specific H1 checkpoint and
  Nerfstudio environment.
- The new SplatAD backend is a separate Renderer; direct checkpoint
  compatibility with the H2 Nerfstudio 1.1.5 backend is not assumed.
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

Stage H3 now prioritizes completing one stable human-driving loop around the
accepted static-8k backend. Environment, acquisition, calibration, the static
baseline, rejected actor ablations, logged-time Renderer, automated preflight,
browser-side trial recording, browser-trial acceptance checking, and scripted
browser-service rehearsal are complete. Static 8k remains the accepted
checkpoint; no further dynamic training starts until driving evidence makes it
necessary.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. keep static 8k as the fixed visual and geometry checkpoint;
2. run `drivability-preflight` after renderer/control changes and preserve its
   JSON plus review images;
3. run `trial-rehearsal` against the live browser service to catch plumbing
   problems before asking for a human trial;
4. run the Level-7 10 Hz browser loop through the full segment with a human;
5. start with the visible movement profile when inspecting whether motion is
   perceptible, then repeat the acceptance run with `safe` if strict envelope
   evidence is needed;
6. accept only steering, throttle, brake, and reset during driving—never ask
   the operator to inspect or compensate for reconstruction defects;
7. preserve `/trial.json`, including browser-reported control-event-to-screen
   p95 latency, reset events, and the five manual drivability gate verdicts;
8. run `trial-check` and preserve the resulting acceptance-check JSON;
9. execute the six separate gates in
   `docs/drivability_acceptance_criteria.md` on this low-interference segment;
10. return dynamic traffic to the main line immediately if it obscures the
   road, creates a false obstacle, or closed-loop autonomous-driving testing
   begins;
11. then fix the offending dynamic object/window with the existing actor bounds
   and timing evidence rather than restarting broad, ungated training.

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
