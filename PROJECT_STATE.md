# Project State — Driving Scene Reconstruction

Last updated: 2026-07-23

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

### Stage H3 Level 8 — World-pose free-driving probe

Completed on 2026-07-22 without retraining:

- added `SplatADWorldRenderer` alongside the unchanged logged renderer;
- separated simulation time from a frozen source-scene anchor at logical frame
  40 / 3.999463 seconds;
- made `SimpleVehicleModel` own future metric x/y/yaw/speed for the probe;
- interpolated all six source camera trajectories to the same anchor time,
  then moved the resulting fixed rig with one rigid transform;
- zeroed source-log linear/angular motion and rolling-shutter metadata instead
  of reusing it for the simulated vehicle;
- rendered lateral offsets -3/-2/-1/0/+1/+2/+3 m and yaw -10/-5/0/+5/+10
  degrees;
- rendered a 3.0-second continuous left turn ending at x=5.654 m, y=1.098 m,
  and yaw=21.978 degrees;
- passed all world-pose plumbing gates over 45 six-camera observations with
  exact reset and 68.46 ms p95 Renderer latency at 0.5 output scale.

Visual review found the main road remains recognizable over +/-3 m with no
large black hole, but close poles, tree edges, curbs, and sidewalks deform as
the pose moves away from the source path. The backend milestone passes; the
full corridor does not. +/-1 m is the current first candidate for a restricted
interactive prototype. Exact evidence is in
`experiments/stage_h3_world_pose_probe.md`.

The initial claim of zero effective camera-time spread in this Level-8 run was
superseded by the independent review and corrected Level-8A run below.

### Stage H3 Level 8A — Reviewed motion paths and restricted browser

Completed on 2026-07-22 without retraining:

- independent review found that SplatAD still added learned per-camera
  time-to-centre adjustments after input rolling-shutter metadata was zeroed;
- disabled that model path, removing the measured 7.036 ms effective scene-time
  spread, and added a same-pose/different-simulation-time pixel-exact gate;
- made anchor initialization transactional and restricted anchors to the real
  six-camera interpolation interval of 0.031291-7.849195 seconds;
- expanded the probe to 206 complete six-camera observations over straight,
  symmetric left/right turns, symmetric left/right lane changes, and full
  braking from 3 m/s;
- passed all plumbing and motion gates: mirrored paths, no straight drift,
  lane-change heading recovery, exact brake-rest pose/pixels, and non-frozen
  moving imagery;
- measured 0.5-scale Renderer latency of 68.14 ms p50, 70.99 ms p95, and
  72.19 ms maximum;
- added a separate `world-browser` on port 8767 with W/S/A/D/R, 2 m/s speed
  cap, 15-degree maximum wheel angle, transactional render-before-state commit,
  and explicit stop-on-boundary behavior;
- GPU/HTTP exercised left acceleration, right recovery, coasting, braking,
  reset, and a deliberate yaw-boundary hit. Warm sampled requests were about
  68-74 ms; physical browser input-to-display remains an operator measurement.

The provisional browser boundary is x +/-6 m, y +/-1 m, and yaw +/-15 degrees.
It is containment for an experiment, not a certified drivable corridor.

This fixed rectangle was superseded by Level 8B below.

### Stage H3 Level 8B — Logged-centreline free-driving tube

Completed on 2026-07-22 without retraining:

- transformed the synchronized recorded rig path into the same anchor-local
  world coordinates used by free driving, producing a 64.595 m centreline;
- replaced the arbitrary 6 m browser rectangle with a provisional tube around
  that centreline. The centreline constrains reconstruction support only; the
  vehicle pose remains entirely controlled by `SimpleVehicleModel` and human
  steering/throttle/brake input;
- exposed road progress and distance from the recorded centreline in the
  browser status;
- ran a 10.0 s GPU/HTTP control sequence through 100 six-camera observations.
  It reached x=18.606 m at 2 m/s without a boundary hit, with maximum distance
  from the centreline of 0.0134 m and a final server control-to-JPEG time of
  68.07 ms at 0.25 scale;
- visually inspected start, 5 s, and 10 s mosaics. The road and intersection
  advance coherently well beyond the former 6 m limit.

The logged centreline is genuine source-data support, but its +/-1 m tube is
still provisional. The next main-line experiment is multi-station visual and
LiDAR road-support checking along this tube, followed by an operator-driven
run. Multi-anchor stitching is only needed for segments that fail those checks.

### Stage H3 Level 8C — Full-corridor keep-or-rebuild decision

Completed on 2026-07-22 without retraining:

- added a reproducible five-station x three-lateral-offset sweep over the full
  64.595 m logged centreline;
- rendered 15 complete six-camera observations at -1/0/+1 m. All frames were
  valid; 0.5-scale Renderer latency was 67.41 ms p50, 70.78 ms p95, and
  73.22 ms maximum;
- visual inspection found a continuous, readable front road at every sampled
  station. Close parked cars, foliage, and some side views still stretch, so
  this is not final 360-degree or closed-loop geometry acceptance;
- retained static-8k as the background for the first restricted human-driving
  prototype instead of starting another training run;
- moved the world-browser spawn to the beginning of the recorded corridor;
- ran a 30 s GPU/HTTP drive through 300 observations, reaching 58.607 m of
  64.595 m with no boundary hit and 0.0182 m maximum centreline distance.

This cheap test answers the immediate keep-or-rebuild question. The next gate
is an operator drive. LiDAR checking, new multi-traversal data, and MTGS-style
reconstruction become the next action only for a concrete segment that harms
driving or when geometry-trustworthy closed-loop evaluation begins. Evidence
is in `experiments/stage_h3_corridor_sweep.md`.

### Stage H3 Level 9A — PandaSet multi-trajectory inventory

Completed on 2026-07-22 without extracting another full scene or training:

- scanned GPS, timestamps, front-camera poses, and semantic availability for
  all 103 scenes directly from the verified PandaSet ZIP;
- found no second traversal near scene 040: scene 072 is nearest at 165.337 m;
- flagged 054+078 for direction-change review, then identified it from its
  trajectory shape and front-camera samples as a partial repeat of the shared
  approach that exposes no second outbound branch; no verified common-intersection
  straight/left/right pilot remains;
- found ten same-direction repeat pairs, generally separated by about 2.9
  hours and visibly changing from daylight to night;
- selected scenes 003+057 as the first coverage pilot: 38.0-38.8 m overlap,
  2.991 m nearest-distance p50, 1.509-degree heading-difference p50, semantics
  on both scenes, and 127.87 m estimated union route length;
- measured front-pose-to-GPS rigid-fit p95 residuals of 0.112 m and 0.118 m,
  sufficient for candidate initialization but not a substitute for static
  LiDAR registration;
- added a dependency-free reproducible inventory script, optional H3/Pillow
  contact sheet, machine-readable report, and six lightweight tests.

This is a metadata and visual candidate-selection result. No multi-sequence
dataparser, shared checkpoint, or MTGS implementation exists yet. Exact evidence
and the minimum pilot gates are in
`experiments/stage_h3_multi_trajectory_inventory.md`.

### Stage H3 Level 9B — External multi-trajectory resource probe

Completed on 2026-07-22 without downloading a full sensor log or training:

- enumerated all 1,043 public Argoverse TbV logs and downloaded only their
  city-frame pose files, totalling 531,619,590 bytes;
- retained 1,039 trajectories longer than 5 m and scanned pairs only within
  the same city;
- generated non-exclusive review queues with 990 same-direction, 301 branch,
  and 168 opposite-direction candidates;
- visually verified the Miami TbV logs `OCa... + QMn...` as a common
  residential approach followed by a right turn versus straight travel;
- measured 116.83/115.05 m shared-path coverage, 0.319 m nearest-distance p50,
  and 0.398/54.939-degree heading-difference p50/p95;
- found no third TbV trajectory within 10 m of that exact junction, so this is
  a two-branch pilot rather than a straight/left/right trio;
- confirmed that the public S3 layout permits per-file timestamp-window
  download and selected two ten-second windows centred on the reviewed branch
  for a small static SplatAD smoke;
- added a reproducible inventory/downloader, trajectory plot, machine-readable
  report, six reviewed source frames outside Git, and four lightweight tests.

This promotes TbV above PandaSet `003+057` for spatial coverage. It does not
claim cross-traversal LiDAR registration, a working TbV parser, or a trained
multi-log checkpoint. Exact evidence and the external-resource comparison are
in `experiments/stage_h3_external_multi_trajectory_resources.md`.

### Stage H3 Level 9C — MTGS published-block trajectory probe

Completed on 2026-07-22 without downloading a complete road block or model:

- audited registered metadata for all six released MTGS road blocks;
- confirmed every block has eight surround cameras and 3-6 traversals covering
  approximately 57-105 m per traversal;
- selected the smallest 3.98 GB Singapore block
  `365000_144000_365100_144080` as a normal-road candidate;
- measured three same-direction 84-87 m routes with gentle 11-16 degree bends;
- confirmed that official training traversals 4 and 5 are 5.17 m apart at
  nearest-distance p50, while evaluation traversal 3 lies between them at
  2.36/2.87 m p50;
- found 270 accepted ego frames, 2,160 valid camera images, and prepared LiDAR,
  masks, boxes, and instance data;
- indexed the release tar and identified a selectively downloadable
  773,241,943-byte checkpoint for this block;
- recorded the official unseen-traversal-3 metrics as PSNR 22.658, SSIM 0.673,
  and LPIPS 0.259, without treating them as host or driving measurements;
- confirmed that official MTGS dependencies conflict with the accepted H3
  environment and that its guide calls for at least 40 GB training VRAM.

This removes intersection branching as a first-pilot requirement and promotes
the released MTGS block to the cheapest next checkpoint-load gate. It does not
claim that the checkpoint fits the host's 24 GB GPU, renders a usable driving
corridor, or replaces SplatAD. Exact evidence and stop conditions are in
`experiments/stage_h3_mtgs_published_block_probe.md`.

### Stage H3 Level 9D — TbV multi-traversal SplatAD pilot

Completed on 2026-07-22 in the accepted H3 environment:

- downloaded only 1,612 public S3 objects and 277,728,040 bytes for the two
  selected ten-second windows: 1,400 images, 200 LiDAR sweeps, calibration,
  maps, and full pose files;
- added a minimal multi-traversal parser that keeps the two logs in their
  shared Miami city frame, maps both to a local 0-10 second interval, returns
  no actors, and namespaces all 14 camera plus two LiDAR sensor IDs;
- loaded every TbV LiDAR Feather as one aggregate ego-frame sweep, avoiding
  the invalid AV2 upper/lower split and missing-point path;
- passed the 0.9-train data gate with 1,260 images, 180 sweeps, 18,133,002
  finite points, and no velocity calculation across logs;
- passed the shared-route registration gate with symmetric static-dominant
  nearest-neighbour residuals of 0.1094 m p50, 0.2408 m p90, and 0.3346 m p95;
- completed a 100-step save/reload smoke and rendered all 140 held-out camera
  views at 25.08 views/s with no non-finite saved image;
- completed a 2,000-step, 0.5-scale, 750,000-seed pilot with 2,672,901 final
  Gaussians and an 887,205,110-byte checkpoint;
- reloaded that checkpoint and rendered all 140 held-out views at 24.35
  views/s, measuring PSNR 20.2621, SSIM 0.6753, and LPIPS 0.5193;
- visually recovered recognizable road, lane markings, buildings, trees, and
  parked vehicles on both routes, while retaining softness, floaters, and
  blurred/merged vehicle detail.

This displaces MTGS environment setup as the immediate next action. It proves
that the existing SplatAD stack can fit both observed windows, not that the
combined model supports counterfactual driving. Exact commands, artifacts,
failure evidence, and acceptance boundary are in
`experiments/stage_h3_tbv_splatad_pilot.md`.

### Stage H3 Level 9E — TbV world-pose corridor probe

Completed on 2026-07-22 in the accepted H3 environment:

- added an experiment-local renderer that keeps one joint checkpoint loaded,
  synchronizes the seven-camera rigs, preserves traversal-specific sensor and
  appearance IDs, freezes scene time, and accepts metre-scale branch-local
  poses;
- selected a shared anchor with 0.8388 m cross-route distance, 13.173 degrees
  heading difference, 42 matches, and 31.629 m shared-route span;
- rendered the same 36 observations and 252 camera views at 2k and 8k over the
  common -20/-10/-2 m entrance, straight +5/+20/+40 m, right +5/+15/+30 m,
  and -1/0/+1 m offsets;
- found that 2k passed spatial/plumbing gates but failed the visual driving
  gate because road floaters, granular side/rear views, and vehicle ghosts
  remained;
- exact-resumed model, optimizer, scheduler, and global step from 1,999 to
  7,999, reaching the 5,000,000-Gaussian cap and a 1,650,493,750-byte
  checkpoint; RNG/dataloader state is not present in the checkpoint and is not
  claimed bitwise exact;
- recovered from one execution-channel SIGTERM by discarding its unsaved
  iterations and using a verified 1,000-step checkpoint chain from the
  original step 1,999; this was not a CUDA or out-of-memory failure;
- rendered the same 140 held-out views at 22.69 views/s and measured PSNR
  23.2130, SSIM 0.7734, and LPIPS 0.3805, improving by +2.9509, +0.0980, and
  -0.1389 respectively over 2k;
- passed all 8k automated gates with 252/252 finite camera renders and measured
  58.48/67.70 ms p50/p95 per seven-camera observation at 0.5 output scale;
- visually retained the shared entrance, both branches, and all three lateral
  offsets while substantially removing the 2k road floaters.

The 8k result is accepted as a restricted front-corridor candidate for a
continuous human-driving trial. It is not certified 360-degree or autonomous
driving: close vegetation and parked vehicles still deform, vehicles remain
baked into static geometry, no lateral ground truth exists, and no continuous
keyboard-to-display run has been completed. Do not add more static iterations
before that trial. Exact evidence is in
`experiments/stage_h3_tbv_world_pose_corridor_probe.md`.

### Stage H3 Level 9F — TbV route adapter and evidence outlet

Completed on 2026-07-22 without retraining:

- added a dependency-light shared-approach/branch adapter around the existing
  vehicle model and logged-centreline support boundary;
- spawns at TbV common progress -20 m, stops within 0.5 m of the shared anchor,
  and requires an explicit straight or right selection before continuing;
- evaluates both branch candidates at the anchor while retaining their
  traversal-specific renderer profiles and route-support margins;
- rejects a candidate beyond the +/-1 m tube without snapping the vehicle pose,
  then rejects all further control until reset;
- added a browser that preserves heterogeneous TbV camera aspect ratios,
  prioritizes the three forward cameras, and exposes `/state.json`,
  `/frame.jpg`, and `/evidence.json`;
- added `route_driving_evidence.v0`, recording world state, control keys,
  route/branch phase, support margins, renderer profile, seven-camera finite
  status, frame SHA-256, latency, reset/branch events, and optional browser
  input-to-image timing;
- ran the actual H3 environment on the RTX 4090 D with checkpoint step 7,999:
  241 sequential samples, two resets, both branch selections, all camera frames
  finite, zero committed support violations, and one intentional rejected
  1.013 m boundary candidate;
- measured the preserved 0.5-scale machine run at 57.61/59.77 ms p50/p95 for
  seven-camera rendering and 82.55/85.01 ms p50/p95 for server control through
  normalized-mosaic JPEG, with 99.46 ms maximum;
- visually found and fixed an initial mosaic-only crop that showed only the top
  of the portrait front-centre frame; preserved normalized mosaics now retain
  the complete forward road view.

This proves route/control/render/evidence plumbing, not human drivability. The
scripted HTTP run has zero browser timing coverage and is not a physical
keyboard-to-display test. Artifacts remain outside Git under
`/home/yawei/stage3_external/artifacts/tbv_branch_pair_driving_adapter/`.
Exact evidence is in `experiments/stage_h3_tbv_driving_adapter.md`.

### Stage H3 Level 9G — TbV cockpit presentation

Completed on 2026-07-23 without retraining:

- replaced the default seven-camera driving mosaic with a calibrated
  cylindrical projection of `ring_front_left`, `ring_front_center`, and
  `ring_front_right`;
- retained the cameras' trained intrinsics and rig rotations, requested a 150
  degree horizontal field of view, and feathered only their calibrated overlap;
- measured 99.9816% projection coverage for the right-turn traversal profile
  and 99.9802% for the straight profile;
- added an ego-up trajectory-support inset showing the logged route tube,
  straight/right paths, current pose, lateral offset, and remaining +/-1 m
  margin; it is explicitly not presented as overhead RGB or certified free
  space;
- moved the aspect-ratio-preserving seven-camera mosaic to `/diagnostic`, where
  it remains available for reconstruction coverage, deformation, and ghost
  inspection without occupying the driving view;
- raised the configurable route-adapter speed cap from 2.0 to 4.0 m/s;
- ran 25 consecutive host control samples to 4.0 m/s with seven finite camera
  frames, zero support violations, and a 0.9296 m minimum remaining corridor
  margin;
- measured renderer latency at 57.35/60.33 ms p50/p95 and complete server
  control-to-cockpit-JPEG latency at 97.40/100.34 ms p50/p95;
- directly inspected the common-approach, straight-profile, and diagnostic
  outputs. Road topology and the route inset remained legible, but camera
  overlap seams, sensor-profile colour changes, reconstructed blur, and baked
  vehicles remain visible.

This completes the requested presentation split and a machine smoke, not the
physical human trial. Exact evidence and artifact paths are in
`experiments/stage_h3_tbv_cockpit_presentation.md`.

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

The separate Level-8 backend path now supports:

```text
HumanControl
→ SimpleVehicleModel owns simulated x/y/yaw/speed/time
→ SplatADWorldRenderer at one frozen static scene time
→ time-synchronized fixed six-camera rig
→ six RGB views from the simulated future world pose
→ six symmetric/straight/braking trajectory artifacts plus JSON evidence
→ a separate manual-default browser bounded by the logged-centreline tube
```

This path proves that steering changes the future simulated trajectory and is
now wired into a restricted browser. The sampled +/-1 m logged-centreline tube
is accepted for a first human-driving prototype, but it is not certified for
closed-loop autonomous-driving evaluation; the full +/-3 m probe remains a
coverage diagnostic.

The Level-9G TbV path now supports:

```text
Branch-local world pose plus traversal route role
→ experiment-local TbVWorldRenderer at one frozen static scene time
→ one synchronized seven-camera rig with traversal-specific appearance IDs
→ common entrance, straight route, or right-turn route at +/-1 m
→ SimpleVehicleModel continuous control from common progress -20 m
→ an explicit anchor stop followed by straight or right branch selection
→ seven finite RGB arrays plus a calibrated three-front-camera cylindrical view
→ an ego-up logged-trajectory support inset and separate seven-camera diagnostic
→ fail-closed route support and per-control machine-readable JSON evidence
```

This is connected to a dedicated manual browser but not the common `Renderer`
protocol. A GPU/HTTP machine rehearsal has exercised both branch transitions
and the route boundary. A separate 4.0 m/s cockpit smoke passed at approximately
10 Hz server-side. A real operator keyboard-to-display trial remains required
before any human-drivability claim.

This is the first repository state where simulated ego motion changes pixels
produced by the trained reconstruction checkpoint. The logged browser loop now
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
- The world-pose probe freezes one static scene time and disables source
  rolling-shutter motion. Simulated-vehicle rolling shutter is not implemented.
- The world-pose backend renders +/-3 m and a 21.98-degree turn. The sampled
  +/-1 m corridor is sufficient for a restricted human trial, not a certified
  visual/geometry corridor for autonomous-driving evaluation.
- The world browser uses distance to the complete logged centreline and starts
  at its recorded beginning. It stops at the provisional tube boundary and
  requires reset; no reverse gear or collision model exists.
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
- No released PandaSet trajectory is close enough to join directly to scene
  040; the nearest observed track is about 165.3 m away.
- The current archive scan found same-direction repeat traversals but no
  verified same-intersection straight/left/right or opposite-direction set.
- The selected 003+057 pair changes from daylight to night. A shared model must
  namespace appearance by traversal, and its apparent 3 m GPS offset still
  needs static-LiDAR registration before it counts as added road coverage.
- TbV supplies no object annotations. Its selected branch pair has different
  parked/moving vehicles; the static pilot softens and merges some vehicle
  detail and cannot stand in for a dynamic-traffic solution.
- The selected TbV location has straight and right-turn traversals but no
  observed left-turn traversal. It expands the driving topology without yet
  providing a three-way action set.
- A route branch is not required for the first expanded driving pilot. Longer
  straight or gently curving observed roads remain valid targets.
- The TbV 8,000-step checkpoint has passed a sparse seven-camera world-pose
  probe and a machine-driven browser rehearsal over both branches and +/-1 m,
  but not a real operator trial, physical input/display latency, wider
  departure, or lateral ground-truth test. The evidence outlet deliberately
  reports zero browser timing coverage for the machine rehearsal.
- TbV 8k held-out quality still differs by traversal (PSNR 24.32 versus 22.11).
  Close foliage and parked vehicles deform, and no-annotation vehicle ghosts
  remain a rejection condition if they block the lane or create a false
  obstacle during the continuous trial.
- The selected MTGS block is trajectory- and checkpoint-qualified but has not
  been visually reviewed or loaded on this host. Official training guidance
  asks for at least 40 GB VRAM, versus the available 24 GB.
- MTGS requires a separate environment because its Nerfstudio, gsplat, NumPy,
  and tyro versions conflict with the accepted `h3_splatad` environment.
- The selected MTGS block contains substantial annotated traffic. Its dynamic
  reconstruction may be an advantage over static TbV, but false obstacles or
  actor ghosts remain a driving rejection condition.
- Free AV2/TbV and MTGS data use is non-commercial under CC BY-NC-SA 4.0; this
  does not establish commercial data rights for a later product.

## 5. Current Next Action — Stage H3

Stage H3 now prioritizes the real operator trial for the completed TbV Miami
`OCa... + QMn...` route adapter. The accepted scene-040 static-8k checkpoint
and its world-coordinate browser remain fixed regression evidence. The
released MTGS Singapore checkpoint remains an isolated fallback, while
PandaSet `003+057` remains a same-direction parser/alignment control.

Both PandaSet scene-040 static-8k and the new TbV static-8k candidate remain
fixed; they have different data and acceptance boundaries. The agreed
technical direction is recorded in
`docs/drivable_reconstruction_model_strategy.md`: retain SplatAD as the primary
interactive renderer, use NeuRAD only for a matched quality comparison, use
MTGS-style multi-traversal reconstruction when spatial coverage is the
limitation, and borrow UniSim's compositional closed-loop concepts without
treating generated completion as observed ground truth. NeuRAD, MTGS, and
UniSim are not currently integrated.

The adapter and cockpit presentation are complete. The next action is
deliberately narrow: use the cylindrical front cockpit to run straight and
right as separate human reset trials, capture browser request-to-image and
physical input-to-image timing, inspect the transition when the renderer
changes traversal profile, and reject any segment where panorama seams, baked
vehicles, permanent geometry, or temporal artifacts alter the driving
decision. Keep `/diagnostic` open only for post-drive inspection. Do not train
beyond 8k before this gate.

See `docs/stage_h3_stable_drivable_reconstruction_plan.md` for the detailed
plan. The short version is:

1. keep static 8k fixed and retain the completed world-pose probe as a
   regression gate;
2. retain the completed symmetric path and brake-to-rest suite as a regression
   gate;
3. retain the completed five-station y=-1/0/+1 m visual sweep as the cheap
   keep-or-rebuild gate;
4. retain the completed 103-scene trajectory inventory and its negative
   scene-040/multi-direction findings;
5. retain PandaSet 003+057 only as a low-risk same-direction parser/alignment
   control;
6. retain the completed TbV download, registration, 8k checkpoint, held-out
   render, and 36-pose sweep as the multi-traversal regression gate;
7. retain the completed route-constrained TbV browser/evidence adapter and run
   a continuous +/-1 m straight/right-turn human trial before more training;
8. keep the released MTGS checkpoint as a separate-environment fallback if the
   TbV continuous trial exposes a model/data limitation rather than plumbing;
9. keep the implemented provisional scene-040 world browser and operator trial
   as regression/acceptance work rather than coupling them to this new scene;
10. return dynamic actors to the main line when they obscure the road, create a
    false obstacle, or responsive traffic/closed-loop agent evaluation begins;
11. never ask the operator to inspect or compensate for reconstruction defects
    while driving.

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
