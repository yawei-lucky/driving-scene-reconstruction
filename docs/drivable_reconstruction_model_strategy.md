# Drivable Reconstruction Model Strategy

Date: 2026-07-22

Status: **accepted technical direction, not an implementation-completion
record**.

## Decision

The project will not replace the current reconstruction stack merely because
the earlier logged-time browser did not produce a genuinely free vehicle
trajectory. The accepted scene-040 static-8k SplatAD checkpoint remains the
fixed visual and geometry baseline while the new world-space browser is
evaluated inside its provisional coverage boundary.

The longer-term reconstruction target is:

```text
multi-traversal, multi-camera driving data
+ LiDAR and fused ego poses
→ metric static Gaussian background
+ object-centric dynamic Gaussian layers and trajectories
→ world-space vehicle pose
→ one rigid six-camera rig
→ real-time multi-view camera and LiDAR rendering
```

The named methods below have different roles. They are not four competing
implementations that should all be integrated at once.

| Role | Method | Current decision |
| --- | --- | --- |
| Primary interactive renderer | SplatAD | Retain as the main engineering base. Keep static-8k fixed until a measured probe or a driving blocker justifies model or data changes. |
| Quality-oriented comparison | NeuRAD | Use only as a controlled comparison on the same data and pose queries when SplatAD quality or extrapolation is in doubt. It is not the default interactive backend. |
| Spatial-coverage direction | MTGS-style multi-traversal reconstruction | Use the multi-traversal principle when a single pass cannot support the required driving envelope. This is a data-and-representation direction, not a claim that MTGS is already integrated. |
| Closed-loop architecture reference | UniSim | Borrow the separation of static world, controllable actors, novel ego trajectories, and sensor outputs. Any learned completion of unseen regions must be identified as generated rather than observed reconstruction. |

## Why The Current Checkpoint Is Retained

The accepted PandaSet scene-040 static-8k checkpoint has already established a
useful static foundation:

- 48 held-out views measured PSNR 26.6605, SSIM 0.8145, and LPIPS 0.2818;
- all 480 logged camera views rendered finite RGB;
- all 126 fixed nearby-pose queries rendered finite RGB;
- cuboid-excluded static LiDAR error measured 0.0614 m p50, 0.6265 m p90,
  and 1.4925 m p95;
- the connected Renderer measured about 13.4 warmed six-camera observations/s
  at 0.5 output scale.

These results show that the checkpoint is a credible static background and
renderer baseline. They do not prove a broad drivable region. In particular,
finite RGB only proves that rendering completed; it does not prove that road
geometry and appearance remain trustworthy at a large offset.

The actor-aware 8k candidates were rejected because they degraded moving-object
crops, geometry, or latency. Therefore the current accepted checkpoint is not
a reliable dynamic scene model. Exact local evidence is recorded in:

- `../experiments/stage_h3_static_8k_and_actor_ablations.md`;
- `../experiments/stage_h3_actor_alignment_and_timing.md`;
- `../experiments/stage_h3_logged_renderer_mvp.md`.

## The Three Separate Bottlenecks

### 1. Simulator integration

The current `LoggedEgoOffsetController` advances along the recorded trajectory
and treats human control as a bounded local offset. The Renderer selects a
logical log frame from `EgoState.time` and then transforms that frame's camera
rig. Consequently, steering can rotate the rendered view without owning the
vehicle's future world trajectory.

This is not evidence that the reconstruction checkpoint cannot render a true
turn. It means the following state concepts must first be separated:

```text
simulation_time
scene/log_time
world_ego_pose(x, y, yaw)
vehicle speed and control state
fixed camera-rig extrinsics
```

The vehicle model must own the absolute world pose. Source log time remains
useful for birth pose, coverage lookup, dynamic-scene time, and comparison, but
must not prescribe the driven path.

### 2. Reconstruction representation

The accepted checkpoint models the static world well enough to continue, but
its dynamic actor layers are not accepted. The final representation requires:

- a LiDAR-grounded, metric static background;
- independent vehicle and pedestrian representations with valid time ranges;
- calibrated actor trajectories and sensor timing;
- asynchronous camera, rolling-shutter, and per-point LiDAR time handling;
- camera and LiDAR rendering from the same requested world-space state;
- spatial partitioning or streaming for routes longer than one short scene.

SplatAD already addresses several of these requirements at the model-family
level, which is why it remains the primary renderer. The current static
checkpoint should not be confused with the complete capability of the model
family.

### 3. Spatial data coverage

Scene 040 is one approximately 7.9-second, 80-frame traversal. Six cameras
provide angular coverage around camera centres on that trajectory. They do not
provide camera centres in neighbouring lanes, unseen branches, or behind every
occluder.

This distinction is fundamental:

```text
surround-view angular coverage != two-dimensional drivable-area coverage
```

No reconstruction method can faithfully recover a never-observed building
face, road branch, or occluded surface. A learned model may synthesize a
plausible answer, but that region is no longer strictly reconstructed evidence.
High-freedom driving while preserving real-scene fidelity therefore requires
the data collection to cover the target driving envelope: multiple lanes,
multiple passes, and each required intersection branch.

## Method Roles And Boundaries

### SplatAD — primary renderer

SplatAD is designed for autonomous-driving camera and LiDAR rendering with 3D
Gaussian Splatting. Its model includes static and dynamic scene components and
accounts for driving-sensor effects such as rolling shutter. It matches the
project's existing pinned environment and real-time rendering requirement.

Decision:

- keep SplatAD as the primary implementation path;
- keep scene-040 static-8k as the regression checkpoint;
- repair or replace its dynamic-object path only after a driving-relevant
  blocker or a controlled actor experiment justifies the work;
- do not claim unrestricted ego extrapolation from the paper or the current
  checkpoint.

### NeuRAD — controlled quality comparison

NeuRAD jointly models driving-camera and LiDAR observations and supports
changes to ego and actor poses. It is useful for determining whether a failure
comes from the current SplatAD representation/rasterization or from the data
itself.

Decision:

- do not replace the interactive renderer pre-emptively;
- run NeuRAD only on the same scene, split, world-pose queries, output scale,
  and evaluation gates as SplatAD;
- promote it only if the controlled result materially improves the required
  driving envelope at an acceptable latency.

### MTGS — multi-traversal spatial-coverage direction

Multi-Traversal Gaussian Splatting builds a shared static representation from
repeated traversals while separating traversal-specific content. Its relevant
lesson for this project is that viewpoint coverage must be expanded in the
data and scene representation, rather than expecting longer optimization of a
single traversal to create missing views.

Decision:

- first measure the actual scene-040 driving envelope;
- if the required lane or branch lies outside that envelope, acquire or select
  multi-traversal data before another long model run;
- evaluate an MTGS-style shared static background together with the project's
  real-time sensor renderer instead of assuming one named model supplies the
  entire simulator.

### UniSim — closed-loop architecture reference

UniSim is conceptually close to the product goal: a recorded log becomes a
compositional static/dynamic scene, the ego path and actors can be changed, and
camera/LiDAR observations are rendered for closed-loop simulation. Its learned
priors and completion are useful references for visually repairing unseen
regions.

Decision:

- borrow its scene composition and closed-loop interface concepts;
- do not treat generated completion as measured ground truth;
- mark or reject low-confidence/generated regions during strict driving-agent
  evaluation;
- do not add a general world-model renderer to the current critical path.

## Execution Order

### Step 1 — world-coordinate free-driving probe

Keep static-8k fixed and change only simulator/renderer semantics:

1. separate simulation time from source-log time;
2. let the vehicle model own absolute `x`, `y`, `yaw`, and speed;
3. derive all six camera poses from one world-space ego pose and fixed rig
   extrinsics;
4. stop copying source-log motion into free-driving rolling-shutter metadata;
5. preserve reset and deterministic replay evidence.

This step addresses the current "view turns but the vehicle path does not"
failure without prejudging the reconstruction model.

### Step 2 — measure the supported driving envelope

Probe the current checkpoint with both isolated poses and continuous vehicle
paths:

- lateral offsets at 0, +/-1 m, +/-2 m, and +/-3 m;
- yaw changes at 0, +/-5 degrees, +/-10 degrees, and larger values only while
  the preceding ring remains valid;
- lane-change curves, braking to a fixed world position, and left/right turn
  trajectories;
- all six cameras from one rigid pose;
- road/lane/curb continuity, holes, floaters, geometry, camera consistency,
  reset repeatability, and control-to-image latency.

The result must distinguish:

- render completion;
- visually usable human-driving coverage;
- geometry-trustworthy closed-loop evaluation coverage.

### Step 3 — choose data expansion before model expansion

If the current checkpoint fails inside the required driving envelope, classify
the failure:

- if observed surfaces fail near known views, compare SplatAD with NeuRAD and
  inspect calibration, timing, capacity, and losses;
- if the requested view exposes surfaces never observed in scene 040, obtain
  multi-lane, multi-pass, or multi-branch data and use an MTGS-style shared
  scene representation;
- do not spend a 15k/30k training budget to solve missing spatial coverage.

### Step 4 — return dynamics when they affect driving

Dynamic-object reconstruction returns to the critical path when:

- a blurred or baked object hides the road or lane boundary;
- a residue looks like a false obstacle;
- responsive traffic is required; or
- an autonomous-driving agent begins closed-loop evaluation.

At that point, use an object-centric dynamic layer with correct validity time,
trajectory, timing, and geometry gates. Do not ask the human driver to inspect
or compensate for reconstruction defects while driving.

## Promotion Gates

A model or data change is promoted only when it improves a fixed,
driving-relevant test under matched conditions. At minimum record:

- dataset, scene, traversal set, sensor set, split, and preprocessing version;
- code revision, model configuration, checkpoint, and training budget;
- tested world-pose envelope and continuous paths;
- image, geometry, temporal, multi-camera, and latency results;
- whether each evaluated region was observed, interpolated, extrapolated, or
  generated;
- the human-driving decision gate affected by the change.

Generic PSNR, SSIM, or LPIPS improvement alone does not justify changing the
interactive renderer.

## Implementation Status Boundary

As of 2026-07-22:

- SplatAD static-8k and logged-time bounded-offset rendering are implemented;
- the separate world-coordinate Renderer, symmetric vehicle-path probe, and
  restricted browser are implemented and GPU-tested;
- the broad scene-040 driving envelope is not certified;
- the accepted dynamic actor model does not exist;
- NeuRAD comparison, MTGS-style multi-traversal reconstruction, and UniSim-like
  completion are research directions, not completed integrations.

This boundary must remain explicit in `PROJECT_STATE.md` and experiment
reports.

The first implementation evidence is recorded in
`../experiments/stage_h3_world_pose_probe.md`.

## Primary References

Accessed 2026-07-22:

- SplatAD, CVPR 2025:
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.pdf>
- SplatAD and NeuRAD official implementation:
  <https://github.com/georghess/neurad-studio>
- NeuRAD, CVPR 2024:
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf>
- Multi-Traversal Gaussian Splatting:
  <https://arxiv.org/abs/2503.12552>
- MTGS official implementation:
  <https://github.com/OpenDriveLab/MTGS>
- UniSim, CVPR 2023:
  <https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf>
