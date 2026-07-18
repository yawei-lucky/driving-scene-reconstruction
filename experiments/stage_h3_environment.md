# Stage H3 Environment Preparation

Date: 2026-07-18

## Outcome

H3-0A passed on the project host. A separate SplatAD/NeuRAD environment now
exists outside Git, its important source revisions and compatibility versions
are pinned, and synthetic camera and LiDAR CUDA kernels execute successfully on
the RTX 4090 D.

This result establishes the training and test base only. No PandaSet data was
downloaded, no PandaSet scene was parsed, and no SplatAD checkpoint was trained
or evaluated.

## Host And Artifact Layout

Observed during the final acceptance run:

```text
host: stf-precision-3680
GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
driver: 580.95.05
filesystem: /dev/nvme1n1p2
free space: 264 GiB
```

The H3 external root is:

```text
/home/yawei/stage3_external/
├── artifacts/
├── cache/
│   ├── matplotlib/
│   ├── pip/
│   └── torch_extensions/
├── code/
├── data/
├── envs/h3_splatad/
└── outputs/
```

The existing H1/H2 environment and artifacts remain separate under
`/home/yawei/miniforge3/envs/wayve_scenes_env` and
`/home/yawei/stage1_external`.

## Pinned Environment

Core runtime:

| Component | Accepted version |
| --- | --- |
| Python | 3.10.20 |
| CUDA toolkit / compiler | 11.8 / 11.8.89 |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| NumPy | 1.24.4 |
| setuptools | 69.5.1 |
| tiny-cuda-nn | 1.6 |
| neurad-studio | 0.1.0 |
| custom SplatAD gsplat | 1.0.0 |
| PandaSet devkit | 0.3.dev0 |
| dataclass-wizard | 0.35.1 |

Pinned source revisions:

| Source | Commit |
| --- | --- |
| neurad-studio | `e6f7e4e509b828a952d8584b7165f7844711ecb2` |
| PandaSet devkit | `59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0` |
| SplatAD gsplat fork | `6e31ad766d39e0c33f9034a2ed772d51364b2343` |
| neurad-studio viser fork | `57142e42df8edd4de33fd60a08d6bb6c35970aa1` |
| tiny-cuda-nn | `8e6e242f36dd197134c9b9275a8e5108a8e3af78` |

Recreate or check the environment with:

```bash
scripts/setup_stage_h3_environment.sh
scripts/check_stage_h3_environment.sh
```

The setup command does not acquire PandaSet. An existing complete environment
is checked rather than reinstalled; `--repair` explicitly reapplies the pinned
packages.

## Acceptance Evidence

The final `scripts/check_stage_h3_environment.sh` run passed:

```text
PyTorch CUDA device: NVIDIA GeForce RTX 4090 D
tiny-cuda-nn forward: (128, 4), finite
camera rasterizer: (1, 64, 64, 3), alpha_max=0.9999
LiDAR rasterizer: (1, 8, 32, 5), alpha_max=0.9999
pip check: no broken requirements
methods: splatad, neurad
dataparser: pandaset-data
PandaSet cameras: front, front_left, front_right, back, left, right
default PandaSet LiDAR: Pandar64
PandaSet sequence length: 80 frames
```

These are actual GPU executions, not import-only checks. The first development
smoke also compiled the custom CUDA extensions. Later checks reused the cached
kernels.

The existing H1 checkpoint was separately regression-tested after the H3
environment was installed:

```bash
scripts/run_stage_h2_scene_094.sh smoke \
  --output-dir /tmp/dsr_h3_old_environment_regression \
  --output-scale 0.125 \
  --forward 0 --left 0 --yaw-degrees 0 \
  --cameras front-forward
```

It loaded checkpoint step 7999 and produced
`/tmp/dsr_h3_old_environment_regression/front-forward.png`. The cold process
took 118.513 seconds because it included environment/model startup and CUDA
work. It is not a warmed render-latency measurement.

The Waymo dataparser prints a missing-dependencies warning when the command
registry is imported. That path is intentionally deferred; Waymo-specific
packages were not installed into the PandaSet-first environment.

## Installation Failures And Recovery

The failed steps are preserved because they identify reproducibility risks:

1. The first PyTorch wheel installation exceeded a 120-second command timeout
   after downloading most artifacts and left a partial installation. Removing
   the partial packages and repeating with a longer timeout succeeded.
2. PyTorch dependency resolution temporarily installed NumPy 2.2.6, which is
   incompatible with neurad-studio's `numpy<2.0` and numba 0.57. The final
   environment pins NumPy 1.24.4.
3. The attempted tiny-cuda-nn tag `v1.7` does not exist. The upstream tag list
   was inspected and the environment now uses the exact v1.6 commit.
4. tiny-cuda-nn build isolation could not import the already installed PyTorch.
   Installing the exact source with `--no-build-isolation` succeeded.
5. neurad-studio dependency resolution installed `dataclass-wizard` 1.0.0.
   `pip check` reported no broken requirements, but `ns-train` failed because
   zod still imports the removed `JSONSerializable` API. Pinning
   `dataclass-wizard==0.35.1` fixed the runtime.

The last failure is especially important: package metadata alone does not prove
this research stack works. The repository checker therefore verifies the
command registry and executes the CUDA kernels.

## Upstream Capability Audit

The inspected neurad-studio revision supports more than a static camera-only
3DGS baseline:

- SplatAD camera and LiDAR rendering;
- LiDAR depth, intensity, ray-drop, and line-of-sight losses;
- actor Gaussians and actor trajectories derived from 3D cuboids;
- static/dynamic seed-point separation using actor boxes;
- camera and LiDAR rolling-shutter compensation;
- six PandaSet camera choices.

The audit also found limits that change the first pilot design:

- PandaSet contains Pandar64 and PandarGT, but the parser defaults to
  `Pandar64` only.
- elevation mapping, azimuth resolution, missing-point handling, and the
  mature raster layout are implemented for Pandar64. Multi-LiDAR missing-point
  handling still carries an upstream TODO.
- the parser skips duplicated front-LiDAR cuboids when building actor tracks;
  this is not equivalent to fully training with both LiDARs.
- PandaSet semantic labels are point-cloud labels and are available only for
  selected scenes. They are not dense image masks.
- each PandaSet sequence contains 80 frames, approximately eight seconds at
  10 Hz. This is suitable for a pilot but not a large free-roam world.
- the inspected upstream repository explains how to resume a checkpoint but
  does not publish an exact-sequence model-zoo checkpoint in its README or
  documentation.

## Directional Risks

### 1. Do not confuse dataset richness with implemented use

The first reliable baseline should be:

```text
6 cameras + Pandar64 + fused poses + 3D cuboid actor tracks
```

PandarGT should be added only after a separate calibration, timing, raster
layout, and ablation audit. Claiming "two-LiDAR reconstruction" before that
would overstate the actual method path.

### 2. Do not reduce the dynamic problem to masking alone

SplatAD already has an object layer and time-varying actor trajectories. The
primary pilot should use this dynamic-aware path. A masked static-background
run remains useful as a diagnostic and fallback, especially when actor
annotations leak, but it should not silently become the only target.

### 3. The achievable simulator is log-local

A fitted driving-scene checkpoint can reproduce observed space and interpolate
near the logged trajectory. It cannot reliably invent unseen streets, geometry
behind occluders, or large human deviations. The correct H3 product remains:

```text
logged-trajectory playback + conservative human-control offsets
```

It is not unrestricted open-world driving.

### 4. The H2 renderer is not yet a SplatAD adapter

The current `NerfstudioRenderer` was validated with Nerfstudio 1.1.5
Splatfacto. SplatAD adds time, actor, LiDAR, and rolling-shutter state and comes
from a forked Nerfstudio codebase. A PandaSet checkpoint must first be trained
or acquired, then loaded through a separate compatibility/adapter task. Direct
checkpoint compatibility must not be assumed.

### 5. Cold startup and warm latency are different

CUDA compilation and model loading can take tens of seconds. H3 evaluation
must explicitly record:

- first-process startup;
- one-time kernel compilation;
- warmed render latency;
- display/transport latency.

Reporting only the best warmed FPS or the first cold render would both be
misleading.

### 6. Storage is now a real gate

Only 264 GiB was free at final inspection. The data archive, extracted sensor
files, processed copies, checkpoints, and renders can multiply the nominal
download size. PandaSet acquisition must not begin until source packaging,
license, archive size, checksum, extraction size, and a cleanup policy are
recorded.

## Next Gate

The read-only H3-0B acquisition audit is now recorded in
`experiments/stage_h3_dataset_foundation.md`. The next gate is:

1. obtain explicit approval for the dataset terms and 44.5-GB download;
2. download and verify the pinned archive;
3. extract and triage one scene;
4. load one frame without training;
5. preserve camera/LiDAR calibration overlays before any optimization;
6. run at most a 100-step end-to-end smoke only after the data and calibration
   gates pass.
