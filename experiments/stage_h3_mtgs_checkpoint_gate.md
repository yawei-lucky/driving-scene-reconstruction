# Stage H3 MTGS Released-Checkpoint Gate

Date: 2026-07-25

## Question

Can one officially released MTGS multi-traversal block load and render on the
project's 24 GB RTX 4090 D without training, and does a first fixed-time
world-pose grid retain enough road structure to justify a minimal wide,
high-speed driving adapter?

This follows the metadata and source audit in
`stage_h3_mtgs_published_block_probe.md`. It is an inference and visual
coverage gate, not an MTGS training attempt.

## Pinned Inputs

- official source commit:
  `7ab67a3e386e5a4830017819324922a7bb9f26f7`;
- block: `road_block-365000_144000_365100_144080`;
- official block archive size: `3,983,371,020` bytes;
- official block archive SHA-256:
  `d75c7e4ed0ec675d1ee7c656aa1695738f04b154bc30482ded6706661470808c`;
- official released checkpoint: `step-000030000.ckpt`, `773,241,943`
  bytes;
- checkpoint CPU inspection: step 30,000, 301 pipeline tensors;
- host: NVIDIA GeForce RTX 4090 D, 24,564 MiB, driver 580.95.05.

The source, data, checkpoint, environment, and artifacts remain outside Git
under `/home/yawei/stage3_external`.

## Isolated Environment

The accepted SplatAD environment was not changed. MTGS was installed in
`/home/yawei/stage3_external/envs/mtgs` with Python 3.9.23,
Torch 2.0.1+cu118, torchvision 0.15.2+cu118, Nerfstudio 1.1.5, gsplat 1.4.0,
and the official tiny-cuda-nn source dependency.

The first tiny-cuda-nn build reached its final link and failed because
`-lcuda` was not on the build library search path. One retry with the host
driver and CUDA stub directories in `LIBRARY_PATH` completed both gsplat and
tiny-cuda-nn. Runtime import succeeds. tiny-cuda-nn reports that it was built
for compute capability 86 while this GPU is 89, so the timing here may be
slower than a fully native build.

Disk use after the gate was approximately 7.9 GB for the environment, 8.1 GB
for the downloaded plus extracted data area, and 738 MB for the checkpoint
run directory.

## Checkpoint-Load Gate

The probe imports the official YAML config, points it at the verified block and
checkpoint, disables unused mask/depth/quality-metric inputs, uses on-demand
eval caching, loads on CUDA, and renders one fixed eval front camera ten times.

Results:

| Measurement | Result |
| --- | ---: |
| Pipeline plus checkpoint load | 19.72 s |
| Output shape | 960x540 RGB |
| Finite output | yes |
| Warm render p50 | 13.52 ms / 73.98 FPS |
| Warm render p95 | 13.71 ms / 72.93 FPS |
| Peak CUDA allocated | 1.287 GiB |
| Peak CUDA reserved | 1.455 GiB |
| 21.5 GiB safety gate | pass |

Manual review found the road surface, multiple lane markings, curb, fence, and
gentle bend readable. Foliage and distant construction remain soft. This
clears the environment, checkpoint compatibility, 24 GB memory, and first
front-render risks only.

## Wide-Corridor Probe

From the same eval camera and scene time, the probe translates the world pose
in the camera's right/forward basis while retaining its heading:

- forward offsets: 0, 15, and 30 m;
- lateral offsets: -5, -3, 0, +3, and +5 m;
- total: 15 poses.

All 15 RGB images were finite. Per-pose rendering measured 10.04 ms p50 and
12.94 ms p95 after checkpoint load.

Manual contact-sheet review found:

- the road, lane directions, and curb remain continuous across every sampled
  row and lateral offset;
- +/-3 m is visually comfortable enough to attempt a continuous lane-change
  smoke;
- +/-5 m remains interpretable, but close foliage, signs, and curbs visibly
  stretch or smear;
- no inspected pose produced a blocking false obstacle on the road;
- fixed time means this says nothing about the correctness or responsiveness
  of dynamic actors.

Decision: pass the isolated checkpoint and first spatial-coverage gate. This is
stronger spatial evidence than the current TbV +/-1 m route, but it is not yet
continuous driving evidence.

## Artifacts

- report:
  `/home/yawei/stage3_external/artifacts/mtgs_checkpoint_gate_20260725/mtgs_checkpoint_gate.json`;
- observed pose:
  `/home/yawei/stage3_external/artifacts/mtgs_checkpoint_gate_20260725/observed_front_probe.jpg`;
- corridor contact sheet:
  `/home/yawei/stage3_external/artifacts/mtgs_checkpoint_gate_20260725/corridor_contact_sheet.jpg`;
- 15 individual corridor images under `corridor_frames/` in that artifact
  directory.

The repository entry points are:

```bash
scripts/run_stage_h3_mtgs_gate.sh verify-assets
scripts/run_stage_h3_mtgs_gate.sh checkpoint-gate
scripts/run_stage_h3_mtgs_gate.sh corridor-probe
```

## Next Gate

Keep the checkpoint and scene time fixed. Render one continuous front-camera
path at about 10-12 m/s over a bounded +/-4 m lateral envelope, recording pose,
support, finite output, and render timing. If the video keeps road boundaries
and false obstacles decision-safe, reuse the small TbV vehicle/evidence
contract for keyboard control. Do not start MTGS training on this 24 GB host.

Status on 2026-07-25: the continuous-video and standalone-adapter parts of this
gate were completed in `stage_h3_mtgs_continuous_drive.md`. Local keyboard and
renderer integration remains pending.
