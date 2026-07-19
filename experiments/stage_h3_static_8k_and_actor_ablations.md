# Stage H3 Scene 040 — Static 8k And Actor Ablations

Date: 2026-07-19

Host: `stf-precision-3680`

Dataset: PandaSet scene `040`, fixed 0.9 temporal split

Environment: `/home/yawei/stage3_external/envs/h3_splatad`

## Question

Can the accepted 2,000-step SplatAD pilot be improved into a stable nearby-pose
rendering baseline, and does preserving explicit vehicle actor layers improve
the dynamic regions that remain weak?

All large checkpoints, renders, videos, and metric reports remain outside Git
under `/home/yawei/stage3_external`.

## Static 8k Baseline

The accepted step-1,999 pilot was resumed for exactly 6,000 additional
iterations. Optimizer and scheduler state were restored before the training
loop, producing step 7,999 without resetting the learned representation.

Artifacts:

```text
config:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_static_8000/splatad/2026-07-19_resume_2k_to_8k/config.yml

checkpoint:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_static_8000/splatad/2026-07-19_resume_2k_to_8k/nerfstudio_models/step-000007999.ckpt

geometry:
/home/yawei/stage3_external/artifacts/scene_040_geometry_gate_8000/scene_040_geometry_report.json

temporal:
/home/yawei/stage3_external/artifacts/scene_040_temporal_gate_8000/scene_040_temporal_report.json
```

Observed result:

| Measure | 2k pilot | Static 8k |
|---|---:|---:|
| Held-out PSNR | 24.7109 | 26.6605 |
| Held-out SSIM | 0.7392 | 0.8145 |
| Held-out LPIPS | 0.4475 | 0.2818 |
| Static LiDAR absolute error p50 | 0.1430 m | 0.0614 m |
| Static LiDAR absolute error p90 | 1.1203 m | 0.6265 m |
| Static LiDAR absolute error p95 | 2.1069 m | 1.4925 m |

All 126 fixed nearby-pose views were finite. Warm single-camera latency was
28.01 ms p95. Sequential six-camera latency was 153.19 ms p95, about 6.5
complete rigs/s, so the 10 Hz six-camera gate did not pass.

The 480-view logged sequence was finite. Per-camera excess-warp p95 was
0.00457-0.00758. Manual inspection showed continuous output rather than
flashes, but nearby vehicles remained blurred and the automated flow coverage
was inconclusive for some side views.

Decision: **accept static 8k as the current best H3 checkpoint, but not as a
completed stable-drivable product**.

## Stationary-And-Moving Object-Layer Pilot

The first actor ablation assigned 91 stationary rigid cuboids and 7 moving
vehicles to separate actor layers. A custom MCMC relocation strategy sampled
donors only from the same actor ID, preventing refinement from recycling an
actor completely into another actor or the background.

Artifacts:

```text
run:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_vehicle_objects_8000/splatad/2026-07-19_stationary_moving_actor_aware

geometry:
/home/yawei/stage3_external/artifacts/scene_040_vehicle_objects_8000_geometry/scene_040_geometry_report.json

temporal:
/home/yawei/stage3_external/artifacts/scene_040_vehicle_objects_8000_temporal/scene_040_temporal_report.json

paired comparison:
/home/yawei/stage3_external/artifacts/scene_040_vehicle_objects_8000_comparison/vehicle_pilot_comparison.json
```

All 98 actor IDs survived, but actors consumed 2,113,920 of the 5,000,000
final Gaussians. Background capacity fell to 2,886,080.

| Measure | Static 8k | 98-object candidate |
|---|---:|---:|
| Held-out PSNR | 26.6605 | 24.0924 |
| Held-out SSIM | 0.8145 | 0.7783 |
| Held-out LPIPS | 0.2818 | 0.3546 |
| 1,099 vehicle crops PSNR | 28.8555 | 25.7734 |
| 1,099 vehicle crops LPIPS | 0.1530 | 0.3017 |
| 40 moving crops PSNR | 21.3726 | 17.3117 |
| 40 moving crops LPIPS | 0.2157 | 0.4451 |
| Static LiDAR error p50 / p90 | 0.0614 / 0.6265 m | 0.2904 / 1.7868 m |
| Sequential rig latency p95 | 153.19 ms | 177.52 ms |

The candidate blurred recognizable parked vehicles into broad low-detail
regions. The small set of apparent crop improvements mostly covered road or
vegetation inside projected cuboids rather than visibly improved vehicles.

Decision: **reject**. Preserving actor IDs did not preserve actor quality, and
the stationary actor seed allocation starved the static background.

## Moving-Only Actor-Aware Ablation

An independent reviewer requested one minimal, falsifiable ablation:

- keep only the original 7 moving vehicle actors;
- keep stationary cuboids in the static background;
- retain actor-aware MCMC;
- keep the same scene, split, seed cap, 5M final cap, and 8k steps.

Artifacts:

```text
run:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_moving_actor_aware_8000/splatad/2026-07-19_moving_only_actor_aware

geometry:
/home/yawei/stage3_external/artifacts/scene_040_moving_actor_aware_8000_geometry/scene_040_geometry_report.json

temporal:
/home/yawei/stage3_external/artifacts/scene_040_moving_actor_aware_8000_temporal/scene_040_temporal_report.json

paired comparison:
/home/yawei/stage3_external/artifacts/scene_040_moving_actor_aware_8000_comparison/vehicle_pilot_comparison.json
```

Training completed in 558.3 seconds. All 7 actors survived with at least 1,030
Gaussians. Actors used only 41,130 Gaussians, leaving 4,958,870 for the
background.

| Measure | Static 8k | Moving-only candidate |
|---|---:|---:|
| Held-out PSNR | 26.6605 | 26.4825 |
| Held-out SSIM | 0.8145 | 0.8078 |
| Held-out LPIPS | 0.2818 | 0.2875 |
| 40 moving crops PSNR | 21.3726 | 19.7167 |
| 40 moving crops LPIPS | 0.2157 | 0.3649 |
| 27 paired moving crops >=32 px, PSNR delta | — | -1.8474 dB |
| 27 paired moving crops >=32 px, LPIPS delta | — | +0.1824 |
| Static LiDAR error p50 / p90 | 0.0614 / 0.6265 m | 0.0699 / 0.8529 m |
| Sequential rig latency p95 | 153.19 ms | 153.13 ms |

All six excess-warp p95 values were at most 0.00683 and render latency matched
the baseline. Nevertheless, the candidate failed the held-out SSIM, moving
crop, and LiDAR gates. Only one of 27 >=32-pixel paired moving crops improved
LPIPS. The worst-pair contact sheet shows annotated moving cars being replaced
by low-detail gray/green blobs.

Decision: **reject and stop the current actor-aware training branch**. The
failure remains after removing stationary actor capacity pressure, so the
remaining cause is in moving-actor construction, supervision, timing,
coordinate transforms, or trajectory optimization rather than training
duration alone.

## Current Conclusion

The static 8k checkpoint is the accepted H3 baseline. Longer training materially
improved static appearance and metric geometry, but neither tested actor-layer
variant improved dynamic vehicles.

The next research action is diagnostic, not another long training run:
validate each moving cuboid against the original camera images at exact sensor
timestamps and render actor-only/background-only layers. This should determine
whether the gray blobs originate before optimization (incorrect cuboid-camera
time/pose or actor-local transform) or during optimization (insufficient or
misassigned image supervision).

## Validation Notes

- The first geometry invocation accidentally used the H1 environment and was
  rejected before evaluation because its Nerfstudio fork lacked the required
  API.
- A second invocation used the H3 Python but omitted the accepted CUDA toolkit
  and extension cache variables; gsplat attempted an incompatible rebuild and
  failed before rendering.
- The recorded geometry results above use the H3 environment, CUDA toolkit,
  and cached extensions exported by `scripts/run_stage_h3_pandaset_040.sh`.
- Python compilation, `git diff --check`, and all 16 lightweight unit tests
  passed after the implementation changes.
