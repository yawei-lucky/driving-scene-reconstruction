# Stage H3 PandaSet Scene 040 SplatAD Pilot

Date: 2026-07-19

Status: Level 2 2,000-step visual pilot passed; nearby-pose geometry, temporal
stability, and dynamic-object generalization remain open.

## Purpose

The Level 1 smoke proved the six-camera/Pandar64/actor/checkpoint/render
pipeline but produced only broad colored Gaussian blobs. This Level 2 run asks
whether a still-bounded optimization can recover recognizable multi-camera
static structure before spending an 8,000-step budget.

## Configuration

The command is preserved by:

```bash
scripts/run_stage_h3_pandaset_040.sh pilot
```

Reproducibility key:

```text
dataset archive revision: e2e123aea3b3132c67f4b395ec6120f63e190271
dataset archive SHA-256: 6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e
scene: 040, all 80 timestamps
training split: 0.9 linspace per sensor
train: 72 timestamps, 432 camera images, 72 Pandar64 sweeps
test: 8 timestamps, 48 camera images, 8 Pandar64 sweeps
cameras: all six
LiDAR: Pandar64
actor tracks: 7 cuboid-derived trajectories
LiDAR downsample factor: 0.5
maximum seed points: 750,000
iterations: 2,000
method/environment: pinned H3 SplatAD environment
```

The fixed test timestamps are frames 09, 19, 29, 39, 49, 59, 69, and 78 for
each camera. The denser 0.9 split is intentionally stability-oriented. Its
metrics must not be compared directly with the smoke's 0.5 split or the
camera-only Wayve held-out protocol.

## Training Result

```text
exit code: 0
wall time: 199.6 seconds
steady iterations near completion: about 34-40 ms
checkpoint step: 1,999
checkpoint size: 912,125,396 bytes
```

Checkpoint:

```text
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_pilot_2000/splatad/2026-07-19_train90_seed750k/nerfstudio_models/step-000001999.ckpt
SHA-256: 93fa8c927759749e51759406627e2cf850f7b9731d78e29cc6388dc4240283e4
```

Config:

```text
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_pilot_2000/splatad/2026-07-19_train90_seed750k/config.yml
SHA-256: 07923f9b9a5dddc6f5958a160651b20c48c2a2f29a616d8da780239ffc178e15
```

The training process did not preserve a peak-VRAM sample. That measurement
remains mandatory before raising the seed budget or starting a longer run.

The only repeated warning was the known PandaSet back-camera crop from
1920x1080 source data to 1920x820 calibrated output. The upstream model cropped
the reference consistently.

## Fixed Test Render

The command is:

```bash
scripts/run_stage_h3_pandaset_040.sh render-pilot
```

Observed result:

```text
checkpoint loaded: step 1,999
views: 48 = 8 timestamps x 6 cameras
render loop: about 4 seconds
reported rate: 12.62 images/s
full command wall time: 45.2 seconds
```

Metrics:

| Metric | Minimum | Mean | Maximum |
| --- | ---: | ---: | ---: |
| PSNR ↑ | 21.5483 | 24.7109 | 28.3072 |
| SSIM ↑ | 0.6426 | 0.7392 | 0.8293 |
| LPIPS ↓ | 0.3469 | 0.4475 | 0.5996 |

External artifacts:

```text
/home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render/metrics.pkl
/home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render/test
/home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render/scene_040_frame_39_gt_100_2000.jpg
/home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render/scene_040_front_2000_progression.jpg
```

## Visual Finding

Frame 39 exists in both evaluation runs and provides a direct visual
progression even though aggregate metrics use different splits:

- at 100 steps, every view is dominated by multicolored fuzzy splats;
- at 2,000 steps, road direction, building faces, sidewalks, trees, signals,
  parked vehicles, and rear/side scene layout are recognizable in all six
  cameras;
- the reconstruction remains softer than the reference, especially on road
  texture, tree/sky boundaries, thin poles, windows, and nearby vehicles;
- the frame-69 front view shows localized blur and deformation around vehicles
  and high-frequency background detail;
- rendered depth has coherent broad layers, but it has not yet been compared
  quantitatively with held-out Pandar64 ranges.

This means the multi-sensor SplatAD direction is viable and the Level 2 visual
gate passes. It does not yet establish stable drivable rendering because the
current evidence uses logged held-out camera poses only.

## Decision

Do not jump directly from this result to cockpit UI or claim a 360-degree
driving simulator. Before an 8,000-step baseline:

1. render a fixed nearby-pose grid around several logged timestamps and check
   the front/side/rear views for floaters, holes, and ground deformation;
2. compare rendered depth against held-out Pandar64 in static road, curb,
   facade, pole, and vehicle regions;
3. measure temporal flicker along the logged trajectory and camera-to-camera
   structural consistency;
4. separate stationary-background quality from the 7 dynamic actor tracks and
   record ghost/residue area;
5. record peak VRAM and warm render latency under the accepted resolution.

If those geometry and temporal gates pass, an 8,000-step scene-040 baseline is
justified and can reuse this exact data/split/preprocessing key. If they fail,
fix timestamp handling, seed geometry, loss scheduling, or actor/background
separation before adding optimization steps.
