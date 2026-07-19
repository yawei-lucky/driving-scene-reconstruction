# Stage H3 PandaSet Scene 040 SplatAD Smoke

Date: 2026-07-19

Status: Level 0 data/calibration gate and Level 1 100-step smoke passed; visual
quality gate not yet passed.

## Purpose

This experiment answers a narrow question: can the pinned H3 environment load
one real PandaSet scene, preserve six-camera/LiDAR/actor timing and calibration,
train SplatAD on the RTX 4090 D, save a checkpoint, reload it, and render the
held-out set?

It is not a quality baseline. The 100-step budget, 0.25 data downsampling, and
250,000-point seed cap were deliberately chosen to expose integration failures
without committing to a long optimization.

## Reproducibility Key

```text
dataset: PandaSet community mirror at e2e123aea3b3132c67f4b395ec6120f63e190271
archive SHA-256: 6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e
scene: 040, all 80 frames
cameras: front, front-left, front-right, left, right, back
LiDAR baseline: Pandar64
dynamic representation: PandaSet cuboid-derived actor trajectories
method: SplatAD from neurad-studio e6f7e4e509b828a952d8584b7165f7844711ecb2
environment: /home/yawei/stage3_external/envs/h3_splatad
training split: upstream PandaSet parser default, 0.5
```

The data license is CC BY 4.0 plus the additional terms stored in the archive.
The user explicitly accepted those terms and the 44.5-GB acquisition on
2026-07-19. The mirrored uploader is not affiliated with the PandaSet creators,
so the source revision and provenance remain part of the run key.

## Scene Selection

All 76 semseg-capable scenes were triaged from their 80-frame cuboid files and
ego trajectories. Contact sheets were then inspected for the strongest
low-dynamic candidates.

Relevant candidates:

| Scene | Ego travel | Mean speed | Cuboids/frame | Dynamic/frame | Near dynamic/frame | Visual finding |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 149 | 49.9 m | 6.3 m/s | 54.5 | 1.6 | 0.0 | Night, low exposure, motion blur; rejected |
| 069 | low-dynamic candidate | — | — | — | — | Night; rejected |
| 028 | daylight candidate | — | — | — | — | Clear, but more near-field parked vehicles |
| **040** | **about 66 m** | **8.21 m/s** | **108.55** | **4.55** | **0.3 in triage** | Clear daylight, stable exposure, visible road/curbs/buildings; selected |

This establishes a reusable rule: annotation density alone cannot select a
reconstruction scene. Photometric quality, road visibility, and camera overlap
must be checked visually before training.

Only scene `040` was extracted:

```text
/home/yawei/stage3_external/data/pandaset/040
744 files
411 MiB on disk
```

## Data And Calibration Gate

The reproducible inspection command is:

```bash
scripts/run_stage_h3_pandaset_040.sh data-gate
```

Observed scene contents:

- 80 frames for each of six 1920x1080 cameras, including poses and timestamps;
- 80 LiDAR frames, poses, and timestamps;
- frame 0 contains 111,572 Pandar64 points and 67,798 PandarGT points;
- 80 cuboid frames and 80 point-cloud semantic-label frames;
- 108.55 cuboids/frame and 4.55 dynamic cuboids/frame on average after removing
  PandarGT annotation duplicates;
- 7.899 seconds duration, 66.62 m front-camera pose path, and 8.213 m/s mean GPS
  speed.

The sensors are not simultaneous. Relative to the front camera, mean absolute
timestamp offsets are:

| Sensor | Mean absolute offset | Maximum absolute offset |
| --- | ---: | ---: |
| front-left camera | 10.797 ms | 10.806 ms |
| front-right camera | 10.782 ms | 10.791 ms |
| left camera | 31.356 ms | 31.381 ms |
| right camera | 31.287 ms | 31.311 ms |
| back camera | 50.041 ms | 50.081 ms |
| LiDAR frame timestamp | 49.837 ms | 49.874 ms |

The calibration overlay at frames 0, 40, and 79 places Pandar64 returns on
road surfaces, building faces, trees, poles, curbs, and vehicles without a
systematic extrinsic failure. The offsets above mean future work must preserve
per-sensor and per-point timestamps; same-index samples must not be treated as
exactly simultaneous.

External evidence:

```text
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_data_report.json
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_six_camera_keyframes.jpg
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_pandar64_camera_overlay.jpg
```

The exact neurad-studio dataparser also passed before training:

```text
train cameras: 240 = 40 timestamps x 6 cameras
point clouds: 40 Pandar64 sweeps
point-cloud time arrays: 40
actor trajectories: 7
sensor names: 6 cameras + Pandar64
add_missing_points: true
dataparser_scale: 1.0
```

## Level 1 Training

The accepted command is preserved by:

```bash
scripts/run_stage_h3_pandaset_040.sh smoke
```

Material settings:

```text
max iterations: 100
data downsample factor: 0.25
maximum seed points: 250,000
train split fraction: 0.5
six cameras: enabled
Pandar64: enabled
missing-point augmentation: enabled
actor trajectories: enabled
```

Result:

```text
exit code: 0
wall time: 139.4 seconds
initialization and step 0: about 97 seconds
steady iterations after warm-up: about 13-25 ms
checkpoint step: 99
checkpoint size: 297,232,596 bytes
```

Checkpoint:

```text
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_smoke_100/splatad/2026-07-19_100step/nerfstudio_models/step-000000099.ckpt
SHA-256: 43addfd7cd5e5f6369c796e58fa87adc96044c74d90875779c8ac690b6a80276
```

Config:

```text
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_smoke_100/splatad/2026-07-19_100step/config.yml
SHA-256: 75b4714d3e0bacd9812b367c6f1ecbf1886dedddd9ac8c0ed61f359ab6ca4f7f
```

The parser warned that the 1920x1080 back image is cropped to 1920x820 after
distortion correction. This matches neurad-studio's explicit 260-pixel
PandaSet back-camera bottom crop. It is recorded for visual acceptance rather
than treated as an unexplained mismatch.

## Checkpoint Reload And Held-Out Rendering

The checkpoint was loaded from disk and all 240 held-out images were rendered
with RGB, reference RGB, and depth:

```bash
scripts/run_stage_h3_pandaset_040.sh render-smoke
```

Result:

```text
checkpoint loaded: step 99
held-out images: 240 = 40 timestamps x 6 cameras
render loop: about 19 seconds
reported render rate: 12.40 images/s
full command wall time: 55.1 seconds
```

Mean metrics across the 240 held-out images:

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| ---: | ---: | ---: |
| 15.9365 | 0.6185 | 0.8402 |

Ranges:

```text
PSNR: 12.9388 to 21.8969
SSIM: 0.5201 to 0.7208
LPIPS: 0.7622 to 0.8977
```

Metrics and renders:

```text
/home/yawei/stage3_external/artifacts/scene_040_smoke_100_render/metrics.pkl
/home/yawei/stage3_external/artifacts/scene_040_smoke_100_render/test
```

## Visual Finding And Decision

The source images are clear and the LiDAR projection is geometrically
plausible. The 100-step RGB output is not usable: it is dominated by broad,
multicolored Gaussian blobs, and road/building/vehicle structure is only
weakly legible. LPIPS 0.8402 agrees with that failure. The more favorable
PSNR/SSIM values mainly reward low-frequency color and blur and must not be
used alone as evidence of driving-scene quality.

Therefore:

- **data gate:** passed;
- **calibration gate:** passed for the first pilot, with asynchronous timing
  explicitly recorded;
- **GPU/checkpoint/render gate:** passed;
- **visual quality gate:** not passed at 100 steps, as expected;
- **8k baseline authorization:** not yet justified.

The decision at this point was to run one Level 2 1k-2k scene-040 pilot with a
fixed temporal holdout and increased training/LiDAR/seed coverage. That run was
subsequently completed on 2026-07-19 and recovered recognizable six-camera
static structure. See `stage_h3_scene_040_pilot.md`; nearby-pose geometry and
temporal gates still precede an 8k baseline.
