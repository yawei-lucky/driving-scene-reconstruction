# Stage H3 HUGSIM And Generative Repair Gate

Date: 2026-07-28

## Question

Can HUGSIM or a generative method remove vehicle ghosts and general blur from
the existing TbV static-8k result without repeating the rejected
mask/deletion/cross-visit repair path?

This is a one-hour capability gate. It does not promote a new checkpoint or
change the default driving renderer.

## Method Triage

Primary sources were inspected on 2026-07-28:

- HUGSIM paper and official repository:
  https://arxiv.org/abs/2412.01718 and
  https://github.com/hyzhou404/HUGSIM
- ProPainter official repository:
  https://github.com/sczhou/ProPainter
- SGD diffusion-prior paper:
  https://arxiv.org/abs/2403.20079
- DiffuEraser official repository:
  https://github.com/lixiaowen-xw/DiffuEraser

The methods solve different problems:

| Method | What it actually supplies | Fit for this gate |
| --- | --- | --- |
| HUGSIM | A new structured 3DGS reconstruction split into ground, non-ground static background, and dynamic vehicles, with physical ground/vehicle constraints | Strong architectural direction for ghosts and extrapolated lanes, but not a postprocessor for a SplatAD checkpoint |
| ProPainter | Learned flow propagation plus transformer video inpainting from frame-wise masks | Smallest temporal object-removal smoke; selected |
| SGD | A fine-tuned diffusion model regularizes 3DGS at unseen views using adjacent frames and LiDAR depth | More 3D-aware, but requires diffusion fine-tuning and reconstruction retraining |
| DiffuEraser | Diffusion video inpainting with temporal attention and a ProPainter prior | Deferred: its base assets add roughly 4 GB at minimum and it is much heavier than the first discriminator |

HUGSIM explicitly treats diffusion-added supervision as vulnerable to
multi-view inconsistency and increased training cost. Its alternative is
physical structure, not a generative cleanup pass. The official pipeline also
requires data preparation, ground training, scene training, and export at
iteration 30,000. The existing local checkout at
`/home/yawei/HUGSIM` is commit
`adeca402cad4af8635e13d0a105e2fee6a14de85`; it was inspected but not
modified. Its unrelated untracked `pixi.toml.smoke-backup` remains untouched.

## Executed ProPainter Smoke

The official ProPainter repository was cloned outside Git at:

```text
/home/yawei/stage3_external/code/ProPainter
commit e870e79321c31b733e2031af5aa2fb1fe3ac7eec
```

The official v0.1.0 inference weights were downloaded automatically:

| Weight | Bytes | SHA-256 |
| --- | ---: | --- |
| `raft-things.pth` | 21,108,000 | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| `recurrent_flow_completion.pth` | 20,348,681 | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |
| `ProPainter.pth` | 157,780,510 | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |

ProPainter code and models are restricted to non-commercial use under the
NTU S-Lab License 1.0. This gate therefore establishes research feasibility,
not product licensing.

The input was 32 consecutive 10 Hz observed-pose renders from the promoted
tile-2 static-8k checkpoint:

```text
window: 315970599.1999272 to 315970602.29992723 seconds
source render: 1550x1798
inpainting resolution: 640x744, fp16
source masks: existing Mask R-CNN traffic masks, aligned and inverted
mask fraction: 5.22% mean, 8.84% max
```

Input preparation used the new fail-closed CUDA 11.8 launcher:

```bash
scripts/run_stage_h3_environment.sh python \
  scripts/prepare_stage_h3_tbv_generative_repair.py \
  --config /home/yawei/stage3_external/outputs/tbv_long_route_tiles/tbv_long_route_tile_2_static_8000/splatad/2026-07-25_resume_2k_to_8k/config.yml \
  --data-root /home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_frozen_20260725 \
  --traffic-mask-root /home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_vehicle_masks \
  --window-start-seconds 315970599.1999272 \
  --window-end-seconds 315970602.29992723 \
  --sample-count 32 \
  --output-dir /home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_input_20260728
```

ProPainter reused the already installed HUGSIM pixi runtime only as an
isolated dependency environment. A host-side probe reported PyTorch
`2.4.1+cu121`, CUDA 12.1, a visible GPU, and
`NVIDIA GeForce RTX 4090 D`:

```bash
cd /home/yawei/stage3_external/code/ProPainter
/home/yawei/.pixi/bin/pixi run \
  --manifest-path /home/yawei/HUGSIM/pixi.toml \
  python inference_propainter.py \
  --video /home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_input_20260728/static_8k \
  --mask /home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_input_20260728/removal_masks \
  --output /home/yawei/stage3_external/artifacts/tbv_tile2_propainter_20260728 \
  --width 640 --height 744 --fp16 --save_frames \
  --save_fps 10 --subvideo_length 32 \
  --neighbor_length 8 --ref_stride 8 --raft_iter 12
```

## Result

All 32 input, mask, and output frames were present. The evidence video is
2560x744, 10 fps, 32 frames, and 3.2 seconds:

```text
/home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_audit_20260728/static8k_vs_propainter_h264.mp4
/home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_audit_20260728/static8k_vs_propainter_contact.jpg
/home/yawei/stage3_external/artifacts/tbv_tile2_generative_repair_audit_20260728/generative_repair_audit.json
```

The JSON points to the audit script's MPEG-4 intermediate; the listed H.264
copy has identical 32-frame content and is the presentation-compatible video.

Measured at the 640x744 inference resolution:

| Screen | Result |
| --- | ---: |
| Pixels changed outside the four-pixel dilated mask | 0.0% |
| Outside-mask RGB MAE | 0.0 / 255 |
| Inside-mask change MAE, p50 | 14.366 / 255 |
| Naive masked consecutive-frame MAE, static-8k p50 | 16.334 / 255 |
| Naive masked consecutive-frame MAE, ProPainter p50 | 9.285 / 255 |

The last metric drops by about 43%, but it is not an optical-flow-warped truth
metric and includes ego motion.

Direct visual inspection finds:

- prominent vehicle/ghost shapes are removed;
- the filled road is plausible at a glance and more temporally stable than
  frame-independent filling;
- large hidden regions are soft and smeared rather than recovered as sharp,
  evidenced geometry;
- original foliage/building blur is unchanged outside traffic masks;
- intermittent source detections make the masks unsuitable as ground truth.

## Decision

ProPainter passes as an **offline temporal-removal capability demonstration**.
It does not pass as a simulator repair:

1. its output is tied to one rendered camera sequence and cannot be reprojected
   consistently when the driver changes pose;
2. no hidden-background ground truth exists behind the removed vehicles;
3. it can erase a false obstacle visually while inventing lane, curb, or
   free-space pixels with no corresponding 3D evidence;
4. it does not address vegetation or general reconstruction blur.

Do not insert ProPainter after the live renderer and do not train a new default
checkpoint from this 32-frame output. If learned completion is continued, use
it only as masked pseudo-supervision and require cross-view consistency,
held-out lane/free-space preservation, and a continuous off-centre driving
gate before promotion.

For the actual reconstruction line, the more credible next discriminator is a
small HUGSIM-style structured reconstruction: predict semantics and 3D tracks,
separate traffic from static ground/background, train a ground-constrained
model, and compare it with static-8k at identical poses. TbV is not an
official HUGSIM input format, so that step needs a bounded AV2/TbV adapter; it
is not an environment-install task and was not started in this gate.
