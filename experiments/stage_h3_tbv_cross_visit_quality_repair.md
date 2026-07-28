# Stage H3 TbV Cross-Visit Quality-Repair Pilot

Date: 2026-07-28

## Decision

The pilot partially succeeds.

Cross-visit observed-background transfer materially reduces vehicle/background
mismatch where the second traversal actually observes the occluded road or
scene. On those evidence-backed pixels, the candidate improves mean MAE by
28.4%, PSNR by 2.74 dB, and gradient MAE by 8.8% relative to the accepted
static-8k checkpoint. Ordinary observed static pixels also improve slightly.

This does not establish a complete quality repair. The transferred regions are
smoother, vegetation remains mixed or blurred, and the full-frame raw-RGB
metrics are slightly worse because they compare a cleaned road against source
frames that still contain vehicles. Keep static-8k as the default regression.
Keep the new step-9,999 checkpoint as an optional tile-2 candidate pending one
short continuous-route visual A/B; do not promote it across the three-tile
route yet.

## Question

Can the second traversal provide missing static appearance and prevent the
traffic-mask treatments from turning vehicles into dark or diffuse road
smears, while retaining the ordinary road, lane, building, and vegetation
quality of static-8k?

This follows the rejected image-only and direct projected-LiDAR pilots. It
tests two different uses of repeated data:

1. cross-visit LiDAR persistence rescues recurrent geometry before projected
   traffic points are removed;
2. cross-visit RGB transfer supplies only donor-observed background inside a
   target traffic mask, leaving unsupported pixels masked.

## Cross-Visit LiDAR Persistence Gate

Both tile-2 visits were transformed to the common city frame. A point was
marked persistent when the other visit occupied a voxel within a one-voxel
Chebyshev neighborhood of its 0.20 m voxel.

The 197-frame persistence set found 97.51% and 98.32% persistent returns in the
two visits. When combined with the projected traffic mask, it rescued
1,222,686 of 1,488,573 deletion candidates (82.14%) and reduced the actual
removed fraction from 7.66% to 1.37%. The projection audit passed.

An 8,000-step persistence/high-resolution candidate nevertheless failed the
same seven-pose raw-RGB comparison:

| Mean metric | Static 8k | Persistence candidate 8k |
| --- | ---: | ---: |
| PSNR | 23.641 dB | 21.061 dB |
| Gradient MAE | 1.676 | 1.766 |
| Detail retention | 0.469 | 0.449 |

It removed some solid vehicle bodies but left dark smears and softer foliage.
The candidate is rejected. This shows that better geometry selection alone
does not restore appearance hidden by traffic.

The two released TbV log prefixes were also checked for the official AV2
`annotations.feather`; both object paths returned HTTP 404 on 2026-07-28.
This release therefore provides no actor cuboids for an actor-aware repair.

## Rejected Naive Inpainting

A bounded OpenCV Telea preview filled the traffic masks directly from their
image neighborhoods. The filled road contained conspicuous dark/gray blobs,
so the method was rejected before training. It is preserved as negative visual
evidence rather than treated as reconstructed background:

`/home/yawei/stage3_external/artifacts/tbv_inpaint_preview_telea_r5_20260728/inpaint_contact.jpg`

Its SHA-256 is
`1426237e3d455d8119f7f0f7c8b660731b54ecab25279c5a4710f72044a74e66`.

## Conservative Observed-Background Transfer

The implemented transfer:

- finds nearby poses from the other visit in city coordinates;
- requires a small pose/yaw difference;
- estimates a same-camera SIFT/RANSAC homography;
- accepts only geometrically supported matches;
- copies donor RGB only where the target traffic mask is excluded and the
  warped donor mask is valid;
- erodes the donor support at seams and leaves all unsupported pixels masked;
- supports multiple donor candidates, resumable generation, and atomic files.

The generated tile-2 tree contains 1,570 parallel RGB images and 1,570 residual
loss masks. Of the source pixels, 6.675% were excluded by traffic masks on
average; the other traversal safely filled 42.97% of those excluded pixels,
leaving 4.423% of all pixels excluded on average. A total of 1,252 image
records had an accepted transfer record. This count can include a
geometrically accepted attempt with little fill as well as safely reused
completed output; it is a generation result, not a count of independent
ground-truth frames.

The parser now treats RGB overrides, residual image-loss masks, projected
LiDAR masks, and LiDAR-persistence masks as separate inputs. Evaluation
explicitly restores raw source RGB and skips training-only masks, so the
candidate cannot score against its own reconstructed training images.

## Training

The accepted tile-2 static-8k step-7,999 checkpoint was resumed exactly for
2,000 iterations with optimizer and scheduler state restored. Training used:

- cross-visit RGB at half native downsample factor;
- residual image masks aligned to that RGB;
- projected LiDAR masks gated by cross-visit persistence;
- the existing five-million-Gaussian capacity;
- no scheduled LiDAR evaluation during the bounded resume.

The first resume attempt failed before step 0 because the residual image masks
were initially resolved relative to the raw data root after RGB remapping. The
failed run is preserved at timestamp
`2026-07-28_resume_static8k_cross_visit_hires`; the path mapping was corrected
and covered by tests before the retry.

The completed checkpoint is:

`/home/yawei/stage3_external/outputs/tbv_long_route_tiles/tbv_long_route_tile_2_cross_visit_hires_10000/splatad/2026-07-28_resume_static8k_cross_visit_hires_retry/nerfstudio_models/step-000009999.ckpt`

It is 1,650,505,270 bytes with SHA-256
`64eb6d3a544fac62a44db5613b9e14150185292897219ddcbc3f33c69b2ada81`.

## Raw-RGB A/B

Seven identical observed front-camera poses were rendered at the same
1798 x 1550 evaluation resolution. Both checkpoints were evaluated against the
original, unmodified camera images.

| Full-frame mean metric | Static 8k | Cross-visit 10k |
| --- | ---: | ---: |
| PSNR | 23.641 dB | 23.250 dB |
| Gradient MAE | 1.676 | 1.691 |
| Detail retention | 0.469 | 0.465 |

The candidate is 0.39 dB worse in full-frame PSNR. This metric is intentionally
retained, but it cannot by itself decide the repair: the raw reference contains
the source vehicle, while the desired static driving background may not.
Manual review found several vehicle bodies reduced without the black-road
smears of the persistence-only candidate. Foliage remained broadly similar
and mixed.

## Regional Evidence

The same renders were therefore measured in two declared regions.

### Originally observed static background

These are pixels outside the original traffic masks, using the original image
as reference.

| Mean metric | Static 8k | Cross-visit 10k | Change |
| --- | ---: | ---: | ---: |
| MAE | 10.289 | 9.794 | 4.81% lower |
| PSNR | 24.322 dB | 24.606 dB | +0.284 dB |
| Gradient MAE | 1.576 | 1.581 | 0.27% higher |
| Detail retention | 0.5088 | 0.5123 | 0.68% higher |

Ordinary static appearance is not being traded away wholesale. Intensity and
detail improve slightly, while edge error is effectively flat.

### Cross-visit safe-fill pixels

These are only pixels for which the other traversal passed the geometric gates
and observed unmasked RGB. Donor RGB is an evidence-backed background proxy,
not hidden ground truth.

| Mean metric | Static 8k | Cross-visit 10k | Change |
| --- | ---: | ---: | ---: |
| MAE | 39.995 | 28.638 | 28.4% lower |
| PSNR | 14.167 dB | 16.910 dB | +2.743 dB |
| Gradient MAE | 5.218 | 4.759 | 8.8% lower |
| Detail retention | 0.559 | 0.479 | smoother |

This is positive evidence for vehicle-ghost/background repair, with an
explicit texture cost. It does not prove vegetation repair because safe-fill
pixels are traffic-selected and the candidate retains less high-frequency
detail there.

## Promotion Boundary

- Retain tile-2 static-8k as the default and three-tile regression.
- Retain cross-visit 10k as an optional candidate, not a promoted checkpoint.
- Do not repeat broad silhouette deletion or naive local inpainting.
- Run one bounded continuous tile-2 route A/B at identical poses before any
  default switch. Reject the candidate if it introduces flicker, dark patches,
  false road boundaries, or vegetation instability.
- If false vehicles still block driving after this treatment, move to an
  actor-annotated source or the existing MTGS path. More TbV static-mask
  threshold tuning cannot supply missing actor/background truth.
- Treat vegetation separately: repeated high-resolution views, stronger
  multi-view coverage, or a renderer/model ablation is required. The current
  pilot provides only a small ordinary-background detail gain.

## Artifacts

LiDAR persistence:

`/home/yawei/stage3_external/data/tbv_long_route_tile_2_persistence_020_n1_20260728`

The persistence manifest SHA-256 is
`9c9dfe66611fb8b33ff16c04d2f181bc9775652fd195e12d027c119116d39602`.

Cross-visit RGB and residual masks:

`/home/yawei/stage3_external/data/tbv_long_route_tile_2_cross_visit_rgb_20260728`

The cross-visit manifest SHA-256 is
`efc12765edeba300e3e3d18866fc7c4ff340fc6141a5442559047508685163cc`.

Fair A/B and regional evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_2_quality_ab_cross_visit_hires_10k_20260728`

The `quality_ab.json`, `quality_regions.json`, and contact-sheet SHA-256 values
are:

`ec5941b2414023c258b6667e84351ee0734ad9f1e722af1ea932d7e26b5d7495`,
`5b8ad95511fb4afd5e492c591dec8ab4556ee19cdb01f3f6a75a9f0377918921`,
and
`175e12dcc488987ad20b5582b3a15b4ba17caf1f02dfe67793480505a71dc2ec`.

## Reproduce The Final Training And Evaluation

After generating the persistence and cross-visit input trees:

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh cross-visit-quality-2
scripts/run_stage_h3_tbv_long_route_tiles.sh cross-visit-quality-ab-2

/home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/audit_stage_h3_tbv_quality_regions.py \
  --quality-ab-json \
  /home/yawei/stage3_external/artifacts/tbv_long_route_tile_2_quality_ab_cross_visit_hires_10k_20260728/quality_ab.json \
  --data-root /home/yawei/stage3_external/data/tbv_long_route_tiles_2_3 \
  --traffic-mask-root \
  /home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_vehicle_masks \
  --cross-visit-root \
  /home/yawei/stage3_external/data/tbv_long_route_tile_2_cross_visit_rgb_20260728 \
  --baseline-label static_8k \
  --candidate-label cross_visit_hires_10k \
  --output-json \
  /home/yawei/stage3_external/artifacts/tbv_long_route_tile_2_quality_ab_cross_visit_hires_10k_20260728/quality_regions.json
```

Completed checkpoints and evidence directories are reused by default. Set
`H3_ALLOW_RETRAIN=1` only when intentionally replacing the bounded candidate.
