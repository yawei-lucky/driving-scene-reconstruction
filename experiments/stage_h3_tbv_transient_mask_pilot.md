# Stage H3 TbV Image-Space Transient-Mask Pilot

Date: 2026-07-26

Status: the proposed projected-LiDAR follow-up has now been completed and
rejected at 2,000 steps. See `stage_h3_tbv_rgb_lidar_mask_pilot.md`. The
historical image-only measurements below remain unchanged; its former “next
experiment” wording is superseded by that follow-up result.

## Question

Can a minimal image-space traffic mask remove the false-obstacle residue from
the selected TbV tile-2/tile-3 pair without damaging the road geometry enough
to block the approximately 180 m two-tile auto-drive?

This is a bounded 2,000-step discriminator against the existing unmasked
2,000-step pair. A negative result stops before 8,000 steps.

## Implementation

The pilot uses torchvision's official
`MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` weights to exclude `person`,
`bicycle`, `car`, `motorcycle`, `bus`, `train`, and `truck` pixels from the RGB
loss. Detections require score 0.60, masks use threshold 0.50, and the excluded
region is dilated by 6 pixels.

The generator wrote valid-pixel PNG masks for all 2,468 downloaded camera
images while mirroring the source data layout. The run took 94.81 seconds on
the project RTX 4090 D with batch size 8. The excluded pixel fraction was
3.71% p50, 30.16% p95, and 8.50% mean. The detector reported 10,063 cars, 653
people, 469 trucks, 98 motorcycles, 65 bicycles, 32 buses, and 23 trains.
These are per-image detections, not tracked actor counts.

Manual review of the front-camera contact sheet found generally useful traffic
coverage, plus expected detector errors: the visible ego hood is sometimes
classified as a vehicle and one pedestrian road sign is classified as a
person. The front-camera training crop removes the hood region, but the masks
remain detector output rather than actor truth.

The TbV dataparser now accepts an optional mask root and refuses incomplete
mask sets. SplatAD already consumes image masks in its RGB loss, but its
downscale path cropped RGB after undistortion without applying the same crop to
the mask. The first tile-2 run therefore failed at step 0 on a shape assertion.
That failed run is preserved at:

`/home/yawei/stage3_external/outputs/tbv_long_route_tiles/tbv_long_route_tile_2_masked_2000_failed_mask_shape_20260726`

`MaskAlignedSplatADModel` makes only that missing crop alignment. No loss
weight, seed count, window, camera selection, or other model behavior changed.

## Training

Both selected tiles were trained independently from scratch to step 1,999 with
the same two traversals and windows as the unmasked 2,000-step baseline.

| Tile | Training masks | Checkpoint bytes | SHA-256 |
| --- | ---: | ---: | --- |
| 2 | 1,570 | 562,507,318 | `6a0414e6b8e294b4ca8f1c6c9d06e6e6900436c694ac49dc51d709885b457ba9` |
| 3 | 1,342 | 562,489,654 | `53d86277047957fe77e15170749094cacee808b4a66631c6016ab9d8912fa151` |

The training-mask count is the tile-specific train-plus-eval image count, not
the 2,468-image mask inventory.

## Results

### Five matched observed poses

| Measurement | Unmasked 2k | Masked 2k | Change |
| --- | ---: | ---: | ---: |
| Tile-2 PSNR p50 | 23.527 dB | 22.532 dB | -0.995 dB |
| Tile-3 PSNR p50 | 22.774 dB | 22.344 dB | -0.430 dB |
| Tile-2/tile-3 RGB MAE p50 | 9.835 / 255 | 11.880 / 255 | +20.8% |
| Pairwise pixel-error p95 p50 | 30 / 255 | 32 / 255 | +2 / 255 |

### Continuous overlap

Both masked models produced all 32 frames over corresponding
18.7522/18.7524 m paths at 12 m/s and 20 fps. The hard-switch, smooth-blend,
and side-by-side videos decode completely.

| Measurement | Unmasked 2k | Masked 2k |
| --- | ---: | ---: |
| Model-to-model RGB MAE p50 | 9.977 / 255 | 11.544 / 255 |
| Hard-output temporal MAE p50 / p95 | 5.771 / 10.451 | 5.583 / 10.111 |
| Blend-output temporal MAE p50 / p95 | 5.409 / 9.051 | 5.208 / 8.788 |
| Hard transition-frame delta | 11.109 / 255 | 12.097 / 255 |
| Blended same-frame delta | 4.949 / 255 | 4.947 / 255 |

The 5.626 m image blend still hides most of the instantaneous cut. It does not
make the two reconstructions agree.

## Manual Visual Decision

The masks visibly remove several parked or moving vehicle bodies and the
strongest central black vehicle-shaped residue from tile 2. Road boundaries,
yellow center lines, buildings, poles, and the `-1/0/+1 m` views remain
readable. This supports the basic premise that traffic should not be baked
into the static background.

The pilot nevertheless fails promotion. Removed vehicle regions become
diffuse road-coloured smears, tile 2 remains noticeably softer than tile 3,
and both matched-pose and continuous model-to-model errors get worse. The
remaining residue is consistent with undecomposed LiDAR traffic points and
missing appearance supervision, which image-only masks cannot solve.

Therefore:

- retain the original 8,000-step pair as the static-geometry regression;
- retain this 2,000-step masked pair as negative/diagnostic evidence;
- do not spend on masked 8,000-step training;
- do not build or claim the approximately 180 m two-tile drive yet;
- next exclude synchronized LiDAR points projected into the same traffic
  masks, then rerun only this 2,000-step pair and the same probes.

That next experiment must preserve the visible obstacle reduction while
recovering the unmasked 2,000-step road sharpness and model-to-model error.

## Artifacts

Mask inventory:

- manifest:
  `/home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_vehicle_masks/vehicle_mask_manifest.json`;
- review contact:
  `/home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_vehicle_masks/front_center_vehicle_mask_contact.jpg`.

Their SHA-256 values are
`ce4e58534b1ac13bc1c31ef4d44118e8f7c2a2a9e352718c7c7f274febf7fbdc`
and
`a0d329c7b5e643f8e81e28edb59f2c56e920000a1f3269cfc6180902b6f60117`.
The downloaded weights SHA-256 is
`73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`.

Matched-pose evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_masked_2000_20260726`

The JSON/contact SHA-256 values are
`2ca29f56a4fd291cf6c1b34ea06e991957419f66597b39dcdd0f5ceb0064394a`
and
`6b66d9d047143c14a72fc65e32751221050736141525407dd791687b4c6a7afc`.

Continuous evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_masked_2000_20260726`

The continuous JSON, hard-switch video, smooth-blend video, lateral contact,
and unmasked-left/masked-right comparison video SHA-256 values are:
`01238587b55c7e09b392f06d296436114935003eb68c5b1a0ea099d32c91712d`,
`706c7e214ce7714d4c645428cb7b7ffe78f1965d75447af4aa639efe7e8e760a`,
`049c51103f2298c629c24cf61c28b5c40603e3502b569e0fb0f92ba3bc8ce526`,
`da8a09678e4c4343f45bcaeb718d5ae6b5043dfa21906072a9b80cc511658247`,
and
`65cfeab2c3ebb9e6d3c16937de8613988297446dc07ea1a25c7f8603a80f7a38`.

## Reproduce

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh mask-data
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-2
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-3
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-continuous-2000
```

The mask generator and evidence directories refuse completed-output
overwrite. Completed checkpoints are reused unless `H3_ALLOW_RETRAIN=1` is
explicitly set.
