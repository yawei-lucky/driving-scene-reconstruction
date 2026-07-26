# Stage H3 TbV RGB + Projected-LiDAR Transient-Mask Pilot

Date: 2026-07-26

## Decision

The direct projection method is implemented and technically valid, but it is
rejected as a reconstruction-quality treatment.

Projecting synchronized LiDAR returns into the existing seven-camera traffic
masks removed 7.66% of tile-2 points and 9.41% of tile-3 points. Two independent
2,000-step checkpoints and the same matched-pose and continuous probes all ran
to completion. However, cross-tile RGB disagreement was worse than both the
unmasked and image-only-mask baselines, and manual review found a stronger
reddish/brown central smear. The pair therefore stops before 8,000 steps and
before the approximately 180 m two-tile drive.

## Question

Can synchronized image-guided LiDAR exclusion remove traffic geometry left by
the image-only pilot while recovering its lost road sharpness and cross-tile
consistency?

This is the bounded follow-up required by
`stage_h3_tbv_transient_mask_pilot.md`. It changes only the LiDAR points used
to initialize/train the same two tile models; data windows, camera masks,
training length, and seam probes remain fixed.

## Implementation And Audit

For every aggregate LiDAR sweep, the parser selects the nearest downloaded
image from each of the seven ring cameras within 50 ms. It uses AV2's official
motion-compensated ego-to-image projection and removes a LiDAR return if any
valid camera projection lands on a zero-valued pixel in the existing traffic
mask. The policy is an image-detector exclusion, not an actor track or semantic
3D label.

An initial audit used AV2's convenience image lookup, which silently imposes a
25 ms bound. It found a matching front-centre image for only 16 of 197 frames,
so that run was rejected and preserved as an audit failure rather than model
evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_rgb_lidar_mask_audit_failed_sparse_av2_25ms_20260726`

The corrected implementation explicitly selects the closest downloaded image
inside the configured 50 ms bound. Tile 2 then used all seven cameras for all
197 frames. Tile 3 missed four camera/frame pairs at a window boundary but
retained 99.65% projection coverage.

| Audit | Tile 2 | Tile 3 |
| --- | ---: | ---: |
| LiDAR frames | 197 | 165 |
| Input points | 19,431,663 | 15,987,841 |
| Removed points | 1,488,573 | 1,504,698 |
| Removed fraction | 7.66% | 9.41% |
| Camera/frame coverage | 100.00% | 99.65% |
| Maximum image/LiDAR delta | 48.129 ms | 47.967 ms |

The standalone tile-2 audit passed its technical gates: at least one point was
removed and retained, removal stayed below 25%, all seven cameras participated,
and camera/frame coverage exceeded 95%. Those gates verify projection plumbing,
not reconstruction quality.

## Training

Both tiles were trained independently from scratch through step 1,999 in the
accepted H3 environment on the project RTX 4090 D. Both the existing RGB masks
and the new projected-LiDAR filter were enabled.

| Tile | Checkpoint bytes | SHA-256 |
| --- | ---: | --- |
| 2 | 562,507,318 | `4a20f37bf0b56252a5aa6f5376ab55eb3628e5661fee58af98811f8c5c66ef9e` |
| 3 | 562,489,654 | `033c04754294e97a4050fccafbfee04adea434d644379db1e0dcaa2e941aae34` |

## Results

### Five matched observed poses

| Measurement | Unmasked 2k | RGB mask 2k | RGB + LiDAR mask 2k |
| --- | ---: | ---: | ---: |
| Tile-2 PSNR p50 | 23.527 dB | 22.532 dB | 22.524 dB |
| Tile-3 PSNR p50 | 22.774 dB | 22.344 dB | 21.999 dB |
| Cross-tile RGB MAE p50 | 9.835 / 255 | 11.880 / 255 | 13.769 / 255 |
| Pairwise pixel-error p95 p50 | 30 / 255 | 32 / 255 | 37 / 255 |

### Continuous 18.75 m overlap

All three variants produced 32 frames at 12 m/s and 20 fps. The joint-filter
pair retained non-black `-1/0/+1 m` views and passed the mechanical video/path
gates.

| Measurement | Unmasked 2k | RGB mask 2k | RGB + LiDAR mask 2k |
| --- | ---: | ---: | ---: |
| Cross-model RGB MAE p50 | 9.977 / 255 | 11.544 / 255 | 13.582 / 255 |
| Hard transition-frame delta | 11.109 / 255 | 12.097 / 255 | 14.305 / 255 |
| Blended same-frame delta | 4.949 / 255 | 4.947 / 255 | 5.094 / 255 |
| Hard temporal MAE p50 / p95 | 5.771 / 10.451 | 5.583 / 10.111 | 5.635 / 9.994 |
| Blend temporal MAE p50 / p95 | 5.409 / 9.051 | 5.208 / 8.788 | 5.094 / 8.633 |

The image blend still softens the instantaneous cut, but it does not make the
two reconstructions agree.

## Manual Visual Decision

Road, lane markings, buildings, and the three lateral offsets remain readable.
Vehicles are largely suppressed. Nevertheless, the joint-filter tile-2 render
has a conspicuous reddish/brown circular smear near the road centre, side
regions remain blurred, and tile 2 is softer than tile 3. The three-way video
shows no driving-relevant improvement over the image-only result.

Direct observation supports rejecting this exact policy. A likely explanation,
not a separately proven causal result, is that a dilated 2D silhouette removes
valid static returns behind detector regions as well as traffic returns, while
masked RGB pixels provide no appearance supervision for the newly unsupported
region. Threshold polishing is unlikely to fix that structural ambiguity.

Therefore:

- keep the original unmasked 8,000-step pair as the static-geometry regression;
- keep both masked 2,000-step pairs as negative diagnostic evidence;
- do not train this joint-filter pair to 8,000 steps;
- do not build the approximately 180 m drive from it;
- use one bounded cross-traversal 3D-persistence pilot as the next gate.

The next pilot should transform both traversals' point clouds into the common
city frame, retain static points supported within a small radius or voxel by
the other traversal, and treat non-recurrent points as transient candidates.
The existing image masks can remain a secondary cue. This tests actual
cross-visit persistence instead of deleting every return behind a 2D traffic
silhouette. If one 2,000-step tile pair still fails, stop refining static TbV
masks and pivot the long-route renderer/data choice toward actor-aware
reconstruction.

## Probe Runtime Note

The matched-pose probe reloads the configured LiDAR filtering and records its
full statistics. The continuous RGB-only probe initially repeated that
training-only work and was terminated by the command runtime boundary. Its
partial output is preserved at:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_rgb_lidar_masked_2000_failed_eval_filter_timeout_20260726`

The completed continuous probe disables duplicate point filtering only while
loading the already-trained checkpoint for RGB evaluation. Its JSON explicitly
records `checkpoint_trained_with_filter=true` and
`evaluation_filter_skipped=true`; checkpoint weights and rendered camera poses
are unchanged.

## Artifacts

Corrected tile-2 projection audit:

`/home/yawei/stage3_external/artifacts/tbv_long_route_rgb_lidar_mask_audit_20260726`

The audit JSON SHA-256 is
`9555cd9fb2bd6d5a18934e773aa00bf03e23ec24a6edaa9b89aa31f0ac555cb9`.

Matched-pose evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_rgb_lidar_masked_2000_20260726`

The JSON and contact-sheet SHA-256 values are
`07004da212d97bb422fd2a86220938967dc22015f531a129109791090d24442b`
and
`e4420ed2c0f9a9763574398c2083965ca081385714c98d585616bc09812653ba`.

Continuous evidence:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_rgb_lidar_masked_2000_20260726`

The continuous JSON, hard-switch video, smooth-blend video, lateral contact
sheet, and unmasked/image-only/joint three-way video SHA-256 values are:

`6f702d3e4dfeb797c7b401a2f2846bde7953a16180546df255dada47e0d80370`,
`e1ccb06cae28e25d0d8300aa435568b39c234ebee461f0b33a603fc4f3df2003`,
`e834eadf910835d5ff069bb8a0aaaea1a2ce75e53bd53a0dcecb9f5e67cc7729`,
`0af25fece665e1f3df5782f423894c978063fa653bae0fb83d9fb632e9d74e46`,
and
`d59ad40d89bd5976b27bb6a0f0f52ddc3ccf4ffe50f36dd721138084a1a48813`.

## Reproduce

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-mask-audit-2
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-2
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-3
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-continuous-2000
```

Audit/evidence directories refuse completed-output overwrite. Completed
checkpoints are reused unless `H3_ALLOW_RETRAIN=1` is explicitly set.
