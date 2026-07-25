# Stage H3 TbV Adjacent-Tile Seam Pilot

Date: 2026-07-25

Status: the prescribed 2,000/8,000-step continuous follow-up is complete in
`experiments/stage_h3_tbv_adjacent_tile_continuous.md`. This file remains the
historical 100/500-step promotion record.

## Question

Can two genuinely adjacent 100 m TbV reconstruction tiles be downloaded,
trained independently, and rendered at identical world poses through their
20 m overlap?

The gate is deliberately smaller than long-route driving or checkpoint
streaming. It tests the data boundary, independent model coordinates, and
first visual seam evidence.

## Data

The previously audited tile-2/tile-3 plan was downloaded to one deduplicated
external directory:

- 2,805 unique objects;
- 424,193,048 source bytes;
- all seven cameras at stride 2, every selected LiDAR sweep, poses,
  calibration, and maps;
- four traversal/tile windows covering tile 2 at 160–260 m and tile 3 at
  240–340 m.

The download tool checks every object size before replacing its temporary file.
The shared storage directory does not merge the training sets: the TbV parser
now applies per-traversal absolute start/end times, so the two checkpoints use
independent tile windows.

Manifest:
`/home/yawei/stage3_external/data/tbv_long_route_tiles_2_3/selection_manifest.json`

Manifest SHA-256:
`aaf11e88b37a8b6b85d4681648ad311d48ba790f9e0cb8942b473aa15538435a`

## Independent Training

Both tiles were trained from scratch in the accepted H3 SplatAD environment on
the RTX 4090 D. Each uses both repeated traversals, 0.9 train split,
0.25 image downsampling, 250,000 maximum seed points, no actors, and metre-scale
city poses.

The first gate trained 100 steps per tile. Because the resulting images were
still dominated by coloured particles, a bounded 500-step repeat was run
before considering a 2,000-step investment.

| Tile | Step | Checkpoint bytes | SHA-256 |
| --- | ---: | ---: | --- |
| 2 | 99 | 289,305,206 | `dab851db266d013e4844578065dd49bb7c6b22c78602529d7b46d43859caeda9` |
| 3 | 99 | 289,287,542 | `dc1307d296e9360d448bacd4e46e1e2285e82d0fb09c9200c22f99e32e790ef9` |
| 2 | 499 | 289,305,206 | `b2334cfe5a43df0af6a36d51c6012550e4d7052e587b855ca3307d80827eb81a` |
| 3 | 499 | 289,287,542 | `31a4f32d52c0926901c456e30a7e28e64a59cbd3e205ea9af5f029b3be15faab` |

## Matched World-Pose Probe

The probe finds exact `ring_front_center` filenames present in both model
windows. It renders five shared reference-traversal timestamps spanning the
overlap. Each model uses its own world-to-model transform, but each query
represents the same original city-frame camera pose.

| Measurement | 100 steps | 500 steps |
| --- | ---: | ---: |
| Tile-2 observed-pose PSNR p50 | 15.115 dB | 19.873 dB |
| Tile-3 observed-pose PSNR p50 | 15.442 dB | 20.902 dB |
| Tile-2/tile-3 RGB MAE p50 | 26.566 / 255 | 18.037 / 255 |
| Tile-2/tile-3 per-frame pixel-error p95 p50 | 65 / 255 | 45 / 255 |
| Warm render p50, tile 2 / tile 3 | 25.82 / 25.44 ms | 25.23 / 25.49 ms |

At both checkpoints all requested RGBs are finite, use scale 1.0, and are not
mostly black. The first tile-2 frame includes cold-start work; render timing is
diagnostic rather than a latency acceptance gate.

## Visual Decision

The 100-step pair passes only the integration gate. Its coloured particles do
not support a meaningful visual seam decision.

At 500 steps, both models recover the same forward road, left low-rise
building, right building, utility poles, and major horizon structure at all
five matched poses. There is no visible coordinate or topology jump. The
pairwise error falls about 32% from the 100-step result, which is sufficient
to promote this exact adjacent pair to a quality-bearing 2,000-step test.

The 500-step images remain soft and granular. Vehicles and some vegetation
differ or smear, and an instantaneous model switch would still be visible.
Therefore this passes the adjacent-tile method/promotion gate, not the final
visual seam, free-driving, or traffic-truth gate.

## Artifacts

100-step:

- JSON:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_20260725/tbv_tile_seam.json`;
- contact sheet:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_20260725/tile_2_3_overlap_contact_sheet.jpg`;
- JSON/contact SHA-256:
  `b59f5e5532ad8e8fabedcc71d32937ac6f91ae117d99659b34a9962e67fb025c`,
  `cb2b80411ab3ad61589038f7ce1ed8c6d7287ba7c8374effc28372ecf1c2a0bb`.

500-step:

- JSON:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_500_20260725/tbv_tile_seam.json`;
- contact sheet:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_500_20260725/tile_2_3_overlap_contact_sheet.jpg`;
- JSON/contact SHA-256:
  `0aa3cb0ea581fd0b945c90ef8ffdf59320f5e3b963d7cad1d967441fc85cef41`,
  `0eda7adfbc1893ffa2e0dbc37a2b30fc8f9c5e4677ba643eee44db85f4c6b643`.

Reproduce or reuse:

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh download
scripts/run_stage_h3_tbv_long_route_tiles.sh smoke-2
scripts/run_stage_h3_tbv_long_route_tiles.sh smoke-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam
scripts/run_stage_h3_tbv_long_route_tiles.sh pilot-2
scripts/run_stage_h3_tbv_long_route_tiles.sh pilot-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-500
```

Completed checkpoints are reused unless `H3_ALLOW_RETRAIN=1` is explicitly
set. Evidence directories refuse non-empty overwrite.

## Next Gate

Train only tiles 2 and 3 to 2,000 steps. Then:

1. repeat the five matched observed-pose comparison;
2. render a continuous centreline sequence through the complete 20 m overlap;
3. compare hard switching with a short overlap blend;
4. add `-1/0/+1 m` counterfactual front views;
5. reject the pair if the switch changes road geometry, creates a false
   obstacle, or leaves traffic ghosts that affect driving.

Do not train the other five tiles or build checkpoint streaming until this
quality-bearing seam video passes.
