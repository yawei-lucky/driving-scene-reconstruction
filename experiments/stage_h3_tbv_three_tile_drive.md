# Stage H3 TbV Three-Tile 260 m Drive Pilot

Date: 2026-07-28

## Question

Can the accepted repeated Miami route be extended from the tile-2/tile-3
adjacent seam to a small but useful approximately 250 m driving result, without
pretending that all seven tiles or real-time streaming are already solved?

The bounded choice was tiles 2, 3, and 4:

- tile 2: route progress 160–260 m;
- tile 3: route progress 240–340 m;
- tile 4: route progress 320–420 m.

Their union is 260 m and contains two real 20 m overlaps.

## Data Isolation

Adding tile 4 exposed a reproducibility issue in the existing shared raw-data
directory. The downloader applies camera stride independently to every
window. Combining a new overlapping window therefore added opposite-phase
camera frames inside the old tile-3 time window. Its configured dataset grew
from 1,359 to 1,411 camera indices and the unchanged checkpoint correctly
rejected the mismatched camera-velocity tensor.

The checkpoint was not damaged. A new manifest-subset tool hard-linked the
original four tile-2/tile-3 windows into:

`/home/yawei/stage3_external/data/tbv_long_route_tiles_2_3_frozen_20260725`

It contains 2,805 objects and 424,193,048 logical bytes. Tile 2/3 checkpoints
now explicitly load that frozen view; tile 4 loads the expanded six-window
view. The initial failed seam output is retained rather than overwritten.

## Tile 4 Reconstruction

The tile-4 metadata gate passed before training:

- repeat-route nearest-distance p50/p95: 0.301/0.626 m;
- heading difference p50/p95: 0.182/0.918 degrees;
- supported reference samples: 100%.

A new unmasked SplatAD model was trained to step 1,999 and exact-resumed to
step 7,999. The first resume attempt reached 5,000,000 Gaussians, then failed
at step 6,000 when periodic LiDAR evaluation requested another 4.47 GiB with
only about 4.09 GiB free. The failed run is preserved at:

`/home/yawei/stage3_external/outputs/tbv_long_route_tiles/tbv_long_route_tile_4_static_8000/splatad/2026-07-28_resume_2k_to_8k`

The actual host process audit after the failed process exited found no
competing compute job. At idle the RTX 4090 D reported 1,426 MiB used and
22,634 MiB free; the only listed GPU process was GNOME Remote Desktop at
396 MiB. That small display allocation was not treated as a training resource
to reclaim.

This was an evaluation-peak failure, not evidence that RGB inference cannot
fit. The exact resume was repeated from the same 2k checkpoint with only
periodic LiDAR evaluation disabled; it completed step 7,999 and saved:

`/home/yawei/stage3_external/outputs/tbv_long_route_tiles/tbv_long_route_tile_4_static_8000/splatad/2026-07-28_resume_2k_to_8k_no_eval/nerfstudio_models/step-000007999.ckpt`

The checkpoint is 1,650,474,934 bytes with SHA-256
`7374720eb6bc18d13be834c99301360d05696d9001b4adbee43cece60f78f4c4`.
Optimizer, scheduler, model, and global-step restoration passed. No final
LiDAR evaluation is claimed.

## Tile-3/Tile-4 Seam

Five exact shared front-camera poses were rendered through both models.

| Measurement | 2,000 steps | 8,000 steps |
| --- | ---: | ---: |
| Tile-3 PSNR p50 | 23.343 dB | 22.872 dB |
| Tile-4 PSNR p50 | 23.124 dB | 24.272 dB |
| Tile-3/tile-4 RGB MAE p50 | 10.513 / 255 | 11.969 / 255 |
| Pixel-error p95 p50 | 31 / 255 | 34 / 255 |
| Warm render p50, tile 3 / tile 4 | 25.82 / 29.07 ms | 31.16 / 35.70 ms |

All frames were finite, metre-scale, and not mostly black. Manual comparison
retained the same road, curb, high-rise, low building, palm-tree, wire, and
horizon layout with no coordinate/topology jump. The 8k tile-4 image became
sharper, but independent-model RGB consistency became worse, not better.
Longer training is therefore not a seam-consistency treatment.

## Simulated Drive

The new `SceneTile` route contract records each checkpoint, its exact data
view, source log/time window, dataparser transform, supported route interval,
overlap, appearance profile, lateral support, and evidence path. It rejects
gaps and triple overlaps and uses smoothstep weights over the central 35–65%
of each real overlap.

The drive uses:

- a 2.8 m-wheelbase kinematic bicycle model;
- a pure-pursuit simulated-human controller;
- 12 m/s requested cruise speed;
- smooth centre/left/centre/right/centre targets inside the +/-1 m support;
- logged source-time progression derived from route progress;
- sequential checkpoint loading followed by offline overlap composition;
- front-centre RGB only at 20 fps.

Results:

| Measurement | Result |
| --- | ---: |
| Route length | 260.0 m |
| Frames / video duration | 494 / 24.70 s |
| Maximum speed | 12.00 m/s |
| Left / right excursion | +0.578 / -0.544 m |
| Minimum corridor reserve | 0.422 m |
| Maximum heading error | 1.771 degrees |
| Boundary hits | 0 |
| Endpoint brake | 0.711 normalized maximum |
| Rendered frames, tile 2 / 3 / 4 | 205 / 166 / 189 |
| Warm render p50, tile 2 / 3 / 4 | 35.90 / 32.57 / 33.32 ms |
| Maximum source-time endpoint clamp | 0.141 s |
| Decoded video frames | 494 / 494 |

A post-run frame diagnostic measured temporal RGB MAE p50/p95/max of
6.135/11.439/14.381 per 255. The inspected transition samples were not global
outliers. Both overlaps remain visually continuous enough to follow the road.

## Decision

Pass this as the first **long-route coverage and control pilot**:

- it is a genuine contiguous 260 m recorded route, not a loop;
- a real vehicle state and controller complete the route at 12 m/s;
- all three 8k checkpoints render sequentially;
- both real overlaps transition without a topology jump.

Do not pass it as equivalent-realism or a live scalable simulator:

- checkpoint loading and blending are offline, not live prefetch/eviction;
- only one front camera is rendered;
- dark/translucent vehicle ghosts can still look like false obstacles;
- foliage and nearby vehicles smear;
- the validated lateral tube remains +/-1 m;
- no collision truth or final LiDAR gate exists.

The data-volume slot is now large enough to stop adding tiles temporarily.
The next action is the agreed usability phase: evaluate the complete coverage
pack at matched held-out poses for road/free-space geometry, false obstacles,
driving decisions, and support-boundary honesty. Do not train tiles 0/1/5/6
until that evidence says the 260 m route is useful.

## Artifacts

- drive report:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_three_tile_drive_8000_20260728/tbv_three_tile_drive.json`;
- H.264 video:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_three_tile_drive_8000_20260728/tbv_260m_drive.mp4`;
- route contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_three_tile_drive_8000_20260728/route_contact.jpg`;
- seam contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_three_tile_drive_8000_20260728/seam_contact.jpg`;
- 8k tile-3/tile-4 seam:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_3_4_seam_8000_isolated_20260728`.

The report, video, route contact, seam contact, and 8k seam-report SHA-256
values are respectively:
`a137b564a5fcf20528180a34ac90a39390b45821bd4d401070e13c246307de23`,
`d248d9aaed72d41c0dcbb8912a09e2d248223810c36fd95a7919b34121557298`,
`5e34dd35b00913327e61f3b3ee4d26161e60a6801d32c1f7e08ab6f360e612da`,
`774412a6c4f112afddba0a1e306bbddb3a0e0de2f68ebc6910cc8e437acc5c65`,
and
`90e3f5bfd65d5ea208f2c8cbc398139d4f15668eb56125d9c69d4171115ff790`.

## Reproduce

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh download
scripts/run_stage_h3_tbv_long_route_tiles.sh freeze-2-3-data
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-4
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-3-4-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh static-4
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-3-4-8000
scripts/run_stage_h3_tbv_long_route_tiles.sh three-tile-drive-8000
```

Completed checkpoints and non-empty evidence directories are reused/refused
respectively unless an explicit new output is selected.
