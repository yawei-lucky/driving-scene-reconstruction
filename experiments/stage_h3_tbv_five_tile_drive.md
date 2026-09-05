# Stage H3 TbV Five-Tile 420 m Drive Pilot

Date: 2026-09-05

## Question

After the 260 m pilot proved technically usable but still felt short in an
actual tool trial, can the same real repeated Miami route be extended without
changing renderer, controls, or support claims?

The bounded extension adds tiles 5 and 6 to the accepted tiles 2/3/4:

- tile 2: route progress 160–260 m;
- tile 3: 240–340 m;
- tile 4: 320–420 m;
- tile 5: 400–500 m;
- tile 6: 480–580 m.

Their union is 420 m with four real 20 m overlaps. This is still only part of
the measured 610.072 m repeated route and must not be reported as the complete
route.

## Data And Reconstruction

Tiles 5 and 6 use separate immutable two-window data roots, avoiding the
camera-stride reproducibility problem previously found when overlapping
windows were appended to one shared directory:

- tile 5: 1,269 objects / 199,763,172 bytes;
- tile 6: 1,443 objects / 228,150,144 bytes.

Both unmasked SplatAD models trained to step 1,999 and then exactly resumed to
step 7,999. Periodic LiDAR evaluation was disabled during the resume, matching
the accepted tile-4 workaround for the 5-million-Gaussian evaluation peak.
Both resumes restored model, optimizer, scheduler, and global-step state and
completed on the RTX 4090 D.

The final checkpoints are:

- tile 5: 1,650,471,862 bytes, SHA-256
  `d8ceb78aeb87e512902f96af045e0873e5a9216125e66598f97b204dc3508adf`;
- tile 6: 1,650,482,998 bytes, SHA-256
  `fc2c557e31889d870d3274eff64572fbb8dc4ac2f8791eb2faa75dc0c83149b4`.

No final LiDAR evaluation is claimed.

## Generalized Drive Entry Point

The former fixed three-tile runner now accepts a repeatable
`ID,START,END,CONFIG,DATA_ROOT` tile specification. Legacy tile-2/3/4 arguments
remain supported. Route length, simulation time allowance, active render
indices, manifest, annotation, report name, and output video name are derived
from the supplied tile set.

The launcher now exposes independent download, 2k, 8k, four-tile, and
five-tile modes for the extension. Large data, checkpoints, frames, and videos
remain outside Git.

## Simulated Drive Result

The same kinematic bicycle model and simulated-human pure-pursuit controller
completed the 420 m union:

| Measurement | Result |
| --- | ---: |
| Route interval / length | 160–580 m / 420.0 m |
| Frames / video duration | 761 / 38.00 s |
| Requested / maximum speed | 12.0 / 12.0 m/s |
| Left / right excursion | +0.543 / -0.565 m |
| Minimum corridor reserve | 0.435 m |
| Maximum heading error | 1.301 degrees |
| Boundary hits | 0 |
| Endpoint brake | 0.711 normalized maximum |
| Decoded video frames | 761 / 761 |

All five checkpoints were step 7,999 and rendered every required frame. Warm
front-camera render p50 values for tiles 2/3/4/5/6 were respectively
34.87/31.64/32.61/32.15/31.32 ms. The maximum source-time clamp was 0.141 s.

The composed sequence measured temporal RGB MAE p50/p95/max of
6.281/11.857/15.811 per 255. Entering and leaving the new tile-5/tile-6 blend
measured 8.674 and 9.701 per 255, both below the full sequence p95. Manual
frame inspection retained the road, curb, buildings, and forward topology
through both new seams. Foliage softness, vehicle/point ghosts, and appearance
change between independent models remain visible.

## Decision

Pass this as a larger **offline route-coverage and simulated-control result**:

- genuine contiguous recorded road, not a loop or unrelated-scene splice;
- 420 m and 38 s at 12 m/s, versus the prior 260 m and 24.7 s;
- four overlap transitions with no support hit or road-topology break;
- a reusable N-tile runner instead of another fixed route script.

Do not pass it as equivalent realism or a mature large-scale simulator:

- checkpoint loading and composition are still offline;
- only the front-center camera is rendered;
- the support tube remains +/-1 m;
- static reconstruction retains traffic ghosts and has no collision truth;
- 420 m is still short of a minute at road speed and is not the full 610 m
  measured repeat.

Finishing tiles 0/1 would add only 160 m, or about 13 s at 12 m/s. It is a
bounded closure option, not the main answer to a future 1–2 km route target.

## Artifacts

- report:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_five_tile_drive_8000_20260905/tbv_tiled_drive.json`;
- H.264 video:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_five_tile_drive_8000_20260905/tbv_420m_drive.mp4`.

Their SHA-256 values are respectively
`68e1ea46fb27cb88ac6cdcf0e9dbfc99b8b97301057cd694ce528624618c0a1d`
and
`d0012b83e38e5d86b372b3caa722fb72bbb2be60b92a4b6e0ae9d5b7433650d8`.

## Reproduce

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh download-5
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-5
scripts/run_stage_h3_tbv_long_route_tiles.sh static-5
scripts/run_stage_h3_tbv_long_route_tiles.sh four-tile-drive-8000
scripts/run_stage_h3_tbv_long_route_tiles.sh download-6
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-6
scripts/run_stage_h3_tbv_long_route_tiles.sh static-6
scripts/run_stage_h3_tbv_long_route_tiles.sh five-tile-drive-8000
```

Completed checkpoints and non-empty evidence directories are reused/refused
respectively unless an explicit new output path is selected.
