# Stage H3 TbV Long-Route Tile Audit

Date: 2026-07-25

## Question

Can an existing public TbV route provide enough repeated, contiguous coverage
for a real long-route reconstruction pilot, without looping one short scene or
concatenating unrelated road blocks?

This is a data and alignment gate. It does not train a reconstruction or claim
that the route is already drivable.

## Selected Route

The complete TbV pose inventory selected two Miami Spring 2020 traversals:

- reference:
  `V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020`;
- repeat:
  `cTrSOEc1gW3XqELP562UlUFJYCmlRoa9__Spring_2020`.

Their full pose-track lengths are 640.934 m and 666.906 m. The longest
continuous same-direction supported run is 610.072 m and contains 521
downsampled pose samples. Support requires:

- nearest cross-traversal distance at most 3 m;
- heading difference at most 15 degrees;
- non-decreasing nearest repeat-trajectory indices.

Measured over the supported run:

| Measurement | p50 | p95 | maximum |
| --- | ---: | ---: | ---: |
| Nearest traversal distance | 0.440 m | 0.878 m | 2.598 m |
| Heading difference | 0.298° | 2.404° | 4.107° |

No nearest-index regression occurs inside the accepted run. This establishes a
long, repeated, same-direction route. It does not establish adjacent-lane
coverage, branching, or a safe `±4–5 m` lateral envelope.

## Tiling Contract

The supported run yields seven complete 100 m tiles with 20 m overlaps and an
80 m stride:

```text
tile 0:   0–100 m
tile 1:  80–180 m
tile 2: 160–260 m
tile 3: 240–340 m
tile 4: 320–420 m
tile 5: 400–500 m
tile 6: 480–580 m
```

Tiles 2 and 3 are the selected first adjacent pair. Their shared interval is
240–260 m:

| Tile | Samples | Support | Distance p50 / p95 / max | Heading p95 |
| --- | ---: | ---: | --- | ---: |
| 2, 160–260 m | 103 | 100% | 0.434 / 0.877 / 0.949 m | 3.627° |
| 3, 240–340 m | 88 | 100% | 0.265 / 0.643 / 0.712 m | 0.819° |

Both tiles pass the metadata gate. The pair is genuinely adjacent in one
continuous physical route; it is not a synthetic join.

## Source-Image Review

Only eight `ring_front_center` source images were downloaded: both traversals
at route progress 160, 240, 260, and 340 m. Their camera-to-pose deltas are
7.5–22.5 ms.

Manual review of the contact sheet found corresponding permanent evidence at
all four stations: road direction, intersections, towers, buildings, power
lines, and palms. The 240 m and 260 m overlap stations are recognizable across
both traversals, and illumination is close enough for an appearance-aware
two-traversal pilot.

Vehicles differ between traversals. The pilot must mask or otherwise separate
transient traffic instead of learning all observed vehicles as one static
background. The source review is evidence for route identity and appearance,
not a reconstruction-quality result.

## Bounded Payload Plan

The exact tile-2/tile-3 plan includes both traversals, all seven ring cameras at
stride 2, every LiDAR sweep in the four time windows, calibration, maps, and
pose files:

- 2,805 unique objects;
- 424,193,048 bytes;
- four 7.54–10.57 s traversal/tile windows.

The full payload was listed but not downloaded. No training was performed.
This preserves a cheap stop point before the adjacent-tile seam is proven.

## Artifacts

- JSON:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_audit_20260725_v3/tbv_long_route_tile_audit.json`;
- JSON SHA-256:
  `9d107f34d34179a3bb2aa7c8b086d46a649d28220ea0ba7b9f22d9d8ee2c5397`;
- route/tile plot:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_audit_20260725_v3/long_route_tiles.png`;
- source contact sheet:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_audit_20260725_v3/adjacent_tile_source_contact_sheet.jpg`;
- contact-sheet SHA-256:
  `108d4bb165bf9cf6219b413ec2c7519d6f7e158aa446bd2eaedce2baa85f5e34`.

Reproduce the audit in the accepted H3 environment:

```bash
env MPLCONFIGDIR=/tmp \
  /home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/audit_stage_h3_tbv_long_route_tiles.py \
  --download-source-images \
  --plan-tile-payload
```

The default output directory is intentionally non-overwriting. Choose a new
`--output-dir` when repeating the command.

## Decision And Next Gate

Pass the long-route metadata and adjacent-data gate. The next bounded
experiment is:

1. download only the planned 424.2 MB payload;
2. reuse the existing TbV SplatAD parser and train separate 100-step smoke
   checkpoints for tiles 2 and 3;
3. render matched world poses through their 20 m overlap;
4. inspect geometry, appearance, traffic ghosts, and the image transition at
   the real seam.

Do not start 2,000-step training, all-seven-tile reconstruction, checkpoint
streaming, or long-route simulated driving until this one adjacent seam passes.
