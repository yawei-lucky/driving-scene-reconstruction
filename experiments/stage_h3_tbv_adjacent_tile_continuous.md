# Stage H3 TbV Adjacent-Tile Continuous Seam Gate

Date: 2026-07-25

## Question

After the 100/500-step integration pilot, can the selected TbV tile-2/tile-3
pair support a quality-bearing transition through its real 20 m overlap at
12 m/s, including small lateral camera offsets?

This is still a two-tile gate. It does not test all seven route tiles, live
checkpoint residency, physical human input, or dynamic-traffic truth.

## Implementation

The long-route wrapper now exposes:

- independent 2,000-step training for tiles 2 and 3;
- exact 2,000-to-8,000-step resume for only those two checkpoints;
- matched observed-pose seam probes at 2,000 and 8,000 steps;
- a continuous overlap probe with hard switching, smooth blending, and
  `-1/0/+1 m` front views.

The continuous probe independently loads each checkpoint and keeps its own
dataparser transform and time coordinate. It finds 22 exact shared
`ring_front_center` frames, interpolates corresponding camera poses in each
model, and renders the same 18.75 m city-frame path at 12 m/s and 20 fps.
Positive lateral offset means route-left.

The models are loaded sequentially. The two videos are offline evidence, not a
live checkpoint-streaming implementation.

## Training

Both 2,000-step models were trained from scratch with the same two traversals
and parser windows as the 500-step pilot. Only this pair was then resumed to
step 7,999. The exact-resume audits confirm restoration of model, optimizer,
scheduler, and global-step state at training step 2,000. The source checkpoint
does not preserve RNG or dataloader state, so the continuation is not bitwise
equivalent to one uninterrupted run.

| Tile | Step | Checkpoint bytes | SHA-256 |
| --- | ---: | ---: | --- |
| 2 | 1,999 | 562,507,318 | `2a7a6d896a0f64308f2e5e9e6f6f4aeb5c85fba21c29add3e742387c0b81670e` |
| 3 | 1,999 | 562,489,654 | `fc8d274d484d9ed15fa7be235fb5ab1fa4c8555fd913a7b029e0c13dd2c953c6` |
| 2 | 7,999 | 1,650,505,270 | `f68ddc2e7af418ddfcaeafbbda3739f1ba6714cba4942a47516771bdfe3abd12` |
| 3 | 7,999 | 1,650,487,606 | `afbcb7f55948594f401ba45ef106649b6681276d54bc56239c7af716577f3c57` |

Exact-resume audit SHA-256, tile 2 / tile 3:
`0f17378fe361564a477de88ca5486af6d24b1d2ded6f1e14e8b9c81089106761`,
`c63cc3643afb74b3f0e00423e0a51aca09b8c19874e9e2b8cad758ccc277ed17`.

## Results

### Matched observed poses

Five exact reference-traversal poses were rendered through both checkpoints.

| Measurement | 500 steps | 2,000 steps | 8,000 steps |
| --- | ---: | ---: | ---: |
| Tile-2 PSNR p50 | 19.873 dB | 23.527 dB | 24.059 dB |
| Tile-3 PSNR p50 | 20.902 dB | 22.774 dB | 24.845 dB |
| Tile-2/tile-3 RGB MAE p50 | 18.037 / 255 | 9.835 / 255 | 9.420 / 255 |
| Pairwise pixel-error p95 p50 | 45 / 255 | 30 / 255 | 30 / 255 |
| Warm render p50, tile 2 / tile 3 | 25.23 / 25.49 ms | 26.57 / 28.53 ms | 30.83 / 37.46 ms |

The first tile-2 render includes cold-start work. These timings are diagnostic,
not an end-to-end display-latency acceptance result.

### Continuous overlap

Both models produced 32 finite frames over corresponding 18.7522/18.7524 m
paths. Their inferred path lengths differ by only 0.00022 m. Both videos
decoded all 32 frames.

| Measurement | 2,000 steps | 8,000 steps |
| --- | ---: | ---: |
| Model-to-model RGB MAE p50 | 9.977 / 255 | 9.499 / 255 |
| Model-to-model pixel-error p95 p50 | 34.45 / 255 | 30.00 / 255 |
| Hard-output temporal MAE p50 / p95 | 5.771 / 10.451 | 4.893 / 9.436 |
| Blend-output temporal MAE p50 / p95 | 5.409 / 9.051 | 4.893 / 8.109 |
| Hard transition-frame delta | 11.109 / 255 | 10.834 / 255 |
| Blended same-frame delta | 4.949 / 255 | 4.635 / 255 |

The smoothstep transition spans 35-65% of the overlap, or 5.626 m. The
`-1/0/+1 m` probes at overlap start, middle, and end were all finite and not
mostly black.

## Manual Visual Decision

At 8,000 steps, both tiles retain the same road surface, lane layout,
buildings, utility poles, overhead wires, and horizon through the transition.
There is no observed coordinate or topology jump. The `-1/0/+1 m` views remain
navigable, and the mid-blend frame does not introduce an obvious double
building edge. This passes the static-background geometry seam.

It does not pass complete driving-scene truth. Vehicles, dark floating blobs,
and local sharpness differ between the independently trained tiles. More
training improves observed-pose PSNR but reduces model-to-model RGB MAE by
only about 5% from 2,000 to 8,000 steps. A central vehicle-shaped smear can be
read as a false obstacle, while pixel blending merely softens the transition.

Therefore:

- retain the two 8,000-step checkpoints as the adjacent-tile geometry
  regression;
- do not spend on the other five tiles yet;
- do not claim a 610 m drive or credible dynamic traffic;
- run one minimal transient-suppression/static-background comparison on this
  exact pair before building the approximately 180 m two-tile auto-drive.

## Failure Preserved

The first 2,000-step continuous run rendered 31 tile-2 frames and then rejected
the final query. At AV2's approximately `3.16e17 ns` timestamps, IEEE-754
rounding placed the last interpolated timestamp tens of nanoseconds beyond the
exact endpoint. This was a probe bug, not a model failure.

The incomplete output was preserved at:

`/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_2000_failed_float_endpoint_20260725`

The probe now clamps only endpoint discrepancies within 128 ns and rejects a
1,000 ns out-of-range query. Dependency-light tests cover both cases.

## Artifacts

2,000-step:

- observed-pose JSON/contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_2000_20260725`;
- continuous JSON/videos/lateral contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_2000_20260725`.

The 2,000-step continuous JSON, hard video, blend video, and lateral contact
SHA-256 values are:
`a81ec149d78b294f85a5813b869d1d46594ff43c4fd21257503c266bbdb602a8`,
`af2b229824f4d5cf688f8ea839dcfaadd895faa1f748e5c9e31a23d28c0a1892`,
`70a421bd34ae6f00cd53350e8b31e8292690c0fa72971a8433578af5a2c979b6`,
and
`83de7aeb3553a121d13f774dfd927451054506e084607ef826e7a3520d6f2b13`.

8,000-step:

- observed-pose JSON/contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_seam_8000_20260725`;
- continuous JSON:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_8000_20260725/tbv_tile_continuous.json`;
- hard-switch video:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_8000_20260725/hard_switch.mp4`;
- smooth-blend video:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_8000_20260725/smooth_blend.mp4`;
- lateral contact:
  `/home/yawei/stage3_external/artifacts/tbv_long_route_tile_continuous_8000_20260725/lateral_minus1_0_plus1_contact.jpg`.

The 8,000-step observed JSON/contact and continuous JSON/hard/blend/lateral
SHA-256 values are:
`0f08392c904b9c15ced66c18c86eb6d1a644858dc0566a040e85b01a51ee4c2f`,
`957bb7c48323f968a0511fe5d9514739365639115d40c306aa8112f7a84e4c27`,
`e0eca65c1295ee74b3b0c688a4221bb5ae01be99082df496d68247ed22f9107a`,
`ded4a75ee6a6eac08abaaac59f6b704b669aa70c2c24cb50410f795efbd3ecc2`,
`4e4b29ce9f9ac167ad211a82f32864b51454d3189643bb41db5f76c3af2ea4d7`,
and
`321fc69452c6ccf80e86c08d4e6ee257979e56e44153744469cd9006c1f2b187`.

## Reproduce

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-2
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh continuous-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh static-2
scripts/run_stage_h3_tbv_long_route_tiles.sh static-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-8000
scripts/run_stage_h3_tbv_long_route_tiles.sh continuous-8000
```

Completed checkpoints are reused unless `H3_ALLOW_RETRAIN=1` is explicitly
set. Evidence directories refuse non-empty overwrite.
