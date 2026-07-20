# Stage H3 Scene 040 — Seed Projection And Painting Audit

Date: 2026-07-20

## Question

Does assigning moving-actor LiDAR seeds at each point's own timestamp, rather
than at the scan centre, improve source alignment enough to authorize another
actor training run?

## Reproducible Audit

The audit uses the accepted 0.9 temporal split, calibrated cuboid timing, raw
Pandar64 world points, actor 0/1, PandaSet point-cloud semantics, and the
back-camera source images. It compares:

1. actor membership at the front-camera/scan-centre timestamp;
2. membership at each point's retained `pc[:, 4]` timestamp;
3. current fixed-camera seed painting with point-time actor propagation and
   rolling-shutter row-time projection.

The point timestamps remain float64 until the front-camera epoch is removed.
The observed offsets are required to remain inside +/-60 ms. Back-camera
projection uses the parser's actual 820-row crop and `H - 1` row normalization.
PandaSet semantic class 13 is the independent vehicle-point signal. Source
overlays are qualitative because the sequence has no dense 2D vehicle masks.

```text
script:
/home/yawei/driving-scene-reconstruction/scripts/audit_stage_h3_seed_projection.py

report and overlays:
/home/yawei/stage3_external/artifacts/scene_040_seed_projection_audit_v3
```

The sampled train/held-out pairs were frames 18/19, 38/39, 58/59, 68/69, and
77/78. Actor 0/1 are only visible in the back camera and occupy small image
regions; actor 1 has non-empty held-out evidence only at frames 19 and 39.

## Assignment Result

| Split | Method | Assigned points | Vehicle semantic precision |
|---|---|---:|---:|
| train | scan centre | 392 | 87.76% |
| train | per point | 360 | 90.28% |
| held out | scan centre | 369 | 89.16% |
| held out | per point | 349 | 87.97% |

The train subset improves, but the independent held-out subset loses 20 usable
points and 1.19 percentage points of semantic precision. Both source overlays
still land on the same small distant vehicles and have no consistent visible
improvement.

Decision: **do not change training seed semantics and do not train a new
checkpoint**. A train-only gain is insufficient when the held-out direction is
negative.

## Painting Result

For the same exact-assignment points, the second audit compares the current
fixed-pose projection used by `paint_points` with point-time actor propagation
and rolling-shutter row-time projection.

| Split | Comparable points | weighted frame median / p95 displacement | RGB L1 / 255 |
|---|---:|---:|---:|
| train | 360 | 0.573 / 2.597 px | 9.13 |
| held out | 349 | 0.375 / 1.547 px | 8.75 |

This confirms a real implementation approximation, but not a demonstrated
quality failure: most changes are subpixel to roughly 2-3 pixels, RGB
difference measures disagreement rather than correctness, and the overlays do
not show a repeatable background-to-vehicle correction. The audit also covers
the back-camera branch rather than the datamanager's complete top-k
multi-camera overwrite order.

Decision: **do not expand this into a cross-datamanager painting rewrite yet**.
The benefit is small and directionally unproven.

## Independent Review

The independent reviewer verified:

- world/local and local/world rotation directions;
- PandaSet OpenCV z-forward projection;
- the 820-row back-camera crop and `H - 1` rolling-shutter formula;
- raw point/semantic row alignment after the Pandar64 mask;
- the seed half-extent convention used by SplatAD.

The reviewer independently rejected both training authorization and a painting
rewrite. The accepted static 8k remains the only H3 visual checkpoint.

## Final Ten-Second Result

No stronger checkpoint was produced in this cycle. The final result is
therefore an exact 10-second, 80-frame, 8 FPS packaging of the accepted static
8k six-camera evaluation:

```text
/home/yawei/stage3_external/artifacts/scene_040_temporal_gate_8000/scene_040_final_accepted_static8k_10s.mp4

duration: 10.000000 seconds
resolution: 1920 x 720
frames: 80
```

## Next Discriminating Test

Actor 0/1 are too small and back-camera-only to represent the general dynamic
layer. Before another training run, repeat the same semantic/projection audit
on a larger actor visible in a front or side camera, then require held-out
semantic precision and visible source alignment to improve in the same
direction. Long or full-scene training remains blocked.
