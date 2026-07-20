# Stage H3 Scene 040 — Actor Alignment And Timing Audit

Date: 2026-07-20

## Question

Why did the moving-only actor model preserve all seven actor IDs but render
nearby cars as road- or wall-like blobs, and can one small correction improve
the result without another blind 8,000-step run?

## Independent Root-Cause Audit

Checkpoint-local analysis found that actor IDs survived while actor-local
geometry did not:

- at step 7,999, actor 0 had 26,477 Gaussians but only 2.29% were inside its
  padded cuboid; active-inside was 3.49%;
- actor 1 had 4,879 Gaussians, 0% inside, and 0% active-inside;
- their local-coordinate p95 magnitudes reached 34-290 m while their padded
  half-extents were only about 1.2 x 2.6 x 1.2 m;
- the failure was already present at step 2,000: actor 0 had 78.2% outside and
  actor 1 had 99.9% outside;
- trajectory optimization drift was much smaller: actor 0/1 translation p95
  was 0.336/0.123 m.

The pinned `ADMCMCStrategy` inherits positional noise injection on every step
but has no actor-bound handling. The alternative `ADDefaultStrategy` explicitly
prunes actor Gaussians outside `dynamic_actors.actor_bounds()`. Same-ID
relocation therefore preserved identity counts while actor-local points
random-walked tens to hundreds of metres away.

Evidence:

```text
/home/yawei/stage3_external/artifacts/scene_040_actor_alignment_audit_unconstrained/actor_alignment_audit.json
/home/yawei/stage3_external/artifacts/scene_040_actor_alignment_audit_unconstrained/actor_alignment_audit.jpg
```

## Spatial-Constraint Test

`ActorAwareMCMCStrategy` was extended to project actor-local means into each
padded cuboid after MCMC noise injection and clear optimizer moments for
projected points. A fresh moving-only 1,999-final-step run kept all seven
actors spatially contained.

```text
run:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_moving_constrained_2000/splatad/2026-07-20_actor_bounds

audit:
/home/yawei/stage3_external/artifacts/scene_040_actor_alignment_audit_constrained_2000
```

At step 1,999, all actors had approximately 100% of their Gaussians inside
their cuboids, but only actors 0, 1, and 5 had any opacity above 0.005. A
one-step diagnostic continuation triggered the scheduled step-2,000 MCMC
relocation. Actor 0 active fraction rose from 0.485 to 0.743 and actor 5 from
0.733 to 0.977, but actors 1, 2, 3, 4, and 6 remained at or below the active
threshold. This continuation initialized model weights from step 1,999 but
intentionally reset optimizer and scheduler state; it is a mechanism probe,
not an exact-resume quality comparison.

The fixed calibrated-reference crop evaluation of that step-2,000 checkpoint
reported:

| Measure | Result |
|---|---:|
| rendered views / finite failures | 480 / 0 |
| moving crops | 39 |
| moving-crop PSNR mean | 18.2289 |
| moving-crop LPIPS mean | 0.4882 |
| warmed render latency p50 / p95 | 21.73 / 23.39 ms |

```text
/home/yawei/stage3_external/artifacts/scene_040_moving_constrained_step2000_temporal/scene_040_temporal_report.json
```

Decision: **retain the spatial constraint as a correctness guard, reject the
checkpoint as a visual candidate**. It prevents catastrophic geometry escape
but does not solve missing actor supervision or opacity collapse. The last
post-MCMC constraint pass still projected 12,464 of 39,658 actor Gaussians
(31.4%) back inside; the time-corrected run projected 20,147 of 43,694
(46.1%). The hard boundary is therefore still opposing large positional noise
rather than producing a naturally stable actor model.

## Cuboid-Time Audit

The parser's legacy cuboid-time correction used
`sequence.lidar.poses`, even though the same upstream parser says those poses
are unreliable when loading LiDAR. It also applied that pose forward to world
coordinates instead of transforming world positions into the calibrated
LiDAR frame.

The corrected path reconstructs LiDAR-to-world from the synchronized front
camera and lidar-to-camera extrinsics, inverts it, and computes azimuth in the
resulting LiDAR frame. After excluding 69 `sensor_id==1` front-facing sibling
cuboids that `_cuboids_to_trajectories` skips, the 253 moving-cuboid
observations that enter training had this absolute corrected-minus-legacy time
difference:

| Percentile | Difference |
|---|---:|
| p50 | 46.973 ms |
| p90 | 55.852 ms |
| p95 | 56.108 ms |
| max | 56.363 ms |

The behavior is serialized explicitly by
`PandaSetVehicleObjectParserConfig.use_calibrated_lidar_frame_for_cuboid_time`.
Old checkpoints default to legacy semantics; new corrected-time checkpoints
remain reproducible rather than changing meaning when current code changes.
The crop evaluator always uses one calibrated annotation reference, independent
of checkpoint semantics.

A fresh boundary-plus-corrected-time run retained active Gaussians for five of
seven actors, but its visual result still failed:

| Measure | Result |
|---|---:|
| rendered views / finite failures | 480 / 0 |
| moving crops | 39 |
| moving-crop PSNR mean | 18.5608 |
| moving-crop LPIPS mean | 0.5331 |
| warmed render latency p50 / p95 | 21.42 / 22.97 ms |

```text
run:
/home/yawei/stage3_external/outputs/pandaset_h3/scene_040_splatad_moving_constrained_timed_2000/splatad/2026-07-20_actor_bounds_and_time_v2

evaluation:
/home/yawei/stage3_external/artifacts/scene_040_moving_constrained_timed_v2_2000_temporal
```

Decision: **retain the calibrated and explicitly serialized time correction,
reject the checkpoint**. The correction changes seed assignment and actor
survival, but it does not by itself recover clear vehicles.

The old 2,000-step report used the legacy crop reference and contained 40
moving crops, so its 18.8072 PSNR / 0.4639 LPIPS is directional context rather
than a strict paired comparison with the new 39-crop calibrated-reference
reports. The conclusions above do not rely on claiming a paired improvement.

## Visual Evidence

The comparison below contains GT/prediction vehicle crops for the legacy and
bounded 2,000-step runs. Geometry containment changes some individual cars but
does not consistently improve perceptual quality.

```text
/home/yawei/stage3_external/artifacts/scene_040_moving_constrained_2000_temporal/moving_actor_old_vs_bounded_2k.jpg
```

Six-camera and per-camera GT/prediction videos for the final time-corrected
candidate are under:

```text
/home/yawei/stage3_external/artifacts/scene_040_moving_constrained_timed_v2_2000_temporal
```

## Current Conclusion And Next Test

The audit separated three effects:

1. actor-local MCMC escape is the proven cause of the road/wall-scale blobs;
2. cuboid timing has a real roughly 50-56 ms systematic error and affects actor
   survival;
3. even after both corrections, several actors fade or remain perceptually
   wrong, so another 8,000/30,000-step run is not justified.

The next smallest high-value test is **per-point LiDAR seed timing and
assignment for actor 0/1**. The source point cloud carries roughly one scan of
per-point time, but initial actor assignment currently evaluates the whole
scan at its center timestamp. Project corrected per-point seeds into the exact
camera rows before training. Only if seed pixels align should a small
actor/window optimization be authorized.

After that diagnostic, add an `exists_at_time` render mask: the current SplatAD
render path discards trajectory presence and can render an actor at its nearest
pose outside its valid annotation/extrapolation interval. This is a separate
ghost-vehicle correctness issue, not the cause of the hundred-metre local
geometry escape.
