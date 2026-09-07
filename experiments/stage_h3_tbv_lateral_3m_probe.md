# Stage H3 TbV 420 m Large-Lateral Diagnostic

Date: 2026-09-07

## Question

Does the five-tile TbV route remain visually smooth and structurally plausible
when the simulated vehicle makes full-lane-scale lateral movements, even though
the observed-data support claim remains only +/-1 m?

This is intentionally a stress test, not an attempt to enlarge the declared
drivable boundary. The existing 420 m, five-checkpoint route and 12 m/s
simulated-human controller are unchanged. Only the lateral programme changes
from approximately +/-0.55 m to +/-3 m. A separate +/-4 m mechanical envelope
lets the diagnostic finish without relabelling any `SceneTile` support field.

## Run

The project launcher used the pinned CUDA 11.8 environment and the five
step-7,999 static checkpoints:

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh five-tile-lateral-probe-8000
```

The first attempt completed the five tile renders but the root filesystem
filled while saving composed JPEGs. That failed condition was not treated as a
GPU or renderer failure. Only the incomplete run directory and the project's
re-downloadable pip cache were removed; data and checkpoints were preserved.
The clean rerun completed.

## Motion And Decode Result

| Measurement | Result |
| --- | ---: |
| Route / speed | 420.0 m / 12.0 m/s |
| Frames / duration | 761 / 38.05 s |
| Left / right peak | +2.984 / -2.985 m |
| Frames beyond trusted +/-1 m | 264 / 761 (34.69%) |
| Maximum support exceedance | 1.985 m |
| Maximum heading error | 4.586 degrees |
| Maximum normalized steering | 0.0595 |
| Boundary hits | 0 |
| Decoded frames | 761 / 761 |

The result therefore proves that the control/render pipeline can request and
produce continuous counterfactual poses at full-lane scale. It does not prove
that the reconstructed content at those poses is correct.

## Smoothness And Visual Review

Playback and ego motion are broadly smooth:

- the video stays at 20 fps with all 761 frames decoded;
- there are no black frames, transport corruption, hard camera cuts, or road
  topology teleportation;
- the cosine lateral programme reaches its left peak near 11.95 s and right
  peak near 22.70 s without a control discontinuity;
- temporal RGB MAE rises moderately from the +/-0.55 m baseline
  p50/p95/maximum of 6.281/11.857/15.811 to
  6.714/12.655/17.314 per 255.

Content plausibility does not pass at +/-3 m:

- on the left sweep, perspective changes continuously, but road texture,
  parked vehicles, wires, and foliage stretch or soften more than in the
  baseline;
- around 19.5–22.8 s at the right peak, an unobserved foreground traffic/tree
  region becomes a large dark malformed occluder, while a sign appears to
  float above it;
- entering the tile-4/tile-5 blend at 22.55–23.05 s produces RGB MAE 13.854 at
  frame 451, versus 5.667 in the small-offset baseline. The malformed
  foreground changes rapidly as the independent tile begins contributing;
- the tile-5/tile-6 transition remains continuous, but trees and local
  brightness visibly breathe/morph.

The practical verdict is **smooth camera motion, unreliable wide-offset
content**. The main failure is spatial coverage and hidden-surface appearance,
not frame delivery or vehicle-control smoothness. Keep +/-1 m as the current
TbV acceptance boundary. Full-lane movement needs additional laterally offset
observations or a scene whose real traversals cover adjacent lanes; renderer
extrapolation alone is not enough.

## Videos

All outputs are under:

`/home/yawei/stage3_external/artifacts/tbv_long_route_five_tile_lateral_3m_20260907`

- raw full probe, 37,247,704 bytes:
  `tbv_420m_lateral_3m_probe.mp4`;
- full baseline-versus-probe comparison, 17,437,060 bytes:
  `tbv_420m_lateral_comparison.mp4`;
- left sweep, 7.5–13.5 s, 3,024,843 bytes:
  `clip_a_left_sweep.mp4`;
- right sweep plus tile-4/tile-5 transition, 19–26 s, 3,965,076 bytes:
  `clip_b_right_sweep_and_tile_4_5.mp4`;
- tile-5/tile-6 transition, 28.5–30.8 s, 1,385,809 bytes:
  `clip_c_tile_5_6_seam.mp4`.

The SHA-256 values of the raw probe and full comparison are respectively
`662c340b263d39867409b901f21a2edc27304e5b89827f00c8f0958d9f7a44f1`
and
`f5ee97a63007d58fa49dee6473dd68ebbdd77d065d9922b890d800d75c58b2cf`.
The machine-readable run report is `tbv_tiled_lateral_probe.json`, SHA-256
`e58c2ab6fa36b55a6966807dcc3d01b79509a1e9e7067c66229005e6255b70c9`.
