# Stage H3 PandaSet Multi-Trajectory Inventory

Date: 2026-07-22

Status: metadata pilot passed; multi-sequence reconstruction is designed but
not yet implemented or trained.

## Question

Can the PandaSet data already present on the project host provide another
traversal around scene 040, or a small same-intersection straight/left/right
set, for the first multi-trajectory SplatAD/MTGS-style pilot?

This is a candidate-selection experiment, not a reconstruction-quality result.

## Method

The scan read only GPS, timestamps, front-camera poses, and archive entry names
from the already verified 44.52 GB PandaSet ZIP. It did not extract all scenes,
alter the accepted scene-040 checkpoint, or run a GPU training job.

```bash
python3 scripts/analyze_stage_h3_pandaset_trajectories.py \
  --output-json /home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/trajectory_inventory.json
```

The script:

- loads all 103 `meta/gps.json` tracks directly from the ZIP;
- projects latitude/longitude into a shared local metric plane;
- measures pairwise nearest distance, bidirectional overlap, and local heading
  difference;
- fits each scene's front-camera local x/y poses to the GPS frame with one 2D
  rigid transform;
- records point-semantic availability without treating semseg as an image mask;
- optionally produces a five-frame front-camera contact sheet.

Default candidate thresholds are 5 m for repeated-path overlap, at least 20
overlapping samples in both trajectories, and 15 m for direction-change
review. A pair is sent to review when at least 20% of valid nearby heading
matches differ by more than 20 degrees, its p95 difference reaches 45 degrees,
or its median indicates opposite travel. Review candidates are not
automatically treated as verified branches.

GPS is only an initial candidate and registration signal. It cannot by itself
prove that a measured lateral separation is a second lane rather than
cross-session localization bias.

## Results

### Scene 040 cannot be expanded from this archive

Scene 040 has no neighbouring traversal in the existing PandaSet release. Its
nearest other tracks are:

```text
072: 165.337 m
039: 171.542 m
073: 327.732 m
```

Therefore no second sequence can be joined directly to the accepted 64.6 m
scene-040 corridor. Static-8k remains the accepted checkpoint for that corridor;
multi-trajectory work must use a separate pilot location.

### No same-intersection multi-direction set was verified

The broader direction-change diagnostic flagged one pair, `054+078`: minimum
GPS distance 1.883 m, heading-difference p50/p95 11.656/52.854 degrees, with
26.1% of valid nearby matches above 20 degrees. Its common 5 m section is only
about 6.4-6.5 m and scene 054 fails the initial pose/GPS fit gate. Trajectory
shape and front-camera samples show a daylight/night partial repeat of the
shared approach; scene 078 ends before it establishes a distinct second
outbound branch.

No opposite-direction candidate remains. The archive contains individual
turning clips, but not a verified common location with separate straight,
left-turn, and right-turn traversals. A stationary clip near a moving route is
not counted as multi-direction driving evidence.

This is a negative result under explicit GPS/direction thresholds, not a claim
that every image in PandaSet has been semantically mapped.

### Same-direction repeats do exist

Ten pairs passed the same-direction repeat gate. The strong pairs were recorded
about 2.84-2.93 hours apart. Contact-sheet inspection shows a consistent domain
change: the first traversal is daylight and the repeat is dark/nighttime.

The near-coincident `032+070` pair is the simplest parser/alignment control:

```text
overlap samples within 5 m: 78 / 76
overlap length:             66.12 / 66.49 m
nearest-distance p50:       0.320 m
heading-difference p50:     0.603 degrees
estimated union length:     70.40 m
```

It verifies multi-traversal loading but does not materially expand camera-centre
coverage.

## Selected Expansion Pilot: 003 + 057

`003+057` is the smallest useful coverage pilot because it has a substantial
shared registration segment, an offset trajectory, new route length on both
sides, and point semantics in both scenes:

```text
minimum GPS distance:                     2.342 m
overlap samples within 5 m:               39 / 30
overlap length:                           38.80 / 38.02 m
overlap nearest-distance p50 / p95:       2.991 / 3.571 m
heading-difference p50 / p95:             1.509 / 2.196 degrees
time between traversal starts:            2.844 hours
scene lengths:                            75.24 / 90.65 m
estimated shared-route union length:      127.87 m
point-cloud semantic annotations:         both scenes
front-pose to GPS rigid-fit p95 residual: 0.112 / 0.118 m
```

Visual inspection confirms that both front cameras show the same broad road,
with scene 003 in daylight and scene 057 at night. It also shows real traffic
and parked-car differences, so the static pilot cannot treat all observed
content as persistent geometry.

The machine-readable report and contact sheet from this run are outside Git:

```text
/home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/trajectory_inventory.json
/home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/front_contact_sheet.jpg
/home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/direction_review_054_078.jpg
```

## Minimum SplatAD / MTGS-Style Pilot

The next implementation should remain a repo-local SplatAD extension. MTGS is
not installed, and this result does not claim an MTGS integration.

Use only the shared slice `003[0:39] + 057[50:80]` first. It is 69 timestamps,
414 six-camera images, and 69 Pandar64 sweeps. The complete two-scene extraction
is still below 1 GiB, so there is no reason to extract other candidates yet.

The minimum adapter should:

1. parse each sequence independently with the pinned PandaSet parser;
2. fit the front-camera local poses into one GPS-derived ENU frame, then
   recenter the selected pair near its shared-section centroid before float32
   model input;
3. remove cuboid/semantic dynamic points in the shared section and refine the
   initial alignment with static LiDAR returns;
4. transform camera, LiDAR, trajectory, and world-velocity fields into the
   common frame before merging parser outputs;
5. disable actor training for the first shared-static-background result;
6. namespace sensor IDs by `(traversal, sensor)` so SplatAD has separate
   appearance embeddings for the daylight and night traversals;
7. use deterministic per-traversal holdouts and report each traversal
   separately.

The existing upstream parser accepts one `sequence: str` and assumes 80 frames.
Concatenating directories or pretending there is one 160-frame sequence would
break coordinate, timestamp, sensor-index, and split semantics.

### Pre-training gates

- static-LiDAR registration residual at most 0.20 m p50 and 0.50 m p90;
- at least 30 m valid shared static overlap;
- aligned centreline separation remains between about 1.5 and 4 m;
- all merged poses, timestamps, velocities, and sensor IDs are finite and
  internally consistent;
- no cross-traversal endpoint is used to estimate vehicle velocity.

If LiDAR registration collapses the apparent separation below 1 m, demote
`003+057` to a repeated-route control rather than claiming expanded coverage.

### Training gates

Run a 100-step static SplatAD smoke first at 0.25 data scale and the existing
250,000-point seed cap. It only needs to save, reload, and render finite held-out
views from both traversals. It is not a visual-quality result.

Only after that passes, run the existing 2,000-step pilot budget with:

- per-traversal held-out PSNR, SSIM, and LPIPS;
- shared-section static-LiDAR residuals;
- three shared stations rendered from the scene-003 line, the midpoint, and
  the scene-057 line;
- explicit inspection for double roads, duplicated curbs, holes, and exposure
  leakage;
- warmed six-camera latency.

The scene-040 static-8k checkpoint remains unchanged and accepted throughout.

## Decision

The present PandaSet archive can support a small same-direction multi-traversal
experiment, but it cannot expand scene 040 and does not contain a verified
multi-direction intersection set. Proceed with `003+057` only as a shared
static-background and spatial-coverage pilot. Keep `032+070` as the low-baseline
alignment control if the first adapter result is ambiguous.
