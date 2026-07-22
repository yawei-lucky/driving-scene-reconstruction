# Stage H3 External Multi-Trajectory Resource Probe

Date: 2026-07-22

Status: metadata and image review passed for one real straight/right branch
pair; no sensor-window extraction, parser adapter, or training run has started.

## Decision

Do not force the PandaSet `003+057` same-direction pair to carry the spatial
coverage goal. Use it only as a low-risk multi-log parser control if needed.

The first coverage-oriented pilot should use this Argoverse Trust, but Verify
(TbV) pair in Miami:

```text
OCaNX1bQSmlP3jEQH80C0TZYzZhKLV81__Spring_2020
QMnNKZiFaxnuGQmxpGkZFdM2EE7uWqDQ__Spring_2020
```

Both traversals share a northbound residential approach and then split: the
first turns right and the second continues straight. This is the first
verified common-intersection branch pair found for the project. It is not a
straight/left/right trio; no other TbV trajectory passes within 10 m of this
exact branch point in the metadata scan.

## Why TbV

The official TbV release contains 1,043 vehicle logs, averages about 54 seconds
per log, and provides seven panoramic ring cameras, two 32-beam LiDARs,
calibration, local maps, and 6-DOF ego pose in a global city coordinate system.
It has no object annotations. The full extracted release is about 922 GB, but
the public S3 layout exposes individual files, so candidate selection and a
small timestamp window do not require the full release.

This is a better fit for the immediate spatial question than the current
PandaSet archive:

- the logs are substantially longer;
- city-frame poses make cross-log trajectory screening direct;
- many logs revisit the same roads across seasons;
- the selected pair changes the available route at a real intersection;
- the two selected logs are both daylight `Spring_2020`, reducing the first
  pilot's appearance-domain difficulty.

The tradeoff is one small parser adapter and no 3D boxes for dynamic filtering.
The pinned Argoverse 2 parser already supplies most calibration, camera, LiDAR,
and city-pose logic, but TbV has a different directory root, no annotations,
and ego-frame aggregate LiDAR sweeps without per-return `offset_ns`. TbV's
`laser_number` values run from 0 through 31, so the existing AV2 parser's
`<32`/`>=32` upper/lower split cannot be reused: it would create an empty lower
sensor and send the points through the wrong transform path.

## Metadata-Only Scan

The new inventory tool enumerates the public S3 log prefixes, identifies the
city from each map object name, and downloads only
`city_SE3_egovehicle.feather`. It never downloads camera or LiDAR sequences.

```bash
/home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/analyze_stage_h3_tbv_trajectories.py --prepare \
  --candidate-plot \
  /home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/branch_candidates.png
```

Observed result:

```text
logs listed:                  1,043
pose files downloaded:       1,043
pose bytes:                   531,619,590
usable moving trajectories:  1,039
same-direction candidates:   990
branch-review candidates:    301
opposite-direction pairs:    168
```

The three candidate counts are heuristic and not mutually exclusive. They are
review queues, not 1,459 verified road relationships. The branch queue requires
at least 15 m of shared path, at least 15 m outside the overlap for both logs,
and a local heading change. Trajectory plots and source imagery are still
required before promoting a pair.

Raw city counts match the official distribution:

```text
ATX 80, DTW 139, MIA 349, PAO 21, PIT 318, WDC 136
```

Four logs are shorter than the scan's 5 m moving-track gate.

## Selected Pair Evidence

The selected pair measures:

```text
minimum centreline distance:            0.061 m
shared-path lengths within 3 m:          116.83 / 115.05 m
nearest-distance p50 in shared path:     0.319 m
heading-difference p50 / p95:            0.398 / 54.939 degrees
full-log route lengths:                  533.35 / 486.65 m
estimated full-log union route length:   904.94 m
```

The longest continuous matched runs are 117.62 m and 115.97 m. The first log
then turns east; the second continues north. Six front-centre frames at about
three seconds before the split, at the split, and three seconds after it show:

- the same tree-lined residential approach and matching permanent buildings;
- a real right turn in `OCa...` and a straight traversal in `QMn...`;
- broadly matched daylight and foliage;
- different parked and moving vehicles, so a shared static model must not
  assume that every observed car is persistent geometry.

This visual review verifies the route relationship. It does not verify
cross-traversal LiDAR registration or reconstruction quality.

Artifacts remain outside Git:

```text
/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/metadata/manifest.json
/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/trajectory_inventory.json
/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/branch_candidates.png
/home/yawei/stage3_external/artifacts/tbv_trajectory_inventory/review_ocan_qmnn/
```

## Minimum Pilot

Download only these ten-second intervals, each centred on the visually checked
split:

```text
OCa...  315972566.15 to 315972576.15 seconds
QMn...  315968138.15 to 315968148.15 seconds
```

The original six-second review crop was too short for the pilot gates. On the
pose scan these expanded windows cover 71.68/125.61 m total, 35.88/39.25 m
within 3 m of the other route, and 35.80/86.36 m outside that overlap. Thus the
window selection itself clears the 30 m shared and 25 m per-route non-shared
pose gates; static-LiDAR registration remains unverified.

Start with every second camera frame (10 Hz), all seven ring cameras, the 10 Hz
LiDAR sweeps, calibration, the local map, and the complete pose file. This is a
small straight/right branch smoke, not the final route-length experiment.

The smallest implementation is:

1. add a `tbv-data` adapter around the existing Argoverse 2 parser logic;
2. return an empty actor set, load each already ego-motion-compensated TbV sweep
   as one aggregate ego-frame LiDAR, use its timestamp as scan-centre time, and
   bypass AV2's dual-LiDAR split and missing-point transform paths;
3. parse each log in its original Miami city frame, crop by timestamp, merge,
   and centre only once around the intersection;
4. namespace camera and LiDAR sensor IDs by `(traversal, sensor)` so SplatAD's
   appearance embedding cannot mix the two passes;
5. refine the city-pose initialization with shared static LiDAR, while keeping
   the two branch-only regions;
6. disable actor training and run a 100-step, 0.25-scale save/reload/finite
   smoke before any quality run.

Pre-training gates:

- both windows load seven calibrated cameras and LiDAR without missing-time
  surprises;
- at least 30 m shared static overlap remains;
- static-LiDAR registration residual is at most 0.20 m p50 and 0.50 m p90;
- both routes extend at least 25 m beyond the common approach;
- all traversal-specific sensor IDs, timestamps, and poses are finite;
- no velocity is estimated across the log boundary.

After a passing smoke, run one matched 2,000-step single-log versus two-log
comparison. Render the common approach, the straight branch, the right branch,
and one counterfactual path that drives into each. Reject the result if parked
car ghosts obscure the road or if the shared curb/building geometry doubles.
Only then consider MTGS-style transient nodes or a broader time window.

## Other Useful Resources

| Resource | What is directly useful | Decision now |
| --- | --- | --- |
| AV2 Sensor | 1,000 annotated 15-second logs; seven ring cameras, LiDAR, city poses, maps; current H3 environment and parser already support it | Best parser/control fallback, but no branch pair has yet been verified |
| MTGS release | Released code, checkpoints, and six calibrated nuPlan multi-traversal blocks; shared geometry plus traversal-specific appearance/transient design | Use a smallest published block only as a method/format smoke, not as proof of branch coverage |
| MARS | 66 locations and 1.4M frames; location 24 is officially labelled `intersection, multiple direction`; nuScenes-style schema | Strong later benchmark, but the location-24 archive is roughly 150 GB in the current listing and does not safely fit the host's remaining 158 GB with extraction/training headroom |
| Boreas-RT | Repeated routes, 128-beam LiDAR, centimetre-level poses, granular public S3, CC BY 4.0 | Good opposite-direction geometry fallback, but its single front camera is a poor fit for the project's 360-degree human-driving view |
| Oxford RobotCar | More than 100 repeated routes and small GPS/VO packages, including alternate-route logs | Cheap independent branch probe, now lower priority because TbV already yielded a verified pair |

MARS's project repository and Hugging Face card expose different Creative
Commons labels. Treat the stricter Hugging Face `CC-BY-NC-ND-4.0` label as the
operative boundary unless the authors clarify it. TbV/AV2 and released MTGS
data are non-commercial research resources under CC BY-NC-SA 4.0. Boreas data
uses CC BY 4.0. Code licenses are separate from data licenses.

## Current Resource Order

1. TbV `OCa... + QMn...`: real two-branch pilot, already metadata/image
   verified, granular download.
2. PandaSet `003+057`: multi-log adapter/alignment control if TbV adapter work
   becomes ambiguous; do not call it a branch pilot.
3. AV2 Sensor: native-parser annotated fallback.
4. Small official MTGS block: method and checkpoint smoke only.
5. MARS location 24: later high-confidence multi-direction benchmark after
   storage is expanded or the archive can be streamed selectively.

## Primary Sources

Accessed 2026-07-22:

- TbV official user guide:
  <https://argoverse.github.io/user-guide/datasets/map_change_detection.html>
- Argoverse 2 code and dataset family:
  <https://github.com/argoverse/av2-api>
- Argoverse terms:
  <https://argoverse.github.io/user-guide/terms_and_conditions.html>
- MTGS official code and release status: <https://github.com/OpenDriveLab/MTGS>
- MTGS paper: <https://arxiv.org/abs/2503.12552>
- MARS official multitraversal page:
  <https://ai4ce.github.io/MARS/projects/multitraversal/>
- MARS official dataset card:
  <https://huggingface.co/datasets/ai4ce-drive/MARS>
- Boreas-RT paper: <https://arxiv.org/abs/2602.16870>
- Boreas/Boreas-RT devkit and data license:
  <https://github.com/utiasASRL/pyboreas>
- Oxford RobotCar official site:
  <https://robotcar-dataset.robots.ox.ac.uk/>
