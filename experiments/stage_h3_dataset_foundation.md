# Stage H3 PandaSet Data Foundation

Date: 2026-07-19

Status: acquisition, integrity, one-scene extraction, data loading, and
calibration gates passed for scene 040.

## Purpose

This record covers the H3-0B gate before any large download or training run:
current source provenance, license, exact archive metadata, extraction budget,
per-sequence packaging, and annotation coverage.

## Execution Result

The user explicitly accepted the recorded CC BY 4.0 plus additional PandaSet
terms and the 44.5-GB acquisition on 2026-07-19.

The pinned archive was downloaded to:

```text
/home/yawei/stage3_external/artifacts/pandaset-e2e123aea3b3132c67f4b395ec6120f63e190271.zip
```

Validated result:

```text
size: 44,520,528,731 bytes
SHA-256: 6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e
unzip integrity test: no errors
```

The long resumable transfer was interrupted by the execution environment and
the local file contained 25,044,133 appended bytes after the valid ZIP end.
This was not silently accepted. Fixed-source range hashes were compared at
offsets from the beginning through the final 528,731 bytes, the ZIP64 end
records were located at the audited expected boundary, and only the proven
appended tail was truncated. The resulting exact-size file then matched the
recorded full SHA-256 and passed `unzip -tq`.

All 76 semseg-capable scenes were triaged by ego motion, cuboid density, and
near-field dynamic density, followed by contact-sheet inspection. Scene 149
ranked well numerically but was rejected because it is dark and blurred.
Scene 040 was selected because it is daylight, has stable exposure, clear road
and curb structure, and usable six-camera coverage.

Only scene 040 was extracted:

```text
/home/yawei/stage3_external/data/pandaset/040
744 files
411 MiB on disk
```

The reproducible data/calibration inspection is:

```bash
scripts/run_stage_h3_pandaset_040.sh data-gate
```

It loaded 80 frames for every one of the six cameras, both LiDAR streams,
poses, timestamps, cuboids, GPS, and point-cloud semantics. The accepted
baseline uses Pandar64 only. Timestamp offsets relative to the front camera
range from about 10.8 ms for the front-side cameras to about 50.0 ms for the
back camera and LiDAR. These offsets are expected asynchronous acquisition and
must remain visible in later temporal modeling.

Calibration evidence:

```text
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_data_report.json
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_six_camera_keyframes.jpg
/home/yawei/stage3_external/artifacts/scene_040_calibration/scene_040_pandar64_camera_overlay.jpg
```

Projected static Pandar64 returns align plausibly with roads, building faces,
trees, poles, curbs, and parked vehicles at frames 0, 40, and 79. No systematic
extrinsic failure blocks the Level 1 smoke. Exact measurements and the
subsequent training result are in `stage_h3_scene_040_smoke.md`.

## Current Sources

The official PandaSet page still describes 100+ eight-second scenes, six
cameras, Pandar64, PandarGT, GPS/IMU, 3D boxes, and point-cloud segmentation.
Its visible download link returned HTTP 404 during this audit.

The pinned neurad-studio README directs users to the Hugging Face dataset
`georghess/pandaset`. That repository is a community mirror: its card explicitly
states that the uploader is not affiliated with the PandaSet creators. It
currently contains one complete `pandaset.zip`, not per-sequence archives.

Primary/source pages inspected on 2026-07-18:

- PandaSet official page: <https://pandaset.org/>
- PandaSet devkit: <https://github.com/scaleapi/pandaset-devkit>
- neurad-studio: <https://github.com/georghess/neurad-studio>
- neurad-linked mirror: <https://huggingface.co/datasets/georghess/pandaset>

The mirror is usable for research, but its community provenance and single-file
packaging must remain visible in experiment reports.

## Exact Mirror Revision

Hugging Face metadata returned:

```text
repository commit: e2e123aea3b3132c67f4b395ec6120f63e190271
file: pandaset.zip
size: 44,520,528,731 bytes
LFS SHA-256 oid: 6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e
Xet hash: 89e5c639d294ae849ba026653cc1e839a1b31ac10497547cea56da6d46d57607
HTTP range support: yes
```

The source API was:

```text
https://huggingface.co/api/datasets/georghess/pandaset/tree/main
```

Any eventual download must use the recorded repository revision and validate
the 44,520,528,731-byte file against the LFS SHA-256 oid before extraction.

## Archive Audit Without Data Download

Only the ZIP end record and 9,090,487-byte central directory were read through
HTTP byte ranges. No image or point-cloud payload was acquired.

Observed archive structure:

```text
entries: 75,758
scenes: 103
compressed entry payload: 44,504,071,771 bytes
uncompressed entry payload: 44,732,715,419 bytes
scenes with point-cloud semantic labels: 76
frames per scene: 80
```

The archive content is already dominated by JPEG and gzip-compressed files, so
full extraction barely changes its size. Keeping the archive and all 103
extracted scenes simultaneously would require approximately:

```text
89,253,244,150 bytes = 83.12 GiB
```

This 83.12-GiB figure is a worst case, not the pilot requirement. Scene
compressed payloads range from 357,266,306 to 473,762,546 bytes and extracted
payloads range from 359,497,852 to 475,771,588 bytes. Standard ZIP tools can
extract one `pandaset/<scene>/` directory after the full archive is downloaded.
The archive plus the largest one-scene extraction therefore needs at most:

```text
44,996,300,319 bytes = 41.91 GiB
```

Against the 264 GiB free-space observation, this leaves approximately 222.09
GiB before processed data, checkpoints, logs, caches, and renders. The standard
mirror still requires the full ZIP download. HTTP range extraction could avoid
that download, but there is no accepted repository tool for it; building an
unverified custom extractor now would add data-corruption and reproducibility
risk to the critical path.

Scenes with point-cloud semantic annotations:

```text
001 002 003 005 011 013 015 016 017 019 021 023 024 027 028 029
030 032 033 034 035 037 038 039 040 041 042 043 044 046 052 053
054 056 057 058 064 065 066 067 069 070 071 072 073 077 078 080
084 088 089 090 094 095 097 098 101 102 103 105 106 109 110 112
113 115 116 117 119 120 122 123 124 139 149 158
```

This list is a filter for scene triage, not a reason by itself to select a
scene. Actor density, near-field occlusion, road visibility, lighting, and
camera overlap still require data inspection.

## License Gate

A 2,179-byte compressed `LICENSE.txt` entry was read directly from scene 001
without acquiring sensor payloads. It states that the dataset is provided under
CC BY 4.0 unless labeled otherwise, with additional dataset terms. Those terms
include attribution, privacy/non-identification restrictions, restrictions on
using licensor names or marks, termination provisions, indemnification, and
legal compliance obligations.

Downloading or using the dataset constitutes agreement to those terms,
including when acting for an organization. The repository must not silently
accept them on the user's behalf. Explicit approval is required before the
44.5-GB acquisition begins.

## Audit-Era Gate Decision

Passed:

- exact source revision, file size, hash, and range capability recorded;
- archive entry count, scene count, extraction size, and semantic coverage
  measured;
- current disk can hold the archive, one selectively extracted scene, and a
  constrained pilot with about 222 GiB remaining before outputs;
- license text exists inside the mirrored archive and matches the CC BY 4.0
  label with additional terms.

At the time of the read-only audit, these items were not yet passed:

- explicit acceptance of the dataset terms and large download;
- full-file checksum validation;
- sequence visual/dynamic-density triage;
- one-frame camera/LiDAR/cuboid load;
- timestamp and calibration validation;
- camera/LiDAR overlays;
- SplatAD one-batch or <=100-step smoke.

All listed acquisition, data, calibration, and Level 1 smoke items were
subsequently completed on 2026-07-19. This wording is retained to distinguish
the original pre-download decision from the later execution result.

## Executed Acquisition Path

The approved run followed the audited path instead of adding an untested
remote-ZIP extractor:

1. downloaded the pinned archive once to
   `/home/yawei/stage3_external/artifacts`;
2. verified exact byte size, SHA-256, and ZIP integrity before extraction;
3. triaged the 76 semseg-capable scenes using annotations and contact sheets;
4. extracted only `pandaset/040` to
   `/home/yawei/stage3_external/data/pandaset/040`;
5. kept the raw archive, one extracted pilot scene, and bounded outputs;
6. used six cameras, Pandar64, fused poses, and cuboid actor tracks;
7. passed the full data and calibration-overlay gates;
8. limited the first end-to-end SplatAD run to 100 iterations.
