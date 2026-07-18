# Stage H3 PandaSet Data Foundation

Date: 2026-07-18

Status: acquisition audit complete; download and data loading not started.

## Purpose

This record covers the H3-0B gate before any large download or training run:
current source provenance, license, exact archive metadata, extraction budget,
per-sequence packaging, and annotation coverage.

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

## Gate Decision

Passed:

- exact source revision, file size, hash, and range capability recorded;
- archive entry count, scene count, extraction size, and semantic coverage
  measured;
- current disk can hold the archive, one selectively extracted scene, and a
  constrained pilot with about 222 GiB remaining before outputs;
- license text exists inside the mirrored archive and matches the CC BY 4.0
  label with additional terms.

Not yet passed:

- explicit acceptance of the dataset terms and large download;
- full-file checksum validation;
- sequence visual/dynamic-density triage;
- one-frame camera/LiDAR/cuboid load;
- timestamp and calibration validation;
- camera/LiDAR overlays;
- SplatAD one-batch or <=100-step smoke.

## Recommended Acquisition Path

Use the full pinned archive rather than adding an untested remote-ZIP extractor
to the critical path. After explicit license/download approval:

1. download once to `/home/yawei/stage3_external/artifacts`;
2. verify byte size and SHA-256 before extraction;
3. extract only the selected `pandaset/<scene>/` directory to
   `/home/yawei/stage3_external/data`;
4. keep only one raw and one processed pilot scene;
5. triage the 76 semseg-capable scenes first, but select on visual and dynamic
   criteria;
6. start with six cameras, Pandar64, fused poses, and cuboid actor tracks;
7. stop after one-frame and calibration-overlay gates if alignment is wrong;
8. run no more than 100 training iterations for the first end-to-end smoke.
