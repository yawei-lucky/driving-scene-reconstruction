# Stage H3 MTGS Published-Block Probe

Date: 2026-07-22

Status: all six released road-block trajectories were inspected from official
metadata. One block is now a credible normal-road driving pilot. No complete
data block, checkpoint payload, image sequence, environment, GPU model load, or
training run was downloaded or executed.

## Corrected Question

An intersection branch is useful but not required for the next simulator
increment. A longer straight or gently curving road is sufficient if it gives
the human driver a continuous observed corridor and, ideally, wider lateral
coverage than the current PandaSet scene-040 result.

Under that criterion, the MTGS release is not merely a method smoke. Its
smallest released Singapore block is a real pilot candidate.

## Evidence Collection

The audit used only official release material:

- cloned the MTGS repository at commit
  `7ab67a3e386e5a4830017819324922a7bb9f26f7`;
- read the six paper configurations and official train/eval traversal lists;
- downloaded the 35,935,479-byte `nuplan_log_infos.jsonl` index;
- fetched only the first 4 MB of five data archives and a partial prefix of the
  sixth, because `video_scene_dict_registered.pkl` is the first archive member;
- extracted 2.4-7.5 MB of registered trajectory metadata per block;
- loaded the pickles with a restricted unpickler that allowed only NumPy,
  `datetime.date`, and `pyquaternion.Quaternion` constructors;
- indexed the uncompressed checkpoint tar through 512-byte HTTP range reads;
- fetched only 88,469 bytes of the selected checkpoint's config, transform,
  training log, and evaluation JSON.

This avoided downloading the 35.7 GB data release or 6.18 GB checkpoint tar.
Temporary audit material remains under `/tmp/mtgs-route-audit.2AdgZC` and is
not project evidence that must survive a reboot; the measurements below are
the durable record.

## Six Released Blocks

Route length uses registered unskipped ego XY samples. Bend is the difference
between smoothed start and end trajectory tangents. The labels describe
trajectory geometry only, not visual reconstruction quality.

| Road block | City | Compressed data | Traversals | Route length | Direct trajectory observation |
| --- | --- | ---: | ---: | ---: | --- |
| `331220_4690660_331190_4690710` | Boston | 4.33 GB | 6 | 76-80 m | nearly straight, two directions, several parallel paths |
| `365000_144000_365100_144080` | Singapore | 3.98 GB | 3 | 84-87 m | all same direction, continuous gentle 11-16 degree bend |
| `365530_143960_365630_144060` | Singapore | 10.7 GB | 4 | 99-105 m | mostly straight and bidirectional; one traversal makes an 8.3 m excursion |
| `587400_4475700_587480_4475800` | Pittsburgh | 6.02 GB | 4 | 83-84 m | nearly straight, two direction groups |
| `587640_4475600_587710_4475660` | Pittsburgh | 5.21 GB | 6 | 79-89 m | nearly straight, two direction groups and parallel paths |
| `587860_4475510_587910_4475570` | Pittsburgh | 5.42 GB | 6 | 57-75 m | nearly straight, two direction groups and parallel paths |

All six metadata files expose the same eight cameras:

```text
CAM_F0, CAM_L0, CAM_L1, CAM_L2, CAM_R0, CAM_R1, CAM_R2, CAM_B0
```

## Selected Block

Promote `road_block-365000_144000_365100_144080` from method smoke to the
first MTGS normal-road candidate.

Its three traversals are:

| Traversal | Usable frames | Duration | Route | Bend | Role in official paper task |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 80 | 7.9 s | 84.3 m | 10.9 degrees | evaluation |
| 4 | 116 | 11.5 s | 86.6 m | 15.7 degrees | training |
| 5 | 74 | 7.3 s | 83.7 m | 13.3 degrees | training |

The official two-traversal task trains on 4 and 5 and evaluates on 3. This is
especially useful for the simulator question:

- training traversals 4 and 5 are 5.17 m apart at nearest-point p50 and 7.42 m
  at p95;
- traversal 3 lies between them, at 2.36 m p50 from traversal 4 and 2.87 m p50
  from traversal 5;
- traversal 3 has 55.7 m and 49.1 m of path within 3 m of traversals 4 and 5;
- all three travel in the same direction and retain eight valid cameras at
  every accepted frame;
- the complete registered metadata contains 270 ego frames and 2,160 camera
  images.

This is a stronger first test of a wider observed driving corridor than a pair
of perfectly coincident repeat runs. It still does not prove that arbitrary
poses between the three tracks render cleanly.

The block is not an empty-road case. Each traversal contains 21-27 unique
annotated tracks and averages 9.5-14.7 boxes per frame across vehicles,
pedestrians, and bicycles. The prepared masks, boxes, instance point clouds,
and MTGS dynamic nodes are therefore useful, but visual review must reject
false obstacles or actor ghosts before driving acceptance.

## Released Checkpoint

The selected 30,000-step MTGS checkpoint is a single 773,241,943-byte member
inside the 6.18 GB uncompressed checkpoint tar. Because the outer archive is
plain tar, that member plus its 13 KB config and small metadata can be fetched
by byte range without downloading the other five checkpoints.

The official evaluation JSON reports:

```text
all evaluation images: PSNR 25.629, SSIM 0.783, LPIPS 0.233
unseen traversal 3:    PSNR 22.658, SSIM 0.673, LPIPS 0.259
reported render FPS:   145.53
```

These are official MTGS evaluator results, not measurements on this host. The
FPS is not an eight-camera browser observation and cannot be compared directly
with the project's Renderer latency. The training log spans about 73 minutes
for 30,000 steps, but does not identify the GPU model.

## Runtime Fit

The data is a good fit; the official training environment is not yet a proven
fit:

- MTGS asks for Python 3.9, Nerfstudio 1.1.5, gsplat 1.4.0, NumPy 1.26.3, and
  tyro 0.8.4;
- the accepted H3 environment uses the neurad-studio fork, gsplat 1.0.0, NumPy
  1.24.4, and tyro 1.0.15;
- installing MTGS into the H3 environment would risk the accepted SplatAD
  setup, so it needs a separate environment;
- the official installation guide says training may require at least 40 GB of
  GPU memory, while this host has 24 GB;
- checkpoint inference may use materially less memory, but that has not been
  measured.

Therefore do not start with training or a SplatAD data conversion. The cheapest
decision gate is checkpoint-only inference in an isolated environment.

## Revised Minimum Pilot

1. Confirm extraction headroom, then download only the 3.98 GB Singapore data
   block and byte-range extract its 773 MB checkpoint and small config files.
2. Create a separate MTGS environment; do not modify `h3_splatad`.
3. Load the checkpoint on the RTX 4090 D and record startup time and peak VRAM.
4. Render one front-camera frame, then all eight cameras at 0.5 scale from an
   observed pose; require finite output and no out-of-memory failure.
5. Render five stations along traversal 3 and offsets toward traversals 4 and
   5. Reject doubled static geometry, broken road surface, and traffic ghosts
   that change the driving decision.
6. Only after those gates, connect the checkpoint through a separate Renderer
   backend and test the approximately 84 m gentle curve as a human-driven
   corridor.

If checkpoint inference does not fit 24 GB or the visual corridor fails, stop
the MTGS path without retraining. Resume the already selected TbV/SplatAD pilot.

## Decision

The corrected resource order is now:

1. MTGS smallest Singapore block: cheapest existing-checkpoint gate for an
   approximately 84 m, eight-camera, multi-trajectory gentle curve;
2. TbV `OCa... + QMn...`: best SplatAD-native data candidate and still the only
   visually verified branch pair;
3. PandaSet `003+057`: same-direction parser/alignment control.

This is a gate order, not a claim that MTGS has replaced SplatAD or already
works on the project host.

## Primary Sources

Accessed 2026-07-22:

- MTGS official repository: <https://github.com/OpenDriveLab/MTGS>
- official data blocks:
  <https://huggingface.co/datasets/OpenDriveLab/MTGS/tree/main/MTGS_paper_data>
- official checkpoints:
  <https://huggingface.co/datasets/OpenDriveLab/MTGS/tree/main/MTGS_paper_ckpts>
- official data preparation guide:
  <https://github.com/OpenDriveLab/MTGS/blob/main/docs/prepare_dataset.md>
- official installation guide:
  <https://github.com/OpenDriveLab/MTGS/blob/main/docs/install.md>
- official running guide:
  <https://github.com/OpenDriveLab/MTGS/blob/main/docs/running.md>
- MTGS paper: <https://arxiv.org/abs/2503.12552>
