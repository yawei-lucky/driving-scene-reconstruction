# Stage H3 TbV World-Pose Corridor Probe

Date: 2026-07-22

Status: the 8,000-step checkpoint passes a restricted front-view corridor
candidate gate over the shared entrance, straight branch, right-turn branch,
and +/-1 m offsets. It is ready for a small continuous human-driving trial,
not certified 360-degree driving or autonomous-driving evaluation.

## Question And Decision

The 2,000-step two-traversal pilot had already shown recognizable observed
views. This follow-up asked the narrower driving question: when one calibrated
seven-camera rig is moved through a common world frame, does the reconstruction
remain usable on the shared entrance, both route branches, and small lateral
departures?

The answer is now **yes for a route-constrained first human trial**. The probe
covers 60 m from the first common station to the far straight station and 50 m
to the far right-turn station. The two sampled branch endpoints are about 41 m
apart in the common frame, so this is real branched spatial coverage rather
than two appearance IDs rendering the same centreline. Do not train beyond 8k
before the continuous trial. If baked vehicles become false obstacles, the
next change must address transient actors or data/model composition rather
than add more static iterations.

## Renderer And Pose Gate

`examples/stage_h3_tbv_world_pose_probe.py` is deliberately experiment-local.
It:

- loads one joint SplatAD checkpoint once;
- combines the train and evaluation camera records into 100 synchronized
  samples for each of seven cameras and two traversals;
- preserves traversal-specific camera sensor and appearance IDs;
- places the two calibrated rigs in one branch-local metre-scale world frame;
- freezes model scene time and zeros source rolling-shutter motion metadata;
- selects the last shared-route region before divergence, retreated by 2 m;
- renders every station at -1, 0, and +1 m lateral offset.

The selected route anchor has 0.8388 m cross-traversal distance and 13.173
degrees heading difference. Forty-two matched route samples cover 31.629 m.
At the three common stations, corresponding route centres differ by only
0.460-0.554 m. Available route coverage relative to the anchor is:

```text
right turn:  -29.885 .. +41.206 m
straight:    -55.657 .. +68.682 m
```

The bounded probe uses:

```text
common entrance:  -20, -10, -2 m on both traversals
straight branch:   +5, +20, +40 m
right branch:      +5, +15, +30 m
lateral offsets:   -1, 0, +1 m
```

That is 36 seven-camera observations and 252 camera renders per checkpoint.
There is no ground truth for the counterfactual lateral views; the automated
gates check finite output, pose plumbing, non-black front views, and actual
pixel changes, while the driving verdict remains visual and task-specific.

## 2,000-Step Probe

The first probe loaded:

```text
/home/yawei/stage3_external/outputs/tbv_h3/
  tbv_branch_pair_splatad_pilot_2000/splatad/
  2026-07-22_train90_seed750k/nerfstudio_models/step-000001999.ckpt
```

All technical gates passed. At 0.5 output scale, the seven-camera observation
latency was 39.82 ms p50 and 44.47 ms p95. The lateral front-view mean absolute
pixel difference was 21.97/28.98 on a 0-255 scale at p50/p95.

The visual coverage result was positive: the shared entrance, +40 m straight
station, +30 m right-turn station, and +/-1 m offsets all retained the road and
major permanent structure without obvious cross-traversal curb or building
doubling. The quality result failed driving acceptance: point-like floaters and
dark splats entered the road, side/rear views were granular, and vehicles were
blurred or baked into the static scene. This justified one bounded longer run;
it did not yet justify browser integration.

## Exact Resume To 8,000 Steps

The accepted H3 environment and neurad-studio commit
`e6f7e4e509b828a952d8584b7165f7844711ecb2` ran on the project host's RTX 4090 D.
The intended operation was an exact model/optimizer/scheduler/global-step
resume from step 1,999 for 6,000 additional iterations.

The first one-shot command was terminated by the execution channel with
SIGTERM (exit 143) at roughly step 3,800. It was not a CUDA or out-of-memory
failure. Only step 2,000 had been saved, so the unsaved work was excluded.
The accepted chain restarted from step 1,999 and used recoverable 1,000-step
segments:

```text
1999 -> 2999 -> 3999 -> 4999 -> 5999 -> 6999 -> 7999
```

Every segment restored model, optimizer, scheduler, and global-step state.
The source checkpoint format does not preserve RNG or dataloader state, so the
result is not bitwise identical to an uninterrupted run. The final model
reached the configured 5,000,000-Gaussian cap and produced:

```text
checkpoint step:   7,999
checkpoint bytes:  1,650,493,750
```

The final checkpoint and exact-resume audit are outside Git at:

```text
/home/yawei/stage3_external/outputs/tbv_h3/
  tbv_branch_pair_splatad_static_8000/splatad/
  2026-07-22_resume_2k_to_8k/
```

## Held-Out Result

Both checkpoints were evaluated on the same 140 held-out RGB views: 70 per
traversal, ten per camera and traversal, using the pilot's 0.9 per-sensor
linspace split. These numbers are only comparable within this TbV experiment.

| checkpoint | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| 2k | 20.2621 | 0.6753 | 0.5193 |
| 8k | 23.2130 | 0.7734 | 0.3805 |
| change | +2.9509 | +0.0980 | -0.1389 |

The 8k per-traversal means were:

```text
OCa right turn:  24.3192 / 0.7981 / 0.3482
QMn straight:    22.1068 / 0.7486 / 0.4128
```

All 140 views rendered at 22.69 views/s in the measured `ns-render` run. The
metric file is:

```text
/home/yawei/stage3_external/artifacts/
  tbv_branch_pair_static_8000_render/metrics.pkl
```

## 8,000-Step World-Pose Result

The identical 36-pose probe again passed every automated gate. At 0.5 output
scale:

```text
finite camera renders:                 252 / 252
seven-camera observation latency p50:  58.48 ms
seven-camera observation latency p95:  67.70 ms
lateral front pixel MAD p50 / p95:     20.89 / 29.87
```

Agent visual review found a material improvement over 2k:

- the shared entrance is coherent for both traversal namespaces and all three
  lateral offsets;
- straight +5/+20/+40 m retains a readable road, curb, and forward route;
- right +5/+15/+30 m retains the turn, lane markings, curb, and buildings;
- the large road-surface floaters seen at 2k are substantially removed;
- all seven directions are recognizable at the sampled centre poses.

The remaining boundary is important. Tree crowns and close parked vehicles
still deform; a dark warped vehicle/vegetation region remains near the far
straight view; and the white van and other vehicles are baked into static
geometry because TbV provides no actor annotations. The lane itself stays
open in the sampled front views, but this has not yet been exercised as a
continuous keyboard-to-display drive. It therefore passes only the restricted
front-corridor candidate gate.

## Reproduction And Artifacts

The stable one-shot entry points are:

```bash
scripts/run_stage_h3_tbv_pilot.sh world-pose-probe
scripts/run_stage_h3_tbv_pilot.sh static-8k
scripts/run_stage_h3_tbv_pilot.sh render-static-8k
scripts/run_stage_h3_tbv_pilot.sh world-pose-probe-8k
```

They reuse completed artifacts unless the corresponding explicit rerun flag is
set. The probe reports and visual sheets are outside Git:

```text
/home/yawei/stage3_external/artifacts/tbv_branch_pair_world_pose_probe/
/home/yawei/stage3_external/artifacts/
  tbv_branch_pair_static_8000_world_pose_probe/
```

The machine-readable 8k report is
`tbv_world_pose_probe.json`; the primary visual evidence is
`common_approach_front.jpg`, `straight_branch_front.jpg`,
`right_branch_front.jpg`, the `seven_camera/` mosaics, and
`tbv_branch_world_routes.jpg`.

## Next Gate

Keep this 8k checkpoint fixed. The next implementation should be a minimal
route-constrained TbV driving adapter: spawn at the -20 m common station,
clamp the initial trial to +/-1 m, offer straight versus right-turn routing at
the shared anchor, and record continuous backend plus physical
keyboard-to-display latency. Do not describe the result as safe 360-degree
driving, and do not ask an operator to compensate for a reconstruction defect.
