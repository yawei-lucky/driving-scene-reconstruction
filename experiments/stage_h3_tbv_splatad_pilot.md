# Stage H3 TbV Multi-Traversal SplatAD Pilot

Date: 2026-07-22

Historical status at completion: the bounded two-traversal data, parser,
registration, 100-step smoke, and 2,000-step visual pilot passed. The model had
not yet been connected to a world-pose driving probe, so this record did not
claim a drivable-corridor acceptance.

Follow-up: the later world-pose probe and exact resume to 8,000 steps are
recorded separately in `stage_h3_tbv_world_pose_corridor_probe.md`. They do not
change the 2,000-step observations below.

## Outcome

The selected Miami TbV pair works in the existing `h3_splatad` environment.
No MTGS environment was needed. A single static SplatAD model now loads both
traversals in their common city frame, keeps traversal-specific appearance
IDs separate, saves and reloads checkpoints, and renders both the straight and
right-turn windows.

The 2,000-step result recovers recognizable roads, markings, buildings,
foliage, and parked vehicles. It is visibly soft and contains point-like
floaters and blurred/merged vehicle detail. This is enough to continue with a
counterfactual world-pose corridor probe, but not enough to claim unrestricted
driving or correct dynamic traffic.

## Data Window

Only the already reviewed ten-second intervals were downloaded:

```text
OCaNX1bQSmlP3jEQH80C0TZYzZhKLV81__Spring_2020
315972566.15 .. 315972576.15 seconds

QMnNKZiFaxnuGQmxpGkZFdM2EE7uWqDQ__Spring_2020
315968138.15 .. 315968148.15 seconds
```

The public per-object S3 layout produced:

```text
selected objects:       1,612
selected bytes:         277,728,040
camera images:          1,400 (100 x 7 x 2)
LiDAR sweeps:           200 (100 x 2)
camera sampling:        every other 20 Hz frame, independently per camera
```

Calibration, local map files, and the complete pose Feather were retained for
each log. The exact selection is outside Git at:

```text
/home/yawei/stage3_external/data/tbv_branch_pilot/selection_manifest.json
```

The downloader is resumable by expected object size:

```bash
scripts/run_stage_h3_tbv_pilot.sh download
```

## Parser Design

`scripts/stage_h3_tbv_dataparser.py` reuses the installed AV2 loader for camera
calibration and city-pose lookup, but does not reuse AV2 Sensor's dual-LiDAR
logic. Each TbV Feather is loaded as one already ego-motion-compensated
ego-frame aggregate sweep with zero relative point time. This avoids the known
incorrect `laser_number < 32` / `>= 32` split, which would make the second TbV
sensor empty.

The parser also:

- returns an empty actor set because TbV has no 3D object annotations;
- maps both windows to a common local 0-10 second interval while retaining
  their original Miami city poses;
- centres the combined scene only once;
- assigns 14 traversal-specific camera IDs and two traversal-specific LiDAR
  IDs, preventing velocity estimation or appearance lookup across a log
  boundary;
- uses a 0.9 linspace train split independently for each sensor.

The formal data gate observed:

```text
train cameras:             1,260
train LiDAR sweeps:        180
train LiDAR points:        18,133,002
local duration:            9.942499 s
camera sensor IDs:         14
LiDAR sensor IDs:          2
actor trajectories:        0
finite camera/LiDAR/point: yes
```

## Cross-Traversal Registration Gate

The gate selects LiDAR sweeps whose ego origins lie within 3 m of the other
route, keeps points at 3-55 m range and -3 to +6 m sensor height, samples every
fifth return, voxelizes at 0.20 m, and computes symmetric cross-traversal
nearest-neighbour distances. It is a static-dominant registration diagnostic,
not a semantic dynamic-object filter.

```text
shared-origin sweeps:        45 / 29
voxel points:                209,563 / 167,355
distance samples:            376,918
nearest-neighbour p50:       0.109414 m
nearest-neighbour p90:       0.240764 m
nearest-neighbour p95:       0.334617 m
fraction within 0.5 m:       0.977382
gate:                        PASS (p50 <= 0.20, p90 <= 0.50 m)
```

The machine-readable report is:

```text
/home/yawei/stage3_external/artifacts/tbv_branch_pair_data_gate.json
```

Regenerate it with:

```bash
scripts/run_stage_h3_tbv_pilot.sh data-gate
```

## Training And Reload Results

Both runs used neurad-studio commit
`e6f7e4e509b828a952d8584b7165f7844711ecb2`, PyTorch 2.0.1+cu118, the pinned
custom gsplat, and the RTX 4090 D.

### 100-step plumbing smoke

```text
downsample factor:    0.25
maximum seed points:  250,000
checkpoint step:      99
checkpoint bytes:     289,293,686
held-out RGB views:   140
render throughput:    25.08 views/s
PSNR / SSIM / LPIPS:  13.3220 / 0.5143 / 0.8298
finite saved images:  420 / 420 (RGB, reference, depth)
```

The output is colourful and blurry, as expected for a plumbing smoke. It is
not a visual candidate.

### 2,000-step visual pilot

```text
downsample factor:    0.5
maximum seed points:  750,000
checkpoint step:      1,999
checkpoint bytes:     887,205,110
final Gaussians:      2,672,901
held-out RGB views:   140 (70 per traversal)
render throughput:    24.35 views/s
PSNR / SSIM / LPIPS:  20.2621 / 0.6753 / 0.5193
finite saved images:  420 / 420 (RGB, reference, depth)
```

Per-traversal held-out means were:

```text
OCa...  21.2828 / 0.6918 / 0.4939
QMn...  19.2414 / 0.6588 / 0.5448
```

These values use this pilot's dataset and 0.9 per-sensor interpolation split;
they are not directly comparable with the PandaSet or Wayve evaluations. Peak
training VRAM and exact wall time were not recorded and are not claimed.

The checkpoints and renders remain outside Git:

```text
/home/yawei/stage3_external/outputs/tbv_h3/
/home/yawei/stage3_external/artifacts/tbv_branch_pair_smoke_100_render/
/home/yawei/stage3_external/artifacts/tbv_branch_pair_pilot_2000_render/
```

The first training attempt selected `/usr/bin/nvcc` from CUDA 11.5 and failed
before step 0 because it does not support `compute_89`. The TbV entrypoint now
exports the same CUDA 11.8 paths and architecture settings as the accepted
PandaSet H3 entrypoint; the unchanged retry completed. This was an environment
entrypoint error, not a TbV data or model failure.

## Decision And Next Gate

Continue with TbV before spending time on MTGS environment setup. The immediate
implementation should be a minimal seven-camera TbV world-pose renderer that:

1. loads the accepted 2,000-step checkpoint;
2. freezes one source time and moves the calibrated rig in the combined city
   frame;
3. selects the matching traversal-specific appearance IDs on the straight and
   right-turn branches;
4. samples the common approach, both observed branches, and bounded lateral
   offsets;
5. rejects the candidate if parked-car ghosts obscure the usable road or curb/
   building geometry doubles across passes.

Do not train longer before that counterfactual corridor test. If the 2,000-step
model already fails away from observed poses, more iterations will not answer
the coverage question.
