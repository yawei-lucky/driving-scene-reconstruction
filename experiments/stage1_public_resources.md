# Stage 1 Public Resources

Last updated: 2026-07-02

## 1. Execution Goal

Run the first real experiment using openly available autonomous-driving resources and mature open-source reconstruction tools.

The first-stage loop is:

```text
public dataset
→ mature open-source reconstruction code
→ train or run baseline
→ render reconstructed / novel views
→ inspect visual and driving-relevant failures
```

The priority is experimental output, not building a custom system from scratch.

## 2. Resource Eligibility Decision

### Primary path: PandaSet + neurad-studio

Use this first because it is the strongest match for the requirement that resources be open / openly available.

| Resource | Role | Availability / license status | Stage-1 decision |
|---|---|---|---|
| PandaSet | Dataset | Official site describes it as an open-source AV dataset for academic and commercial use. HuggingFace mirror lists `cc-by-4.0`. | Use first |
| neurad-studio | Model codebase | Public GitHub repo, Apache-2.0. Official code for NeuRAD and SplatAD. | Use first |
| SplatAD | 3DGS-based AD renderer | Included in neurad-studio. Designed for camera + lidar rendering in AD scenes. | First baseline |
| NeuRAD | NeRF-based AD renderer | Included in neurad-studio. Dynamic AD neural rendering baseline. | Second baseline |
| Nerfstudio | Framework dependency | Public GitHub repo, Apache-2.0. neurad-studio builds on it. | Dependency / fallback |
| PandaSet devkit | Dataset reader | Public GitHub repo with documented camera, lidar, pose, timestamp, cuboid access. | Optional utility |

### Secondary path: WayveScenes101 + Nerfstudio / Splatfacto

WayveScenes101 is highly relevant for novel-view synthesis because it provides held-out off-axis evaluation cameras and Nerfstudio integration. However, its dataset license is non-commercial. Therefore:

```text
WayveScenes101 = research-only secondary track, not the strict open-source first path.
```

Use it after the PandaSet / neurad-studio path is running, or when the task specifically needs held-out off-axis evaluation.

## 3. Verified Resource Inventory

### PandaSet

Source URLs:

```text
https://pandaset.org/
https://huggingface.co/datasets/georghess/pandaset
https://github.com/scaleapi/pandaset-devkit
```

Useful facts:

```text
- 100+ scenes
- 8 seconds per scene
- 6 cameras
- mechanical spinning lidar
- forward-facing lidar
- GPS / IMU
- 3D cuboid annotations
- semantic segmentation annotations
- HuggingFace package size: about 44.5 GB
```

Stage-1 local target:

```text
/data/external/driving_scene_reconstruction/datasets/pandaset
```

### neurad-studio

Source URL:

```text
https://github.com/georghess/neurad-studio
```

Useful facts:

```text
- Apache-2.0 license
- official code release for NeuRAD and SplatAD
- supports PandaSet quickstart
- includes `splatad` and `neurad` training methods
- supports multiple AD datasets, including PandaSet, nuScenes, Argoverse 2, KITTIMOT, and Waymo v2
```

Stage-1 local target:

```text
/data/external/driving_scene_reconstruction/code/neurad-studio
```

### Nerfstudio

Source URL:

```text
https://github.com/nerfstudio-project/nerfstudio
```

Useful facts:

```text
- Apache-2.0 license
- provides training, viewer, rendering, and evaluation CLI
- provides Splatfacto and Nerfacto baselines
```

### WayveScenes101

Source URL:

```text
https://github.com/wayveai/wayve_scenes
```

Useful facts:

```text
- codebase is MIT licensed
- dataset license is non-commercial
- 101 scenes
- 20 seconds per scene
- 5 synchronized cameras
- COLMAP-format camera poses
- held-out off-axis evaluation camera
- simple Nerfstudio integration
```

Stage-1 decision:

```text
Keep as secondary research-only track.
```

## 4. First Execution Commands

Fetch open resources:

```bash
bash scripts/fetch_stage1_resources.sh --all-open
```

This should clone:

```text
neurad-studio
nerfstudio
pandaset-devkit
```

and download:

```text
georghess/pandaset:pandaset.zip
```

Run the first baseline:

```bash
bash scripts/run_stage1_pandaset_neurad_studio.sh --method splatad
```

Fallback baseline:

```bash
bash scripts/run_stage1_pandaset_neurad_studio.sh --method neurad
```

## 5. Expected Local Output

Do not commit these directories:

```text
/data/external/driving_scene_reconstruction/
outputs/stage1_pandaset_neurad_studio/
```

Expected local artifacts:

```text
outputs/stage1_pandaset_neurad_studio/
  run_commands.sh
  environment.txt
  train.log
  render_help.txt
  eval.json              # if ns-eval succeeds
  result_summary.md
  failure_notes.md
```

## 6. First Result Criteria

The first successful result is:

```text
PandaSet sequence
+ SplatAD or NeuRAD trained with neurad-studio
+ at least one rendered output or viewer-confirmed reconstruction
+ notes on visual failures and dynamic-object failures
```

A stronger result is:

```text
SplatAD result
+ NeuRAD result
+ side-by-side comparison
+ short judgment on which baseline is better for driving-scene view extrapolation
```

## 7. Failure Notes Template

Use this template after the first run:

```text
case_id:
dataset:
sequence:
method:
training_steps:
render_type:
main_success:
main_failures:
  - blur:
  - ghosting:
  - geometry_distortion:
  - object_disappearance:
  - dynamic_object_popping:
  - occlusion_error:
  - lane_or_road_boundary_error:
acceptance_decision: accept / down-weight / reject
next_best_action:
```

## 8. Current Status

```text
status: ready_to_download_and_run
primary_dataset: PandaSet
primary_codebase: neurad-studio
primary_method: splatad
fallback_method: neurad
secondary_dataset: WayveScenes101 research-only
```
