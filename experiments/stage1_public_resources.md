# Stage 1 Public Resources

Last updated: 2026-07-02

> **Historical Stage 1 planning record.**
>
> The resource choices and imperative wording below document the Stage 1
> decision at that time; they are not current execution instructions. See the
> repository root `README.md` and `PROJECT_STATE.md` for current priorities.

## 1. Execution Goal

Run the first real experiment using public / openly available autonomous-driving resources and mature open-source reconstruction tools.

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

The project should distinguish three things:

```text
open-source code: software repository license, such as MIT or Apache-2.0
public research dataset: dataset is downloadable / usable for research, but may be non-commercial
commercial-friendly dataset: dataset terms are broader and more product-friendly
```

The previous version over-penalized WayveScenes101 because its dataset license is non-commercial. That was too strict for the current research stage. WayveScenes101 is public, directly designed for autonomous-driving novel-view synthesis, and should be the first experimental path.

### Primary path A: WayveScenes101 + Nerfstudio / Splatfacto

Use this first for view extrapolation and held-out/off-axis evaluation.

| Resource | Role | Availability / license status | Stage-1 decision |
|---|---|---|---|
| WayveScenes101 code | Dataset tooling | Public GitHub repo, MIT license | Use first |
| WayveScenes101 dataset | NVS / reconstruction dataset | Publicly downloadable, non-commercial research license | Use first for research experiments |
| Nerfstudio | Reconstruction framework | Public GitHub repo, Apache-2.0 | Primary framework |
| Splatfacto | 3DGS baseline | Included in Nerfstudio | First baseline |
| Nerfacto | NeRF-style baseline | Included in Nerfstudio | Fallback baseline |

### Primary path B: PandaSet + neurad-studio

Run this in parallel or immediately after the first WayveScenes101 result.

| Resource | Role | Availability / license status | Stage-1 decision |
|---|---|---|---|
| PandaSet | Dataset | Official site describes it as an open-source AV dataset for academic and commercial use. HuggingFace mirror lists `cc-by-4.0`. | Use as commercial-friendly / multi-sensor track |
| neurad-studio | Model codebase | Public GitHub repo, Apache-2.0. Official code for NeuRAD and SplatAD. | Use as dynamic AD baseline |
| SplatAD | 3DGS-based AD renderer | Included in neurad-studio. Designed for camera + lidar rendering in AD scenes. | First PandaSet baseline |
| NeuRAD | NeRF-based AD renderer | Included in neurad-studio. Dynamic AD neural rendering baseline. | Second PandaSet baseline |
| PandaSet devkit | Dataset reader | Public GitHub repo with documented camera, lidar, pose, timestamp, cuboid access. | Optional utility |

## 3. Verified Resource Inventory

### WayveScenes101

Source URL:

```text
https://github.com/wayveai/wayve_scenes
```

Useful facts:

```text
- codebase is MIT licensed
- dataset is publicly downloadable from Google Drive through the repo
- dataset license is non-commercial research use
- 101 scenes
- 20 seconds per scene
- 101,000 images
- 5 time-synchronized cameras
- 10 Hz camera stream
- COLMAP-format camera poses / calibration
- separate held-out evaluation camera for off-axis reconstruction measurement
- simple Nerfstudio integration
```

Stage-1 local target:

```text
/data/external/driving_scene_reconstruction/datasets/wayve_scenes_101
```

Stage-1 decision:

```text
Use first for novel-view synthesis and off-axis view-extrapolation experiments.
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

Stage-1 local target:

```text
/data/external/driving_scene_reconstruction/code/nerfstudio
```

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

## 4. First Execution Commands

Fetch the primary WayveScenes101 research resources:

```bash
bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Select one downloaded scene directory and train the first mature baseline:

```bash
export WAYVE_SCENE_DIR=/data/external/driving_scene_reconstruction/datasets/wayve_scenes_101/<scene_dir>
ns-train splatfacto --data "$WAYVE_SCENE_DIR"
```

Fallback WayveScenes101 baseline:

```bash
ns-train nerfacto --data "$WAYVE_SCENE_DIR"
```

Render and evaluate after Nerfstudio writes a config:

```bash
ns-render interpolate --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/interpolate.mp4
ns-eval --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/eval.json
```

Fetch and run the PandaSet / neurad-studio track:

```bash
UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --pandaset
bash scripts/run_stage1_pandaset_neurad_studio.sh --method splatad
bash scripts/run_stage1_pandaset_neurad_studio.sh --method neurad
```

Fetch both tracks:

```bash
UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --all-public
```

## 5. Expected Local Output

Do not commit these directories:

```text
/data/external/driving_scene_reconstruction/
outputs/stage1_wayvescenes101_nerfstudio/
outputs/stage1_pandaset_neurad_studio/
```

Expected local WayveScenes101 artifacts:

```text
outputs/stage1_wayvescenes101_nerfstudio/
  run_commands.sh
  environment.txt
  train.log
  render_help.txt
  result_summary.md
  failure_notes.md
```

Expected local PandaSet artifacts:

```text
outputs/stage1_pandaset_neurad_studio/
  stage1_pandaset_splatad/
    run_commands.sh
    environment.txt
    train.log
    render_help.txt
    eval.json              # if evaluation succeeds
    result_summary.md
    failure_notes.md
```

## 6. First Result Criteria

The first successful result should be:

```text
WayveScenes101 scene
+ Splatfacto or Nerfacto trained with Nerfstudio
+ at least one rendered novel / held-out / off-axis view
+ notes on visual failures, geometry failures, and dynamic-object failures
```

A stronger result is:

```text
WayveScenes101 Splatfacto result
+ WayveScenes101 Nerfacto result
+ PandaSet SplatAD or NeuRAD result
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
primary_dataset: WayveScenes101
primary_codebase: Nerfstudio
primary_method: splatfacto
fallback_method: nerfacto
parallel_dataset: PandaSet
parallel_codebase: neurad-studio
parallel_methods: splatad, neurad
```
