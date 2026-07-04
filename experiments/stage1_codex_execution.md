# Stage 1A Codex Execution Notes

Date: 2026-07-04

## Scope

Task: verify public resources and prepare the first WayveScenes101 run.

Priority path:

```text
WayveScenes101 dataset + Nerfstudio / Splatfacto
```

Backup / parallel path:

```text
PandaSet + neurad-studio / SplatAD or NeuRAD
```

Important license distinction:

- WayveScenes101 code is public and MIT licensed.
- WayveScenes101 dataset is public / downloadable for non-commercial research use.
- PandaSet remains the more commercial-friendly / multi-sensor backup track.

No raw datasets, videos, checkpoints, rendered outputs, or large binaries were committed.

## Repository And Issue Context

Local checkout:

```text
C:\Users\Yawei\Documents\<workspace>
branch: main
remote: https://github.com/yawei-lucky/driving-scene-reconstruction.git
```

GitHub issue #1 was visible through the GitHub REST API:

```text
title: Stage 1: Run public driving scene reconstruction baselines on WayveScenes101 and PandaSet
state: open
priority: WayveScenes101 + Nerfstudio / Splatfacto first; PandaSet + neurad-studio as parallel track
```

## Environment Summary

Local Codex host:

```text
path: C:\Users\Yawei\Documents\<workspace>
git: available at C:\Program Files\Git\cmd\git.exe
bash: not on PATH, but Git Bash works at C:\Program Files\Git\bin\bash.exe
python: not found on PATH
ns-train: not found
nvidia-smi: not found
disk: C: 609G total, 464G free from Git Bash df
```

Remote SSH workbench `shidi`:

```text
host: stf-precision-3680
user: yawei
os: Ubuntu 22.04 kernel 6.8.0-124-generic
python: /usr/bin/python -> Python 2.7.18
python3: /usr/bin/python3 -> Python 3.10.12
git: /usr/bin/git
ns-train: not found
gpu: NVIDIA GeForce RTX 4090 D, driver 580.95.05, CUDA 13.0
root disk: /dev/nvme1n1p2, 1.9T total, 8.8G available
data disk: /dev/nvme0n1p2 mounted at /data, 1.9T total, 11G available
```

GPU is available on `shidi`, but disk is not sufficient for a full dataset run. The requested 200GB free-disk threshold is not met.

## Commands Run

Local initial checks:

```bash
pwd
ls
git status
python --version
df -h
which git || true
which python || true
which ns-train || true
which nvidia-smi || true
nvidia-smi || true
```

Observed local blockers:

```text
python: command not found
ns-train: not found
nvidia-smi: not found
bash: not on PATH
```

Git Bash was used for script checks:

```bash
bash -n scripts/fetch_stage1_resources.sh
bash -n scripts/run_stage1_pandaset_neurad_studio.sh
bash scripts/fetch_stage1_resources.sh --help
```

Results:

```text
PASS: bash -n scripts/fetch_stage1_resources.sh
PASS: bash -n scripts/run_stage1_pandaset_neurad_studio.sh
PASS: bash scripts/fetch_stage1_resources.sh --help when run through Git Bash login shell
```

Remote workbench checks:

```bash
ssh shidi "pwd; ls; git status 2>/dev/null || true; python --version; df -h; which git || true; which python || true; which ns-train || true; which nvidia-smi || true; nvidia-smi || true"
```

Results:

```text
PASS: SSH access works
PASS: git is available
PASS: GPU is visible through nvidia-smi
FAIL/BLOCKER: only 8.8G available on / and 11G available on /data
FAIL/BLOCKER: ns-train not found
CAUTION: default python is Python 2.7.18; use a Python 3 / conda environment for actual setup
```

Remote repository checkout:

```bash
git clone https://github.com/yawei-lucky/driving-scene-reconstruction.git ~/driving-scene-reconstruction
```

Remote script validation:

```bash
cd ~/driving-scene-reconstruction
bash -n scripts/fetch_stage1_resources.sh
bash -n scripts/run_stage1_pandaset_neurad_studio.sh
bash scripts/fetch_stage1_resources.sh --help
```

Results:

```text
PASS: both shell scripts pass bash syntax validation on shidi
PASS: fetch script help prints expected options and license notes
```

WayveScenes101 lightweight public resource check:

```bash
cd ~/driving-scene-reconstruction
SKIP_WAYVE_DATA_DOWNLOAD=1 bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Result:

```text
FAIL: default RESOURCE_ROOT=/data/external/driving_scene_reconstruction could not be created
reason: mkdir: cannot create directory '/data/external': Permission denied
```

Retry with a user-writable resource root:

```bash
cd ~/driving-scene-reconstruction
RESOURCE_ROOT=~/stage1_external SKIP_WAYVE_DATA_DOWNLOAD=1 bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Result:

```text
PASS: cloned https://github.com/nerfstudio-project/nerfstudio.git
PASS: cloned https://github.com/wayveai/wayve_scenes.git
PASS: skipped WayveScenes101 dataset download because SKIP_WAYVE_DATA_DOWNLOAD=1
```

This confirms internet access to GitHub and the public code path. Full dataset download was intentionally not attempted.

## Upstream WayveScenes101 Inspection

Cloned upstream code path:

```text
~/stage1_external/code/wayve_scenes
```

Relevant upstream files inspected:

```text
README.md
download.sh
tutorial/dataset_usage.ipynb
tutorial/nerfstudio_adapter.ipynb
tutorial/evaluate.ipynb
src/wayve_scenes/evaluation.py
```

Dataset download command documented by upstream:

```bash
bash download.sh /path/to/wayve_scenes_101
```

The repository script wraps the same upstream command as:

```bash
bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Expected dataset directory after download and unzip:

```text
wayve_scenes_101/
  scene_001/
    colmap_sparse/rig/
      cameras.bin
      images.bin
      points3D.bin
    images/
      front-forward/
      left-backward/
      left-forward/
      right-backward/
      right-forward/
    masks/
      front-forward/
      left-backward/
      left-forward/
      right-backward/
      right-forward/
  scene_002/
  ...
```

Small subset support:

```text
The README says users may download all scenes or only a subset from Google Drive.
The upstream download.sh itself contains a fixed LINKS array and has no subset flag.
For a scripted subset, edit/copy the LINKS list or manually download selected scenes from the official Google Drive folder.
```

Nerfstudio preparation from upstream tutorial:

```python
from pathlib import Path
from wayve_scenes.utils import colmap_utils

dataset_root = "/path/to/wayve_scenes_101"
scene_name = "scene_096"

recon_dir = Path(f"{dataset_root}/{scene_name}/colmap_sparse/rig/")
output_dir = Path(f"{dataset_root}/{scene_name}/")

colmap_utils.colmap_to_json(recon_dir=recon_dir, output_dir=output_dir, use_masks=True)
```

Upstream tutorial training command:

```bash
export DATASET_ROOT=/path/to/wayve_scenes_101/
export SCENE_NAME=scene_096
export SCENE_PATH=$DATASET_ROOT/$SCENE_NAME

ns-train nerfacto --data $SCENE_PATH --pipeline.model.camera-optimizer.mode off
```

Stage-1 first Splatfacto command, matching the project priority:

```bash
ns-train splatfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

Render after training:

```bash
ns-render interpolate --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/interpolate.mp4
```

Repository-level Nerfstudio evaluation command:

```bash
ns-eval --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/eval.json
```

Official WayveScenes101 evaluation expects rendered predictions to mirror the dataset image tree:

```text
/path/to/model_predictions/<scene_name>/images/<camera>/<image_name>.jpeg
```

Then run:

```python
from wayve_scenes.evaluation import evaluate_submission

metrics_dict_all, metrics_dict_train, metrics_dict_test = evaluate_submission(
    dir_pred="/path/to/model_predictions/",
    dir_target="/path/to/wayve_scenes_101/",
)
```

The official train cameras are:

```text
left-forward
right-forward
left-backward
right-backward
```

The held-out test camera is:

```text
front-forward
```

## Blockers

Heavy execution should not start on the current environments.

Blockers:

```text
1. Local host has no python, no ns-train, and no visible NVIDIA GPU.
2. Remote shidi has a GPU, but available disk is far below the required 200GB threshold.
3. Remote shidi does not have ns-train on PATH.
4. Remote shidi default python is Python 2.7.18; a Python 3 conda environment must be active before real setup.
5. Default /data/external resource root is not writable by user yawei.
```

## Exact Next Command For A GPU Machine

After moving to a GPU machine with at least 200GB free disk, a writable resource root, Python 3, and Nerfstudio available, run:

```bash
cd ~/driving-scene-reconstruction
RESOURCE_ROOT=/data/external/driving_scene_reconstruction \
SKIP_WAYVE_DATA_DOWNLOAD=0 \
bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Then select one scene and run the first baseline:

```bash
export WAYVE_SCENE_DIR=/data/external/driving_scene_reconstruction/datasets/wayve_scenes_101/<scene_dir>
ns-train splatfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

Fallback baseline:

```bash
ns-train nerfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

## Next Best Action

Free or attach at least 200GB of disk on the GPU workbench, make the resource root writable, and install/activate a Python 3 Nerfstudio environment. After that, run the WayveScenes101 fetch command above and start with one scene using Splatfacto.
