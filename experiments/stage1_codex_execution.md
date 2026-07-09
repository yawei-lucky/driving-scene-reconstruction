# Stage 1A Codex Execution Notes

Date: 2026-07-09

## 2026-07-09 Linux Workspace Check

Task: verify public resources and prepare the first WayveScenes101 run from the local Linux checkout.

Repository checkout:

```text
path: /home/yawei/driving-scene-reconstruction
branch: main
remote: https://github.com/yawei-lucky/driving-scene-reconstruction.git
initial git status: clean
```

Important license distinction:

- WayveScenes101 code is public and MIT licensed.
- WayveScenes101 dataset is public / downloadable for non-commercial research use.
- The dataset license is separate from the code license.
- No raw datasets, videos, checkpoints, rendered outputs, or large binaries were committed.

### Environment Summary

```text
pwd before repo switch: /home/yawei
repo path: /home/yawei/driving-scene-reconstruction
python --version: Python 3.13.12
git: /usr/bin/git
python: /home/yawei/miniforge3/bin/python
ns-train: not found
ns-render: not found
ns-eval: not found
nvidia-smi: /usr/bin/nvidia-smi
gpu status in sandbox: nvidia-smi failed because it could not communicate with the NVIDIA driver
gpu status outside sandbox: NVIDIA GeForce RTX 4090 D, driver 580.95.05, CUDA 13.0
root/home disk: /dev/nvme1n1p2, 1.9T total, 330G available
data disk: /dev/nvme0n1p2 mounted at /data, 1.9T total, 11G available
stage1_external size: 190M
```

The `/home/yawei` filesystem has enough free space for the 200GB threshold. The script default resource root is under `/data`, where only 11G is available, so heavy dataset download should use `/home/yawei/stage1_external` instead of the default path. GPU access is available outside the Codex sandbox; the sandboxed `nvidia-smi` failure should not be treated as a machine-level GPU blocker.

GitHub issue #1 was requested for context, but `gh` is not installed:

```text
/bin/bash: line 1: gh: command not found
```

### Commands Run

Environment inspection:

```bash
pwd
ls
git status
python --version
df -h
which git
which python
which ns-train
which nvidia-smi
nvidia-smi
```

Results:

```text
PASS: /home/yawei exists and contains driving-scene-reconstruction
PASS: repo git status is clean on main and up to date with origin/main
PASS: Python 3.13.12 is available
PASS: git is available
FAIL/BLOCKER: ns-train is not on PATH
PASS: nvidia-smi works outside sandbox: RTX 4090 D, driver 580.95.05, CUDA 13.0
CAUTION: sandboxed nvidia-smi cannot communicate with the NVIDIA driver
CAUTION: /data has only 11G available
PASS: /home/yawei has about 330G available
```

Script validation:

```bash
bash -n scripts/fetch_stage1_resources.sh
bash -n scripts/run_stage1_pandaset_neurad_studio.sh
bash scripts/fetch_stage1_resources.sh --help
```

Results:

```text
PASS: bash -n scripts/fetch_stage1_resources.sh
PASS: bash -n scripts/run_stage1_pandaset_neurad_studio.sh
PASS: fetch script help printed expected options and license notes
```

WayveScenes101 lightweight public code-path check:

```bash
SKIP_WAYVE_DATA_DOWNLOAD=1 RESOURCE_ROOT=/home/yawei/stage1_external bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

The first sandboxed attempt failed because outbound GitHub access was blocked:

```text
[update] /home/yawei/stage1_external/code/nerfstudio
fatal: unable to access 'https://github.com/nerfstudio-project/nerfstudio.git/': Couldn't connect to server
```

After approved network access, the same command passed:

```text
[update] /home/yawei/stage1_external/code/nerfstudio
Already up to date.
[update] /home/yawei/stage1_external/code/wayve_scenes
Already up to date.
[wayvescenes101] Skipping dataset download because SKIP_WAYVE_DATA_DOWNLOAD=1
```

This confirms internet access to GitHub when allowed and confirms that the WayveScenes101 public code path is reachable. The full dataset was intentionally not downloaded.

### Upstream WayveScenes101 Inspection

Cloned upstream paths:

```text
WayveScenes101: /home/yawei/stage1_external/code/wayve_scenes
Nerfstudio: /home/yawei/stage1_external/code/nerfstudio
WayveScenes101 commit: 2fc2fa9606b328cfa3dbfa11ae1c6ace99c240eb
Nerfstudio commit: 50e0e3c70c775e89333256213363badbf074f29d
```

Files inspected:

```text
README.md
download.sh
tutorial/dataset_usage.ipynb
tutorial/nerfstudio_adapter.ipynb
tutorial/evaluate.ipynb
src/wayve_scenes/evaluation.py
```

Exact upstream dataset download command:

```bash
bash download.sh /path/to/wayve_scenes_101
```

Repository wrapper command:

```bash
RESOURCE_ROOT=/data/external/driving_scene_reconstruction bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Writable local resource-root command for this machine:

```bash
RESOURCE_ROOT=/home/yawei/stage1_external bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

Expected dataset directory after download and extraction:

```text
wayve_scenes_101/
  scene_001/
    colmap_sparse/rig/
    images/
      front-forward/
      left-forward/
      right-forward/
      left-backward/
      right-backward/
    masks/
      front-forward/
      left-forward/
      right-forward/
      left-backward/
      right-backward/
```

Subset download status:

```text
The upstream README says users may download all scenes or only a subset from the official Google Drive folder.
The upstream download.sh itself has a fixed LINKS array and no subset flag.
For a scripted subset run, use the official Google Drive file links for selected scenes or copy/edit the LINKS array in a local scratch script outside the repo.
```

Official extraction command from the dataset usage tutorial:

```bash
export DATA_ROOT=/path/to/wayve_scenes_101/
unzip "$DATA_ROOT/*.zip"
```

Nerfstudio preparation from the upstream adapter tutorial:

```python
from pathlib import Path
from wayve_scenes.utils import colmap_utils

dataset_root = "/path/to/wayve_scenes_101"
scene_name = "scene_096"

recon_dir = Path(f"{dataset_root}/{scene_name}/colmap_sparse/rig/")
output_dir = Path(f"{dataset_root}/{scene_name}/")

colmap_utils.colmap_to_json(recon_dir=recon_dir, output_dir=output_dir, use_masks=True)
```

Official tutorial training command:

```bash
export DATASET_ROOT=/path/to/wayve_scenes_101/
export SCENE_NAME=scene_096
export SCENE_PATH=$DATASET_ROOT/$SCENE_NAME

ns-train nerfacto --data $SCENE_PATH --pipeline.model.camera-optimizer.mode off
```

Stage-1 first Splatfacto command, matching this repository's priority:

```bash
export WAYVE_SCENE_DIR=/home/yawei/stage1_external/datasets/wayve_scenes_101/<scene_dir>
ns-train splatfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

Render and Nerfstudio evaluation after training writes a config:

```bash
ns-render interpolate --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/interpolate.mp4
ns-eval --load-config <config.yml> --output-path outputs/stage1_wayvescenes101_nerfstudio/eval.json
```

Official WayveScenes101 evaluation expects predictions to mirror the dataset tree:

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

### Blockers

Heavy execution should not start until Nerfstudio is installed / activated and data download is intentionally launched into `/home/yawei/stage1_external`.

```text
1. ns-train, ns-render, and ns-eval are not on PATH.
2. /data has only 11G available, which is below the 200GB threshold for full WayveScenes101 download.
3. Use /home/yawei/stage1_external for heavy resources; /home/yawei has about 330G available.
4. GitHub issue #1 could not be inspected because gh is not installed.
5. jq is not installed, so notebook inspection used rg text search instead of structured notebook extraction.
```

### Exact Next Command For A GPU Machine

On this machine, after activating/installing the needed Python/Nerfstudio/WayveScenes101 environment, use the `/home/yawei` resource root because `/data` is full:

```bash
cd ~/driving-scene-reconstruction
RESOURCE_ROOT=/home/yawei/stage1_external \
SKIP_WAYVE_DATA_DOWNLOAD=0 \
bash scripts/fetch_stage1_resources.sh --wayvescenes101
```

After the official download finishes, extract the downloaded scene zips and prepare one scene for Nerfstudio:

```bash
export DATA_ROOT=/home/yawei/stage1_external/datasets/wayve_scenes_101
unzip "$DATA_ROOT/*.zip"

export SCENE_NAME=scene_096
python - <<'PY'
from pathlib import Path
from wayve_scenes.utils import colmap_utils
import os

dataset_root = os.environ["DATA_ROOT"]
scene_name = os.environ["SCENE_NAME"]
recon_dir = Path(f"{dataset_root}/{scene_name}/colmap_sparse/rig/")
output_dir = Path(f"{dataset_root}/{scene_name}/")
colmap_utils.colmap_to_json(recon_dir=recon_dir, output_dir=output_dir, use_masks=True)
PY

export WAYVE_SCENE_DIR="$DATA_ROOT/$SCENE_NAME"
ns-train splatfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

Fallback baseline:

```bash
ns-train nerfacto --data "$WAYVE_SCENE_DIR" --pipeline.model.camera-optimizer.mode off
```

### Next Best Action

Install or activate the Nerfstudio / WayveScenes101 Python environment, then run the exact next command above using `/home/yawei/stage1_external`. Keep using WayveScenes101 + Nerfstudio / Splatfacto as the first path; use PandaSet + neurad-studio only as the backup / parallel track.

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
