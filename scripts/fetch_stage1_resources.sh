#!/usr/bin/env bash
set -euo pipefail

# Fetch Stage-1 public resources for driving scene reconstruction.
#
# Default open-resource path:
#   PandaSet + neurad-studio + SplatAD / NeuRAD
#
# Usage:
#   bash scripts/fetch_stage1_resources.sh --all-open
#   RESOURCE_ROOT=/data/external/driving_scene_reconstruction bash scripts/fetch_stage1_resources.sh --all-open
#   UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --pandaset
#
# Notes:
#   - This script downloads large external resources. Do not commit them to GitHub.
#   - PandaSet on HuggingFace is about 44.5 GB.
#   - WayveScenes101 is useful but non-commercial; it is not included in --all-open.

RESOURCE_ROOT="${RESOURCE_ROOT:-/data/external/driving_scene_reconstruction}"
CODE_ROOT="${CODE_ROOT:-${RESOURCE_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${RESOURCE_ROOT}/datasets}"
PANDASET_HF_ROOT="${PANDASET_HF_ROOT:-${DATASET_ROOT}/pandaset_hf}"
PANDASET_DIR="${PANDASET_DIR:-${DATASET_ROOT}/pandaset}"
UNZIP_PANDASET="${UNZIP_PANDASET:-0}"

FETCH_CODE=0
FETCH_PANDASET=0
FETCH_WAYVE_RESEARCH=0

print_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fetch_stage1_resources.sh [options]

Options:
  --all-open          Fetch open first-path resources: code + PandaSet.
  --code              Clone / update code repositories only.
  --pandaset          Download PandaSet from HuggingFace only.
  --wayve-research    Clone WayveScenes101 code and print dataset download command.
                      Note: WayveScenes101 dataset is non-commercial.
  -h, --help          Show this message.

Environment variables:
  RESOURCE_ROOT       Root directory for external resources.
                      Default: /data/external/driving_scene_reconstruction
  CODE_ROOT           Code checkout directory.
  DATASET_ROOT        Dataset directory.
  PANDASET_HF_ROOT    HuggingFace snapshot target.
  PANDASET_DIR        Unzipped PandaSet target.
  UNZIP_PANDASET      Set to 1 to unzip pandaset.zip after download.
USAGE
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-open)
      FETCH_CODE=1
      FETCH_PANDASET=1
      shift
      ;;
    --code)
      FETCH_CODE=1
      shift
      ;;
    --pandaset)
      FETCH_PANDASET=1
      shift
      ;;
    --wayve-research)
      FETCH_WAYVE_RESEARCH=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    echo "[update] ${dir}"
    git -C "${dir}" pull --ff-only
  else
    echo "[clone] ${url} -> ${dir}"
    git clone "${url}" "${dir}"
  fi
}

mkdir -p "${CODE_ROOT}" "${DATASET_ROOT}"
need_cmd git
need_cmd python

if [[ "${FETCH_CODE}" -eq 1 ]]; then
  clone_or_update "https://github.com/georghess/neurad-studio.git" "${CODE_ROOT}/neurad-studio"
  clone_or_update "https://github.com/nerfstudio-project/nerfstudio.git" "${CODE_ROOT}/nerfstudio"
  clone_or_update "https://github.com/scaleapi/pandaset-devkit.git" "${CODE_ROOT}/pandaset-devkit"
fi

if [[ "${FETCH_PANDASET}" -eq 1 ]]; then
  echo "[pandaset] Installing HuggingFace download client if needed"
  python -m pip install -U huggingface_hub

  echo "[pandaset] Downloading georghess/pandaset -> ${PANDASET_HF_ROOT}"
  mkdir -p "${PANDASET_HF_ROOT}"
  PANDASET_HF_ROOT="${PANDASET_HF_ROOT}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

local_dir = os.environ["PANDASET_HF_ROOT"]
snapshot_download(
    repo_id="georghess/pandaset",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["pandaset.zip", "README.md", ".gitattributes"],
)
print(f"Downloaded PandaSet snapshot to: {local_dir}")
PY

  if [[ "${UNZIP_PANDASET}" == "1" ]]; then
    need_cmd unzip
    echo "[pandaset] Unzipping pandaset.zip -> ${PANDASET_DIR}"
    mkdir -p "${PANDASET_DIR}"
    unzip -n "${PANDASET_HF_ROOT}/pandaset.zip" -d "${PANDASET_DIR}"
  else
    echo "[pandaset] Download complete. Set UNZIP_PANDASET=1 to unzip automatically."
  fi
fi

if [[ "${FETCH_WAYVE_RESEARCH}" -eq 1 ]]; then
  clone_or_update "https://github.com/wayveai/wayve_scenes.git" "${CODE_ROOT}/wayve_scenes"
  cat <<EOF

[wayvescenes101]
WayveScenes101 code is MIT licensed, but the dataset license is non-commercial.
Use it as a secondary research-only track.

To download manually after reviewing the license:
  cd ${CODE_ROOT}/wayve_scenes
  bash download.sh ${DATASET_ROOT}/wayve_scenes_101
EOF
fi

cat <<EOF

Done.

Resource root:
  ${RESOURCE_ROOT}

Code root:
  ${CODE_ROOT}

Dataset root:
  ${DATASET_ROOT}

Next command:
  bash scripts/run_stage1_pandaset_neurad_studio.sh --method splatad
EOF
