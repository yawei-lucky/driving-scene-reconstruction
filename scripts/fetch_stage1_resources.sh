#!/usr/bin/env bash
set -euo pipefail

# Fetch Stage-1 public resources for driving scene reconstruction.
#
# Primary research path:
#   WayveScenes101 + Nerfstudio / Splatfacto
#
# Parallel commercial-friendly / multi-sensor path:
#   PandaSet + neurad-studio + SplatAD / NeuRAD
#
# Usage:
#   bash scripts/fetch_stage1_resources.sh --wayvescenes101
#   bash scripts/fetch_stage1_resources.sh --pandaset
#   UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --all-public
#   SKIP_WAYVE_DATA_DOWNLOAD=1 bash scripts/fetch_stage1_resources.sh --wayvescenes101
#
# Notes:
#   - This script downloads large external resources. Do not commit them to GitHub.
#   - WayveScenes101 dataset is public and research-usable, but non-commercial.
#   - PandaSet on HuggingFace is about 44.5 GB.

RESOURCE_ROOT="${RESOURCE_ROOT:-/data/external/driving_scene_reconstruction}"
CODE_ROOT="${CODE_ROOT:-${RESOURCE_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${RESOURCE_ROOT}/datasets}"

WAYVE_CODE_DIR="${WAYVE_CODE_DIR:-${CODE_ROOT}/wayve_scenes}"
WAYVE_DATA_DIR="${WAYVE_DATA_DIR:-${DATASET_ROOT}/wayve_scenes_101}"
SKIP_WAYVE_DATA_DOWNLOAD="${SKIP_WAYVE_DATA_DOWNLOAD:-0}"

PANDASET_HF_ROOT="${PANDASET_HF_ROOT:-${DATASET_ROOT}/pandaset_hf}"
PANDASET_DIR="${PANDASET_DIR:-${DATASET_ROOT}/pandaset}"
UNZIP_PANDASET="${UNZIP_PANDASET:-0}"

FETCH_WAYVE=0
FETCH_PANDASET=0
FETCH_CODE_COMMON=0

print_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fetch_stage1_resources.sh [options]

Options:
  --wayvescenes101  Fetch primary research path:
                    WayveScenes101 code + Nerfstudio, then run upstream dataset download.
  --pandaset        Fetch PandaSet / neurad-studio path.
  --all-public      Fetch both WayveScenes101 and PandaSet tracks.
  --all-open        Backward-compatible alias for --all-public.
  --code            Clone / update common code repositories only.
  -h, --help        Show this message.

Environment variables:
  RESOURCE_ROOT                 Root directory for external resources.
                                Default: /data/external/driving_scene_reconstruction
  CODE_ROOT                     Code checkout directory.
  DATASET_ROOT                  Dataset directory.

  WAYVE_CODE_DIR                WayveScenes101 code checkout path.
  WAYVE_DATA_DIR                WayveScenes101 dataset target path.
  SKIP_WAYVE_DATA_DOWNLOAD      Set to 1 to clone Wayve code but skip dataset download.

  PANDASET_HF_ROOT              HuggingFace snapshot target.
  PANDASET_DIR                  Unzipped PandaSet target.
  UNZIP_PANDASET                Set to 1 to unzip pandaset.zip after download.

License notes:
  WayveScenes101 code is MIT licensed. The dataset is public and research-usable,
  but under a non-commercial dataset license.
  PandaSet is the broader commercial-friendly / multi-sensor path.
USAGE
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wayvescenes101)
      FETCH_WAYVE=1
      FETCH_CODE_COMMON=1
      shift
      ;;
    --pandaset)
      FETCH_PANDASET=1
      FETCH_CODE_COMMON=1
      shift
      ;;
    --all-public|--all-open)
      FETCH_WAYVE=1
      FETCH_PANDASET=1
      FETCH_CODE_COMMON=1
      shift
      ;;
    --code)
      FETCH_CODE_COMMON=1
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

if [[ "${FETCH_CODE_COMMON}" -eq 1 ]]; then
  clone_or_update "https://github.com/nerfstudio-project/nerfstudio.git" "${CODE_ROOT}/nerfstudio"
fi

if [[ "${FETCH_WAYVE}" -eq 1 ]]; then
  clone_or_update "https://github.com/wayveai/wayve_scenes.git" "${WAYVE_CODE_DIR}"

  if [[ "${SKIP_WAYVE_DATA_DOWNLOAD}" == "1" ]]; then
    echo "[wayvescenes101] Skipping dataset download because SKIP_WAYVE_DATA_DOWNLOAD=1"
  else
    echo "[wayvescenes101] Installing gdown if needed"
    python -m pip install -U gdown

    echo "[wayvescenes101] Downloading dataset with upstream download.sh -> ${WAYVE_DATA_DIR}"
    mkdir -p "${WAYVE_DATA_DIR}"
    (
      cd "${WAYVE_CODE_DIR}"
      bash download.sh "${WAYVE_DATA_DIR}"
    )
  fi
fi

if [[ "${FETCH_PANDASET}" -eq 1 ]]; then
  clone_or_update "https://github.com/georghess/neurad-studio.git" "${CODE_ROOT}/neurad-studio"
  clone_or_update "https://github.com/scaleapi/pandaset-devkit.git" "${CODE_ROOT}/pandaset-devkit"

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

cat <<EOF

Done.

Resource root:
  ${RESOURCE_ROOT}

Code root:
  ${CODE_ROOT}

Dataset root:
  ${DATASET_ROOT}

Primary next command:
  bash scripts/run_stage1_wayvescenes101_nerfstudio.sh --method splatfacto

Parallel PandaSet command:
  bash scripts/run_stage1_pandaset_neurad_studio.sh --method splatad
EOF
