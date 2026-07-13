#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_DIR="${SCENE_DIR:-/home/yawei/stage1_external/datasets/wayve_scenes_101/scene_094_official_split}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/yawei/stage1_external/outputs/wayvescenes101_h1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scene_094_h1_big}"
METHOD="${METHOD:-splatfacto-big}"
TIMESTAMP="${TIMESTAMP:-run_v1}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-30000}"
LOAD_DIR="${LOAD_DIR:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
CONDA_ENV="${CONDA_ENV:-wayve_scenes_env}"
CONDA_ROOT="${CONDA_ROOT:-/home/yawei/miniforge3}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"

usage() {
  cat <<'EOF'
Usage: scripts/run_stage_h1_wayvescenes101_splatfacto.sh [options]

Options:
  --scene-dir PATH          Prepared WayveScenes101 scene directory.
  --output-dir PATH         Nerfstudio output root.
  --experiment-name NAME   Stable experiment name.
  --method NAME            Nerfstudio method preset (default: splatfacto-big).
  --timestamp NAME         Stable run directory name (default: run_v1).
  --iterations N           Absolute target iteration count (default: 30000).
  --load-dir PATH           Resume from a Nerfstudio checkpoint directory.
  -h, --help                Show this help.

Environment overrides: METHOD, TIMESTAMP, CUDA_HOME, CONDA_ENV, CONDA_ROOT, MIN_FREE_GB,
CUDA_VISIBLE_DEVICES, TORCH_CUDA_ARCH_LIST.
EOF
}

while (($#)); do
  case "$1" in
    --scene-dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --iterations)
      MAX_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --load-dir)
      LOAD_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$MAX_NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--iterations must be a positive integer" >&2
  exit 2
fi

reject_managed_path() {
  local label="$1"
  local candidate
  candidate="$(realpath -m "$2")"
  case "$candidate" in
    "$REPO_ROOT"|"$REPO_ROOT"/*|/data|/data/*)
      echo "$label must be outside the Git repository and /data: $candidate" >&2
      exit 1
      ;;
  esac
  printf '%s' "$candidate"
}

OUTPUT_DIR="$(reject_managed_path OUTPUT_DIR "$OUTPUT_DIR")"

if [[ ! -f "${SCENE_DIR}/transforms.json" ]]; then
  echo "Missing prepared scene transforms: ${SCENE_DIR}/transforms.json" >&2
  exit 1
fi

NS_TRAIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/ns-train"
if [[ ! -x "$NS_TRAIN" ]]; then
  echo "ns-train not found: $NS_TRAIN" >&2
  exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA toolkit not found: ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
free_kb="$(df --output=avail "$OUTPUT_DIR" | tail -1 | tr -d ' ')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if ((free_kb < required_kb)); then
  echo "Insufficient free disk at $OUTPUT_DIR: require ${MIN_FREE_GB}GB" >&2
  exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"
export TMPDIR="${TMPDIR:-/home/yawei/tmp}"
TMPDIR="$(reject_managed_path TMPDIR "$TMPDIR")"
mkdir -p "$TMPDIR"

shopt -s nullglob
existing_runs=("$OUTPUT_DIR/$EXPERIMENT_NAME"/*/"$TIMESTAMP")
shopt -u nullglob
if [[ -z "$LOAD_DIR" && ${#existing_runs[@]} -ne 0 ]]; then
  echo "Refusing fresh training into existing run: ${existing_runs[0]}" >&2
  echo "Use a new --timestamp, or pass --load-dir to resume." >&2
  exit 1
fi

args=(
  "$METHOD"
  --data "$SCENE_DIR"
  --max-num-iterations "$MAX_NUM_ITERATIONS"
  --vis tensorboard
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$EXPERIMENT_NAME"
  --timestamp "$TIMESTAMP"
  --machine.seed 42
  --steps-per-save 1000
  --steps-per-eval-image 500
  --steps-per-eval-all-images 3000
  --save-only-latest-checkpoint True
  --load-scheduler True
  --logging.steps-per-log 20
  --pipeline.datamanager.cache-images cpu
  --pipeline.datamanager.cache-images-type uint8
  --pipeline.datamanager.images-on-gpu False
  --pipeline.datamanager.masks-on-gpu False
  --pipeline.model.camera-optimizer.mode off
)

if [[ -n "$LOAD_DIR" ]]; then
  LOAD_DIR="$(reject_managed_path LOAD_DIR "$LOAD_DIR")"
  if [[ ! -d "$LOAD_DIR" ]]; then
    echo "Checkpoint directory not found: $LOAD_DIR" >&2
    exit 1
  fi
  shopt -s nullglob
  checkpoints=("$LOAD_DIR"/step-*.ckpt)
  shopt -u nullglob
  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No step-*.ckpt checkpoint found in: $LOAD_DIR" >&2
    exit 1
  fi
  latest_checkpoint="${checkpoints[${#checkpoints[@]} - 1]}"
  checkpoint_name="$(basename "$latest_checkpoint")"
  if [[ ! "$checkpoint_name" =~ ^step-([0-9]+)\.ckpt$ ]]; then
    echo "Cannot parse checkpoint step: $checkpoint_name" >&2
    exit 1
  fi
  loaded_step="$((10#${BASH_REMATCH[1]}))"
  if ((MAX_NUM_ITERATIONS <= loaded_step)); then
    echo "--iterations must exceed loaded checkpoint step $loaded_step" >&2
    exit 1
  fi
  args+=(--load-dir "$LOAD_DIR")
fi

echo "scene_dir=$SCENE_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "experiment_name=$EXPERIMENT_NAME"
echo "method=$METHOD"
echo "timestamp=$TIMESTAMP"
echo "target_max_num_iterations=$MAX_NUM_ITERATIONS"
echo "cuda_home=$CUDA_HOME"
echo "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
printf 'command='
printf '%q ' "$NS_TRAIN" "${args[@]}"
printf '\n'

exec "$NS_TRAIN" "${args[@]}"
