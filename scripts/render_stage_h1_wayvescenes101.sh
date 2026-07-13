#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${1:-}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/yawei/stage1_external/artifacts/scene_094/latest}"
DOWNSCALE_FACTOR="${DOWNSCALE_FACTOR:-1}"
FPS="${FPS:-10}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_WAYVE_EVAL="${RUN_WAYVE_EVAL:-1}"
RUN_DATASET_RENDER="${RUN_DATASET_RENDER:-1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
CONDA_ENV="${CONDA_ENV:-wayve_scenes_env}"
CONDA_ROOT="${CONDA_ROOT:-/home/yawei/miniforge3}"

usage() {
  cat <<'EOF'
Usage: CONFIG=/path/to/config.yml scripts/render_stage_h1_wayvescenes101.sh
   or: scripts/render_stage_h1_wayvescenes101.sh /path/to/config.yml

Environment overrides:
  ARTIFACT_ROOT        External artifact destination.
  DOWNSCALE_FACTOR     Dataset render downscale (default: 1; Wayve eval requires 1).
  FPS                  Reference video frame rate (default: 10).
  RUN_EVAL             Set to 0 to skip ns-eval.
  RUN_WAYVE_EVAL       Set to 0 to skip official Wayve evaluation.
  RUN_DATASET_RENDER   Set to 0 to reuse rendered dataset frames.
EOF
}

if [[ -z "$CONFIG" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

ARTIFACT_ROOT="$(realpath -m "$ARTIFACT_ROOT")"
case "$ARTIFACT_ROOT" in
  "$REPO_ROOT"|"$REPO_ROOT"/*|/data|/data/*)
    echo "ARTIFACT_ROOT must be outside the Git repository and /data" >&2
    exit 1
    ;;
esac

if [[ "$RUN_WAYVE_EVAL" == "1" && "$DOWNSCALE_FACTOR" != "1" ]]; then
  echo "Official Wayve evaluation requires DOWNSCALE_FACTOR=1" >&2
  exit 1
fi

NS_EVAL="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/ns-eval"
NS_RENDER="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/ns-render"
PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
for executable in "$NS_EVAL" "$NS_RENDER" "$PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="Agg"

mkdir -p "$ARTIFACT_ROOT/metrics" "$ARTIFACT_ROOT/eval_frames" \
  "$ARTIFACT_ROOT/videos"

echo "config=$CONFIG"
echo "artifact_root=$ARTIFACT_ROOT"

if [[ "$RUN_EVAL" == "1" ]]; then
  "$NS_EVAL" \
    --load-config "$CONFIG" \
    --output-path "$ARTIFACT_ROOT/metrics/metrics.json" \
    --render-output-path "$ARTIFACT_ROOT/eval_frames"
fi

if [[ "$RUN_DATASET_RENDER" == "1" ]]; then
  if [[ -e "$ARTIFACT_ROOT/dataset" ]]; then
    echo "Refusing to mix renders with existing dataset directory: $ARTIFACT_ROOT/dataset" >&2
    exit 1
  fi
  render_staging="$(mktemp -d "$ARTIFACT_ROOT/.dataset.render.XXXXXX")"
  cleanup_render_staging() {
    if [[ -n "${render_staging:-}" && -d "$render_staging" ]]; then
      rm -rf -- "$render_staging"
    fi
  }
  trap cleanup_render_staging EXIT
  "$NS_RENDER" dataset \
    --load-config "$CONFIG" \
    --output-path "$render_staging" \
    --split train+test \
    --rendered-output-names rgb gt-rgb \
    --image-format jpeg \
    --jpeg-quality 95 \
    --downscale-factor "$DOWNSCALE_FACTOR"
  mv -- "$render_staging" "$ARTIFACT_ROOT/dataset"
  render_staging=""
  trap - EXIT
elif [[ ! -d "$ARTIFACT_ROOT/dataset" ]]; then
  echo "Rendered dataset directory not found: $ARTIFACT_ROOT/dataset" >&2
  exit 1
fi

"$PYTHON" "$REPO_ROOT/scripts/build_stage_h1_reference_videos.py" \
  --render-root "$ARTIFACT_ROOT/dataset" \
  --output-dir "$ARTIFACT_ROOT/videos" \
  --video-width 960 \
  --fps "$FPS" \
  --comparison-camera front-forward

if [[ "$RUN_WAYVE_EVAL" == "1" ]]; then
  "$PYTHON" "$REPO_ROOT/scripts/evaluate_stage_h1_wayve.py" \
    --render-root "$ARTIFACT_ROOT/dataset" \
    --prediction-root "$ARTIFACT_ROOT/predictions" \
    --target-root /home/yawei/stage1_external/datasets/wayve_scenes_101 \
    --output-path "$ARTIFACT_ROOT/metrics/wayve_metrics.json" \
    --scene scene_094
fi
