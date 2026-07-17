#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-smoke}"
if (($#)); then
  shift
fi

CONDA_ROOT="${CONDA_ROOT:-/home/yawei/miniforge3}"
CONDA_ENV="${CONDA_ENV:-wayve_scenes_env}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
CONFIG="${CONFIG:-/home/yawei/stage1_external/outputs/wayvescenes101_h1/scene_094_h1_big/splatfacto/run_v2/config.yml}"
PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Nerfstudio config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA compiler not found: $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MPLBACKEND="Agg"

case "$MODE" in
  smoke)
    exec "$PYTHON" "$REPO_ROOT/examples/reconstruction_renderer_smoke.py" \
      --config "$CONFIG" "$@"
    ;;
  interactive)
    exec "$PYTHON" "$REPO_ROOT/examples/interactive_reconstruction.py" \
      --config "$CONFIG" "$@"
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage:
  scripts/run_stage_h2_scene_094.sh smoke [example options]
  scripts/run_stage_h2_scene_094.sh interactive [example options]

Environment:
  CONFIG, CONDA_ROOT, CONDA_ENV, CUDA_HOME, TORCH_CUDA_ARCH_LIST
EOF
    ;;
  *)
    echo "Unknown mode: $MODE (expected smoke or interactive)" >&2
    exit 2
    ;;
esac
