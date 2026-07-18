#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
PYTHON="${H3_ENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Stage H3 Python interpreter not found: $PYTHON" >&2
  echo "Run scripts/setup_stage_h3_environment.sh first." >&2
  exit 1
fi
if [[ ! -x "${H3_ENV}/bin/nvcc" ]]; then
  echo "Stage H3 CUDA compiler not found: ${H3_ENV}/bin/nvcc" >&2
  exit 1
fi

export H3_ROOT
export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/torch_extensions}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${H3_ROOT}/cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${H3_ROOT}/cache/matplotlib}"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

echo "Host and storage:"
echo "  host: $(hostname)"
echo "  H3 root: $H3_ROOT"
df -h "$H3_ROOT" | tail -n 1
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader
"${H3_ENV}/bin/nvcc" --version | tail -n 1

"$PYTHON" "$REPO_ROOT/scripts/check_stage_h3_environment.py" \
  --h3-root "$H3_ROOT"

"$PYTHON" -m pip check

METHOD_HELP="$("${H3_ENV}/bin/ns-train" --help 2>&1)"
if [[ "$METHOD_HELP" != *"splatad"* || "$METHOD_HELP" != *"neurad"* ]]; then
  echo "ns-train does not expose both splatad and neurad." >&2
  exit 1
fi

PANDASET_HELP="$("${H3_ENV}/bin/ns-train" splatad pandaset-data --help 2>&1)"
if [[ "$PANDASET_HELP" != *"Pandar64"* || "$PANDASET_HELP" != *"PandarGT"* ]]; then
  echo "PandaSet parser help does not expose the expected LiDAR choices." >&2
  exit 1
fi

echo "PASS: ns-train exposes splatad, neurad, and pandaset-data."
