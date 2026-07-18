#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
CONDA_BIN="${CONDA_BIN:-/home/yawei/miniforge3/bin/conda}"
REPAIR=0
export H3_ROOT H3_ENV

NEURAD_COMMIT="e6f7e4e509b828a952d8584b7165f7844711ecb2"
PANDASET_COMMIT="59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0"
GSPLAT_COMMIT="6e31ad766d39e0c33f9034a2ed772d51364b2343"
VISER_COMMIT="57142e42df8edd4de33fd60a08d6bb6c35970aa1"
TCNN_COMMIT="8e6e242f36dd197134c9b9275a8e5108a8e3af78"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_stage_h3_environment.sh [--repair]

Creates or repairs the isolated H3 SplatAD environment, then runs the
synthetic GPU acceptance check. It does not download PandaSet.

Environment:
  H3_ROOT   External code/data/output root
  H3_ENV    Conda prefix (default: $H3_ROOT/envs/h3_splatad)
  CONDA_BIN Conda executable
EOF
}

if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
  shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (($#)); then
  usage >&2
  exit 2
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable not found: $CONDA_BIN" >&2
  exit 1
fi

CODE_ROOT="${H3_ROOT}/code"
NEURAD_DIR="${CODE_ROOT}/neurad-studio"
PANDASET_DIR="${CODE_ROOT}/pandaset-devkit"
GSPLAT_DIR="${CODE_ROOT}/splatad-gsplat"
VISER_DIR="${CODE_ROOT}/viser-neurad"

clone_at_commit() {
  local url="$1"
  local target="$2"
  local commit="$3"

  if [[ ! -d "${target}/.git" ]]; then
    git clone "$url" "$target"
  fi
  if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo "Refusing to change dirty upstream checkout: $target" >&2
    exit 1
  fi
  if ! git -C "$target" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$target" fetch origin "$commit"
  fi
  git -C "$target" checkout --detach "$commit"
}

mkdir -p \
  "$CODE_ROOT" \
  "${H3_ROOT}/data" \
  "${H3_ROOT}/outputs" \
  "${H3_ROOT}/artifacts" \
  "${H3_ROOT}/cache/pip" \
  "${H3_ROOT}/cache/torch_extensions" \
  "${H3_ROOT}/cache/matplotlib" \
  "${H3_ROOT}/envs"

clone_at_commit \
  https://github.com/georghess/neurad-studio.git \
  "$NEURAD_DIR" \
  "$NEURAD_COMMIT"
clone_at_commit \
  https://github.com/scaleapi/pandaset-devkit.git \
  "$PANDASET_DIR" \
  "$PANDASET_COMMIT"
clone_at_commit \
  https://github.com/carlinds/splatad.git \
  "$GSPLAT_DIR" \
  "$GSPLAT_COMMIT"
clone_at_commit \
  https://github.com/atonderski/viser.git \
  "$VISER_DIR" \
  "$VISER_COMMIT"

if [[ -x "${H3_ENV}/bin/ns-train" && "$REPAIR" -eq 0 ]]; then
  echo "Existing H3 environment found; running acceptance check."
  exec "$REPO_ROOT/scripts/check_stage_h3_environment.sh"
fi

if [[ ! -x "${H3_ENV}/bin/python" ]]; then
  "$CONDA_BIN" create \
    --prefix "$H3_ENV" \
    --yes \
    python=3.10.20 \
    pip
fi

"$CONDA_BIN" install \
  --prefix "$H3_ENV" \
  --yes \
  --channel nvidia/label/cuda-11.8.0 \
  cuda-toolkit=11.8.0

PYTHON="${H3_ENV}/bin/python"
export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${H3_ROOT}/cache/torch_extensions"
export PIP_CACHE_DIR="${H3_ROOT}/cache/pip"
export MPLCONFIGDIR="${H3_ROOT}/cache/matplotlib"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install \
  "setuptools==69.5.1" \
  "numpy==1.24.4" \
  "dataclass-wizard==0.35.1" \
  dill \
  ninja
"$PYTHON" -m pip install \
  "torch==2.0.1+cu118" \
  "torchvision==0.15.2+cu118" \
  --extra-index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install \
  --no-build-isolation \
  "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn.git@${TCNN_COMMIT}#subdirectory=bindings/torch"
"$PYTHON" -m pip install --editable "$NEURAD_DIR"

# neurad-studio's direct URL dependencies are not commit-pinned in pyproject.
# Install the audited local revisions last so the final environment is fixed.
"$PYTHON" -m pip install --no-deps --editable "${PANDASET_DIR}/python"
"$PYTHON" -m pip install --no-deps --editable "$VISER_DIR"
"$PYTHON" -m pip install --no-build-isolation --no-deps --editable "$GSPLAT_DIR"

# zod accepts newer dataclass-wizard releases by metadata but still imports an
# API removed in 1.0. Reapply the tested compatibility pins after resolution.
"$PYTHON" -m pip install \
  "numpy==1.24.4" \
  "dataclass-wizard==0.35.1" \
  "setuptools==69.5.1"
"$PYTHON" -m pip check

exec "$REPO_ROOT/scripts/check_stage_h3_environment.sh"
