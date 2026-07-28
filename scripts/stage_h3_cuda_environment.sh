#!/usr/bin/env bash
# Shared CUDA contract for the SplatAD/NeuRAD Stage H3 environment.
#
# Source this file from project launchers. For one-off commands, use
# scripts/run_stage_h3_environment.sh instead of invoking H3 Python directly.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a launcher; do not execute it directly." >&2
  exit 2
fi

H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
STAGE_H3_PYTHON="${H3_ENV}/bin/python"
STAGE_H3_NVCC="${H3_ENV}/bin/nvcc"

if [[ ! -x "$STAGE_H3_PYTHON" ]]; then
  echo "Stage H3 Python interpreter not found: $STAGE_H3_PYTHON" >&2
  echo "Run scripts/setup_stage_h3_environment.sh first." >&2
  return 1
fi
if [[ ! -x "$STAGE_H3_NVCC" ]]; then
  echo "Stage H3 CUDA compiler not found: $STAGE_H3_NVCC" >&2
  echo "Do not fall back to /usr/bin/nvcc; it is CUDA 11.5 on this host." >&2
  return 1
fi

export H3_ROOT H3_ENV
export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/torch_extensions}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${H3_ROOT}/cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${H3_ROOT}/cache/matplotlib}"

stage_h3_assert_cuda_toolchain() {
  local selected_nvcc selected_real expected_real version_output
  selected_nvcc="$(command -v nvcc || true)"
  if [[ -z "$selected_nvcc" ]]; then
    echo "Stage H3 CUDA compiler is absent from PATH after activation." >&2
    return 1
  fi
  selected_real="$(readlink -f "$selected_nvcc")"
  expected_real="$(readlink -f "$STAGE_H3_NVCC")"
  if [[ "$selected_real" != "$expected_real" ]]; then
    echo "Refusing mixed Stage H3 CUDA toolchains." >&2
    echo "  expected nvcc: $STAGE_H3_NVCC -> $expected_real" >&2
    echo "  selected nvcc: $selected_nvcc -> $selected_real" >&2
    return 1
  fi
  version_output="$("$STAGE_H3_NVCC" --version)"
  if [[ "$version_output" != *"release 11.8,"* ]]; then
    echo "Stage H3 requires CUDA 11.8; selected compiler reports:" >&2
    while IFS= read -r line; do
      echo "  $line" >&2
    done <<< "$version_output"
    return 1
  fi
}

stage_h3_nvcc_release_line() {
  local line
  while IFS= read -r line; do
    if [[ "$line" == *"release "* ]]; then
      echo "$line"
      return 0
    fi
  done < <("$STAGE_H3_NVCC" --version)
  return 1
}

stage_h3_print_cuda_contract() {
  echo "Stage H3 CUDA contract:"
  echo "  python: $STAGE_H3_PYTHON"
  echo "  CUDA_HOME: $CUDA_HOME"
  echo "  nvcc: $(command -v nvcc)"
  echo "  nvcc version: $(stage_h3_nvcc_release_line)"
  echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
  echo "  TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
}

stage_h3_verify_torch_cuda_contract() {
  "$STAGE_H3_PYTHON" - <<'PY'
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

expected = Path(os.environ["CUDA_HOME"]).resolve()
selected = Path(CUDA_HOME).resolve() if CUDA_HOME else None
if torch.version.cuda != "11.8":
    raise SystemExit(
        f"Stage H3 requires torch CUDA 11.8; found {torch.version.cuda!r}"
    )
if selected != expected:
    raise SystemExit(
        f"torch extension CUDA_HOME mismatch: expected {expected}, found {selected}"
    )
print(f"  torch: {torch.__version__}")
print(f"  torch CUDA: {torch.version.cuda}")
print(f"  torch extension CUDA_HOME: {selected}")
print(f"  GPU visible to torch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
PY
}

stage_h3_assert_cuda_toolchain
