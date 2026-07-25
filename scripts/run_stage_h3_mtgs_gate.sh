#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
MTGS_ENV="${MTGS_ENV:-${H3_ROOT}/envs/mtgs}"
MTGS_CODE="${MTGS_CODE:-${H3_ROOT}/code/mtgs_7ab67a3}"
MTGS_DATA_ROOT="${MTGS_DATA_ROOT:-${H3_ROOT}/data/mtgs_paper}"
BLOCK="road_block-365000_144000_365100_144080"
MTGS_RUN_ROOT="${MTGS_RUN_ROOT:-${H3_ROOT}/outputs/mtgs_gate/${BLOCK}}"
MTGS_ARTIFACT_ROOT="${MTGS_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/mtgs_checkpoint_gate_20260725}"
MTGS_DRIVE_ARTIFACT_ROOT="${MTGS_DRIVE_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/mtgs_continuous_drive_20260725_v4}"
CONFIG="${MTGS_RUN_ROOT}/config.yml"
CHECKPOINT="${MTGS_RUN_ROOT}/nerfstudio_models/step-000030000.ckpt"
ROAD_BLOCK_CONFIG="${MTGS_CODE}/nuplan_scripts/configs/mtgs_exp/${BLOCK}.yml"
PYTHON="${MTGS_ENV}/bin/python"

export PYTHONPATH="${MTGS_CODE}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-${H3_ROOT}/envs/h3_splatad}"
export PATH="${MTGS_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/mtgs_torch_extensions}"
export NERFSTUDIO_DATAPARSER_CONFIGS="nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
export NERFSTUDIO_METHOD_CONFIGS="mtgs=mtgs.config.MTGS:method"

MODE="${1:-}"
case "$MODE" in
  verify-assets)
    test "$(stat -c %s "${MTGS_DATA_ROOT}/${BLOCK}.tar.gz")" = "3983371020"
    test "$(sha256sum "${MTGS_DATA_ROOT}/${BLOCK}.tar.gz" | cut -d' ' -f1)" = \
      "d75c7e4ed0ec675d1ee7c656aa1695738f04b154bc30482ded6706661470808c"
    test "$(stat -c %s "$CHECKPOINT")" = "773241943"
    "$PYTHON" -c "import torch; value=torch.load('$CHECKPOINT', map_location='cpu'); assert value['step'] == 30000; print('PASS: MTGS assets are complete')"
    ;;
  checkpoint-gate)
    if [[ ! -x "$PYTHON" || ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      echo "MTGS environment/config/checkpoint is incomplete" >&2
      exit 1
    fi
    cd "$MTGS_CODE"
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_mtgs_checkpoint.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --output-dir "$MTGS_ARTIFACT_ROOT"
    ;;
  corridor-probe)
    if [[ ! -x "$PYTHON" || ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      echo "MTGS environment/config/checkpoint is incomplete" >&2
      exit 1
    fi
    cd "$MTGS_CODE"
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_mtgs_checkpoint.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --output-dir "$MTGS_ARTIFACT_ROOT" \
      --corridor-probe
    ;;
  continuous-drive)
    if [[ ! -x "$PYTHON" || ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      echo "MTGS environment/config/checkpoint is incomplete" >&2
      exit 1
    fi
    cd "$MTGS_CODE"
    "$PYTHON" "$REPO_ROOT/scripts/run_stage_h3_mtgs_continuous_drive.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --output-dir "$MTGS_DRIVE_ARTIFACT_ROOT"
    ;;
  paths)
    echo "environment: $MTGS_ENV"
    echo "code: $MTGS_CODE"
    echo "data: ${MTGS_DATA_ROOT}/MTGS/${BLOCK}"
    echo "config: $CONFIG"
    echo "checkpoint: $CHECKPOINT"
    echo "artifacts: $MTGS_ARTIFACT_ROOT"
    echo "continuous drive artifacts: $MTGS_DRIVE_ARTIFACT_ROOT"
    ;;
  *)
    echo "Usage: $0 {verify-assets|checkpoint-gate|corridor-probe|continuous-drive|paths}" >&2
    exit 2
    ;;
esac
