#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
H3_CODE="${H3_CODE:-${H3_ROOT}/code/neurad-studio}"
DATA_ROOT="${TBV_DATA_ROOT:-${H3_ROOT}/data/tbv_branch_pilot}"
TRAIN_ROOT="${H3_TBV_TRAIN_ROOT:-${H3_ROOT}/outputs/tbv_h3}"
EXPERIMENT="${H3_TBV_EXPERIMENT_NAME:-tbv_branch_pair_splatad_smoke_100}"
RUN_TIMESTAMP="${H3_TBV_RUN_TIMESTAMP:-2026-07-22_100step}"
RUN_ROOT="${TRAIN_ROOT}/${EXPERIMENT}/splatad/${RUN_TIMESTAMP}"
CONFIG="${RUN_ROOT}/config.yml"
CHECKPOINT="${RUN_ROOT}/nerfstudio_models/step-000000099.ckpt"
RENDER_ROOT="${H3_TBV_RENDER_ROOT:-${H3_ROOT}/artifacts/tbv_branch_pair_smoke_100_render}"
PILOT_EXPERIMENT="${H3_TBV_PILOT_EXPERIMENT_NAME:-tbv_branch_pair_splatad_pilot_2000}"
PILOT_TIMESTAMP="${H3_TBV_PILOT_TIMESTAMP:-2026-07-22_train90_seed750k}"
PILOT_RUN_ROOT="${TRAIN_ROOT}/${PILOT_EXPERIMENT}/splatad/${PILOT_TIMESTAMP}"
PILOT_CONFIG="${PILOT_RUN_ROOT}/config.yml"
PILOT_CHECKPOINT="${PILOT_RUN_ROOT}/nerfstudio_models/step-000001999.ckpt"
PILOT_RENDER_ROOT="${H3_TBV_PILOT_RENDER_ROOT:-${H3_ROOT}/artifacts/tbv_branch_pair_pilot_2000_render}"
PYTHON="${H3_ENV}/bin/python"

export PYTHONPATH="${REPO_ROOT}/scripts:${H3_CODE}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${H3_ROOT}/cache/matplotlib}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

MODE="${1:-}"
case "$MODE" in
  download)
    "$PYTHON" "$REPO_ROOT/scripts/download_stage_h3_tbv_window.py" \
      --output-dir "$DATA_ROOT" --workers "${H3_TBV_DOWNLOAD_WORKERS:-16}"
    ;;
  smoke)
    if [[ -f "$CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing TbV 100-step checkpoint: $CHECKPOINT"
      exit 0
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_tbv_smoke.py" \
      --data "$DATA_ROOT" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$EXPERIMENT" \
      --timestamp "$RUN_TIMESTAMP"
    ;;
  data-gate)
    "$PYTHON" "$REPO_ROOT/scripts/inspect_stage_h3_tbv_pilot.py" \
      --data "$DATA_ROOT" \
      --output "${H3_TBV_DATA_GATE:-${H3_ROOT}/artifacts/tbv_branch_pair_data_gate.json}"
    ;;
  render-smoke)
    if [[ ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      echo "TbV smoke config/checkpoint is missing under: $RUN_ROOT" >&2
      exit 1
    fi
    if [[ -f "${RENDER_ROOT}/metrics.pkl" && "${H3_ALLOW_RERENDER:-0}" != "1" ]]; then
      echo "PASS: reusing existing TbV smoke render: $RENDER_ROOT"
      exit 0
    fi
    "${H3_ENV}/bin/ns-render" dataset \
      --load-config "$CONFIG" \
      --output-path "$RENDER_ROOT" \
      --rendered-output-names rgb gt-rgb depth \
      --pose-source test \
      --image-format jpeg \
      --jpeg-quality 95 \
      --calculate-and-save-metrics True
    ;;
  pilot)
    if [[ -f "$PILOT_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing TbV 2,000-step checkpoint: $PILOT_CHECKPOINT"
      exit 0
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_tbv_smoke.py" \
      --data "$DATA_ROOT" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$PILOT_EXPERIMENT" \
      --timestamp "$PILOT_TIMESTAMP" \
      --iterations 2000 \
      --downsample-factor 0.5 \
      --max-num-seed-points 750000
    ;;
  render-pilot)
    if [[ ! -f "$PILOT_CONFIG" || ! -f "$PILOT_CHECKPOINT" ]]; then
      echo "TbV pilot config/checkpoint is missing under: $PILOT_RUN_ROOT" >&2
      exit 1
    fi
    if [[ -f "${PILOT_RENDER_ROOT}/metrics.pkl" && "${H3_ALLOW_RERENDER:-0}" != "1" ]]; then
      echo "PASS: reusing existing TbV pilot render: $PILOT_RENDER_ROOT"
      exit 0
    fi
    "${H3_ENV}/bin/ns-render" dataset \
      --load-config "$PILOT_CONFIG" \
      --output-path "$PILOT_RENDER_ROOT" \
      --rendered-output-names rgb gt-rgb depth \
      --pose-source test \
      --image-format jpeg \
      --jpeg-quality 95 \
      --calculate-and-save-metrics True
    ;;
  paths)
    echo "data: $DATA_ROOT"
    echo "config: $CONFIG"
    echo "checkpoint: $CHECKPOINT"
    echo "render: $RENDER_ROOT"
    echo "pilot config: $PILOT_CONFIG"
    echo "pilot checkpoint: $PILOT_CHECKPOINT"
    echo "pilot render: $PILOT_RENDER_ROOT"
    ;;
  *)
    echo "Usage: $0 {download|data-gate|smoke|render-smoke|pilot|render-pilot|paths}" >&2
    exit 2
    ;;
esac
