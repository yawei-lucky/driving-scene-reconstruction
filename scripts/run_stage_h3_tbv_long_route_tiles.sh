#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
H3_CODE="${H3_CODE:-${H3_ROOT}/code/neurad-studio}"
DATA_ROOT="${H3_TBV_LONG_DATA_ROOT:-${H3_ROOT}/data/tbv_long_route_tiles_2_3}"
TRAIN_ROOT="${H3_TBV_LONG_TRAIN_ROOT:-${H3_ROOT}/outputs/tbv_long_route_tiles}"
ARTIFACT_ROOT="${H3_TBV_LONG_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_20260725}"
PILOT_ARTIFACT_ROOT="${H3_TBV_LONG_PILOT_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_500_20260725}"
PYTHON="${H3_ENV}/bin/python"

REFERENCE="V17LgyVPyrd2yjWS4oEuipUBJQN5X0wZ__Spring_2020"
REPEAT="cTrSOEc1gW3XqELP562UlUFJYCmlRoa9__Spring_2020"
TILE2_REFERENCE_START="315970596.3874255"
TILE2_REFERENCE_END="315970606.9574283"
TILE2_REPEAT_START="315976184.9774825"
TILE2_REPEAT_END="315976194.16245127"
TILE3_REFERENCE_START="315970604.760172"
TILE3_REFERENCE_END="315970613.7774825"
TILE3_REPEAT_START="315976192.6224129"
TILE3_REPEAT_END="315976200.16045"

TILE2_EXPERIMENT="tbv_long_route_tile_2_smoke_100"
TILE3_EXPERIMENT="tbv_long_route_tile_3_smoke_100"
RUN_TIMESTAMP="2026-07-25_100step"
TILE2_RUN="${TRAIN_ROOT}/${TILE2_EXPERIMENT}/splatad/${RUN_TIMESTAMP}"
TILE3_RUN="${TRAIN_ROOT}/${TILE3_EXPERIMENT}/splatad/${RUN_TIMESTAMP}"
TILE2_CONFIG="${TILE2_RUN}/config.yml"
TILE3_CONFIG="${TILE3_RUN}/config.yml"
TILE2_CHECKPOINT="${TILE2_RUN}/nerfstudio_models/step-000000099.ckpt"
TILE3_CHECKPOINT="${TILE3_RUN}/nerfstudio_models/step-000000099.ckpt"
PILOT_TIMESTAMP="2026-07-25_500step"
TILE2_PILOT_EXPERIMENT="tbv_long_route_tile_2_pilot_500"
TILE3_PILOT_EXPERIMENT="tbv_long_route_tile_3_pilot_500"
TILE2_PILOT_RUN="${TRAIN_ROOT}/${TILE2_PILOT_EXPERIMENT}/splatad/${PILOT_TIMESTAMP}"
TILE3_PILOT_RUN="${TRAIN_ROOT}/${TILE3_PILOT_EXPERIMENT}/splatad/${PILOT_TIMESTAMP}"
TILE2_PILOT_CONFIG="${TILE2_PILOT_RUN}/config.yml"
TILE3_PILOT_CONFIG="${TILE3_PILOT_RUN}/config.yml"
TILE2_PILOT_CHECKPOINT="${TILE2_PILOT_RUN}/nerfstudio_models/step-000000499.ckpt"
TILE3_PILOT_CHECKPOINT="${TILE3_PILOT_RUN}/nerfstudio_models/step-000000499.ckpt"

export PYTHONPATH="${REPO_ROOT}/scripts:${H3_CODE}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${H3_ROOT}/cache/matplotlib}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

train_tile() {
  local tile="$1"
  local experiment="$2"
  local checkpoint="$3"
  local reference_start="$4"
  local reference_end="$5"
  local repeat_start="$6"
  local repeat_end="$7"
  local iterations="$8"
  local timestamp="$9"
  if [[ -f "$checkpoint" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
    echo "PASS: reusing tile ${tile} checkpoint: $checkpoint"
    return
  fi
  "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_tbv_smoke.py" \
    --data "$DATA_ROOT" \
    --output-dir "$TRAIN_ROOT" \
    --experiment-name "$experiment" \
    --timestamp "$timestamp" \
    --iterations "$iterations" \
    --sequence "$REFERENCE" \
    --window-start-seconds "$reference_start" \
    --window-end-seconds "$reference_end" \
    --sequence "$REPEAT" \
    --window-start-seconds "$repeat_start" \
    --window-end-seconds "$repeat_end"
}

MODE="${1:-}"
case "$MODE" in
  download)
    "$PYTHON" "$REPO_ROOT/scripts/download_stage_h3_tbv_window.py" \
      --output-dir "$DATA_ROOT" \
      --workers "${H3_TBV_DOWNLOAD_WORKERS:-16}" \
      --window "$REFERENCE,$TILE2_REFERENCE_START,$TILE2_REFERENCE_END" \
      --window "$REPEAT,$TILE2_REPEAT_START,$TILE2_REPEAT_END" \
      --window "$REFERENCE,$TILE3_REFERENCE_START,$TILE3_REFERENCE_END" \
      --window "$REPEAT,$TILE3_REPEAT_START,$TILE3_REPEAT_END"
    ;;
  smoke-2)
    train_tile 2 "$TILE2_EXPERIMENT" "$TILE2_CHECKPOINT" \
      "$TILE2_REFERENCE_START" "$TILE2_REFERENCE_END" \
      "$TILE2_REPEAT_START" "$TILE2_REPEAT_END" 100 "$RUN_TIMESTAMP"
    ;;
  smoke-3)
    train_tile 3 "$TILE3_EXPERIMENT" "$TILE3_CHECKPOINT" \
      "$TILE3_REFERENCE_START" "$TILE3_REFERENCE_END" \
      "$TILE3_REPEAT_START" "$TILE3_REPEAT_END" 100 "$RUN_TIMESTAMP"
    ;;
  pilot-2)
    train_tile 2 "$TILE2_PILOT_EXPERIMENT" "$TILE2_PILOT_CHECKPOINT" \
      "$TILE2_REFERENCE_START" "$TILE2_REFERENCE_END" \
      "$TILE2_REPEAT_START" "$TILE2_REPEAT_END" 500 "$PILOT_TIMESTAMP"
    ;;
  pilot-3)
    train_tile 3 "$TILE3_PILOT_EXPERIMENT" "$TILE3_PILOT_CHECKPOINT" \
      "$TILE3_REFERENCE_START" "$TILE3_REFERENCE_END" \
      "$TILE3_REPEAT_START" "$TILE3_REPEAT_END" 500 "$PILOT_TIMESTAMP"
    ;;
  seam)
    if [[ ! -f "$TILE2_CONFIG" || ! -f "$TILE2_CHECKPOINT" ||
          ! -f "$TILE3_CONFIG" || ! -f "$TILE3_CHECKPOINT" ]]; then
      echo "both tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_CONFIG" \
      --tile-3-config "$TILE3_CONFIG" \
      --output-dir "$ARTIFACT_ROOT"
    ;;
  seam-500)
    if [[ ! -f "$TILE2_PILOT_CONFIG" ||
          ! -f "$TILE2_PILOT_CHECKPOINT" ||
          ! -f "$TILE3_PILOT_CONFIG" ||
          ! -f "$TILE3_PILOT_CHECKPOINT" ]]; then
      echo "both 500-step tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_PILOT_CONFIG" \
      --tile-3-config "$TILE3_PILOT_CONFIG" \
      --expected-checkpoint-step 499 \
      --output-dir "$PILOT_ARTIFACT_ROOT"
    ;;
  paths)
    echo "data: $DATA_ROOT"
    echo "tile 2 config: $TILE2_CONFIG"
    echo "tile 2 checkpoint: $TILE2_CHECKPOINT"
    echo "tile 3 config: $TILE3_CONFIG"
    echo "tile 3 checkpoint: $TILE3_CHECKPOINT"
    echo "seam evidence: $ARTIFACT_ROOT"
    echo "tile 2 pilot config: $TILE2_PILOT_CONFIG"
    echo "tile 2 pilot checkpoint: $TILE2_PILOT_CHECKPOINT"
    echo "tile 3 pilot config: $TILE3_PILOT_CONFIG"
    echo "tile 3 pilot checkpoint: $TILE3_PILOT_CHECKPOINT"
    echo "500-step seam evidence: $PILOT_ARTIFACT_ROOT"
    ;;
  *)
    echo "Usage: $0 {download|smoke-2|smoke-3|seam|pilot-2|pilot-3|seam-500|paths}" >&2
    exit 2
    ;;
esac
