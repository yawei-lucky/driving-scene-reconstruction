#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
DATA_ROOT="${PANDASET_DATA_ROOT:-${H3_ROOT}/data/pandaset}"
SCENE="${PANDASET_SCENE:-040}"
EXPERIMENT="${H3_EXPERIMENT_NAME:-scene_040_splatad_smoke_100}"
RUN_TIMESTAMP="${H3_RUN_TIMESTAMP:-2026-07-19_100step}"
TRAIN_ROOT="${H3_TRAIN_ROOT:-${H3_ROOT}/outputs/pandaset_h3}"
RUN_ROOT="${TRAIN_ROOT}/${EXPERIMENT}/splatad/${RUN_TIMESTAMP}"
CONFIG="${RUN_ROOT}/config.yml"
CHECKPOINT="${RUN_ROOT}/nerfstudio_models/step-000000099.ckpt"
CALIBRATION_ROOT="${H3_CALIBRATION_ROOT:-${H3_ROOT}/artifacts/scene_040_calibration}"
RENDER_ROOT="${H3_RENDER_ROOT:-${H3_ROOT}/artifacts/scene_040_smoke_100_render}"
PILOT_EXPERIMENT="${H3_PILOT_EXPERIMENT_NAME:-scene_040_splatad_pilot_2000}"
PILOT_TIMESTAMP="${H3_PILOT_TIMESTAMP:-2026-07-19_train90_seed750k}"
PILOT_RUN_ROOT="${TRAIN_ROOT}/${PILOT_EXPERIMENT}/splatad/${PILOT_TIMESTAMP}"
PILOT_CONFIG="${PILOT_RUN_ROOT}/config.yml"
PILOT_CHECKPOINT="${PILOT_RUN_ROOT}/nerfstudio_models/step-000001999.ckpt"
PILOT_RENDER_ROOT="${H3_PILOT_RENDER_ROOT:-${H3_ROOT}/artifacts/scene_040_pilot_2000_render}"
PYTHON="${H3_ENV}/bin/python"

usage() {
  echo "Usage: $0 {data-gate|smoke|render-smoke|pilot|render-pilot|paths}" >&2
}

if [[ ! -x "$PYTHON" ]]; then
  echo "Stage H3 Python interpreter not found: $PYTHON" >&2
  echo "Run scripts/setup_stage_h3_environment.sh first." >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/${SCENE}" ]]; then
  echo "PandaSet scene not found: ${DATA_ROOT}/${SCENE}" >&2
  exit 1
fi

export CUDA_HOME="$H3_ENV"
export PATH="${H3_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${H3_ROOT}/cache/matplotlib}"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"

MODE="${1:-}"
case "$MODE" in
  data-gate)
    "$PYTHON" "$REPO_ROOT/scripts/inspect_stage_h3_pandaset.py" \
      --data-root "$DATA_ROOT" \
      --sequence "$SCENE" \
      --frames 0,40,79 \
      --output-dir "$CALIBRATION_ROOT"
    ;;
  smoke)
    if [[ -f "$CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 100-step checkpoint: $CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    "${H3_ENV}/bin/ns-train" splatad \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$EXPERIMENT" \
      --timestamp "$RUN_TIMESTAMP" \
      --vis tensorboard \
      --max-num-iterations 100 \
      --steps-per-save 100 \
      --steps-per-eval-image 50 \
      --steps-per-eval-all-images 100000 \
      --pipeline.calc-fid-steps 999999 \
      --pipeline.datamanager.max-thread-workers 8 \
      --pipeline.datamanager.downsample-factor 0.25 \
      --pipeline.model.max-steps 100 \
      --pipeline.model.max-num-seed-points 250000 \
      pandaset-data \
      --data "$DATA_ROOT" \
      --sequence "$SCENE"
    ;;
  render-smoke)
    if [[ ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      echo "Smoke config/checkpoint is missing under: $RUN_ROOT" >&2
      exit 1
    fi
    if [[ -f "${RENDER_ROOT}/metrics.pkl" && "${H3_ALLOW_RERENDER:-0}" != "1" ]]; then
      echo "PASS: reusing existing test render and metrics: $RENDER_ROOT"
      echo "Set H3_ALLOW_RERENDER=1 only when an intentional rerender is required."
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
      echo "PASS: reusing existing 2,000-step checkpoint: $PILOT_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    "${H3_ENV}/bin/ns-train" splatad \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$PILOT_EXPERIMENT" \
      --timestamp "$PILOT_TIMESTAMP" \
      --vis tensorboard \
      --max-num-iterations 2000 \
      --steps-per-save 1000 \
      --steps-per-eval-image 500 \
      --steps-per-eval-all-images 100000 \
      --pipeline.calc-fid-steps 999999 \
      --pipeline.datamanager.max-thread-workers 8 \
      --pipeline.datamanager.downsample-factor 0.5 \
      --pipeline.model.max-steps 2000 \
      --pipeline.model.max-num-seed-points 750000 \
      pandaset-data \
      --data "$DATA_ROOT" \
      --sequence "$SCENE" \
      --train-split-fraction 0.9
    ;;
  render-pilot)
    if [[ ! -f "$PILOT_CONFIG" || ! -f "$PILOT_CHECKPOINT" ]]; then
      echo "Pilot config/checkpoint is missing under: $PILOT_RUN_ROOT" >&2
      exit 1
    fi
    if [[ -f "${PILOT_RENDER_ROOT}/metrics.pkl" && "${H3_ALLOW_RERENDER:-0}" != "1" ]]; then
      echo "PASS: reusing existing pilot test render and metrics: $PILOT_RENDER_ROOT"
      echo "Set H3_ALLOW_RERENDER=1 only when an intentional rerender is required."
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
    echo "data: $DATA_ROOT/$SCENE"
    echo "calibration: $CALIBRATION_ROOT"
    echo "config: $CONFIG"
    echo "checkpoint: $CHECKPOINT"
    echo "test render: $RENDER_ROOT"
    echo "pilot config: $PILOT_CONFIG"
    echo "pilot checkpoint: $PILOT_CHECKPOINT"
    echo "pilot test render: $PILOT_RENDER_ROOT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
