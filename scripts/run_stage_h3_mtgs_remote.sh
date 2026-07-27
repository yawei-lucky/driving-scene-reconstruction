#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
MTGS_ENV="${MTGS_ENV:-${H3_ROOT}/envs/mtgs}"
MTGS_CODE="${MTGS_CODE:-${H3_ROOT}/code/mtgs_7ab67a3}"
BLOCK="road_block-365000_144000_365100_144080"
MTGS_RUN_ROOT="${MTGS_RUN_ROOT:-${H3_ROOT}/outputs/mtgs_gate/${BLOCK}}"
CONFIG="${MTGS_RUN_ROOT}/config.yml"
CHECKPOINT="${MTGS_RUN_ROOT}/nerfstudio_models/step-000030000.ckpt"
ROAD_BLOCK_CONFIG="${MTGS_CODE}/nuplan_scripts/configs/mtgs_exp/${BLOCK}.yml"
PYTHON="${MTGS_ENV}/bin/python"
CONTROL_HOST="${MTGS_REMOTE_CONTROL_HOST:-0.0.0.0}"
CONTROL_PORT="${MTGS_REMOTE_CONTROL_PORT:-18765}"
VIDEO_PORT="${MTGS_REMOTE_VIDEO_PORT:-19001}"
VIDEO_DESTINATION="${MTGS_REMOTE_VIDEO_DESTINATION:-srt://0.0.0.0:${VIDEO_PORT}?mode=listener&latency=80&transtype=live}"
RECORD_PATH="${MTGS_REMOTE_RECORD:-${H3_ROOT}/artifacts/mtgs_remote_apps_20260726/mtgs_remote_auto_demo.mp4}"
PPT_OUTPUT_DIR="${MTGS_REMOTE_PPT_OUTPUT_DIR:-${H3_ROOT}/artifacts/mtgs_app_control_ppt_20260727_v2}"

export PYTHONPATH="${MTGS_CODE}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-${H3_ROOT}/envs/h3_splatad}"
export PATH="${MTGS_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${H3_ROOT}/cache/mtgs_torch_extensions}"
export NERFSTUDIO_DATAPARSER_CONFIGS="nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
export NERFSTUDIO_METHOD_CONFIGS="mtgs=mtgs.config.MTGS:method"

if [[ ! -x "$PYTHON" || ! -d "$MTGS_CODE" || ! -f "$CHECKPOINT" ]]; then
  echo "MTGS environment/code/checkpoint is incomplete" >&2
  exit 1
fi

token_args=()
if [[ -n "${MTGS_REMOTE_TOKEN:-}" ]]; then
  token_args=(--token "$MTGS_REMOTE_TOKEN")
fi

MODE="${1:-}"
case "$MODE" in
  server)
    cd "$MTGS_CODE"
    exec "$PYTHON" "$REPO_ROOT/apps/mtgs_remote_simulator.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --control-host "$CONTROL_HOST" \
      --control-port "$CONTROL_PORT" \
      --video-destination "$VIDEO_DESTINATION" \
      "${token_args[@]}"
    ;;
  record-demo)
    cd "$MTGS_CODE"
    exec "$PYTHON" "$REPO_ROOT/apps/mtgs_remote_simulator.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --record "$RECORD_PATH" \
      --max-frames 183 \
      --no-control-server
    ;;
  ppt-demo)
    cd "$MTGS_CODE"
    exec "$PYTHON" "$REPO_ROOT/scripts/build_stage_h3_mtgs_app_control_demo.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --road-block-config "$ROAD_BLOCK_CONFIG" \
      --output-dir "$PPT_OUTPUT_DIR"
    ;;
  paths)
    echo "simulator app: $REPO_ROOT/apps/mtgs_remote_simulator.py"
    echo "driver app: $REPO_ROOT/apps/mtgs_remote_driver.py"
    echo "control listen: ws://${CONTROL_HOST}:${CONTROL_PORT}"
    echo "video destination: $VIDEO_DESTINATION"
    echo "record path: $RECORD_PATH"
    echo "PPT demo output: $PPT_OUTPUT_DIR"
    ;;
  *)
    echo "Usage: $0 {server|record-demo|ppt-demo|paths}" >&2
    exit 2
    ;;
esac
