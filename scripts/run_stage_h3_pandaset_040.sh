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
STATIC_EXPERIMENT="${H3_STATIC_EXPERIMENT_NAME:-scene_040_splatad_static_8000}"
STATIC_TIMESTAMP="${H3_STATIC_TIMESTAMP:-2026-07-19_resume_2k_to_8k}"
STATIC_RUN_ROOT="${TRAIN_ROOT}/${STATIC_EXPERIMENT}/splatad/${STATIC_TIMESTAMP}"
STATIC_CONFIG="${STATIC_RUN_ROOT}/config.yml"
STATIC_CHECKPOINT="${STATIC_RUN_ROOT}/nerfstudio_models/step-000007999.ckpt"
LOGGED_RENDER_ROOT="${H3_LOGGED_RENDER_ROOT:-${H3_ROOT}/artifacts/scene_040_logged_renderer_mvp}"
DRIVABILITY_ROOT="${H3_DRIVABILITY_ROOT:-${H3_ROOT}/artifacts/scene_040_drivability_preflight}"
BROWSER_TRIAL_ROOT="${H3_BROWSER_TRIAL_ROOT:-${H3_ROOT}/artifacts/scene_040_browser_trial}"
BROWSER_TRIAL_OUTPUT="${H3_BROWSER_TRIAL_OUTPUT:-${BROWSER_TRIAL_ROOT}/browser_trial.json}"
TRIAL_CHECK_OUTPUT="${H3_TRIAL_CHECK_OUTPUT:-${BROWSER_TRIAL_ROOT}/browser_trial_acceptance_check.json}"
BROWSER_REHEARSAL_ROOT="${H3_BROWSER_REHEARSAL_ROOT:-${H3_ROOT}/artifacts/scene_040_browser_trial_rehearsal}"
BROWSER_REHEARSAL_OUTPUT="${H3_BROWSER_REHEARSAL_OUTPUT:-${BROWSER_REHEARSAL_ROOT}/browser_trial_rehearsal.json}"
BROWSER_REHEARSAL_CHECK_OUTPUT="${H3_BROWSER_REHEARSAL_CHECK_OUTPUT:-${BROWSER_REHEARSAL_ROOT}/browser_trial_acceptance_check.json}"
VEHICLE_EXPERIMENT="${H3_VEHICLE_EXPERIMENT_NAME:-scene_040_splatad_vehicle_objects_8000}"
VEHICLE_TIMESTAMP="${H3_VEHICLE_TIMESTAMP:-2026-07-19_stationary_moving_actor_aware}"
VEHICLE_RUN_ROOT="${TRAIN_ROOT}/${VEHICLE_EXPERIMENT}/splatad/${VEHICLE_TIMESTAMP}"
VEHICLE_CONFIG="${VEHICLE_RUN_ROOT}/config.yml"
VEHICLE_CHECKPOINT="${VEHICLE_RUN_ROOT}/nerfstudio_models/step-000007999.ckpt"
MOVING_EXPERIMENT="${H3_MOVING_EXPERIMENT_NAME:-scene_040_splatad_moving_actor_aware_8000}"
MOVING_TIMESTAMP="${H3_MOVING_TIMESTAMP:-2026-07-19_moving_only_actor_aware}"
MOVING_RUN_ROOT="${TRAIN_ROOT}/${MOVING_EXPERIMENT}/splatad/${MOVING_TIMESTAMP}"
MOVING_CONFIG="${MOVING_RUN_ROOT}/config.yml"
MOVING_CHECKPOINT="${MOVING_RUN_ROOT}/nerfstudio_models/step-000007999.ckpt"
CONSTRAINED_EXPERIMENT="${H3_CONSTRAINED_EXPERIMENT_NAME:-scene_040_splatad_moving_constrained_2000}"
CONSTRAINED_TIMESTAMP="${H3_CONSTRAINED_TIMESTAMP:-2026-07-20_actor_bounds}"
CONSTRAINED_RUN_ROOT="${TRAIN_ROOT}/${CONSTRAINED_EXPERIMENT}/splatad/${CONSTRAINED_TIMESTAMP}"
CONSTRAINED_CONFIG="${CONSTRAINED_RUN_ROOT}/config.yml"
CONSTRAINED_CHECKPOINT="${CONSTRAINED_RUN_ROOT}/nerfstudio_models/step-000001999.ckpt"
TIMED_EXPERIMENT="${H3_TIMED_EXPERIMENT_NAME:-scene_040_splatad_moving_constrained_timed_2000}"
TIMED_TIMESTAMP="${H3_TIMED_TIMESTAMP:-2026-07-20_actor_bounds_and_time_v2}"
TIMED_RUN_ROOT="${TRAIN_ROOT}/${TIMED_EXPERIMENT}/splatad/${TIMED_TIMESTAMP}"
TIMED_CONFIG="${TIMED_RUN_ROOT}/config.yml"
TIMED_CHECKPOINT="${TIMED_RUN_ROOT}/nerfstudio_models/step-000001999.ckpt"
PYTHON="${H3_ENV}/bin/python"

usage() {
  echo "Usage: $0 {data-gate|smoke|render-smoke|pilot|render-pilot|static-8k|logged-renderer-smoke|drivability-preflight|logged-browser|trial-rehearsal|trial-check|vehicle-8k|moving-8k|moving-constrained-2k|moving-constrained-timed-2k|paths}" >&2
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
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

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
  static-8k)
    if [[ -f "$STATIC_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 8,000-step static-background checkpoint: $STATIC_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    if [[ ! -f "$PILOT_CHECKPOINT" ]]; then
      echo "Accepted 2,000-step source checkpoint is missing: $PILOT_CHECKPOINT" >&2
      exit 1
    fi
    # Trainer max_num_iterations counts additional iterations after resume.
    # Loading step 1,999 plus 6,000 iterations produces final step 7,999. The
    # repository entrypoint works around the pinned trainer's optimizer-load
    # ordering bug while preserving the exact optimizer and scheduler state.
    "$PYTHON" "$REPO_ROOT/scripts/resume_stage_h3_exact.py" \
      --source-config "$PILOT_CONFIG" \
      --checkpoint "$PILOT_CHECKPOINT" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$STATIC_EXPERIMENT" \
      --timestamp "$STATIC_TIMESTAMP" \
      --additional-iterations 6000 \
      --model-max-steps 8000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 1000 \
      --keep-all-checkpoints
    ;;
  logged-renderer-smoke)
    if [[ ! -f "$STATIC_CONFIG" || ! -f "$STATIC_CHECKPOINT" ]]; then
      echo "Accepted static-8k config/checkpoint is missing: $STATIC_RUN_ROOT" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/examples/stage_h3_logged_renderer_smoke.py" \
      --config "$STATIC_CONFIG" \
      --output-dir "$LOGGED_RENDER_ROOT" \
      --output-scale "${H3_LOGGED_RENDER_SCALE:-0.5}" \
      --steps "${H3_LOGGED_RENDER_STEPS:-80}" \
      --dt "${H3_LOGGED_RENDER_DT:-0.1}" \
      --movement-profile "${H3_LOGGED_MOVEMENT_PROFILE:-safe}"
    ;;
  drivability-preflight)
    if [[ ! -f "$STATIC_CONFIG" || ! -f "$STATIC_CHECKPOINT" ]]; then
      echo "Accepted static-8k config/checkpoint is missing: $STATIC_RUN_ROOT" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/examples/stage_h3_drivability_preflight.py" \
      --config "$STATIC_CONFIG" \
      --output-dir "$DRIVABILITY_ROOT" \
      --output-scale "${H3_DRIVABILITY_RENDER_SCALE:-0.5}" \
      --steps "${H3_DRIVABILITY_STEPS:-80}" \
      --dt "${H3_LOGGED_RENDER_DT:-0.1}" \
      --movement-profile "${H3_DRIVABILITY_MOVEMENT_PROFILE:-visible}"
    ;;
  logged-browser)
    if [[ ! -f "$STATIC_CONFIG" || ! -f "$STATIC_CHECKPOINT" ]]; then
      echo "Accepted static-8k config/checkpoint is missing: $STATIC_RUN_ROOT" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/examples/stage_h3_logged_browser.py" \
      --config "$STATIC_CONFIG" \
      --output-scale "${H3_BROWSER_RENDER_SCALE:-0.25}" \
      --dt "${H3_LOGGED_RENDER_DT:-0.1}" \
      --movement-profile "${H3_BROWSER_MOVEMENT_PROFILE:-visible}" \
      --time-mode "${H3_BROWSER_TIME_MODE:-manual}" \
      --host "${H3_BROWSER_HOST:-0.0.0.0}" \
      --port "${H3_BROWSER_PORT:-8766}" \
      --trial-output "$BROWSER_TRIAL_OUTPUT"
    ;;
  trial-check)
    "$PYTHON" "$REPO_ROOT/examples/stage_h3_trial_acceptance_check.py" \
      --trial-json "${H3_TRIAL_JSON:-$BROWSER_TRIAL_OUTPUT}" \
      --output "$TRIAL_CHECK_OUTPUT" \
      --expected-scene "$SCENE" \
      --expected-checkpoint-step 7999 \
      --expected-movement-profile "${H3_TRIAL_EXPECTED_MOVEMENT_PROFILE:-visible}" \
      --min-sample-count "${H3_TRIAL_MIN_SAMPLE_COUNT:-70}" \
      --min-reset-count "${H3_TRIAL_MIN_RESET_COUNT:-1}" \
      --min-browser-input-samples "${H3_TRIAL_MIN_INPUT_SAMPLES:-1}" \
      --max-browser-request-to-image-p95-ms "${H3_TRIAL_MAX_REQUEST_TO_IMAGE_P95_MS:-100}" \
      --max-browser-input-to-image-p95-ms "${H3_TRIAL_MAX_INPUT_TO_IMAGE_P95_MS:-100}" \
      --max-server-control-to-jpeg-p95-ms "${H3_TRIAL_MAX_SERVER_TO_JPEG_P95_MS:-100}" \
      --max-camera-time-spread-p95-ms "${H3_TRIAL_MAX_CAMERA_SPREAD_P95_MS:-100}"
    ;;
  trial-rehearsal)
    "$PYTHON" "$REPO_ROOT/examples/stage_h3_browser_trial_rehearsal.py" \
      --base-url "${H3_BROWSER_BASE_URL:-http://127.0.0.1:${H3_BROWSER_PORT:-8766}}" \
      --steps "${H3_REHEARSAL_STEPS:-80}" \
      --output "$BROWSER_REHEARSAL_OUTPUT" \
      --acceptance-output "$BROWSER_REHEARSAL_CHECK_OUTPUT" \
      --expected-movement-profile "${H3_REHEARSAL_EXPECTED_MOVEMENT_PROFILE:-visible}"
    ;;
  vehicle-8k)
    if [[ -f "$VEHICLE_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 8,000-step vehicle-object checkpoint: $VEHICLE_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    if [[ ! -f "$PILOT_CONFIG" ]]; then
      echo "Accepted source config is missing: $PILOT_CONFIG" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_vehicle_objects.py" \
      --source-config "$PILOT_CONFIG" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$VEHICLE_EXPERIMENT" \
      --timestamp "$VEHICLE_TIMESTAMP" \
      --max-num-iterations 8000 \
      --model-max-steps 8000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 1000 \
      --keep-all-checkpoints
    ;;
  moving-8k)
    if [[ -f "$MOVING_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 8,000-step moving-only actor checkpoint: $MOVING_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    if [[ ! -f "$PILOT_CONFIG" ]]; then
      echo "Accepted source config is missing: $PILOT_CONFIG" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_vehicle_objects.py" \
      --source-config "$PILOT_CONFIG" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$MOVING_EXPERIMENT" \
      --timestamp "$MOVING_TIMESTAMP" \
      --max-num-iterations 8000 \
      --model-max-steps 8000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 1000 \
      --keep-all-checkpoints \
      --moving-only
    ;;
  moving-constrained-2k)
    if [[ -f "$CONSTRAINED_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 2,000-step constrained-actor checkpoint: $CONSTRAINED_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_vehicle_objects.py" \
      --source-config "$PILOT_CONFIG" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$CONSTRAINED_EXPERIMENT" \
      --timestamp "$CONSTRAINED_TIMESTAMP" \
      --max-num-iterations 2000 \
      --model-max-steps 2000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 1000 \
      --keep-all-checkpoints \
      --moving-only
    ;;
  moving-constrained-timed-2k)
    if [[ -f "$TIMED_CHECKPOINT" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing existing 2,000-step bounded+timed checkpoint: $TIMED_CHECKPOINT"
      echo "Set H3_ALLOW_RETRAIN=1 only when an intentional rerun is required."
      exit 0
    fi
    "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_vehicle_objects.py" \
      --source-config "$PILOT_CONFIG" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$TIMED_EXPERIMENT" \
      --timestamp "$TIMED_TIMESTAMP" \
      --max-num-iterations 2000 \
      --model-max-steps 2000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 1000 \
      --keep-all-checkpoints \
      --moving-only \
      --calibrated-cuboid-time
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
    echo "static 8k config: $STATIC_CONFIG"
    echo "static 8k checkpoint: $STATIC_CHECKPOINT"
    echo "logged renderer smoke: $LOGGED_RENDER_ROOT"
    echo "drivability preflight: $DRIVABILITY_ROOT"
    echo "browser trial report: $BROWSER_TRIAL_OUTPUT"
    echo "browser trial acceptance check: $TRIAL_CHECK_OUTPUT"
    echo "browser trial rehearsal report: $BROWSER_REHEARSAL_OUTPUT"
    echo "browser trial rehearsal check: $BROWSER_REHEARSAL_CHECK_OUTPUT"
    echo "vehicle 8k config: $VEHICLE_CONFIG"
    echo "vehicle 8k checkpoint: $VEHICLE_CHECKPOINT"
    echo "moving-only 8k config: $MOVING_CONFIG"
    echo "moving-only 8k checkpoint: $MOVING_CHECKPOINT"
    echo "constrained moving-only 2k config: $CONSTRAINED_CONFIG"
    echo "constrained moving-only 2k checkpoint: $CONSTRAINED_CHECKPOINT"
    echo "bounded+timed moving-only 2k config: $TIMED_CONFIG"
    echo "bounded+timed moving-only 2k checkpoint: $TIMED_CHECKPOINT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
