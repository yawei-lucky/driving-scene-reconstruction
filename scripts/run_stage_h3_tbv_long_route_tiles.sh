#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
H3_ENV="${H3_ENV:-${H3_ROOT}/envs/h3_splatad}"
H3_CODE="${H3_CODE:-${H3_ROOT}/code/neurad-studio}"
DATA_ROOT="${H3_TBV_LONG_DATA_ROOT:-${H3_ROOT}/data/tbv_long_route_tiles_2_3}"
TILE23_DATA_ROOT="${H3_TBV_LONG_TILE23_DATA_ROOT:-${H3_ROOT}/data/tbv_long_route_tiles_2_3_frozen_20260725}"
MASK_ROOT="${H3_TBV_LONG_MASK_ROOT:-${H3_ROOT}/data/tbv_long_route_tiles_2_3_vehicle_masks}"
CROSS_VISIT_ROOT="${H3_TBV_LONG_CROSS_VISIT_ROOT:-${H3_ROOT}/data/tbv_long_route_tile_2_cross_visit_rgb_20260728}"
PERSISTENCE_ROOT="${H3_TBV_LONG_PERSISTENCE_ROOT:-${H3_ROOT}/data/tbv_long_route_tile_2_persistence_020_n1_20260728}"
TRAIN_ROOT="${H3_TBV_LONG_TRAIN_ROOT:-${H3_ROOT}/outputs/tbv_long_route_tiles}"
ARTIFACT_ROOT="${H3_TBV_LONG_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_20260725}"
PILOT_ARTIFACT_ROOT="${H3_TBV_LONG_PILOT_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_500_20260725}"
QUALITY_ARTIFACT_ROOT="${H3_TBV_LONG_QUALITY_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_2000_20260725}"
CONTINUOUS_ARTIFACT_ROOT="${H3_TBV_LONG_CONTINUOUS_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_continuous_2000_20260725}"
STATIC_ARTIFACT_ROOT="${H3_TBV_LONG_STATIC_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_8000_20260725}"
STATIC_CONTINUOUS_ROOT="${H3_TBV_LONG_STATIC_CONTINUOUS_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_continuous_8000_20260725}"
MASKED_ARTIFACT_ROOT="${H3_TBV_LONG_MASKED_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_masked_2000_20260726}"
MASKED_CONTINUOUS_ROOT="${H3_TBV_LONG_MASKED_CONTINUOUS_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_continuous_masked_2000_20260726}"
JOINT_MASK_AUDIT_ROOT="${H3_TBV_LONG_JOINT_MASK_AUDIT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_rgb_lidar_mask_audit_20260726}"
JOINT_MASKED_ARTIFACT_ROOT="${H3_TBV_LONG_JOINT_MASKED_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_seam_rgb_lidar_masked_2000_20260726}"
JOINT_MASKED_CONTINUOUS_ROOT="${H3_TBV_LONG_JOINT_MASKED_CONTINUOUS_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_continuous_rgb_lidar_masked_2000_20260726}"
TILE34_QUALITY_ARTIFACT_ROOT="${H3_TBV_LONG_TILE34_QUALITY_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_3_4_seam_2000_isolated_20260728}"
TILE34_STATIC_ARTIFACT_ROOT="${H3_TBV_LONG_TILE34_STATIC_ARTIFACT_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_3_4_seam_8000_isolated_20260728}"
THREE_TILE_DRIVE_ROOT="${H3_TBV_LONG_THREE_TILE_DRIVE_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_three_tile_drive_8000_20260728}"
CROSS_VISIT_AB_ROOT="${H3_TBV_LONG_CROSS_VISIT_AB_ROOT:-${H3_ROOT}/artifacts/tbv_long_route_tile_2_quality_ab_cross_visit_hires_10k_20260728}"
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
TILE4_REFERENCE_START="315970612.43742543"
TILE4_REFERENCE_END="315970620.28742546"
TILE4_REPEAT_START="315976198.81245124"
TILE4_REPEAT_END="315976206.459454"

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
QUALITY_TIMESTAMP="2026-07-25_2000step"
TILE2_QUALITY_EXPERIMENT="tbv_long_route_tile_2_seam_2000"
TILE3_QUALITY_EXPERIMENT="tbv_long_route_tile_3_seam_2000"
TILE2_QUALITY_RUN="${TRAIN_ROOT}/${TILE2_QUALITY_EXPERIMENT}/splatad/${QUALITY_TIMESTAMP}"
TILE3_QUALITY_RUN="${TRAIN_ROOT}/${TILE3_QUALITY_EXPERIMENT}/splatad/${QUALITY_TIMESTAMP}"
TILE2_QUALITY_CONFIG="${TILE2_QUALITY_RUN}/config.yml"
TILE3_QUALITY_CONFIG="${TILE3_QUALITY_RUN}/config.yml"
TILE2_QUALITY_CHECKPOINT="${TILE2_QUALITY_RUN}/nerfstudio_models/step-000001999.ckpt"
TILE3_QUALITY_CHECKPOINT="${TILE3_QUALITY_RUN}/nerfstudio_models/step-000001999.ckpt"
TILE4_QUALITY_TIMESTAMP="2026-07-28_2000step"
TILE4_QUALITY_EXPERIMENT="tbv_long_route_tile_4_seam_2000"
TILE4_QUALITY_RUN="${TRAIN_ROOT}/${TILE4_QUALITY_EXPERIMENT}/splatad/${TILE4_QUALITY_TIMESTAMP}"
TILE4_QUALITY_CONFIG="${TILE4_QUALITY_RUN}/config.yml"
TILE4_QUALITY_CHECKPOINT="${TILE4_QUALITY_RUN}/nerfstudio_models/step-000001999.ckpt"
STATIC_TIMESTAMP="2026-07-25_resume_2k_to_8k"
TILE2_STATIC_EXPERIMENT="tbv_long_route_tile_2_static_8000"
TILE3_STATIC_EXPERIMENT="tbv_long_route_tile_3_static_8000"
TILE2_STATIC_RUN="${TRAIN_ROOT}/${TILE2_STATIC_EXPERIMENT}/splatad/${STATIC_TIMESTAMP}"
TILE3_STATIC_RUN="${TRAIN_ROOT}/${TILE3_STATIC_EXPERIMENT}/splatad/${STATIC_TIMESTAMP}"
TILE2_STATIC_CONFIG="${TILE2_STATIC_RUN}/config.yml"
TILE3_STATIC_CONFIG="${TILE3_STATIC_RUN}/config.yml"
TILE2_STATIC_CHECKPOINT="${TILE2_STATIC_RUN}/nerfstudio_models/step-000007999.ckpt"
CROSS_VISIT_TIMESTAMP="2026-07-28_resume_static8k_cross_visit_hires_retry"
CROSS_VISIT_EXPERIMENT="tbv_long_route_tile_2_cross_visit_hires_10000"
CROSS_VISIT_RUN="${TRAIN_ROOT}/${CROSS_VISIT_EXPERIMENT}/splatad/${CROSS_VISIT_TIMESTAMP}"
CROSS_VISIT_CONFIG="${CROSS_VISIT_RUN}/config.yml"
CROSS_VISIT_CHECKPOINT="${CROSS_VISIT_RUN}/nerfstudio_models/step-000009999.ckpt"
TILE3_STATIC_CHECKPOINT="${TILE3_STATIC_RUN}/nerfstudio_models/step-000007999.ckpt"
TILE4_STATIC_TIMESTAMP="2026-07-28_resume_2k_to_8k_no_eval"
TILE4_STATIC_EXPERIMENT="tbv_long_route_tile_4_static_8000"
TILE4_STATIC_RUN="${TRAIN_ROOT}/${TILE4_STATIC_EXPERIMENT}/splatad/${TILE4_STATIC_TIMESTAMP}"
TILE4_STATIC_CONFIG="${TILE4_STATIC_RUN}/config.yml"
TILE4_STATIC_CHECKPOINT="${TILE4_STATIC_RUN}/nerfstudio_models/step-000007999.ckpt"
MASKED_TIMESTAMP="2026-07-26_masked_2000step"
TILE2_MASKED_EXPERIMENT="tbv_long_route_tile_2_masked_2000"
TILE3_MASKED_EXPERIMENT="tbv_long_route_tile_3_masked_2000"
TILE2_MASKED_RUN="${TRAIN_ROOT}/${TILE2_MASKED_EXPERIMENT}/splatad/${MASKED_TIMESTAMP}"
TILE3_MASKED_RUN="${TRAIN_ROOT}/${TILE3_MASKED_EXPERIMENT}/splatad/${MASKED_TIMESTAMP}"
TILE2_MASKED_CONFIG="${TILE2_MASKED_RUN}/config.yml"
TILE3_MASKED_CONFIG="${TILE3_MASKED_RUN}/config.yml"
TILE2_MASKED_CHECKPOINT="${TILE2_MASKED_RUN}/nerfstudio_models/step-000001999.ckpt"
TILE3_MASKED_CHECKPOINT="${TILE3_MASKED_RUN}/nerfstudio_models/step-000001999.ckpt"
JOINT_MASKED_TIMESTAMP="2026-07-26_rgb_lidar_masked_2000step"
TILE2_JOINT_MASKED_EXPERIMENT="tbv_long_route_tile_2_rgb_lidar_masked_2000"
TILE3_JOINT_MASKED_EXPERIMENT="tbv_long_route_tile_3_rgb_lidar_masked_2000"
TILE2_JOINT_MASKED_RUN="${TRAIN_ROOT}/${TILE2_JOINT_MASKED_EXPERIMENT}/splatad/${JOINT_MASKED_TIMESTAMP}"
TILE3_JOINT_MASKED_RUN="${TRAIN_ROOT}/${TILE3_JOINT_MASKED_EXPERIMENT}/splatad/${JOINT_MASKED_TIMESTAMP}"
TILE2_JOINT_MASKED_CONFIG="${TILE2_JOINT_MASKED_RUN}/config.yml"
TILE3_JOINT_MASKED_CONFIG="${TILE3_JOINT_MASKED_RUN}/config.yml"
TILE2_JOINT_MASKED_CHECKPOINT="${TILE2_JOINT_MASKED_RUN}/nerfstudio_models/step-000001999.ckpt"
TILE3_JOINT_MASKED_CHECKPOINT="${TILE3_JOINT_MASKED_RUN}/nerfstudio_models/step-000001999.ckpt"

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
  local data_root="${10:-$TILE23_DATA_ROOT}"
  if [[ -f "$checkpoint" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
    echo "PASS: reusing tile ${tile} checkpoint: $checkpoint"
    return
  fi
  "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_tbv_smoke.py" \
    --data "$data_root" \
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

resume_tile() {
  local tile="$1"
  local source_config="$2"
  local source_checkpoint="$3"
  local experiment="$4"
  local checkpoint="$5"
  local timestamp="${6:-$STATIC_TIMESTAMP}"
  local eval_interval="${7:-3000}"
  if [[ -f "$checkpoint" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
    echo "PASS: reusing tile ${tile} checkpoint: $checkpoint"
    return
  fi
  if [[ ! -f "$source_config" || ! -f "$source_checkpoint" ]]; then
    echo "tile ${tile} 2,000-step source is missing" >&2
    exit 1
  fi
  "$PYTHON" "$REPO_ROOT/scripts/resume_stage_h3_exact.py" \
    --source-config "$source_config" \
    --checkpoint "$source_checkpoint" \
    --output-dir "$TRAIN_ROOT" \
    --experiment-name "$experiment" \
    --timestamp "$timestamp" \
    --additional-iterations 6000 \
    --model-max-steps 8000 \
    --steps-per-save 6000 \
    --steps-per-eval-image "$eval_interval"
}

train_masked_tile() {
  local tile="$1"
  local experiment="$2"
  local checkpoint="$3"
  local reference_start="$4"
  local reference_end="$5"
  local repeat_start="$6"
  local repeat_end="$7"
  local timestamp="${8:-$MASKED_TIMESTAMP}"
  local mask_lidar_points="${9:-0}"
  if [[ -f "$checkpoint" && "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
    echo "PASS: reusing masked tile ${tile} checkpoint: $checkpoint"
    return
  fi
  if [[ ! -f "$MASK_ROOT/vehicle_mask_manifest.json" ]]; then
    echo "vehicle masks are incomplete: $MASK_ROOT" >&2
    exit 1
  fi
  local lidar_mask_args=()
  if [[ "$mask_lidar_points" == "1" ]]; then
    lidar_mask_args+=(--mask-lidar-points)
  fi
  "$PYTHON" "$REPO_ROOT/scripts/train_stage_h3_tbv_smoke.py" \
    --data "$TILE23_DATA_ROOT" \
    --mask-root "$MASK_ROOT" \
    "${lidar_mask_args[@]}" \
    --output-dir "$TRAIN_ROOT" \
    --experiment-name "$experiment" \
    --timestamp "$timestamp" \
    --iterations 2000 \
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
      --window "$REPEAT,$TILE3_REPEAT_START,$TILE3_REPEAT_END" \
      --window "$REFERENCE,$TILE4_REFERENCE_START,$TILE4_REFERENCE_END" \
      --window "$REPEAT,$TILE4_REPEAT_START,$TILE4_REPEAT_END"
    ;;
  freeze-2-3-data)
    "$PYTHON" \
      "$REPO_ROOT/scripts/materialize_stage_h3_tbv_manifest_subset.py" \
      --manifest "$DATA_ROOT/selection_manifest.json" \
      --window-indices 0,1,2,3 \
      --output-dir "$TILE23_DATA_ROOT"
    ;;
  mask-data)
    "$PYTHON" "$REPO_ROOT/scripts/generate_stage_h3_tbv_vehicle_masks.py" \
      --data-root "$DATA_ROOT" \
      --output-dir "$MASK_ROOT" \
      --batch-size "${H3_TBV_MASK_BATCH_SIZE:-8}"
    ;;
  joint-mask-audit-2)
    if [[ ! -f "$MASK_ROOT/vehicle_mask_manifest.json" ]]; then
      echo "vehicle masks are incomplete: $MASK_ROOT" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/audit_stage_h3_tbv_lidar_masks.py" \
      --data "$DATA_ROOT" \
      --mask-root "$MASK_ROOT" \
      --output-json "$JOINT_MASK_AUDIT_ROOT/tile_2_lidar_mask_audit.json" \
      --sequence "$REFERENCE" \
      --window-start-seconds "$TILE2_REFERENCE_START" \
      --window-end-seconds "$TILE2_REFERENCE_END" \
      --sequence "$REPEAT" \
      --window-start-seconds "$TILE2_REPEAT_START" \
      --window-end-seconds "$TILE2_REPEAT_END"
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
  quality-2)
    train_tile 2 "$TILE2_QUALITY_EXPERIMENT" "$TILE2_QUALITY_CHECKPOINT" \
      "$TILE2_REFERENCE_START" "$TILE2_REFERENCE_END" \
      "$TILE2_REPEAT_START" "$TILE2_REPEAT_END" 2000 "$QUALITY_TIMESTAMP"
    ;;
  quality-3)
    train_tile 3 "$TILE3_QUALITY_EXPERIMENT" "$TILE3_QUALITY_CHECKPOINT" \
      "$TILE3_REFERENCE_START" "$TILE3_REFERENCE_END" \
      "$TILE3_REPEAT_START" "$TILE3_REPEAT_END" 2000 "$QUALITY_TIMESTAMP"
    ;;
  quality-4)
    train_tile 4 "$TILE4_QUALITY_EXPERIMENT" "$TILE4_QUALITY_CHECKPOINT" \
      "$TILE4_REFERENCE_START" "$TILE4_REFERENCE_END" \
      "$TILE4_REPEAT_START" "$TILE4_REPEAT_END" 2000 \
      "$TILE4_QUALITY_TIMESTAMP" "$DATA_ROOT"
    ;;
  static-2)
    resume_tile 2 "$TILE2_QUALITY_CONFIG" "$TILE2_QUALITY_CHECKPOINT" \
      "$TILE2_STATIC_EXPERIMENT" "$TILE2_STATIC_CHECKPOINT"
    ;;
  static-3)
    resume_tile 3 "$TILE3_QUALITY_CONFIG" "$TILE3_QUALITY_CHECKPOINT" \
      "$TILE3_STATIC_EXPERIMENT" "$TILE3_STATIC_CHECKPOINT"
    ;;
  static-4)
    resume_tile 4 "$TILE4_QUALITY_CONFIG" "$TILE4_QUALITY_CHECKPOINT" \
      "$TILE4_STATIC_EXPERIMENT" "$TILE4_STATIC_CHECKPOINT" \
      "$TILE4_STATIC_TIMESTAMP" 999999
    ;;
  cross-visit-quality-2)
    if [[ -f "$CROSS_VISIT_CHECKPOINT" &&
          "${H3_ALLOW_RETRAIN:-0}" != "1" ]]; then
      echo "PASS: reusing cross-visit checkpoint: $CROSS_VISIT_CHECKPOINT"
      exit 0
    fi
    if [[ ! -f "$TILE2_STATIC_CONFIG" ||
          ! -f "$TILE2_STATIC_CHECKPOINT" ||
          ! -f "$CROSS_VISIT_ROOT/cross_visit_manifest.json" ||
          ! -f "$PERSISTENCE_ROOT/persistence_manifest.json" ]]; then
      echo "cross-visit resume inputs are incomplete" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/resume_stage_h3_exact.py" \
      --source-config "$TILE2_STATIC_CONFIG" \
      --checkpoint "$TILE2_STATIC_CHECKPOINT" \
      --output-dir "$TRAIN_ROOT" \
      --experiment-name "$CROSS_VISIT_EXPERIMENT" \
      --timestamp "$CROSS_VISIT_TIMESTAMP" \
      --additional-iterations 2000 \
      --model-max-steps 10000 \
      --steps-per-save 2000 \
      --steps-per-eval-image 999999 \
      --rgb-root "$CROSS_VISIT_ROOT/rgb" \
      --image-mask-root "$CROSS_VISIT_ROOT/valid_masks" \
      --lidar-mask-root "$MASK_ROOT" \
      --lidar-persistence-root "$PERSISTENCE_ROOT" \
      --downsample-factor 0.5 \
      --use-mask-aligned-model
    ;;
  cross-visit-quality-ab-2)
    if [[ ! -f "$CROSS_VISIT_CONFIG" ||
          ! -f "$CROSS_VISIT_CHECKPOINT" ]]; then
      echo "cross-visit 10k config/checkpoint is incomplete" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_quality_ab.py" \
      --baseline-config "$TILE2_STATIC_CONFIG" \
      --candidate-config "$CROSS_VISIT_CONFIG" \
      --baseline-data-root "$DATA_ROOT" \
      --candidate-data-root "$DATA_ROOT" \
      --baseline-label static_8k \
      --candidate-label cross_visit_hires_10k \
      --window-start-seconds "$TILE2_REFERENCE_START" \
      --window-end-seconds "$TILE2_REFERENCE_END" \
      --sample-count 7 \
      --expected-baseline-step 7999 \
      --expected-candidate-step 9999 \
      --output-dir "$CROSS_VISIT_AB_ROOT"
    ;;
  masked-2)
    train_masked_tile 2 "$TILE2_MASKED_EXPERIMENT" \
      "$TILE2_MASKED_CHECKPOINT" \
      "$TILE2_REFERENCE_START" "$TILE2_REFERENCE_END" \
      "$TILE2_REPEAT_START" "$TILE2_REPEAT_END" "$MASKED_TIMESTAMP" 0
    ;;
  masked-3)
    train_masked_tile 3 "$TILE3_MASKED_EXPERIMENT" \
      "$TILE3_MASKED_CHECKPOINT" \
      "$TILE3_REFERENCE_START" "$TILE3_REFERENCE_END" \
      "$TILE3_REPEAT_START" "$TILE3_REPEAT_END" "$MASKED_TIMESTAMP" 0
    ;;
  joint-masked-2)
    train_masked_tile 2 "$TILE2_JOINT_MASKED_EXPERIMENT" \
      "$TILE2_JOINT_MASKED_CHECKPOINT" \
      "$TILE2_REFERENCE_START" "$TILE2_REFERENCE_END" \
      "$TILE2_REPEAT_START" "$TILE2_REPEAT_END" \
      "$JOINT_MASKED_TIMESTAMP" 1
    ;;
  joint-masked-3)
    train_masked_tile 3 "$TILE3_JOINT_MASKED_EXPERIMENT" \
      "$TILE3_JOINT_MASKED_CHECKPOINT" \
      "$TILE3_REFERENCE_START" "$TILE3_REFERENCE_END" \
      "$TILE3_REPEAT_START" "$TILE3_REPEAT_END" \
      "$JOINT_MASKED_TIMESTAMP" 1
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
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
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
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 499 \
      --output-dir "$PILOT_ARTIFACT_ROOT"
    ;;
  seam-2000)
    if [[ ! -f "$TILE2_QUALITY_CONFIG" ||
          ! -f "$TILE2_QUALITY_CHECKPOINT" ||
          ! -f "$TILE3_QUALITY_CONFIG" ||
          ! -f "$TILE3_QUALITY_CHECKPOINT" ]]; then
      echo "both 2,000-step tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_QUALITY_CONFIG" \
      --tile-3-config "$TILE3_QUALITY_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 1999 \
      --output-dir "$QUALITY_ARTIFACT_ROOT"
    ;;
  continuous-2000)
    if [[ ! -f "$TILE2_QUALITY_CONFIG" ||
          ! -f "$TILE2_QUALITY_CHECKPOINT" ||
          ! -f "$TILE3_QUALITY_CONFIG" ||
          ! -f "$TILE3_QUALITY_CHECKPOINT" ]]; then
      echo "both 2,000-step tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_continuous.py" \
      --tile-2-config "$TILE2_QUALITY_CONFIG" \
      --tile-3-config "$TILE3_QUALITY_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --output-dir "$CONTINUOUS_ARTIFACT_ROOT"
    ;;
  seam-3-4-2000)
    if [[ ! -f "$TILE3_QUALITY_CONFIG" ||
          ! -f "$TILE3_QUALITY_CHECKPOINT" ||
          ! -f "$TILE4_QUALITY_CONFIG" ||
          ! -f "$TILE4_QUALITY_CHECKPOINT" ]]; then
      echo "tile 3/4 2,000-step configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE3_QUALITY_CONFIG" \
      --tile-3-config "$TILE4_QUALITY_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$DATA_ROOT" \
      --first-label tile_3 \
      --second-label tile_4 \
      --overlap-start-seconds "$TILE4_REFERENCE_START" \
      --overlap-end-seconds "$TILE3_REFERENCE_END" \
      --expected-checkpoint-step 1999 \
      --output-dir "$TILE34_QUALITY_ARTIFACT_ROOT"
    ;;
  seam-8000)
    if [[ ! -f "$TILE2_STATIC_CONFIG" ||
          ! -f "$TILE2_STATIC_CHECKPOINT" ||
          ! -f "$TILE3_STATIC_CONFIG" ||
          ! -f "$TILE3_STATIC_CHECKPOINT" ]]; then
      echo "both 8,000-step tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_STATIC_CONFIG" \
      --tile-3-config "$TILE3_STATIC_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 7999 \
      --output-dir "$STATIC_ARTIFACT_ROOT"
    ;;
  continuous-8000)
    if [[ ! -f "$TILE2_STATIC_CONFIG" ||
          ! -f "$TILE2_STATIC_CHECKPOINT" ||
          ! -f "$TILE3_STATIC_CONFIG" ||
          ! -f "$TILE3_STATIC_CHECKPOINT" ]]; then
      echo "both 8,000-step tile configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_continuous.py" \
      --tile-2-config "$TILE2_STATIC_CONFIG" \
      --tile-3-config "$TILE3_STATIC_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 7999 \
      --output-dir "$STATIC_CONTINUOUS_ROOT"
    ;;
  seam-3-4-8000)
    if [[ ! -f "$TILE3_STATIC_CONFIG" ||
          ! -f "$TILE3_STATIC_CHECKPOINT" ||
          ! -f "$TILE4_STATIC_CONFIG" ||
          ! -f "$TILE4_STATIC_CHECKPOINT" ]]; then
      echo "tile 3/4 8,000-step configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE3_STATIC_CONFIG" \
      --tile-3-config "$TILE4_STATIC_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$DATA_ROOT" \
      --first-label tile_3 \
      --second-label tile_4 \
      --overlap-start-seconds "$TILE4_REFERENCE_START" \
      --overlap-end-seconds "$TILE3_REFERENCE_END" \
      --expected-checkpoint-step 7999 \
      --output-dir "$TILE34_STATIC_ARTIFACT_ROOT"
    ;;
  three-tile-drive-8000)
    if [[ ! -f "$TILE2_STATIC_CONFIG" ||
          ! -f "$TILE2_STATIC_CHECKPOINT" ||
          ! -f "$TILE3_STATIC_CONFIG" ||
          ! -f "$TILE3_STATIC_CHECKPOINT" ||
          ! -f "$TILE4_STATIC_CONFIG" ||
          ! -f "$TILE4_STATIC_CHECKPOINT" ]]; then
      echo "tile 2/3/4 8,000-step configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/run_stage_h3_tbv_three_tile_drive.py" \
      --tile-2-config "$TILE2_STATIC_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-config "$TILE3_STATIC_CONFIG" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --tile-4-config "$TILE4_STATIC_CONFIG" \
      --tile-4-data-root "$DATA_ROOT" \
      --output-dir "$THREE_TILE_DRIVE_ROOT"
    ;;
  masked-seam-2000)
    if [[ ! -f "$TILE2_MASKED_CONFIG" ||
          ! -f "$TILE2_MASKED_CHECKPOINT" ||
          ! -f "$TILE3_MASKED_CONFIG" ||
          ! -f "$TILE3_MASKED_CHECKPOINT" ]]; then
      echo "both masked 2,000-step configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_MASKED_CONFIG" \
      --tile-3-config "$TILE3_MASKED_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 1999 \
      --output-dir "$MASKED_ARTIFACT_ROOT"
    ;;
  masked-continuous-2000)
    if [[ ! -f "$TILE2_MASKED_CONFIG" ||
          ! -f "$TILE2_MASKED_CHECKPOINT" ||
          ! -f "$TILE3_MASKED_CONFIG" ||
          ! -f "$TILE3_MASKED_CHECKPOINT" ]]; then
      echo "both masked 2,000-step configs/checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_continuous.py" \
      --tile-2-config "$TILE2_MASKED_CONFIG" \
      --tile-3-config "$TILE3_MASKED_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --output-dir "$MASKED_CONTINUOUS_ROOT"
    ;;
  joint-masked-seam-2000)
    if [[ ! -f "$TILE2_JOINT_MASKED_CONFIG" ||
          ! -f "$TILE2_JOINT_MASKED_CHECKPOINT" ||
          ! -f "$TILE3_JOINT_MASKED_CONFIG" ||
          ! -f "$TILE3_JOINT_MASKED_CHECKPOINT" ]]; then
      echo "both RGB+LiDAR masked 2,000-step checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_seam.py" \
      --tile-2-config "$TILE2_JOINT_MASKED_CONFIG" \
      --tile-3-config "$TILE3_JOINT_MASKED_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --expected-checkpoint-step 1999 \
      --output-dir "$JOINT_MASKED_ARTIFACT_ROOT"
    ;;
  joint-masked-continuous-2000)
    if [[ ! -f "$TILE2_JOINT_MASKED_CONFIG" ||
          ! -f "$TILE2_JOINT_MASKED_CHECKPOINT" ||
          ! -f "$TILE3_JOINT_MASKED_CONFIG" ||
          ! -f "$TILE3_JOINT_MASKED_CHECKPOINT" ]]; then
      echo "both RGB+LiDAR masked 2,000-step checkpoints are required" >&2
      exit 1
    fi
    "$PYTHON" "$REPO_ROOT/scripts/probe_stage_h3_tbv_tile_continuous.py" \
      --tile-2-config "$TILE2_JOINT_MASKED_CONFIG" \
      --tile-3-config "$TILE3_JOINT_MASKED_CONFIG" \
      --tile-2-data-root "$TILE23_DATA_ROOT" \
      --tile-3-data-root "$TILE23_DATA_ROOT" \
      --output-dir "$JOINT_MASKED_CONTINUOUS_ROOT"
    ;;
  paths)
    echo "data: $DATA_ROOT"
    echo "frozen tile 2/3 data: $TILE23_DATA_ROOT"
    echo "vehicle masks: $MASK_ROOT"
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
    echo "tile 2 quality config: $TILE2_QUALITY_CONFIG"
    echo "tile 2 quality checkpoint: $TILE2_QUALITY_CHECKPOINT"
    echo "tile 3 quality config: $TILE3_QUALITY_CONFIG"
    echo "tile 3 quality checkpoint: $TILE3_QUALITY_CHECKPOINT"
    echo "tile 4 quality config: $TILE4_QUALITY_CONFIG"
    echo "tile 4 quality checkpoint: $TILE4_QUALITY_CHECKPOINT"
    echo "2,000-step seam evidence: $QUALITY_ARTIFACT_ROOT"
    echo "continuous seam evidence: $CONTINUOUS_ARTIFACT_ROOT"
    echo "tile 2 static config: $TILE2_STATIC_CONFIG"
    echo "tile 2 static checkpoint: $TILE2_STATIC_CHECKPOINT"
    echo "cross-visit config: $CROSS_VISIT_CONFIG"
    echo "cross-visit checkpoint: $CROSS_VISIT_CHECKPOINT"
    echo "cross-visit A/B evidence: $CROSS_VISIT_AB_ROOT"
    echo "tile 3 static config: $TILE3_STATIC_CONFIG"
    echo "tile 3 static checkpoint: $TILE3_STATIC_CHECKPOINT"
    echo "tile 4 static config: $TILE4_STATIC_CONFIG"
    echo "tile 4 static checkpoint: $TILE4_STATIC_CHECKPOINT"
    echo "8,000-step seam evidence: $STATIC_ARTIFACT_ROOT"
    echo "8,000-step continuous evidence: $STATIC_CONTINUOUS_ROOT"
    echo "tile 2 masked config: $TILE2_MASKED_CONFIG"
    echo "tile 2 masked checkpoint: $TILE2_MASKED_CHECKPOINT"
    echo "tile 3 masked config: $TILE3_MASKED_CONFIG"
    echo "tile 3 masked checkpoint: $TILE3_MASKED_CHECKPOINT"
    echo "masked 2,000-step seam evidence: $MASKED_ARTIFACT_ROOT"
    echo "masked 2,000-step continuous evidence: $MASKED_CONTINUOUS_ROOT"
    echo "RGB+LiDAR mask audit: $JOINT_MASK_AUDIT_ROOT"
    echo "tile 2 RGB+LiDAR masked config: $TILE2_JOINT_MASKED_CONFIG"
    echo "tile 2 RGB+LiDAR masked checkpoint: $TILE2_JOINT_MASKED_CHECKPOINT"
    echo "tile 3 RGB+LiDAR masked config: $TILE3_JOINT_MASKED_CONFIG"
    echo "tile 3 RGB+LiDAR masked checkpoint: $TILE3_JOINT_MASKED_CHECKPOINT"
    echo "RGB+LiDAR masked seam evidence: $JOINT_MASKED_ARTIFACT_ROOT"
    echo "RGB+LiDAR masked continuous evidence: $JOINT_MASKED_CONTINUOUS_ROOT"
    echo "tile 3/4 2,000-step seam evidence: $TILE34_QUALITY_ARTIFACT_ROOT"
    echo "tile 3/4 8,000-step seam evidence: $TILE34_STATIC_ARTIFACT_ROOT"
    echo "three-tile drive evidence: $THREE_TILE_DRIVE_ROOT"
    ;;
  *)
    echo "Usage: $0 MODE" >&2
    echo "Modes: download freeze-2-3-data mask-data joint-mask-audit-2" >&2
    echo "       smoke-2 smoke-3 seam" >&2
    echo "       pilot-2 pilot-3 seam-500" >&2
    echo "       quality-2 quality-3 quality-4 seam-2000 seam-3-4-2000" >&2
    echo "       continuous-2000 static-2 static-3 static-4" >&2
    echo "       cross-visit-quality-2" >&2
    echo "       cross-visit-quality-ab-2" >&2
    echo "       seam-8000 seam-3-4-8000 continuous-8000" >&2
    echo "       three-tile-drive-8000 paths" >&2
    echo "       masked-2 masked-3 masked-seam-2000 masked-continuous-2000" >&2
    echo "       joint-masked-2 joint-masked-3 joint-masked-seam-2000" >&2
    echo "       joint-masked-continuous-2000" >&2
    exit 2
    ;;
esac
