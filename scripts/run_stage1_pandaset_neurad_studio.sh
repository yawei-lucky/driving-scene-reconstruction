#!/usr/bin/env bash
set -euo pipefail

# Run Stage-1 PandaSet reconstruction baseline with neurad-studio.
#
# Usage:
#   bash scripts/run_stage1_pandaset_neurad_studio.sh --method splatad
#   bash scripts/run_stage1_pandaset_neurad_studio.sh --method neurad
#
# Expected prior step:
#   UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --all-open
#
# Notes:
#   - This script assumes a CUDA-capable machine for useful training speed.
#   - It intentionally does not commit outputs. Outputs are local artifacts.
#   - If the upstream neurad-studio CLI changes, inspect:
#       python nerfstudio/scripts/train.py <method> pandaset-data --help
#       python nerfstudio/scripts/render.py --help

RESOURCE_ROOT="${RESOURCE_ROOT:-/data/external/driving_scene_reconstruction}"
CODE_ROOT="${CODE_ROOT:-${RESOURCE_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${RESOURCE_ROOT}/datasets}"
NEURAD_ROOT="${NEURAD_ROOT:-${CODE_ROOT}/neurad-studio}"
PANDASET_DIR="${PANDASET_DIR:-${DATASET_ROOT}/pandaset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage1_pandaset_neurad_studio}"
INSTALL_ENV="${INSTALL_ENV:-0}"
VIS_MODE="${VIS_MODE:-viewer}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
METHOD="splatad"

print_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_stage1_pandaset_neurad_studio.sh [options]

Options:
  --method splatad|neurad   Baseline method to run. Default: splatad.
  -h, --help                Show this message.

Environment variables:
  RESOURCE_ROOT             External resource root.
  NEURAD_ROOT               neurad-studio checkout path.
  PANDASET_DIR              Unzipped PandaSet path.
  OUTPUT_ROOT               Local output directory. Default: outputs/stage1_pandaset_neurad_studio
  INSTALL_ENV               Set to 1 to run `pip install -e .` inside neurad-studio before training.
  VIS_MODE                  Nerfstudio visualizer mode. Default: viewer.
  MAX_NUM_ITERATIONS        Optional quick-test iteration cap. Example: 1000
  EXTRA_TRAIN_ARGS          Extra arguments appended to the train command.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ "${METHOD}" != "splatad" && "${METHOD}" != "neurad" ]]; then
  echo "Unsupported method: ${METHOD}. Expected 'splatad' or 'neurad'." >&2
  exit 1
fi

if [[ ! -d "${NEURAD_ROOT}" ]]; then
  echo "neurad-studio not found: ${NEURAD_ROOT}" >&2
  echo "Run: bash scripts/fetch_stage1_resources.sh --code" >&2
  exit 1
fi

if [[ ! -d "${PANDASET_DIR}" ]]; then
  echo "PandaSet directory not found: ${PANDASET_DIR}" >&2
  echo "Run: UNZIP_PANDASET=1 bash scripts/fetch_stage1_resources.sh --pandaset" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_NAME="stage1_pandaset_${METHOD}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

{
  echo "method=${METHOD}"
  echo "resource_root=${RESOURCE_ROOT}"
  echo "neurad_root=${NEURAD_ROOT}"
  echo "pandaset_dir=${PANDASET_DIR}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "install_env=${INSTALL_ENV}"
  echo "vis_mode=${VIS_MODE}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "extra_train_args=${EXTRA_TRAIN_ARGS}"
  echo "date=$(date -Iseconds)"
  echo "git_neurad_commit=$(git -C "${NEURAD_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "python=$(command -v python || true)"
  python --version 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
} | tee "${RUN_DIR}/environment.txt"

cd "${NEURAD_ROOT}"

if [[ "${INSTALL_ENV}" == "1" ]]; then
  echo "[install] pip install -e ."
  python -m pip install -U pip
  python -m pip install -e .
fi

TRAIN_CMD=(python nerfstudio/scripts/train.py "${METHOD}")

if [[ -n "${MAX_NUM_ITERATIONS}" ]]; then
  TRAIN_CMD+=(--max-num-iterations "${MAX_NUM_ITERATIONS}")
fi

TRAIN_CMD+=(--vis "${VIS_MODE}")
TRAIN_CMD+=(--output-dir "${RUN_DIR}")
TRAIN_CMD+=(--experiment-name "${RUN_NAME}")
TRAIN_CMD+=(pandaset-data --data "${PANDASET_DIR}")

# Append optional user-provided raw args last.
# shellcheck disable=SC2206
EXTRA_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
TRAIN_CMD+=("${EXTRA_ARGS_ARRAY[@]}")

printf '%q ' "${TRAIN_CMD[@]}" | tee "${RUN_DIR}/run_commands.sh"
echo | tee -a "${RUN_DIR}/run_commands.sh"
chmod +x "${RUN_DIR}/run_commands.sh"

echo "[train] Running ${METHOD} on PandaSet"
"${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_DIR}/train.log"

CONFIG_PATH="$(find "${RUN_DIR}" -name config.yml | sort | tail -n 1 || true)"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "No config.yml found under ${RUN_DIR}. Training may have failed before checkpoint creation." | tee -a "${RUN_DIR}/result_summary.md"
  exit 1
fi

{
  echo "# Stage 1 PandaSet ${METHOD} Result Summary"
  echo
  echo "config_path: ${CONFIG_PATH}"
  echo "run_dir: ${RUN_DIR}"
  echo
  echo "## Next render / evaluation commands"
  echo
  echo '```bash'
  echo "python nerfstudio/scripts/render.py --help"
  echo "python nerfstudio/scripts/render.py dataset --load-config '${CONFIG_PATH}' --output-path '${RUN_DIR}/renders/dataset'"
  echo "python nerfstudio/scripts/render.py interpolate --load-config '${CONFIG_PATH}' --output-path '${RUN_DIR}/renders/interpolate.mp4'"
  echo "python nerfstudio/scripts/eval.py --load-config '${CONFIG_PATH}' --output-path '${RUN_DIR}/eval.json'"
  echo '```'
  echo
  echo "## Failure notes"
  echo
  echo "Fill this in after rendering:"
  echo
  echo '```text'
  echo "case_id:"
  echo "dataset: PandaSet"
  echo "method: ${METHOD}"
  echo "main_success:"
  echo "main_failures:"
  echo "  - blur:"
  echo "  - ghosting:"
  echo "  - geometry_distortion:"
  echo "  - object_disappearance:"
  echo "  - dynamic_object_popping:"
  echo "  - occlusion_error:"
  echo "  - lane_or_road_boundary_error:"
  echo "acceptance_decision: accept / down-weight / reject"
  echo "next_best_action:"
  echo '```'
} | tee "${RUN_DIR}/result_summary.md"

python nerfstudio/scripts/render.py --help > "${RUN_DIR}/render_help.txt" 2>&1 || true

echo "Done. Summary written to ${RUN_DIR}/result_summary.md"
