#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=stage_h3_cuda_environment.sh
source "$REPO_ROOT/scripts/stage_h3_cuda_environment.sh"

mkdir -p \
  "$TORCH_EXTENSIONS_DIR" \
  "$PIP_CACHE_DIR" \
  "$MPLCONFIGDIR"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_stage_h3_environment.sh --check
  scripts/run_stage_h3_environment.sh --print
  scripts/run_stage_h3_environment.sh python SCRIPT.py [ARGS...]
  scripts/run_stage_h3_environment.sh COMMAND [ARGS...]

Use this wrapper for one-off Stage H3 SplatAD/NeuRAD commands. It pins CUDA
11.8 before Python can import or compile a CUDA extension.
EOF
}

case "${1:-}" in
  --check)
    stage_h3_print_cuda_contract
    stage_h3_verify_torch_cuda_contract
    exit 0
    ;;
  --print)
    stage_h3_print_cuda_contract
    exit 0
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  "")
    usage
    exit 2
    ;;
esac

command_name="$1"
shift
case "$command_name" in
  python|python3)
    command_name="$STAGE_H3_PYTHON"
    ;;
  ns-*)
    if [[ ! -x "${H3_ENV}/bin/${command_name}" ]]; then
      echo "Stage H3 command not found: ${H3_ENV}/bin/${command_name}" >&2
      exit 1
    fi
    command_name="${H3_ENV}/bin/${command_name}"
    ;;
esac

exec "$command_name" "$@"
