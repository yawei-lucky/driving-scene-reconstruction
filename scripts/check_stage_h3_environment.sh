#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H3_ROOT="${H3_ROOT:-/home/yawei/stage3_external}"
# shellcheck source=stage_h3_cuda_environment.sh
source "$REPO_ROOT/scripts/stage_h3_cuda_environment.sh"
PYTHON="$STAGE_H3_PYTHON"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

echo "Host and storage:"
echo "  host: $(hostname)"
echo "  H3 root: $H3_ROOT"
df -h "$H3_ROOT" | tail -n 1
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader
"${H3_ENV}/bin/nvcc" --version | tail -n 1

"$PYTHON" "$REPO_ROOT/scripts/check_stage_h3_environment.py" \
  --h3-root "$H3_ROOT"

"$PYTHON" -m pip check

METHOD_HELP="$("${H3_ENV}/bin/ns-train" --help 2>&1)"
if [[ "$METHOD_HELP" != *"splatad"* || "$METHOD_HELP" != *"neurad"* ]]; then
  echo "ns-train does not expose both splatad and neurad." >&2
  exit 1
fi

PANDASET_HELP="$("${H3_ENV}/bin/ns-train" splatad pandaset-data --help 2>&1)"
if [[ "$PANDASET_HELP" != *"Pandar64"* || "$PANDASET_HELP" != *"PandarGT"* ]]; then
  echo "PandaSet parser help does not expose the expected LiDAR choices." >&2
  exit 1
fi

echo "PASS: ns-train exposes splatad, neurad, and pandaset-data."
