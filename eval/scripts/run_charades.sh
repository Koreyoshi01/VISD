#!/bin/bash
# Public release: Charades-STA evaluation uses the external Time-R1 repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_DATASET="${EVAL_DATASET:-charades}"

bash "${SCRIPT_DIR}/run_tvgbench.sh"
