#!/usr/bin/env bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-vstar videommmu worldsense lrr}"

export MODEL_PATH
export MODEL_NAME

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "RUN_BENCHMARKS=${RUN_BENCHMARKS}"

for benchmark in ${RUN_BENCHMARKS}; do
    case "${benchmark}" in
        vstar)
            bash "${SCRIPT_DIR}/run_vstar.sh"
            ;;
        videommmu)
            bash "${SCRIPT_DIR}/run_videommmu.sh"
            ;;
        worldsense)
            bash "${SCRIPT_DIR}/run_worldsense.sh"
            ;;
        lrr|longvideoreason)
            bash "${SCRIPT_DIR}/run_lrr.sh"
            ;;
        tvg|tvgbench)
            bash "${SCRIPT_DIR}/run_tvgbench.sh"
            ;;
        charades|charades_sta)
            bash "${SCRIPT_DIR}/run_charades.sh"
            ;;
        *)
            echo "Unknown benchmark: ${benchmark}" >&2
            exit 1
            ;;
    esac
done
