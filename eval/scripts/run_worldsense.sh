#!/bin/bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODEL_PATH="${MODEL_PATH:-your_model_path}"

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
EXP_NAME="${MODEL_NAME}_worldsense_eval"
LOG_DIR="${EVAL_DIR}/logs/world_logs/${MODEL_NAME}"

DATA_DIR="${DATA_DIR:-your_worldsense_data_path}"
MODEL_KWARGS="${EVAL_DIR}/config/world_sense.yaml"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N="${N:-1}"
VOTE="${VOTE:-majority_voting}"

# Make videos visible at dataset root for the current dataloader.
if [ -d "${DATA_DIR}/videos" ]; then
    for f in "${DATA_DIR}"/videos/*.mp4; do
        [ -e "${DATA_DIR}/$(basename "$f")" ] || ln -s "$f" "${DATA_DIR}/$(basename "$f")"
    done
fi

export WORLDSENSE_LOG_DIR="${LOG_DIR}"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${EVAL_DIR}/test/test_worldsense.py" \
    --exp_name "${EXP_NAME}" \
    --data_dir "${DATA_DIR}" \
    --model_path "${MODEL_PATH}" \
    --model_kwargs "${MODEL_KWARGS}" \
    --N ${N} \
    --vote "${VOTE}" \
    --think_mode
