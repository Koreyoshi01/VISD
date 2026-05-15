#!/bin/bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_vstar_config.sh"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

SCRIPT_NAME="$(basename "$0")"

log_message() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

on_signal() {
    local signal="$1"
    local code="$2"
    log_message "${SCRIPT_NAME} received ${signal}; exiting with code ${code}."
    exit "${code}"
}

on_exit() {
    local exit_code=$?
    log_message "${SCRIPT_NAME} exited with code ${exit_code}."
}

trap 'on_signal SIGINT 130' INT
trap 'on_signal SIGTERM 143' TERM
trap 'on_signal SIGHUP 129' HUP
trap 'on_exit' EXIT

log_message "${SCRIPT_NAME} started."

NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${EVAL_DIR}/test/test_vstar_multi_images.py" \
    --video_folder "${VIDEO_FOLDER}" \
    --anno_file "${ANNO_FILE}" \
    --result_file "${RESULT_FILE}" \
    --model_path "${MODEL_PATH}" \
    --model_kwargs "${MODEL_KWARGS}" \
    --think_mode

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${EVAL_DIR}/test/eval_vstar_vllm.py" \
    --result_file "${RESULT_FILE}" \
    --model_path "${LLM_PATH}" \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}
