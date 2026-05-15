#!/bin/bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODEL_PATH="${MODEL_PATH:-your_model_path}"

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
DATA_DIR="${DATA_DIR:-your_videommmu_data_path}"
MODEL_KWARGS="${EVAL_DIR}/config/video_mmmu.yaml"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N="${N:-1}"
VOTE="${VOTE:-majority_voting}"
TEMPERATURE="${TEMPERATURE:-}"
RUN_TAG="${RUN_TAG:-$(date '+%m%d_%H%M')}"
TEMP_RAW="${TEMP_RAW:-${TEMPERATURE:-$(awk -F': *' '$1=="temperature"{print $2; exit}' "${MODEL_KWARGS}")}}"
TEMP_TAG="${TEMP_RAW// /}"
EXP_NAME="${MODEL_NAME}_videommmu_eval_t${TEMP_TAG}_n${N}_${VOTE}_${RUN_TAG}"
LOG_DIR="${EVAL_DIR}/logs/videommmu_logs/${MODEL_NAME}"

export VIDEOMMMU_LOG_DIR="${LOG_DIR}"

mkdir -p "${LOG_DIR}"

MODEL_KWARGS_EFFECTIVE="${MODEL_KWARGS}"
if [ -n "${TEMPERATURE}" ]; then
    MODEL_KWARGS_EFFECTIVE="${LOG_DIR}/${EXP_NAME}.model_kwargs.yaml"
    cp "${MODEL_KWARGS}" "${MODEL_KWARGS_EFFECTIVE}"
    if grep -q '^temperature:' "${MODEL_KWARGS_EFFECTIVE}"; then
        sed -i "s/^temperature:.*/temperature: ${TEMPERATURE}/" "${MODEL_KWARGS_EFFECTIVE}"
    else
        printf '\ntemperature: %s\n' "${TEMPERATURE}" >> "${MODEL_KWARGS_EFFECTIVE}"
    fi
fi

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
log_message "Using model kwargs: ${MODEL_KWARGS_EFFECTIVE}"

NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${EVAL_DIR}/test/test_videommmu.py" \
    --exp_name "${EXP_NAME}" \
    --data_dir "${DATA_DIR}" \
    --model_path "${MODEL_PATH}" \
    --model_kwargs "${MODEL_KWARGS_EFFECTIVE}" \
    --N ${N} \
    --vote "${VOTE}" \
    --think_mode
