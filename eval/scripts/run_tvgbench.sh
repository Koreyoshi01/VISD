#!/bin/bash
# Public release: TVG evaluation uses the external Time-R1 repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIME_R1_ROOT="${TIME_R1_ROOT:-your_time_r1_path}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-${GPU_LIST:-0,1,2,3,4,5,6,7}}"

EVAL_DATASET="${EVAL_DATASET:-tvgbench}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOTAL_PIXELS="${TOTAL_PIXELS:-2809856}"
PROMPT_TYPE="${PROMPT_TYPE:-openo3}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
USE_NOTHINK="${USE_NOTHINK:-0}"
USE_VLLM="${USE_VLLM:-1}"
USE_R1_THINKING_PROMPT="${USE_R1_THINKING_PROMPT:-1}"

LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_ROOT}/${MODEL_NAME}/${EVAL_DATASET}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${MODEL_NAME}/${EVAL_DATASET}}"

cd "${TIME_R1_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TIME_R1_LOG_PARENT="${TIME_R1_ROOT}/logs/eval/${MODEL_NAME}"
TIME_R1_DATASET_LINK="${TIME_R1_LOG_PARENT}/${EVAL_DATASET}"
mkdir -p "${TIME_R1_LOG_PARENT}"
if [ ! -e "${TIME_R1_DATASET_LINK}" ]; then
    ln -s "${OUTPUT_DIR}" "${TIME_R1_DATASET_LINK}"
fi
export PYTHONPATH="${TIME_R1_ROOT}:${TIME_R1_ROOT}/src/vllm_inference:${PYTHONPATH:-}"

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS="${NUM_GPUS:-${#GPUS[@]}}"
RUN_ID="$(date +%m%d_%H%M)"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${EVAL_DATASET}_${RUN_ID}.log"

echo "[$(date '+%F %T')] run_tvgbench.sh started." | tee "${LOG_FILE}"
echo "TIME_R1_ROOT=${TIME_R1_ROOT}" | tee -a "${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "MODEL_NAME=${MODEL_NAME}" | tee -a "${LOG_FILE}"
echo "GPU_LIST=${GPU_LIST}" | tee -a "${LOG_FILE}"
echo "EVAL_DATASET=${EVAL_DATASET}" | tee -a "${LOG_FILE}"
echo "SPLIT=${SPLIT}" | tee -a "${LOG_FILE}"
echo "BATCH_SIZE=${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" | tee -a "${LOG_FILE}"
echo "TOTAL_PIXELS=${TOTAL_PIXELS}" | tee -a "${LOG_FILE}"
echo "PROMPT_TYPE=${PROMPT_TYPE}" | tee -a "${LOG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

for ((i=0; i<NUM_GPUS; i++)); do
    gpu="${GPUS[$i]}"
    echo "[$(date '+%F %T')] Launch shard ${i}/${NUM_GPUS} on GPU ${gpu}" | tee -a "${LOG_FILE}"
    cmd=(
        python evaluate.py
        --model_base "${MODEL_PATH}"
        --batch_size "${BATCH_SIZE}"
        --curr_idx "${i}"
        --total_idx "${NUM_GPUS}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --split "${SPLIT}"
        --datasets "${EVAL_DATASET}"
        --output_dir "${OUTPUT_DIR}"
        --total_pixels "${TOTAL_PIXELS}"
        --prompt_type "${PROMPT_TYPE}"
        --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}"
    )
    if [[ "${USE_R1_THINKING_PROMPT}" == "1" ]]; then
        cmd+=(--use_r1_thinking_prompt)
    fi
    if [[ "${USE_VLLM}" == "1" ]]; then
        cmd+=(--use_vllm_inference)
    fi
    if [[ "${USE_NOTHINK}" == "1" ]]; then
        cmd+=(--use_nothink)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" &
done
wait

echo "[$(date '+%F %T')] Inference finished; computing metrics." | tee -a "${LOG_FILE}"
python src/vllm_inference/eval_all.py \
    --model_name "${MODEL_NAME}" \
    --split "${SPLIT}" \
    --dataset "${EVAL_DATASET}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date '+%F %T')] run_tvgbench.sh finished." | tee -a "${LOG_FILE}"
