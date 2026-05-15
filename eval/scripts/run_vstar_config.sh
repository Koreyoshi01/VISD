#!/bin/bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODEL_PATH="${MODEL_PATH:-your_model_path}"
LLM_PATH="${LLM_PATH:-your_llm_judge_model_path}"

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
EXP_NAME="${MODEL_NAME}_vstar_eval"
LOG_DIR="${EVAL_DIR}/logs/vstar_logs/${MODEL_NAME}"
RESULT_FILE="${LOG_DIR}/${EXP_NAME}.json"

DATA_ROOT="${DATA_ROOT:-your_vstar_data_path}"
VIDEO_FOLDER="${DATA_ROOT}/videos"
ANNO_FILE="${DATA_ROOT}/V_STaR_test.json"
MODEL_KWARGS="${EVAL_DIR}/config/vstar.yaml"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
