#!/bin/bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EASY_ROOT="${EASY_ROOT:-your_easyvideor1_path}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${EASY_ROOT}/eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py}"
DATA_DIR_PATH="${DATA_DIR_PATH:-${EASY_ROOT}/eval/data}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"

export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-200000}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

NFRAMES="${NFRAMES:-64}"
FPS="${FPS:-2.0}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
TOTAL_PIXELS="${TOTAL_PIXELS:-33554432}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.001}"
TOP_K="${TOP_K:-1}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

NUM_WORKERS="${NUM_WORKERS:-8}"
LOAD_WORKERS="${LOAD_WORKERS:-16}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
QUEUE_SIZE="${QUEUE_SIZE:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MODE="${MODE:-auto}"
THINKING_MODE="${THINKING_MODE:-1}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/eval/logs/lrr_logs}"
CACHE_DIR="${CACHE_DIR:-${EASY_ROOT}/eval/caches/lrr}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_ROOT}/${MODEL_NAME}/outputs}"
RESULT_DIR="${RESULT_DIR:-${LOG_ROOT}/${MODEL_NAME}/results}"
mkdir -p "$LOG_ROOT" "$OUTPUT_DIR" "$RESULT_DIR"

RUN_ID="$(date +%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/${MODEL_NAME}_longvideoreason_f${NFRAMES}_${RUN_ID}.log"

cd "$EASY_ROOT"

echo "[$(date '+%F %T')] run_lrr.sh started." | tee "$LOG_FILE"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "$LOG_FILE"
echo "MODEL_NAME=${MODEL_NAME}" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "$LOG_FILE"
echo "DATA_DIR_PATH=${DATA_DIR_PATH}" | tee -a "$LOG_FILE"
echo "VIDEO_DIR=${DATA_DIR_PATH}/longvila_videos" | tee -a "$LOG_FILE"
echo "NFRAMES=${NFRAMES} FPS=${FPS} MAX_PIXELS=${MAX_PIXELS} TOTAL_PIXELS=${TOTAL_PIXELS}" | tee -a "$LOG_FILE"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "RESULT_DIR=${RESULT_DIR}" | tee -a "$LOG_FILE"

cmd=(
    python "$EVAL_SCRIPT"
    --mode "$MODE"
    --model_path "$MODEL_PATH"
    --model_family qwen25
    --qwen25_utils_root "${EASY_ROOT}/eval/code/qwen_vl_utils-0.0.8"
    --data_dir_path "$DATA_DIR_PATH"
    --cache_dir "$CACHE_DIR"
    --output_dir "$OUTPUT_DIR"
    --result_dir "$RESULT_DIR"
    --datasets longvideoreason
    --num_gpus "$NUM_GPUS"
    --nframes "$NFRAMES"
    --fps "$FPS"
    --max_pixels "$MAX_PIXELS"
    --total_pixels "$TOTAL_PIXELS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --presence_penalty "$PRESENCE_PENALTY"
    --repetition_penalty "$REPETITION_PENALTY"
    --max_tokens "$MAX_TOKENS"
    --num_workers "$NUM_WORKERS"
    --load_workers "$LOAD_WORKERS"
    --max_concurrent "$MAX_CONCURRENT"
    --queue_size "$QUEUE_SIZE"
    --max_model_len "$MAX_MODEL_LEN"
    --max_num_seqs "$MAX_NUM_SEQS"
    --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu_mem_util "$GPU_MEM_UTIL"
)

if [[ "$THINKING_MODE" == "1" ]]; then
    cmd+=(--thinking_mode)
fi

"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] run_lrr.sh finished." | tee -a "$LOG_FILE"
