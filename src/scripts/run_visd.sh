#!/usr/bin/env bash
# Public release: replace placeholder paths, model paths, dataset paths, and API credentials with local values before running.
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/src/r1-v"
TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-your_conda_env_path}"
TRAIN_ENV_PYTHON="${TRAIN_ENV_PYTHON:-${TRAIN_ENV_ROOT}/bin/python}"
TRAIN_ENV_TORCHRUN="${TRAIN_ENV_TORCHRUN:-${TRAIN_ENV_ROOT}/bin/torchrun}"

cd "${TRAIN_ROOT}"

export VIDEO_PIXELS_FACTOR="${VIDEO_PIXELS_FACTOR:-128}"
export PYTHONPATH="${TRAIN_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# Keep checkpoint/state loading compatible with PyTorch 2.6+ defaults.
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

export CUDA_HOME=/usr/local/cuda
export PATH="${TRAIN_ENV_ROOT}/bin:${PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

NVCC_PATH="$(command -v nvcc || true)"

DEFAULT_WANDB_PROJECT="visd-rl"
DEFAULT_WANDB_API_KEY="__FILL_WANDB_API_KEY__"
DEFAULT_JUDGE_BASE_URL="__FILL_JUDGE_BASE_URL__"
DEFAULT_JUDGE_API_KEY="__FILL_JUDGE_API_KEY__"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-${DEFAULT_WANDB_PROJECT}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-visd-grpo-2node16gpu}"
export WANDB_API_KEY="${WANDB_API_KEY:-${DEFAULT_WANDB_API_KEY}}"
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY || true
fi
if [ -n "${WANDB_BASE_URL:-}" ]; then
  export WANDB_BASE_URL
else
  unset WANDB_BASE_URL || true
fi

MODEL_PATH="${MODEL_PATH:-your_model_path}"
DATASET_PATH="${DATASET_PATH:-your_data_path}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

EXP_NAME="${EXP_NAME:-visd_rl}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results_rl/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.node${RANK:-${SLURM_NODEID:-${NODE_RANK:-0}}}.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"


#   WORLD_SIZE     -> number of nodes
#   RANK           -> node rank
#   NPROC_PER_NODE -> GPUs per node
NNODES="${WORLD_SIZE:-${NNODES:-${SLURM_NNODES:-2}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${RANK:-${NODE_RANK:-${SLURM_NODEID:-0}}}"
MASTER_ADDR_RESOLVED="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12341}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-local_scripts/zero3.json}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cp "${SCRIPT_PATH}" "${OUT_DIR}/launch_script.sh"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=================================================="
echo "Phase2 GRPO Full RL 2-Node 16-GPU Launch"
echo "=================================================="
echo "Repo root:                 ${REPO_ROOT}"
echo "Train root:                ${TRAIN_ROOT}"
echo "Env root:                  ${TRAIN_ENV_ROOT}"
echo "Python:                    ${TRAIN_ENV_PYTHON}"
echo "Torchrun:                  ${TRAIN_ENV_TORCHRUN}"
echo "Env RANK(node_rank):       ${RANK:-<unset>}"
echo "Env NPROC_PER_NODE:        ${NPROC_PER_NODE:-<unset>}"
echo "Node rank:                 ${NODE_RANK}"
echo "NNODES:                    ${NNODES}"
echo "NPROC_PER_NODE:            ${NPROC_PER_NODE}"
echo "MASTER_ADDR:               ${MASTER_ADDR_RESOLVED}"
echo "MASTER_PORT:               ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES:      ${CUDA_VISIBLE_DEVICES}"
echo "CUDA_HOME:                 ${CUDA_HOME}"
echo "Resolved nvcc:             ${NVCC_PATH:-<missing>}"
echo "Model path:                ${MODEL_PATH}"
echo "Dataset path:              ${DATASET_PATH}"
echo "Torch load mode:           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}"
echo "Resume checkpoint:         ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "Output dir:                ${OUT_DIR}"
echo "Run name:                  ${EXP_NAME}"
echo "Base GRPO settings:        Open-o3-Video/VisionCoach defaults"
echo "phase2 mode:               reweighting"
echo "phase2 teacher:            enabled, EMA update rate 0.01 every step"
echo "phase2 reweighting:        top-K local support, K=16, lambda=0.5, clip=0.2"
echo "phase2 schedule:           teacher/judge disabled after step 600"
echo "phase2 judge model:        ${LLM_AS_A_JUDGE_MODEL:-gpt-5.4}"
echo "=================================================="

TORCHRUN_CMD=(
  "${TRAIN_ENV_TORCHRUN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR_RESOLVED}"
  --master_port="${MASTER_PORT}"
  src/open_r1/grpo.py
  --output_dir "${OUT_DIR}"
  --model_name_or_path "${MODEL_PATH}"
  --dataset_name "${DATASET_PATH}"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --max_prompt_length 16384
  --max_completion_length 768
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 1e-6
  --lr_scheduler_type "cosine"
  --weight_decay 0.01
  --bf16
  --logging_steps 1
  --report_to wandb
  --gradient_checkpointing true
  --attn_implementation flash_attention_2
  --max_pixels 401408
  --num_train_epochs 1
  --max_steps -1
  --run_name "${EXP_NAME}"
  --save_strategy steps
  --save_steps 200
  --save_total_limit 100000
  --ignore_data_skip false
  --beta 0.0
  --max_grad_norm 5
  --save_only_model true
  --num_generations 4
  --spatial_iou_mode avg
  --identity_match_mode soft
  --spatial_norm_mode matched
  --correct_tempgate true
  --enable_phase2_teacher true
  --phase2_enable_ref_model false
  --phase2_enable_teacher_model true
  --phase2_teacher_update_mode ema
  --phase2_teacher_update_rate 0.01
  --phase2_teacher_update_interval 1
  --phase2_importance_sampling_level grpo
  --phase2_reweighting_mixing_lambda 0.5
  --phase2_reweighting_weight_mode topk_interpolate
  --phase2_reweighting_topk 16
  --phase2_reweighting_topk_gamma 0.0
  --phase2_reweighting_weight_clip 0.2
  --phase2_reweighting_anneal_steps 600
  --phase2_teacher_disable_after_step 600
  --phase2_process_feedback_disable_after_step 600
  --phase2_teacher_max_prompt_length 18432
  --phase2_process_feedback_enable true
  --phase2_process_feedback_timeout 30.0
  --phase2_process_feedback_scope all_with_hindsight
  --phase2_process_feedback_model "${LLM_AS_A_JUDGE_MODEL:-gpt-5.4}"
  --phase2_process_feedback_base_url "${LLM_AS_A_JUDGE_BASE:-${DEFAULT_JUDGE_BASE_URL}}"
  --phase2_process_feedback_api_key "${LLM_AS_A_JUDGE_API_KEY:-${DEFAULT_JUDGE_API_KEY}}"
  --phase2_process_feedback_max_feedback_chars 1000
)


if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  TORCHRUN_CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TORCHRUN_CMD[@]}"
