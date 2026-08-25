#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

has_huggingface_model_weights() {
    local model_dir="$1"
    [[ -f "${model_dir}/config.json" ]] || return 1
    compgen -G "${model_dir}/*.safetensors" >/dev/null \
        || compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null
}

resolve_reference_checkpoint() {
    local checkpoint_path="$1"
    local resolved_path
    resolved_path="$(resolve_path "${checkpoint_path}")"

    if [[ ! -d "${resolved_path}" ]]; then
        echo "Reference checkpoint directory does not exist: ${resolved_path}" >&2
        exit 1
    fi

    # Accept a directly loadable Hugging Face model directory.
    if has_huggingface_model_weights "${resolved_path}"; then
        RESOLVED_REFERENCE_MODEL_PATH="${resolved_path}"
        return
    fi

    # Accept either global_step_*/actor/huggingface or actor/huggingface when it contains merged weights.
    local actor_dir="${resolved_path}"
    if [[ -d "${resolved_path}/actor" ]]; then
        actor_dir="${resolved_path}/actor"
    fi
    if has_huggingface_model_weights "${actor_dir}/huggingface"; then
        RESOLVED_REFERENCE_MODEL_PATH="${actor_dir}/huggingface"
        return
    fi
    if has_huggingface_model_weights "${actor_dir}/merged_huggingface"; then
        RESOLVED_REFERENCE_MODEL_PATH="${actor_dir}/merged_huggingface"
        return
    fi

    # Raw VERL checkpoints are sharded and cannot be passed directly to from_pretrained.
    # Merge them once and reuse the resulting Hugging Face directory on later launches.
    local backend=""
    if [[ -f "${actor_dir}/fsdp_config.json" ]]; then
        backend="fsdp"
    elif [[ -d "${actor_dir}/dist_ckpt" ]]; then
        backend="megatron"
    fi

    if [[ -z "${backend}" ]]; then
        echo "Unsupported Reference checkpoint layout: ${resolved_path}" >&2
        echo "Expected a Hugging Face model directory, global_step_*/actor, or a raw VERL actor checkpoint." >&2
        exit 1
    fi
    if [[ "${AUTO_MERGE_REFERENCE_CHECKPOINT:-True}" != "True" ]]; then
        echo "Reference checkpoint is a raw ${backend} checkpoint and must be merged first: ${actor_dir}" >&2
        echo "Set AUTO_MERGE_REFERENCE_CHECKPOINT=True or provide its merged Hugging Face directory." >&2
        exit 1
    fi

    local merged_dir="${actor_dir}/merged_huggingface"
    echo "Merging ${backend} Reference checkpoint into Hugging Face format: ${merged_dir}"
    activate_opd_env
    PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
        python -m verl.model_merger merge \
        --backend "${backend}" \
        --local_dir "${actor_dir}" \
        --target_dir "${merged_dir}"

    if ! has_huggingface_model_weights "${merged_dir}"; then
        echo "Reference checkpoint merge completed without loadable model weights: ${merged_dir}" >&2
        exit 1
    fi
    RESOLVED_REFERENCE_MODEL_PATH="${merged_dir}"
}

# Hybrid setting: Qwen3-4B Student, RL-Math Step500 Teacher.
if [[ -z "${ACTOR_MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_ROOT}/Qwen3-4B" ]]; then
        export ACTOR_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B"
    else
        export ACTOR_MODEL_PATH="Qwen/Qwen3-4B"
    fi
fi

# Optional: point this at a merged HF checkpoint, global_step_* directory, or its actor directory.
# Example: /path/to/checkpoint/global_step_50
export REFERENCE_CHECKPOINT_PATH="${REFERENCE_CHECKPOINT_PATH:-}"
export AUTO_MERGE_REFERENCE_CHECKPOINT="${AUTO_MERGE_REFERENCE_CHECKPOINT:-True}"
if [[ -n "${REFERENCE_CHECKPOINT_PATH}" ]]; then
    RESOLVED_REFERENCE_MODEL_PATH=""
    resolve_reference_checkpoint "${REFERENCE_CHECKPOINT_PATH}"
    export REFERENCE_MODEL_PATH="${RESOLVED_REFERENCE_MODEL_PATH}"
else
    export REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-${ACTOR_MODEL_PATH}}"
fi
if [[ -z "${REWARD_MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500" ]]; then
        export REWARD_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500"
    else
        export REWARD_MODEL_PATH="Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"
    fi
fi

export GOPD_DATA_ROOT="${GOPD_DATA_ROOT:-${DATA_ROOT}/G-OPD-Training-Data}"
export TRAIN_DATASET="${TRAIN_DATASET:-${GOPD_DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
if [[ ! -f "${TRAIN_DATASET}" && "${AUTO_DOWNLOAD_GOPD_DATA:-1}" == "1" ]]; then
    mkdir -p "$(dirname "${TRAIN_DATASET}")"
    if ! curl --fail --location --retry 3 \
        "https://huggingface.co/datasets/Keven16/G-OPD-Training-Data/resolve/main/DeepMath-103K/train_filtered_level6.parquet?download=true" \
        --output "${TRAIN_DATASET}.part"; then
        rm -f "${TRAIN_DATASET}.part"
        curl --fail --location --retry 3 \
            "https://hf-mirror.com/datasets/Keven16/G-OPD-Training-Data/resolve/main/DeepMath-103K/train_filtered_level6.parquet?download=true" \
            --output "${TRAIN_DATASET}.part"
    fi
    mv "${TRAIN_DATASET}.part" "${TRAIN_DATASET}"
fi
export TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-DeepMath-103K-level6-57k}"
export TEST_DATASET="${TEST_DATASET:-[\"${DATA_ROOT}/test_data/AMC23/test.parquet\",\"${DATA_ROOT}/test_data/AIME24/test.parquet\",\"${DATA_ROOT}/test_data/AIME25/test.parquet\",\"${DATA_ROOT}/test_data/HMMT24/test.parquet\",\"${DATA_ROOT}/test_data/HMMT25/test.parquet\"]}"

export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export GOPD_ENABLE="${GOPD_ENABLE:-False}"
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.0}"
export USE_KL="${USE_KL:-False}"
export KL_COEF="${KL_COEF:-0.0}"
export KL_TYPE="${KL_TYPE:-low_var_kl}"

# Student rolls out at most 2K response tokens.  OPD is computed only on this
# Student prefix.  For prefixes that reach the boundary, the Teacher samples
# four continuations; the verifier keeps the best correct continuation and the
# Student learns that suffix with token-level cross entropy.
export HANDOFF_OPD_ENABLE="${HANDOFF_OPD_ENABLE:-True}"
export HANDOFF_OPD_CUTOFF="${HANDOFF_OPD_CUTOFF:-2048}"
export HANDOFF_TEACHER_NUM_SAMPLES="${HANDOFF_TEACHER_NUM_SAMPLES:-4}"
export HANDOFF_VERIFIED_SUFFIX_ONLY="${HANDOFF_VERIFIED_SUFFIX_ONLY:-True}"
export HANDOFF_SELECT_BEST_VERIFIED_SUFFIX="${HANDOFF_SELECT_BEST_VERIFIED_SUFFIX:-True}"
export HANDOFF_CORRECT_THRESHOLD="${HANDOFF_CORRECT_THRESHOLD:-0.5}"
export HANDOFF_SFT_WEIGHT="${HANDOFF_SFT_WEIGHT:-1.0}"
export HANDOFF_TEACHER_BACKEND="${HANDOFF_TEACHER_BACKEND:-vllm}"
export HANDOFF_TEACHER_TEMPERATURE="${HANDOFF_TEACHER_TEMPERATURE:-1.0}"
export HANDOFF_TEACHER_TOP_P="${HANDOFF_TEACHER_TOP_P:-0.95}"
export HANDOFF_TEACHER_DTYPE="${HANDOFF_TEACHER_DTYPE:-float32}"
export HANDOFF_TEACHER_GPU_MEMORY_UTILIZATION="${HANDOFF_TEACHER_GPU_MEMORY_UTILIZATION:-0.4}"
export HANDOFF_TEACHER_MAX_MODEL_LEN="${HANDOFF_TEACHER_MAX_MODEL_LEN:-32768}"
export HANDOFF_TEACHER_MAX_NUM_BATCHED_TOKENS="${HANDOFF_TEACHER_MAX_NUM_BATCHED_TOKENS:-32768}"
export HANDOFF_TEACHER_MAX_NUM_SEQS="${HANDOFF_TEACHER_MAX_NUM_SEQS:-64}"
export HANDOFF_TEACHER_ENFORCE_EAGER="${HANDOFF_TEACHER_ENFORCE_EAGER:-False}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1024}"
export N_RESPONSES="${N_RESPONSES:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-16384}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-16384}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export DATA_SEED="${DATA_SEED:-42}"

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-0}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
export MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
export REFERENCE_MODEL_DTYPE="${REFERENCE_MODEL_DTYPE:-fp32}"
export TEACHER_MODEL_DTYPE="${TEACHER_MODEL_DTYPE:-fp32}"

export ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
export ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ="${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-False}"
export REF_LOG_PROB_USE_DYNAMIC_BSZ="${REF_LOG_PROB_USE_DYNAMIC_BSZ:-False}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export TEACHER_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${TEACHER_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
export ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
export FSDP_FORWARD_PREFETCH="${FSDP_FORWARD_PREFETCH:-False}"
export REFERENCE_PARAM_OFFLOAD="${REFERENCE_PARAM_OFFLOAD:-True}"
export TEACHER_PARAM_OFFLOAD="${TEACHER_PARAM_OFFLOAD:-False}"

export ROLLOUT_IS="${ROLLOUT_IS:-token}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"
export ROLLOUT_RS="${ROLLOUT_RS:-null}"
export ROLLOUT_BYPASS_OLD_LOGPROB="${ROLLOUT_BYPASS_OLD_LOGPROB:-False}"

export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
export VAL_TOP_P="${VAL_TOP_P:-1.0}"
export VAL_N_RESPONSES="${VAL_N_RESPONSES:-32}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

export PROJECT_NAME="${PROJECT_NAME:-on-policy-distillation}"
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[\"console\",\"wandb\"]}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"

run_opd "${OPD_RUN_NAME:-opd-2k-handoff-bestof4-correct-sft-qwen3-4b-step500}" "$@"
