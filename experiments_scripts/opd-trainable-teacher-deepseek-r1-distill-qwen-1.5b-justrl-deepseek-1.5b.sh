#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-6000}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console,wandb]}"

# Student keeps the standard OPD objective. Only the Teacher LoRA uses verifier feedback.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export N_RESPONSES="${N_RESPONSES:-4}"
export TRAIN_TEACHER="${TRAIN_TEACHER:-True}"
export TEACHER_LORA_RANK="${TEACHER_LORA_RANK:-8}"
export TEACHER_LORA_ALPHA="${TEACHER_LORA_ALPHA:-16}"
export TEACHER_LR="${TEACHER_LR:-1e-5}"
export TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}"
export TEACHER_ANCHOR_TOP_K="${TEACHER_ANCHOR_TOP_K:-16}"
export TEACHER_UPDATE_INTERVAL="${TEACHER_UPDATE_INTERVAL:-1}"
export TEACHER_UPDATE_EPOCHS="${TEACHER_UPDATE_EPOCHS:-1}"
export TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-1}"
export TEACHER_MAX_GRAD_NORM="${TEACHER_MAX_GRAD_NORM:-1.0}"
export TEACHER_NORM_ADV_BY_STD="${TEACHER_NORM_ADV_BY_STD:-True}"

run_opd "opd-trainable-teacher-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
