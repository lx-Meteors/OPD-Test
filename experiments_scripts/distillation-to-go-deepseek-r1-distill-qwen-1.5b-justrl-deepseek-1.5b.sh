#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export N_RESPONSES="${N_RESPONSES:-4}"
export LOSS_AGG_MODE="token-mean"

export DTG_WEIGHT="${DTG_WEIGHT:-0.1}"
export DTG_BLOCK_SIZE="${DTG_BLOCK_SIZE:-256}"
export DTG_BLOCK_GAMMA="${DTG_BLOCK_GAMMA:-0.95}"
export DTG_OUTCOME_WEIGHT="${DTG_OUTCOME_WEIGHT:-0.25}"
export DTG_NORMALIZE_BY_STD="${DTG_NORMALIZE_BY_STD:-False}"
export DTG_MAX_ABS_ADVANTAGE="${DTG_MAX_ABS_ADVANTAGE:-0.5}"

run_opd "distillation-to-go-w_${DTG_WEIGHT}-b_${DTG_BLOCK_SIZE}-g_${DTG_BLOCK_GAMMA}" \
    "+algorithm.distillation_to_go.enable=True" \
    "+algorithm.distillation_to_go.weight=${DTG_WEIGHT}" \
    "+algorithm.distillation_to_go.block_size=${DTG_BLOCK_SIZE}" \
    "+algorithm.distillation_to_go.block_gamma=${DTG_BLOCK_GAMMA}" \
    "+algorithm.distillation_to_go.outcome_weight=${DTG_OUTCOME_WEIGHT}" \
    "+algorithm.distillation_to_go.normalize_by_std=${DTG_NORMALIZE_BY_STD}" \
    "+algorithm.distillation_to_go.max_abs_advantage=${DTG_MAX_ABS_ADVANTAGE}" \
    "$@"
