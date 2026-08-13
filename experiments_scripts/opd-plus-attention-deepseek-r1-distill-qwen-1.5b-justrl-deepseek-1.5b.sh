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
export PARALLEL_SIZE="${PARALLEL_SIZE:-1}"
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console,wandb]}"
export ATTENTION_DISTILL_LOSS_COEF="${ATTENTION_DISTILL_LOSS_COEF:-0.1}"
export ATTENTION_DISTILL_QUERY_CHUNK_SIZE="${ATTENTION_DISTILL_QUERY_CHUNK_SIZE:-32}"

if [[ "${PARALLEL_SIZE}" != "1" ]]; then
    echo "Attention distillation currently requires PARALLEL_SIZE=1." >&2
    exit 1
fi

run_opd "opd-plus-full-attention-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" \
    "actor_rollout_ref.actor.attention_distill.enable=True" \
    "actor_rollout_ref.actor.attention_distill.loss_coef=${ATTENTION_DISTILL_LOSS_COEF}" \
    "actor_rollout_ref.actor.attention_distill.query_chunk_size=${ATTENTION_DISTILL_QUERY_CHUNK_SIZE}" \
    "actor_rollout_ref.actor.attention_distill.divergence=reverse_kl" \
    "$@"
