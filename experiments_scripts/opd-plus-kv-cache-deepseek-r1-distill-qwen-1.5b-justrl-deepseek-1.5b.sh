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
export KV_CACHE_DISTILL_LOSS_COEF="${KV_CACHE_DISTILL_LOSS_COEF:-0.5}"
export KV_CACHE_DISTILL_LAYER_INDICES="${KV_CACHE_DISTILL_LAYER_INDICES:-[-1]}"
export KV_CACHE_DISTILL_TOKEN_SCOPE="${KV_CACHE_DISTILL_TOKEN_SCOPE:-all}"
export KV_CACHE_DISTILL_TOKEN_CHUNK_SIZE="${KV_CACHE_DISTILL_TOKEN_CHUNK_SIZE:-1024}"
export KV_CACHE_DISTILL_KEY_WEIGHT="${KV_CACHE_DISTILL_KEY_WEIGHT:-1.0}"
export KV_CACHE_DISTILL_VALUE_WEIGHT="${KV_CACHE_DISTILL_VALUE_WEIGHT:-1.0}"

if [[ "${PARALLEL_SIZE}" != "1" ]]; then
    echo "KV-cache distillation currently requires PARALLEL_SIZE=1." >&2
    exit 1
fi

run_opd "opd-plus-kv-cache-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" \
    "actor_rollout_ref.actor.kv_cache_distill.enable=True" \
    "actor_rollout_ref.actor.kv_cache_distill.loss_coef=${KV_CACHE_DISTILL_LOSS_COEF}" \
    "actor_rollout_ref.actor.kv_cache_distill.layer_indices=${KV_CACHE_DISTILL_LAYER_INDICES}" \
    "actor_rollout_ref.actor.kv_cache_distill.token_scope=${KV_CACHE_DISTILL_TOKEN_SCOPE}" \
    "actor_rollout_ref.actor.kv_cache_distill.token_chunk_size=${KV_CACHE_DISTILL_TOKEN_CHUNK_SIZE}" \
    "actor_rollout_ref.actor.kv_cache_distill.key_loss_weight=${KV_CACHE_DISTILL_KEY_WEIGHT}" \
    "actor_rollout_ref.actor.kv_cache_distill.value_loss_weight=${KV_CACHE_DISTILL_VALUE_WEIGHT}" \
    "$@"
