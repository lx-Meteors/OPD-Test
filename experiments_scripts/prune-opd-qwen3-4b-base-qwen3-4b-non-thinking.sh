#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export PRUNE_OPD_METRIC="${PRUNE_OPD_METRIC:-overlap_ratio}"
export PRUNE_OPD_THRESHOLD="${PRUNE_OPD_THRESHOLD:-0.7}"
export PRUNE_OPD_W_DROP="${PRUNE_OPD_W_DROP:-0.01}"
export PRUNE_OPD_W_BASE="${PRUNE_OPD_W_BASE:-0.5}"
export PRUNE_OPD_HIT_RATIO="${PRUNE_OPD_HIT_RATIO:-0.1}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"

run_opd "prune-opd-qwen3-4b-base-qwen3-4b-non-thinking-m_${PRUNE_OPD_METRIC}-t_${PRUNE_OPD_THRESHOLD}-w_${PRUNE_OPD_W_DROP}-b_${PRUNE_OPD_W_BASE}-h_${PRUNE_OPD_HIT_RATIO}" \
    "+actor_rollout_ref.rollout.prune_opd.enable=True" \
    "+actor_rollout_ref.rollout.prune_opd.metric=${PRUNE_OPD_METRIC}" \
    "+actor_rollout_ref.rollout.prune_opd.threshold=${PRUNE_OPD_THRESHOLD}" \
    "+actor_rollout_ref.rollout.prune_opd.w_drop=${PRUNE_OPD_W_DROP}" \
    "+actor_rollout_ref.rollout.prune_opd.w_base=${PRUNE_OPD_W_BASE}" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.enable=True" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.init_len=1024" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.min_len=1024" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.max_len=12288" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.step=100" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.hit_ratio=${PRUNE_OPD_HIT_RATIO}" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.margin=100" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.shrink_patience=3" \
    "$@"
