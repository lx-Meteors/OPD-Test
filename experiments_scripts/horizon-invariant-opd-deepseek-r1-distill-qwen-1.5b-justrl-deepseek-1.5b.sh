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

# This aggregation makes the position-weighted objective exact across dynamic
# micro-batches. Uniform weights recover the scale of token-mean OPD.
export LOSS_AGG_MODE="seq-mean-token-sum"
export HORIZON_OPD_BIN_SIZE="${HORIZON_OPD_BIN_SIZE:-1024}"
export HORIZON_OPD_REFERENCE_STEPS="${HORIZON_OPD_REFERENCE_STEPS:-5}"
export HORIZON_OPD_ALPHA="${HORIZON_OPD_ALPHA:-1.0}"
export HORIZON_OPD_MIN_WEIGHT="${HORIZON_OPD_MIN_WEIGHT:-0.25}"
export HORIZON_OPD_MAX_WEIGHT="${HORIZON_OPD_MAX_WEIGHT:-3.0}"

run_opd "horizon-invariant-opd-a_${HORIZON_OPD_ALPHA}-clip_${HORIZON_OPD_MIN_WEIGHT}_${HORIZON_OPD_MAX_WEIGHT}" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.enable=True" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.bin_size=${HORIZON_OPD_BIN_SIZE}" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.reference_steps=${HORIZON_OPD_REFERENCE_STEPS}" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.alpha=${HORIZON_OPD_ALPHA}" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.min_weight=${HORIZON_OPD_MIN_WEIGHT}" \
    "+actor_rollout_ref.rollout.horizon_invariant_opd.max_weight=${HORIZON_OPD_MAX_WEIGHT}" \
    "$@"
