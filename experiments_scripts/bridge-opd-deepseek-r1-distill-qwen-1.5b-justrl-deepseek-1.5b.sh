#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-/input0/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export N_RESPONSES="${N_RESPONSES:-4}"
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export BRIDGE_OPD_BETA="${BRIDGE_OPD_BETA:-0.2}"
export BRIDGE_OPD_ADAPTIVE_BETA="${BRIDGE_OPD_ADAPTIVE_BETA:-True}"
export BRIDGE_OPD_MIN_ESS_RATIO="${BRIDGE_OPD_MIN_ESS_RATIO:-0.5}"
export BRIDGE_OPD_BETA_SEARCH_STEPS="${BRIDGE_OPD_BETA_SEARCH_STEPS:-12}"

run_opd "bridge-opd-r1-1p5b-justrl-1p5b-b_${BRIDGE_OPD_BETA}-ess_${BRIDGE_OPD_MIN_ESS_RATIO}" \
    "+actor_rollout_ref.rollout.bridge_opd.enable=True" \
    "+actor_rollout_ref.rollout.bridge_opd.beta=${BRIDGE_OPD_BETA}" \
    "+actor_rollout_ref.rollout.bridge_opd.adaptive_beta=${BRIDGE_OPD_ADAPTIVE_BETA}" \
    "+actor_rollout_ref.rollout.bridge_opd.min_ess_ratio=${BRIDGE_OPD_MIN_ESS_RATIO}" \
    "+actor_rollout_ref.rollout.bridge_opd.beta_search_steps=${BRIDGE_OPD_BETA_SEARCH_STEPS}" \
    "$@"
