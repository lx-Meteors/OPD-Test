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

# beta=1.0 is standard OPD; beta=0.0 removes all temperature-explainable signal.
# The default retains a small confidence component because teacher confidence can
# still improve rollout stability. Override with PREFERENCE_OPD_CONFIDENCE_BETA=0
# to run the pure preference-only hypothesis.
export PREFERENCE_OPD_CONFIDENCE_BETA="${PREFERENCE_OPD_CONFIDENCE_BETA:-0.25}"
export PREFERENCE_OPD_MIN_SCALE="${PREFERENCE_OPD_MIN_SCALE:-0.1}"
export PREFERENCE_OPD_MAX_SCALE="${PREFERENCE_OPD_MAX_SCALE:-10.0}"
export PREFERENCE_OPD_NEWTON_STEPS="${PREFERENCE_OPD_NEWTON_STEPS:-6}"

run_opd "preference-opd-beta_${PREFERENCE_OPD_CONFIDENCE_BETA}" \
    "+actor_rollout_ref.rollout.preference_opd.enable=True" \
    "+actor_rollout_ref.rollout.preference_opd.confidence_beta=${PREFERENCE_OPD_CONFIDENCE_BETA}" \
    "+actor_rollout_ref.rollout.preference_opd.min_scale=${PREFERENCE_OPD_MIN_SCALE}" \
    "+actor_rollout_ref.rollout.preference_opd.max_scale=${PREFERENCE_OPD_MAX_SCALE}" \
    "+actor_rollout_ref.rollout.preference_opd.newton_steps=${PREFERENCE_OPD_NEWTON_STEPS}" \
    "$@"

