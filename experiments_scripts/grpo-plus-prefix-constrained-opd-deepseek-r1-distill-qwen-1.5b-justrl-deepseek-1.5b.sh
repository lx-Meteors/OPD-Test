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
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console,wandb]}"

# GRPO is the full-trajectory objective. Frozen-Teacher OPD is only an
# adaptive constraint on the first PREFIX_TOKENS response tokens.
export ADV_ESTIMATOR=token_reward_direct_plus_grpo
export GRPO_OUTCOME_WEIGHT="${GRPO_OUTCOME_WEIGHT:-1.0}"
export N_RESPONSES="${N_RESPONSES:-4}"
export TRAIN_TEACHER=False

export OPD_PREFIX_TOKENS="${OPD_PREFIX_TOKENS:-5000}"
export OPD_CONSTRAINT_TARGET="${OPD_CONSTRAINT_TARGET:-0.02}"
export OPD_CONSTRAINT_INIT_COEF="${OPD_CONSTRAINT_INIT_COEF:-1.0}"
export OPD_CONSTRAINT_DUAL_LR="${OPD_CONSTRAINT_DUAL_LR:-1.0}"
export OPD_CONSTRAINT_MAX_COEF="${OPD_CONSTRAINT_MAX_COEF:-2.0}"

run_opd "grpo-prefix5k-opd-constraint" \
    "actor_rollout_ref.actor.opd_constraint_enable=True" \
    "actor_rollout_ref.actor.opd_constraint_prefix_tokens=${OPD_PREFIX_TOKENS}" \
    "actor_rollout_ref.actor.opd_constraint_target=${OPD_CONSTRAINT_TARGET}" \
    "actor_rollout_ref.actor.opd_constraint_init_coef=${OPD_CONSTRAINT_INIT_COEF}" \
    "actor_rollout_ref.actor.opd_constraint_dual_lr=${OPD_CONSTRAINT_DUAL_LR}" \
    "actor_rollout_ref.actor.opd_constraint_max_coef=${OPD_CONSTRAINT_MAX_COEF}" \
    "$@"
