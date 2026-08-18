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

# Frozen Teacher OPD + verifier GRPO on the same Student rollouts.
export ADV_ESTIMATOR=token_reward_direct_plus_grpo
export GRPO_OUTCOME_WEIGHT="${GRPO_OUTCOME_WEIGHT:-1.0}"
export N_RESPONSES="${N_RESPONSES:-4}"
export TRAIN_TEACHER=False

run_opd "opd-plus-grpo-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
