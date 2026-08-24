#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"

# Reverse-direction standard OPD:
#   Student: Qwen3-4B-Non-Thinking-RL-Math-Step500
#   Teacher: Qwen3-4B base
if [[ -z "${ACTOR_MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500" ]]; then
        export ACTOR_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500"
    else
        export ACTOR_MODEL_PATH="Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"
    fi
fi

if [[ -z "${REWARD_MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_ROOT}/Qwen3-4B" ]]; then
        export REWARD_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B"
    else
        export REWARD_MODEL_PATH="Qwen/Qwen3-4B"
    fi
fi

export GOPD_ENABLE=False
export GOPD_LAMBDA=1.0
export USE_KL=False
export KL_COEF=0.0
export ADV_ESTIMATOR=token_reward_direct
# Loading the 4B student independently on all eight FSDP ranks in fp32 creates a
# large host-memory spike before sharding.  It is also incompatible with
# FlashAttention 2.  Keep every frozen/trainable model in bf16 for this run.
export MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
export REFERENCE_MODEL_DTYPE="${REFERENCE_MODEL_DTYPE:-bf16}"
export TEACHER_MODEL_DTYPE="${TEACHER_MODEL_DTYPE:-bf16}"
export SAVE_FREQ=10
export OPD_RUN_NAME="opd-standard-reverse-step500-student-qwen3-4b-teacher"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
