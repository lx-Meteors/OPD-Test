#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# CFG-guided teacher on top of the topk-16 baseline OPD
# (opd-baseline-qwen3-4b-nonthinking-step1200-qwen3-4b: student Qwen3-4B, teacher
# Qwen3-4B-Non-Thinking-RL-Math-Step1200, 12288 response length, n=4, mbs=64,
# topk=16, only_stu, student_p weights, token_reward_direct). Everything is
# identical to that baseline -- same models, data, sampling, batch sizes,
# sequence lengths, validation, weights -- except the per-cell teacher term:
#
#     reward_k = -w_k * ( log p_k - ((1+g) * log q_full,k - g * log q_free,k) )
#
# where q_free is the teacher scored on [anchor + response] with the prompt
# deleted. g = 0.3 from the local sweep of the pure (unclamped) formula.
# The teacher context window (tchwin) stays 0: the teacher sees the full prefix.
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step1200}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"
export TEACHER_CTX_WINDOW=0
export TEACHER_CFG_GAMMA="${TEACHER_CFG_GAMMA:-0.3}"

run_opd "opd-cfg-qwen3-4b-nonthinking-step1200-qwen3-4b" "$@"
