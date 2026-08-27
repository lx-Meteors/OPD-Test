#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# CFG-guided teacher on top of the 1.5B baseline OPD
# (opd-baseline-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b: student
# DeepSeek-R1-Distill-Qwen-1.5B, teacher JustRL-DeepSeek-1.5B, 12288 response
# length, and the common.sh defaults n=4, mbs=64, topk=16, only_stu, student_p,
# token_reward_direct). Everything matches that baseline except the per-cell
# teacher term in the top-k reward:
#
#     reward_k = -w_k * ( log p_k - ((1+g) * log q_full,k - g * log q_free,k) )
#
# where q_free is the teacher scored on [anchor + response] with the prompt
# deleted (anchor = BOS for the DeepSeek tokenizer). g = 0.3. The teacher
# context window (tchwin) stays 0: the teacher sees the full student prefix.
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export TEACHER_CTX_WINDOW=0
export TEACHER_CFG_GAMMA="${TEACHER_CFG_GAMMA:-0.3}"

run_opd "opd-cfg-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
