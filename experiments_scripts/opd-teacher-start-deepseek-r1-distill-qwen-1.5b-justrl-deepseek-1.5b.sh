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

# The teacher writes this many response tokens. The student rolls out the remainder.
export TEACHER_START_PREFIX_LENGTH="${TEACHER_START_PREFIX_LENGTH:-500}"
export TEACHER_START_MIN_STUDENT_RESPONSE_LENGTH="${TEACHER_START_MIN_STUDENT_RESPONSE_LENGTH:-256}"
export TEACHER_START_LOSS_COEF="${TEACHER_START_LOSS_COEF:-1.0}"
export TEACHER_START_DO_SAMPLE="${TEACHER_START_DO_SAMPLE:-True}"
export TEACHER_START_TEMPERATURE="${TEACHER_START_TEMPERATURE:-1.0}"
export TEACHER_START_TOP_P="${TEACHER_START_TOP_P:-1.0}"

if (( MAX_RESP_LENGTH - TEACHER_START_PREFIX_LENGTH < TEACHER_START_MIN_STUDENT_RESPONSE_LENGTH )); then
    echo "MAX_RESP_LENGTH must leave at least ${TEACHER_START_MIN_STUDENT_RESPONSE_LENGTH} student tokens after the teacher prefix." >&2
    exit 1
fi

run_opd "opd-teacher-start-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" \
    "actor_rollout_ref.rollout.teacher_start.enable=True" \
    "actor_rollout_ref.rollout.teacher_start.prefix_length=${TEACHER_START_PREFIX_LENGTH}" \
    "actor_rollout_ref.rollout.teacher_start.min_student_response_length=${TEACHER_START_MIN_STUDENT_RESPONSE_LENGTH}" \
    "actor_rollout_ref.rollout.teacher_start.do_sample=${TEACHER_START_DO_SAMPLE}" \
    "actor_rollout_ref.rollout.teacher_start.temperature=${TEACHER_START_TEMPERATURE}" \
    "actor_rollout_ref.rollout.teacher_start.top_p=${TEACHER_START_TOP_P}" \
    "actor_rollout_ref.actor.teacher_start.enable=True" \
    "actor_rollout_ref.actor.teacher_start.loss_coef=${TEACHER_START_LOSS_COEF}" \
    "$@"
