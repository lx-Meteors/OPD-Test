#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"

# Continue from the actor trained with OPD on the first 6K response tokens.
# ACTOR_MODEL_PATH must remain a Hugging Face model directory; the full Verl
# checkpoint is restored separately so actor/optimizer state is loaded correctly.
export SOURCE_CHECKPOINT_PATH="${SOURCE_CHECKPOINT_PATH:-/ossfs/workspace/OPD-Test/checkpoint/6k-opd-baseline-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b_token_reward_direct_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_6000-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-08-17_19-44-20/global_step_203}"

# Verl counts total_training_steps globally after a resume. Starting at step 203,
# a target of 403 performs 200 additional optimization steps.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-403}"
export TOTAL_TRAINING_EPOCHS="${TOTAL_TRAINING_EPOCHS:-2}"

# Generate a full 12K response, but optimize OPD only on response positions
# [6000, 12288). Validation remains full-length and is not position-masked.
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export LOSS_POSITION_START="${LOSS_POSITION_START:-6000}"
export LOSS_POSITION_END="${LOSS_POSITION_END:-12288}"

run_opd \
    "opd-suffix-6k-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" \
    "trainer.resume_mode=resume_path" \
    "trainer.resume_from_path=${SOURCE_CHECKPOINT_PATH}" \
    "$@"
