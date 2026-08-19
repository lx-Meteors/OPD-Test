#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"

# Replace PUT_YOUR_RUN_DIR_HERE with the directory name containing global_step_200.
# The path must end at global_step_200; do not append /actor.
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-/ossfs/workspace/code/Prune-OPD/checkpoint/opd-plus-grpo-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b_token_reward_direct_plus_grpo_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_12288-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_stu-rw_student_p-2026-08-18_21-22-40/global_step_200}"

export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console,wandb]}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

# Frozen Teacher OPD + verifier GRPO on the same Student rollouts.
export ADV_ESTIMATOR=token_reward_direct_plus_grpo
export GRPO_OUTCOME_WEIGHT="${GRPO_OUTCOME_WEIGHT:-1.0}"
export N_RESPONSES="${N_RESPONSES:-4}"
export TRAIN_TEACHER=False

if [[ "${RESUME_FROM_PATH%/}" != */global_step_200 ]]; then
    echo "RESUME_FROM_PATH must point to the global_step_200 directory, not its actor/ subdirectory" >&2
    exit 1
fi
if [[ ! -d "${RESUME_FROM_PATH}" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Checkpoint directory does not exist: ${RESUME_FROM_PATH}" >&2
    echo "Edit RESUME_FROM_PATH near the top of this script." >&2
    exit 1
fi

run_opd "opd-plus-grpo-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" \
    "trainer.resume_mode=resume_path" \
    "trainer.resume_from_path=${RESUME_FROM_PATH}" \
    "$@"
