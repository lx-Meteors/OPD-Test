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

# G-OPD uses a frozen base/reference R to extract the dense signal log(T/R).
# The official default mode uses Student init; reward-correction experiments
# may instead use the Teacher's pre-RL base checkpoint.
export G_OPD_REFERENCE_MODEL_PATH="${G_OPD_REFERENCE_MODEL_PATH:-${ACTOR_MODEL_PATH}}"
export G_OPD_LAMBDA="${G_OPD_LAMBDA:-1.25}"

# Keep every other choice aligned with the standard OPD baseline.
export PROJECT_NAME="${PROJECT_NAME:-PruneOPD}"
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console,wandb]}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-g-opd}"
export N_RESPONSES="${N_RESPONSES:-4}"
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
export TOP_K_STRATEGY="only_stu"
export REWARD_WEIGHT_MODE="student_p"
export USE_KL="False"

require_model_if_local_path "${G_OPD_REFERENCE_MODEL_PATH}"

run_opd "g-opd-r1-1p5b-justrl-1p5b-lambda_${G_OPD_LAMBDA}" \
    "+actor_rollout_ref.ref.model.path=${G_OPD_REFERENCE_MODEL_PATH}" \
    "+actor_rollout_ref.rollout.g_opd.enable=True" \
    "+actor_rollout_ref.rollout.g_opd.lambda=${G_OPD_LAMBDA}" \
    "$@"
