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
export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[wandb]}"

# Keep the same rollout count as the OPD baseline: four independent responses
# per prompt. Set-OPD compares this existing response set and adds no branching.
export N_RESPONSES="${N_RESPONSES:-4}"
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-set_opd}"
export SET_OPD_WEIGHT="${SET_OPD_WEIGHT:-0.05}"
export SET_OPD_FEATURE_DIM="${SET_OPD_FEATURE_DIM:-128}"
export SET_OPD_POSITION_BINS="${SET_OPD_POSITION_BINS:-8}"
export SET_OPD_MAX_POSITIONS="${SET_OPD_MAX_POSITIONS:-64}"
export SET_OPD_LOGDET_SCALE="${SET_OPD_LOGDET_SCALE:-1.0}"
export SET_OPD_QUALITY_WEIGHT="${SET_OPD_QUALITY_WEIGHT:-1.0}"
export SET_OPD_DIVERSITY_WEIGHT="${SET_OPD_DIVERSITY_WEIGHT:-1.0}"
export SET_OPD_NORMALIZE_BY_STD="${SET_OPD_NORMALIZE_BY_STD:-True}"
export SET_OPD_CORRECT_THRESHOLD="${SET_OPD_CORRECT_THRESHOLD:-0.5}"

run_opd "set-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
