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

# Match the standard OPD baseline except for the target construction.
export N_RESPONSES="${N_RESPONSES:-4}"
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
export TOP_K_STRATEGY="only_stu"
export REWARD_WEIGHT_MODE="student_p"
export USE_KL="False"

# kappa=0.25 is the tangent counterpart of ExOPD lambda=1.25.
export CHI2_OPD_KAPPA="${CHI2_OPD_KAPPA:-0.25}"
export CHI2_OPD_ADAPTIVE_KAPPA="${CHI2_OPD_ADAPTIVE_KAPPA:-True}"
export CHI2_OPD_MIN_DENSITY_RATIO="${CHI2_OPD_MIN_DENSITY_RATIO:-0.1}"
export CHI2_OPD_MAX_DENSITY_RATIO="${CHI2_OPD_MAX_DENSITY_RATIO:-3.0}"

run_opd "chi2-opd-r1-1p5b-justrl-1p5b-k_${CHI2_OPD_KAPPA}" \
    "+actor_rollout_ref.rollout.chi2_opd.enable=True" \
    "+actor_rollout_ref.rollout.chi2_opd.kappa=${CHI2_OPD_KAPPA}" \
    "+actor_rollout_ref.rollout.chi2_opd.adaptive_kappa=${CHI2_OPD_ADAPTIVE_KAPPA}" \
    "+actor_rollout_ref.rollout.chi2_opd.min_density_ratio=${CHI2_OPD_MIN_DENSITY_RATIO}" \
    "+actor_rollout_ref.rollout.chi2_opd.max_density_ratio=${CHI2_OPD_MAX_DENSITY_RATIO}" \
    "$@"
