#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Controlled ablation for success-conditioned OPD supervision:
#   K_opd = OPD_SUCCESS_HORIZON_RATIO * mean successful response length.
# Student rollout remains full length so terminal correctness is observed. There
# is no Reference model and no G-OPD reward extrapolation in this experiment.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export GOPD_ENABLE=False
export GOPD_LAMBDA=1.0
export USE_KL=False
export KL_COEF=0.0
export OPD_MAX_TOKENS=0
export GOPD_EXTRAPOLATION_MAX_TOKENS=0

export SUCCESS_REWARD_THRESHOLD="${SUCCESS_REWARD_THRESHOLD:-0.5}"
export OPD_SUCCESS_HORIZON_RATIO="${OPD_SUCCESS_HORIZON_RATIO:-1.0}"
export SUCCESS_HORIZON_INIT_TOKENS="${SUCCESS_HORIZON_INIT_TOKENS:-8192}"
export SUCCESS_HORIZON_MIN_TOKENS="${SUCCESS_HORIZON_MIN_TOKENS:-1}"
export SUCCESS_HORIZON_MAX_TOKENS="${SUCCESS_HORIZON_MAX_TOKENS:-${MAX_RESP_LENGTH:-16384}}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
ratio_tag="${OPD_SUCCESS_HORIZON_RATIO//./p}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-opd-success-mean-ratio${ratio_tag}-qwen3-4b-base-teacher-step500-train${TOTAL_TRAINING_STEPS}}"

# extrapolation_ratio is set to 1.0 only to keep the shared horizon state valid;
# GOPD_ENABLE=False means no extrapolation term and no Ref forward are created.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.enable=True" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.init_len=${SUCCESS_HORIZON_INIT_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.min_len=${SUCCESS_HORIZON_MIN_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.max_len=${SUCCESS_HORIZON_MAX_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.reward_threshold=${SUCCESS_REWARD_THRESHOLD}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.opd_ratio=${OPD_SUCCESS_HORIZON_RATIO}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.extrapolation_ratio=1.0" \
    "$@"
