#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Success-conditioned dual horizon:
#   K_opd = mean response length among successful rollouts in the current batch
#   K_ext = floor(GOPD_EXTRAPOLATION_RATIO * K_opd)
# Student rollout and validation remain full length. Teacher forward is physically
# shortened to K_opd and Ref forward to K_ext; outputs are padded neutrally before
# the normal trainer dataflow continues.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
export GOPD_ENABLE=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.25}"
export USE_KL=True
export KL_COEF=0.0

# Search this parameter directly, for example:
#   GOPD_EXTRAPOLATION_RATIO=0.25 bash experiments_scripts/meteor-gopd-success-horizon-qwen3-4b-base-teacher-step500.sh
export GOPD_EXTRAPOLATION_RATIO="${GOPD_EXTRAPOLATION_RATIO:-0.5}"
export SUCCESS_REWARD_THRESHOLD="${SUCCESS_REWARD_THRESHOLD:-0.5}"
export SUCCESS_HORIZON_INIT_TOKENS="${SUCCESS_HORIZON_INIT_TOKENS:-8192}"
export SUCCESS_HORIZON_MIN_TOKENS="${SUCCESS_HORIZON_MIN_TOKENS:-1}"
export SUCCESS_HORIZON_MAX_TOKENS="${SUCCESS_HORIZON_MAX_TOKENS:-${MAX_RESP_LENGTH:-16384}}"

# Static limits are disabled after initialization; the current-batch horizons
# above control both the loss masks and the actual Teacher/Ref forward lengths.
export OPD_MAX_TOKENS=0
export GOPD_EXTRAPOLATION_MAX_TOKENS=0
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
ratio_tag="${GOPD_EXTRAPOLATION_RATIO//./p}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-gopd-success-mean-opd-ext${ratio_tag}-qwen3-4b-base-teacher-step500-train${TOTAL_TRAINING_STEPS}}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.enable=True" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.init_len=${SUCCESS_HORIZON_INIT_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.min_len=${SUCCESS_HORIZON_MIN_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.max_len=${SUCCESS_HORIZON_MAX_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.reward_threshold=${SUCCESS_REWARD_THRESHOLD}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.opd_ratio=1.0" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.extrapolation_ratio=${GOPD_EXTRAPOLATION_RATIO}" \
    "$@"
