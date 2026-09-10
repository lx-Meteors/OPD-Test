#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Standard OPD control. Models, data, rollout, optimizer and validation settings
# are inherited from the shared Qwen3-4B experiment configuration.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
export GOPD_ENABLE=False
export GOPD_LAMBDA=1.0
export OPD_MAX_TOKENS=0
export GOPD_EXTRAPOLATION_MAX_TOKENS=0
export USE_KL=False
export KL_COEF=0.0

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-opd-qwen3-4b-base-teacher-step500-train${TOTAL_TRAINING_STEPS}}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
