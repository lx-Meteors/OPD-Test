#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Standard G-OPD: full-response OPD and full-response Teacher-Ref reward
# extrapolation. The fixed Ref defaults to the Student initialization.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
export GOPD_ENABLE=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.25}"
export OPD_MAX_TOKENS="${OPD_MAX_TOKENS:-0}"
export GOPD_EXTRAPOLATION_MAX_TOKENS="${GOPD_EXTRAPOLATION_MAX_TOKENS:-0}"
export USE_KL=True
export KL_COEF=0.0

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-gopd-qwen3-4b-base-teacher-step500-lambda${GOPD_LAMBDA}-train${TOTAL_TRAINING_STEPS}}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
