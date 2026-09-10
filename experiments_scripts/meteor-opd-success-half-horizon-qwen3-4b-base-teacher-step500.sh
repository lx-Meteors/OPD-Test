#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefix-length control:
#   K_opd = 0.5 * mean response length among successful current-batch rollouts.
# This changes only the OPD supervision horizon; Student rollout remains full.
export OPD_SUCCESS_HORIZON_RATIO="${OPD_SUCCESS_HORIZON_RATIO:-0.5}"
export SUCCESS_HORIZON_INIT_TOKENS="${SUCCESS_HORIZON_INIT_TOKENS:-4096}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-opd-success-half-horizon-qwen3-4b-base-teacher-step500-train${TOTAL_TRAINING_STEPS}}"

exec "${SCRIPT_DIR}/meteor-opd-success-horizon-qwen3-4b-base-teacher-step500.sh" "$@"
