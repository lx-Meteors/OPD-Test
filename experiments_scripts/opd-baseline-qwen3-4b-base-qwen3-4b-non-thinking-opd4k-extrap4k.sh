#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Success-conditioned G-OPD horizons:
#   - OPD horizon: current batch's successful-response mean length
#   - Extrapolation horizon: half of that successful-response mean length
# Training rollout and validation both retain their normal configured lengths.
export OPD_MAX_TOKENS="${OPD_MAX_TOKENS:-8192}"
export GOPD_EXTRAPOLATION_MAX_TOKENS="${GOPD_EXTRAPOLATION_MAX_TOKENS:-4096}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-gopd-success-mean-opd-half-extrap-qwen3-4b-base-teacher-nonthinking-rl-math-step500-train52-lambda-1.25}"

# OPD_MAX_TOKENS is only the fallback loss horizon before a successful sample exists.
exec "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.enable=True" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.init_len=${OPD_MAX_TOKENS}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.min_len=1" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.max_len=${MAX_RESP_LENGTH:-16384}" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.reward_threshold=0.5" \
    "++actor_rollout_ref.rollout.gopd_success_length_horizon.extrapolation_ratio=0.5" \
    "$@"
