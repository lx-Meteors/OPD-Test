#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# G-OPD with asymmetric supervision horizons:
#   - Student-Teacher OPD alignment: first 8192 response tokens
#   - Teacher-Reference reward extrapolation: first 4096 response tokens
# The rollout and validation response lengths remain unchanged.
export OPD_MAX_TOKENS="${OPD_MAX_TOKENS:-8192}"
export GOPD_EXTRAPOLATION_MAX_TOKENS="${GOPD_EXTRAPOLATION_MAX_TOKENS:-4096}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-gopd-opd8k-extrap4k-qwen3-4b-base-teacher-nonthinking-rl-math-step500-train52-lambda-1.25}"

exec "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
