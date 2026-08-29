#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Matched-prefix G-OPD control:
#   - Student-Teacher OPD alignment: first 4096 response tokens
#   - Teacher-Reference reward extrapolation: first 4096 response tokens
# The rollout and validation response lengths remain unchanged.
export OPD_MAX_TOKENS="${OPD_MAX_TOKENS:-4096}"
export GOPD_EXTRAPOLATION_MAX_TOKENS="${GOPD_EXTRAPOLATION_MAX_TOKENS:-4096}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-gopd-opd4k-extrap4k-qwen3-4b-nonthinking-rl-math-step500-lambda-1.25}"

exec "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
