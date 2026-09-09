#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep every baseline G-OPD hyperparameter unchanged.  The only algorithmic
# difference is that Teacher-Reference reward extrapolation is performed in
# the conditional non-EOS distribution; standard Student-Teacher OPD still
# supervises the complete response distribution.
export GOPD_STOP_PRESERVING_EXTRAPOLATION="${GOPD_STOP_PRESERVING_EXTRAPOLATION:-True}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-gopd-stop-preserving-extrapolation-qwen3-4b-nonthinking-rl-math-step500-lambda-1.25}"

exec bash "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
