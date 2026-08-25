#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fair standard-OPD control for the G-OPD reproduction. The shared script supplies identical models, data,
# sampling, optimizer, sequence lengths, batch sizes, validation, and training steps. Only the objective differs.
export GOPD_ENABLE=False
export GOPD_LAMBDA=1.0
export USE_KL=False
export KL_COEF=0.0
export ADV_ESTIMATOR=token_reward_direct
export SAVE_FREQ=10
export OPD_RUN_NAME="${OPD_RUN_NAME:-opd-standard-qwen3-4b-nonthinking-rl-math-step500-paper-config}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
