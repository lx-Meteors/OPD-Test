#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CFG-guided teacher arm. Identical to the standard-OPD control
# (opd-standard-qwen3-4b-nonthinking-rl-math-step500.sh) in models, data, sampling,
# optimizer, sequence lengths, batch sizes, validation, and training steps; only the
# objective differs. The teacher additionally scores every response token with the
# prompt deleted, and that prompt-free pass stands in for the G-OPD reference branch,
# giving the sampled-token advantage
#
#     (1 + g) * log q_full - g * log q_free - log p        with g = 0.3.
#
# gamma = 0.3 comes from the local sweep of the pure (unclamped) formula: flips of the
# teacher's top-1 token stay at 1.8-3.3% and land on high-entropy decision points,
# while the tilted target still sharpens; at 0.5+ the sharpening flips negative on
# harder problems. Two models only -- no reference model is loaded.
export GOPD_ENABLE=False
export USE_KL=False
export KL_COEF=0.0
export ADV_ESTIMATOR=token_reward_direct
export SAVE_FREQ=10
export TEACHER_CFG_GAMMA="${TEACHER_CFG_GAMMA:-0.3}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-cfgopd-standard-qwen3-4b-nonthinking-rl-math-step500}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
