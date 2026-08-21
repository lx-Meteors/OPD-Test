#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step1200}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"

# Equal-compute occupancy replay: 20% of each OPD batch is replaced with
# historical student trajectories. Current student and teacher distributions
# are recomputed on every replayed state.
export OCCUPANCY_REPLAY_ENABLE="${OCCUPANCY_REPLAY_ENABLE:-True}"
export OCCUPANCY_REPLAY_RATIO="${OCCUPANCY_REPLAY_RATIO:-0.2}"
export OCCUPANCY_REPLAY_CAPACITY="${OCCUPANCY_REPLAY_CAPACITY:-1024}"
export OCCUPANCY_REPLAY_WARMUP_STEPS="${OCCUPANCY_REPLAY_WARMUP_STEPS:-2}"
export OCCUPANCY_REPLAY_INSERT_SAMPLES="${OCCUPANCY_REPLAY_INSERT_SAMPLES:-64}"
export OCCUPANCY_REPLAY_SEED="${OCCUPANCY_REPLAY_SEED:-2026}"

run_opd "opd-occupancy-replay-qwen3-4b-qwen3-4b-non-thinking-rl-math-step1200" "$@"
