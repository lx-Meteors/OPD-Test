#!/usr/bin/env bash

set -euo pipefail

OUR_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${OUR_SCRIPTS_DIR}/opd_common.sh"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_ENABLE="${REWARD_MODEL_ENABLE:-False}"
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-0}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"

run_opd "grpo_r1_1p5b_8k" "$@"
