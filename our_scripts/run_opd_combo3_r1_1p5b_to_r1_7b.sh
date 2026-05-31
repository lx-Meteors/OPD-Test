#!/usr/bin/env bash

set -euo pipefail

OUR_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${OUR_SCRIPTS_DIR}/opd_common.sh"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/DeepSeek-R1-Distill-Qwen-7B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"

run_opd "combo3_r1_1p5b_to_r1_7b_12k" "$@"
