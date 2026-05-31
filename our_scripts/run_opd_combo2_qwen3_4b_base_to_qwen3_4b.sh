#!/usr/bin/env bash

set -euo pipefail

OUR_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${OUR_SCRIPTS_DIR}/opd_common.sh"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/Qwen3-4B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/Qwen3-4B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"

run_opd "combo2_qwen3_4b_base_to_qwen3_4b_8k" "$@"
