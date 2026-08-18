#!/usr/bin/env bash

# RKL-sdtw OPD: the teacher q is never touched and no cell is gated or
# clamped. Per position, mu solves H(p^mu) = H(q) (two-sided) and the plain
# reverse-KL field is reweighted by p~ = p^mu:
#
#     r_c = -p~_c (log p_c - log q_c)      ( = (p~/p) * baseline reward )
#
# The direction keeps the raw log-ratio, so the entropy brake stays in the
# signal (the sdt_gain run erased it and ratcheted to H=0.004); p~ enters
# only as a weight <= 1, so the field is bounded by the data range without
# any floor (the 1/mu-gained field grew past |r| ~ 250). At H(p) = H(q) the
# reward equals the baseline's exactly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-rkl_sdtw}"

run_opd "rklsdtw-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
