#!/usr/bin/env bash

# RKL-cdt OPD: rkl_dt's force form, r = -p*(logp - logq~), with the entropy
# bisection replaced by a closed form. In polar coordinates logp = sigma*z
# (level x structure), the detempered teacher is the teacher's structure at
# the student's level:  logq~ = log_softmax(sigma_p * z(logq)).  On the
# tempering fiber, entropy is a transcendental coordinate (hence rkl_dt's
# root-finding) while the log-prob spread sigma is the linear one - the
# matching equation solves itself. Exact zero force on pure temperature fog
# in BOTH directions (rkl_dt's one-sided H(q)>H(p) gate is gone), ties and
# partial forks survive re-leveling (no entropy squeeze), the fixed set is
# the teacher's whole tempering orbit (force self-terminates on structure
# match, no softness chase either way), and confident-wrong positions keep
# the dt funnel: everything except the teacher's choice gets negative force
# and the iterated field hands the position over within a few steps.

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-rkl_cdt}"

run_opd "rklcdt-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
