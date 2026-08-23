#!/usr/bin/env bash

# FQ OPD, one formula:  r = z(logq) - z(logp)  on the top-k support.
# Tempering acts on log probs as the affine group (a*logq + b, a = 1/T), and
# the z-score is its complete invariant, so this distills the teacher's
# structure modulo temperature: pure fog gets exactly zero force (any T,
# two-sided, no gates, no bisection), the fixed set is the teacher's whole
# tempering orbit (force self-terminates on structure match at the student's
# own level), and correction/alternative/fork cells receive their full z-gap
# with no p-weighted dilution.

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-fq}"

run_opd "fq-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
