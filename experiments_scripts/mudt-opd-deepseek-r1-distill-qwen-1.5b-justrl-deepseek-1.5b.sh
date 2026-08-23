#!/usr/bin/env bash

# mu-dt OPD: baseline's own force r = -p*(logp - logq~) with the target moved
# to the mu-detempered teacher,  q~ = q^(1/sqrt(mu)) / Z,
# mu = KL(q||u)/KL(p||u) = (logK - H(q))/(logK - H(p)),  u uniform on the
# top-K cells. mu is the teacher/student knowledge-radius ratio; tempering is
# the q-u geodesic, on which KL(.||u) is ~quadratic, so the radius-matching
# exponent is the closed form mu^(-1/2) - no bisection, no gates, no clamps,
# no hyperparameters (only an IEEE _EPS against division by zero). Equals
# baseline OPD exactly wherever H(p) = H(q), i.e. the correction is
# self-extinguishing. Equivalently G-OPD with the reference model replaced by
# the uniform distribution and lambda measured per token: no third model.
# Real-val audit: fog level force -42% two-sided, healthy tokens untouched
# (median mu = 1.000 at every checkpoint), CW promotion the only rising
# class, fork alt-force keep 0.991, equal-budget overshoot 41.4% -> 36.9%.

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-mu_dt}"

run_opd "mudt-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
