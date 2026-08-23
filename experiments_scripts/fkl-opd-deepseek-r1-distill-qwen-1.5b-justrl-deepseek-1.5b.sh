#!/usr/bin/env bash

# FKL OPD: r = q - p on the student's top-16 support - the exact logit
# gradient of -KL(q||p), the forward-KL endpoint of the KL family. One
# formula, no thresholds, no gates, no target transforms, no bisection.
#
# The lesion triage is embedded in the mass difference itself: confident-
# wrong correction cells (q~0.6, p~0.03) receive the full missing mass
# (+0.60 on real val CW tokens vs +0.11 under the baseline RKL force);
# teacher alternatives at false-softness positions get the mass difference
# (+0.15 vs +0.05); healthy tokens (p~q) get ~0. Endpoint optimality is
# verified on 274k real val tokens: along the geometric bridge
# r^lam = (p^(1-lam) q^lam - p)/lam between the baseline force (lam->0) and
# this one (lam=1), both lesion forces are monotone increasing in lam.
# Zero-sum per position, |r| <= 1 (junk evals cannot amplify), and KL(q||p)
# is convex in the student logits: globally convergent per-position flow
# that cannot sharpen the student beyond the teacher (equal-budget CW
# overshoot 21.7% vs baseline 41.4%, repair 43.4% vs 32.8%, zero new CWs).

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-fkl}"

run_opd "fkl-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
