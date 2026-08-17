#!/usr/bin/env bash

# RKL-dt OPD: the baseline per-cell reward -p*(logp-logq) is unchanged; the
# single edit is the target. Wherever the teacher's cell distribution is
# softer than the student's (off-manifold epistemic noise on student-sampled
# contexts), the teacher is re-tempered along its own temperature family
# (q^lam, rank-preserving) so its entropy matches the student's; where the
# teacher is sharper (the RL "commit" edits), it is left untouched.

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-rkl_dt}"

run_opd "rkldt-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
