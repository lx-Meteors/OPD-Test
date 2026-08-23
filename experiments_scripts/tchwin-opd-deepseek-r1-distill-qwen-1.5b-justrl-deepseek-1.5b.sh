#!/usr/bin/env bash

# Windowed-teacher OPD: baseline student_p reward, unchanged. The only change is
# what the teacher conditions on -- [prompt] + the last TEACHER_CTX_WINDOW
# response tokens, position ids restarted from zero, instead of the whole
# student prefix. The student trajectory is untouched: it still generates and
# scores p under the full context, so the run stays on-policy.
#
# Motivation, measured on step-0 rollouts of the baseline run. The teacher's
# entropy on student-written text inflates with the absolute length of the
# off-policy prefix (exp H 3.12 -> 6.71 from 0-1k to 10-12k) and crosses above
# the student's at ~6k, while on its own text it stays flat (1.82 -> 2.04) and
# below the student's at every depth. So the deep-layer degradation is drift off
# the teacher's manifold, not a property of deep positions: past ~10k the
# teacher is a worse predictor than its own pre-RL base, which is the student
# init. Truncating its view to the last 4096 tokens recovers ~90% of the entropy
# gap, lifts top-16 overlap with the student from 55.9% to 69.8% and cuts
# KL(p||q) by 58% at depth >= 8192, against 0.056 nats of measured distribution
# shift on on-policy text (i.e. the information dropped is nearly free).
#
# Windows below ~1024 make the teacher confidently misaligned instead (at 256:
# entropy falls but overlap drops to 59.9% and KL(p||q) does not improve), so
# 4096 is the operating point and shallow positions are never truncated.
# Costs ~2-2.5x teacher forward: the linear layers scale with the extra tokens
# while the quadratic attention term actually shrinks under chunking.
#
# Note this cannot be done with sliding-window attention in a single pass. Over
# 28 layers a banded mask has a receptive field of 28*(W-1), so the prefix still
# reaches the read-out; measured at W=4096 it moves KL(p||q) the wrong way,
# 0.778 -> 1.049. Re-encoding a truncated context is what produces the effect.

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
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
export TEACHER_CTX_WINDOW="${TEACHER_CTX_WINDOW:-4096}"

run_opd "tchwin-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b" "$@"
