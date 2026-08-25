#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Windowed-teacher arm of the standard-OPD control. Everything -- models, data,
# sampling, optimizer, sequence lengths, batch sizes, validation, training steps -- is
# inherited from opd-standard-qwen3-4b-nonthinking-rl-math-step500.sh. The only change
# is that the teacher conditions on a bounded slice of the student prefix instead of
# all of it.
#
# Why: an off-policy prefix inflates the teacher's entropy monotonically with its
# length, so deep into a long response the teacher is scoring text far off the manifold
# it was trained on and its penalty stops being informative. Truncating its view keeps
# it near that manifold. The student trajectory is untouched; it still generates and is
# scored under the full context.
#
# Two knobs, and the second one is the one that bites:
#
#   W = TEACHER_CTX_WINDOW   minimum prefix the teacher is guaranteed to see
#   S = TEACHER_CTX_SEGMENT  read-out positions sharing one re-encode
#
# Truncation cannot start before the first chunk boundary past W, so the depth at which
# the intervention first does anything is
#
#   onset = W + S = 2048 + 4096 = 6144
#
# Every token shallower than that sees the untruncated prefix and is bit-identical to
# the baseline. This is the whole reason W=2048 rather than 4096: at MAX_RESP_LENGTH
# 16384 both settings re-encode exactly 30720 tokens per sequence, but W=4096 pushes
# the onset out to 8192 and leaves the shallow two thirds of a typical response
# untouched. W=2048 buys a lower onset for free. Going lower is possible (W=1024 puts
# the onset at 5120 and is cheaper still) but offline the teacher becomes confidently
# misaligned below roughly 1024 tokens of context, so 2048 is the safe end of the range.
#
# Watch teacher_ctx/frac_tokens_truncated in W&B. If it sits near zero the intervention
# is configured but inert and this run is just a slower baseline.
export TEACHER_CTX_WINDOW="${TEACHER_CTX_WINDOW:-2048}"
export TEACHER_CTX_SEGMENT="${TEACHER_CTX_SEGMENT:-4096}"

export OPD_RUN_NAME="${OPD_RUN_NAME:-tchwin-opd-standard-qwen3-4b-nonthinking-rl-math-step500}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-standard-qwen3-4b-nonthinking-rl-math-step500.sh" "$@"
