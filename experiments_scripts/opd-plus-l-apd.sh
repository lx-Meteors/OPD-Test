#!/usr/bin/env bash

# Standard student-Top-K OPD plus 0.05 * pairwise L-APD on the same student Top-K.
# Tail and complement opponents are disabled because OPD already identifies the
# absolute probability mass. The actor reuses one forward pass for both objectives.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export L_APD_CANDIDATE_SOURCE="${L_APD_CANDIDATE_SOURCE:-student}"
export L_APD_USE_AS_AUXILIARY="${L_APD_USE_AS_AUXILIARY:-True}"
export L_APD_LOSS_COEF="${L_APD_LOSS_COEF:-0.2}"
export L_APD_TAIL_CANDIDATE="${L_APD_TAIL_CANDIDATE:-False}"
export L_APD_COMPLEMENT_CANDIDATE="${L_APD_COMPLEMENT_CANDIDATE:-False}"

exec "${SCRIPT_DIR}/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh" "$@"
