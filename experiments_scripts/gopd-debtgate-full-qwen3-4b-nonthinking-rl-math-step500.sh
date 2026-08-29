#!/usr/bin/env bash

# Debt-gated G-OPD (quadrant surgery), FULL-trajectory extrapolation - no 4k cutoff.
#
#   adv = (logT - logS) + (lambda - 1) * [ min(d, 0) + max(d, 0) * 1[logS < logT] ],
#   d = logT - logR
#
# The demolition side of the extrapolation residual (d < 0: tokens RL pushed
# down, e.g. premature EOS at d ~ -7) applies unconditionally at every response
# position, so the anti-early-stop channel keeps working in the deep completion
# zone that the first-4k cutoff abandons. The supplement side (d > 0) pays only
# while the student still owes mass on the sampled token (logS < logT) and is
# revoked per token once learned: no per-token length rent, no loop subsidies,
# no stop-delay bias - the one toxic quadrant (supplement fighting the
# alignment term's demolition) is structurally removed.
#
# At initialization S = R, so the gate passes everything and the first steps
# coincide exactly with full G-OPD; the gate then anneals the objective toward
# the demolition-only field as tokens are learned - the preset first-4k
# position boundary is replaced by an intrinsic learning-progress boundary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GOPD_ENABLE=True
export USE_KL=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.25}"
export GOPD_EXTRAPOLATION_MAX_TOKENS=0
export GOPD_DEBT_GATE=True
export OPD_RUN_NAME="gopd-debtgate-full-qwen3-4b-nonthinking-rl-math-step500-lambda-${GOPD_LAMBDA}"
export WANDB_RUN_GROUP="gopd-debtgate-full-lambda-${GOPD_LAMBDA}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
