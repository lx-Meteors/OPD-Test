#!/usr/bin/env bash

# SC-ratio OPD (reference-free replacement for the G-OPD extrapolation term):
#
#   adv_t = (log T - log S) * (1 + w_t),   w_t = clamp(1 - SC_S(s_t)/SC_T(s_t), 0, 1)
#
# where SC = KL(U || pi) = logsumexp(z) - mean(z) - log|V| is the INTUITOR-style
# self-certainty of the full next-token distribution. The bonus stays parallel
# to the live alignment debt (quadrant-3 buyout impossible by construction), is
# capped at doubling it, and retires per state as SC_S -> SC_T. No reference
# model (USE_KL=False: the ref worker is never instantiated), no position
# window, no hand-picked constants.
#
# Baseline-G-OPD harness is otherwise unchanged: n=1 rollout, sampled-token
# logprobs only (top-k 0), same teacher scoring pass. The only new wire is one
# scalar per token (teacher_self_certainty) computed from logits the teacher
# worker already materializes.
#
# Readouts: actor/sc_weight_mean and actor/sc_weight_seg0..3 (raw, unscaled)
# should start around the shallow>deep profile and anneal toward 0;
# actor/sc_weight_zero_frac tracks the clamp (student sharper than teacher).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GOPD_ENABLE=True
export GOPD_SC_RATIO=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.0}"
export USE_KL=False

export OPD_RUN_NAME="${OPD_RUN_NAME:-sc-ratio-opd-qwen3-4b-nonthinking-rl-math-step500}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-sc-ratio-opd-qwen3-4b}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
