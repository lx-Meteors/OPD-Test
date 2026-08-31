#!/usr/bin/env bash

# SC-centered OPD (reference-free directional replacement for the G-OPD
# extrapolation term):
#
#   adv_t = (log T - log S) + ( g_t - mean_traj(g) ),   g_t = log( SC_T(s_t) / SC_S(s_t) )
#
# where SC = KL(U || pi) = logsumexp(z) - mean(z) - log|V| is the INTUITOR-style
# self-certainty and the mean runs over each trajectory's valid response tokens.
# The centered tilt is a zero-sum within-trajectory redistribution of commitment
# toward states where the teacher's distribution shape is relatively clearer:
# pure direction with zero net speed (per-trajectory total force is identically
# zero, so no length rent and no clock speedup by construction). Offline, the
# centering reproduces and strengthens first4k's depth schedule (front-minus-
# deep force contrast +0.082 vs first4k's +0.027) and its deep negative force
# beyond 8k lands exclusively on runaway rows. Single-rollout self-contained,
# no reference model (USE_KL=False: the ref worker is never instantiated), no
# position window, beta = 1 (no free constants).
#
# Baseline-G-OPD harness is otherwise unchanged: n=1 rollout, sampled-token
# logprobs only (top-k 0), same teacher scoring pass; the only wires are the
# per-position self-certainty scalars of the two models.
#
# Readouts (sc_centered/* suite, raw num/den pairs):
#   c_abs_mean          live force budget; pre-registered underpowered kill line:
#                       sustained < 0.02 by ~step 15.
#   c_num_seg0..3       depth schedule (front positive, deep negative claim).
#   conflict_num/active_den   bounded-conflict rate vs the alignment debt (~30%).
#   c_capped_num / cneg_capped_num   runaway targeting (negative-force coverage).
#   c_term_num/term_den terminal-window tilt (watch for early-truncation side
#                       effects; offline ~-0.13 on correct endings).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GOPD_ENABLE=True
export GOPD_SC_CENTERED=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.0}"
export USE_KL=False

export OPD_RUN_NAME="${OPD_RUN_NAME:-sc-centered-opd-qwen3-4b-nonthinking-rl-math-step500}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-sc-centered-opd-qwen3-4b}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
