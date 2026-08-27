#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Asymmetric-reward OPD ("supplement first, dismantle later") on the sampled-token
# channel, strictly comparable with both the standard-OPD control and G-OPD: same
# models (Qwen3-4B student, Step500 teacher), data (DeepMath-103K level6), sampling,
# optimizer, sequence lengths, batch sizes (n=1, mbs=1024, 16384), validation, and
# training steps. The only change is the per-token reward shape:
#
#     r = log q - log p        (sampled token)
#     reward = r               if r >= 0   (supplement side: linear, as in RKL)
#     reward = expm1(k*r)/k    if r <  0   (dismantle side: FKL curve, floor -1/k)
#
# k = OPD_NEG_KAPPA = 1.0 is the canonical setting: the negative side is exactly
# e^r - 1, which together with the linear positive side forms a valid f-divergence.
# Punishments are bounded below by -1, small cleanups keep slope ~1, and the curve
# anneals back to plain RKL as p -> q (full dismantling force returns late).
#
# Expectations written down before launch: score/mean should recover toward zero
# faster than the control; s10 val may LAG the control (dismantling is rate-limited
# early, by design); judge at s20-s40 against the G-OPD band 53.8-54.4 (mean@32).
# Watch actor/entropy and response_length: if entropy dives or length collapses
# early, rerun with OPD_NEG_KAPPA=0.5 (floor -2, retains more early dismantling).
export GOPD_ENABLE=False
export GOPD_LAMBDA=1.0
export USE_KL=False
export KL_COEF=0.0
export ADV_ESTIMATOR=token_reward_direct
export SAVE_FREQ=10
export OPD_NEG_KAPPA="${OPD_NEG_KAPPA:-1.0}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-negfkl-opd-standard-qwen3-4b-nonthinking-rl-math-step500}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
