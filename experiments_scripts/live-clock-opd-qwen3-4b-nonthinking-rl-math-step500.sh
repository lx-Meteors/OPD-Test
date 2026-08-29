#!/usr/bin/env bash

# Live-clock control arm:
#
#   adv_t = lambda * (log T - log S),   lambda = 1.25 (inherited from G-OPD)
#
# G-OPD's advantage is a + (lambda-1)*d with d = log T - log R frozen at init;
# since the student is initialized at R, d equals the live debt a at step 0.
# This arm replaces the frozen d with its live version, i.e. adv = 1.25*a:
# identical to G-OPD at initialization, self-annealing afterwards (no rent by
# construction: the net force is -1.25*KL(S||T) <= 0). It isolates the pure
# force-magnitude ("acceleration") component of G-OPD's early gains. If it
# reproduces G-OPD's val@10 sprint, the frozen reference carries no
# information beyond force scale. Reference-free: USE_KL=False, no ref worker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GOPD_ENABLE=True
export GOPD_LIVE_CLOCK_LAMBDA="${GOPD_LIVE_CLOCK_LAMBDA:-1.25}"
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.0}"
export USE_KL=False

export OPD_RUN_NAME="${OPD_RUN_NAME:-live-clock-${GOPD_LIVE_CLOCK_LAMBDA}-opd-qwen3-4b-nonthinking-rl-math-step500}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-live-clock-opd-qwen3-4b}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
