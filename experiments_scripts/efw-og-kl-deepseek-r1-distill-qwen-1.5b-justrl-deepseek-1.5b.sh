#!/usr/bin/env bash

# EFW x og-kl (三条件蒸馏 on the order-gated KL flow) for
# DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B.
#
# Single-variable comparison against the og-kl script: identical L-APD
# order_gated_kl objective on the pure student top-16 cells, with the per-token
# loss additionally scaled by the frozen edit field
#
#     w(s) = KL(b || q)(s),
#
# estimated on the same student top-16 ids (b = base = the actor's initial
# weights, served by the ref worker; q = teacher). Teaching mass per state =
# (reachable: on-policy sampling) x (RL edited here: sg(w)) x (not learned yet:
# the og-kl residual). At init the student equals b, so the effective loss
# starts at w x residual = w^2: teaching opens quadratically concentrated on
# the states RL actually edited, and re-spreads as high-field states converge.
#
# The only new compute is one ref designated-ids forward per step (it also
# yields ref_log_prob for free).
#
# Watch (on top of the og-kl dashboard): efw/field_* (the field profile),
# efw/field_frac_low (un-anchored surface), efw/b_mass_coverage (candidate-set
# health; if it sags well below ~0.9 the student top-16 no longer covers b),
# efw/low_field_kl_p_b (drift where the loss no longer corrects; the reserved
# EFW_FLOOR knob exists for this, default off), actor/l_apd_efw_field and
# actor/l_apd_efw_token_loss (the weighted loss actually optimized).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-/input0/models}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export TEST_DATASET="${TEST_DATASET:-[\"${DATA_ROOT}/test_data/AMC23/test.parquet\",\"${DATA_ROOT}/test_data/AIME24/test.parquet\",\"${DATA_ROOT}/test_data/AIME25/test.parquet\",\"${DATA_ROOT}/test_data/HMMT24/test.parquet\",\"${DATA_ROOT}/test_data/HMMT25/test.parquet\"]}"

# og-kl settings, verbatim (see og-kl-*.sh for the rationale).
export L_APD_CANDIDATE_SOURCE="${L_APD_CANDIDATE_SOURCE:-student}"
export L_APD_TAIL_CANDIDATE="${L_APD_TAIL_CANDIDATE:-False}"
export L_APD_COMPLEMENT_CANDIDATE="${L_APD_COMPLEMENT_CANDIDATE:-False}"
export L_APD_NORMALIZE_WEIGHTS="${L_APD_NORMALIZE_WEIGHTS:-True}"
export L_APD_WEIGHT_SOURCE="${L_APD_WEIGHT_SOURCE:-student}"
export L_APD_PAIR_DIVERGENCE="${L_APD_PAIR_DIVERGENCE:-order_gated_kl}"

# The one reserved EFW knob (default off): epsilon floor of the field, guarding
# low-field un-anchored drift. Watch efw/low_field_kl_p_b before touching it.
EFW_FLOOR="${EFW_FLOOR:-0.0}"

run_opd "efw-og-kl-r1-1p5b-justrl-1p5b-src_${L_APD_CANDIDATE_SOURCE}-div_${L_APD_PAIR_DIVERGENCE}" \
    "actor_rollout_ref.actor.l_apd.enable=True" \
    "actor_rollout_ref.actor.l_apd.candidate_source=${L_APD_CANDIDATE_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.tail_candidate=${L_APD_TAIL_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.complement_candidate=${L_APD_COMPLEMENT_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.normalize_weights=${L_APD_NORMALIZE_WEIGHTS}" \
    "actor_rollout_ref.actor.l_apd.weight_source=${L_APD_WEIGHT_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.pair_divergence=${L_APD_PAIR_DIVERGENCE}" \
    "+actor_rollout_ref.rollout.efw.enable=True" \
    "+actor_rollout_ref.rollout.efw.floor=${EFW_FLOOR}" \
    "$@"
