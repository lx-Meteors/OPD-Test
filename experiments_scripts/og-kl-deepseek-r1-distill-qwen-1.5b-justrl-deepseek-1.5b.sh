#!/usr/bin/env bash

# Order-gated bidirectional KL distillation (order_gated_kl) for
# DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B.
#
# Single-variable comparison against the OPD baseline: the cells are the
# baseline's pure student top-16 (the sampled token is NOT deduplicated out, no
# tail cell), the per-cell scoring is the baseline's own k1 form -- a frozen
# coefficient sg[w_c (log p_c - log q_c)] carried by the differentiable
# log p_c, i.e. the PPO-wings force at ratio ~= 1 -- and the only new object is
# the weight vector
#
#     w = (1 - lambda) p~ + lambda q~
#
# blending the baseline's student_p (reverse KL) and teacher_p (forward KL)
# weight modes per token. lambda is the top-1 anchored order gap
# max_j |sigma(m_j) - sigma(m_j^T)|, detached: when the teacher agrees with the
# student's mode and its confidence profile, lambda -> 0 and the loss is the
# baseline reverse-KL scoring cell by cell; when the teacher disputes the mode,
# teacher-mass weights keep pulling cells the student has starved (the reverse
# weight p~ ~ 0 dead zone). Both directions share the fixed point p = q on raw
# cell masses, so the gate reroutes the path, never the destination.
#
# Rollout, teacher, optimizer, token budget and batch match the OPD baseline
# script.
#
# Watch: actor/l_apd_order_lambda (should anneal), rev/fwd_kl_est (rev is the
# baseline reward 1:1), student/teacher_covered_prob (P -> Q), mode_agreement
# (should rise with lambda annealing), early grad_norm (k1 scale sanity).

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

# Cells must be the student's own top-k so the set is the baseline's verbatim.
export L_APD_CANDIDATE_SOURCE="${L_APD_CANDIDATE_SOURCE:-student}"
# order_gated_kl operates on the pure top-k cells: no aggregate opponents.
export L_APD_TAIL_CANDIDATE="${L_APD_TAIL_CANDIDATE:-False}"
export L_APD_COMPLEMENT_CANDIDATE="${L_APD_COMPLEMENT_CANDIDATE:-False}"
# Ignored by order_gated_kl (the lambda-blend defines the weights); kept for config completeness.
export L_APD_NORMALIZE_WEIGHTS="${L_APD_NORMALIZE_WEIGHTS:-True}"
export L_APD_WEIGHT_SOURCE="${L_APD_WEIGHT_SOURCE:-student}"
export L_APD_PAIR_DIVERGENCE="${L_APD_PAIR_DIVERGENCE:-order_gated_kl}"

run_opd "og-kl-r1-1p5b-justrl-1p5b-src_${L_APD_CANDIDATE_SOURCE}-div_${L_APD_PAIR_DIVERGENCE}" \
    "actor_rollout_ref.actor.l_apd.enable=True" \
    "actor_rollout_ref.actor.l_apd.candidate_source=${L_APD_CANDIDATE_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.tail_candidate=${L_APD_TAIL_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.complement_candidate=${L_APD_COMPLEMENT_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.normalize_weights=${L_APD_NORMALIZE_WEIGHTS}" \
    "actor_rollout_ref.actor.l_apd.weight_source=${L_APD_WEIGHT_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.pair_divergence=${L_APD_PAIR_DIVERGENCE}" \
    "$@"
