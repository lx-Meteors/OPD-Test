#!/usr/bin/env bash

# L-APD (anchored pairwise distillation) for DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B.
#
# Rollout, teacher, optimizer, token budget and batch match the OPD baseline
# script; only the distillation loss differs, so the two runs are directly
# comparable.

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

# Competitor tokens. The OPD baseline scores the student top-k, so L-APD ranks
# against the same set to keep the two runs comparable.
export L_APD_CANDIDATE_SOURCE="${L_APD_CANDIDATE_SOURCE:-student}"
# Off to match the baseline, which renormalizes strictly inside the top-k and
# carries no aggregate tail term.
export L_APD_TAIL_CANDIDATE="${L_APD_TAIL_CANDIDATE:-False}"
# Aggregated "everything except the anchor" candidate, i.e. the anchor term
# KL_B(q(y_t) || p(y_t)) weighted by q(y_t). Required with the tail candidate off,
# and it makes the weights normalize over the same 16 ids the baseline uses.
export L_APD_COMPLEMENT_CANDIDATE="${L_APD_COMPLEMENT_CANDIDATE:-True}"
export L_APD_NORMALIZE_WEIGHTS="${L_APD_NORMALIZE_WEIGHTS:-True}"
# Per-pair discrepancy. log_ratio keeps only the v = y_t outcome of the reverse KL,
# so the loss is sum_z w(z) log[r_S/r_T] + w(y_t) log[p(y_t)/q(y_t)]. It is not a
# divergence: the teacher side is an additive stop-gradient constant, so the margin
# gradient 1 - r_S never vanishes and never sees the teacher, and the objective has no
# lower bound. Expect actor/pg_loss to fall without bound and
# actor/l_apd_student_anchor_prob to collapse; actor/l_apd_pair_kl still reports the
# honest KL. reverse_kl (bounded, stationary at m = m_T) and forward_kl are the
# alternatives.
export L_APD_PAIR_DIVERGENCE="${L_APD_PAIR_DIVERGENCE:-log_ratio}"

run_opd "l-apd-r1-1p5b-justrl-1p5b-src_${L_APD_CANDIDATE_SOURCE}-tail_${L_APD_TAIL_CANDIDATE}-cmpl_${L_APD_COMPLEMENT_CANDIDATE}-div_${L_APD_PAIR_DIVERGENCE}" \
    "actor_rollout_ref.actor.l_apd.enable=True" \
    "actor_rollout_ref.actor.l_apd.candidate_source=${L_APD_CANDIDATE_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.tail_candidate=${L_APD_TAIL_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.complement_candidate=${L_APD_COMPLEMENT_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.normalize_weights=${L_APD_NORMALIZE_WEIGHTS}" \
    "actor_rollout_ref.actor.l_apd.pair_divergence=${L_APD_PAIR_DIVERGENCE}" \
    "$@"
