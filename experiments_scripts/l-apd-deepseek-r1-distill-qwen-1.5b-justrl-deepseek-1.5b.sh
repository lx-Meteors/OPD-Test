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

# Competitor tokens. Scoring against the student's own top-k puts the anchor
# inside the candidate set 99.4% of the time, so every competitor is a token the
# student already ranks highly: the loss can only reorder the set the student
# already prefers, and never sees a token the teacher wants but the student has
# dropped past rank k. Pairwise agreement stalled at 0.95 from step 140 that way.
# The teacher top-k instead carries the ~1.5% of teacher mass sitting outside the
# student top-k, which is where the teacher still disagrees.
export L_APD_CANDIDATE_SOURCE="${L_APD_CANDIDATE_SOURCE:-teacher}"
# Aggregate opponent. The tail block turns the opponents into a true partition of
# the non-anchor vocabulary: every non-anchor token appears in the loss exactly
# once, either as a named candidate or inside the tail. The complement opponent
# (the old default) contains the named candidates a second time and took
# q(y_t)/Z ~ 0.68 of the weight for re-checking a mostly satisfied constraint;
# it is kept as an ablation only.
export L_APD_TAIL_CANDIDATE="${L_APD_TAIL_CANDIDATE:-True}"
export L_APD_COMPLEMENT_CANDIDATE="${L_APD_COMPLEMENT_CANDIDATE:-False}"
export L_APD_NORMALIZE_WEIGHTS="${L_APD_NORMALIZE_WEIGHTS:-True}"
# Whose probabilities set the per-pair mixture weights. student (default) weights every
# duel by the student's own conditional mass sg[p(o) / (1 - p(y_t))]: closed-loop, so a
# bloated tail or a wrongly favoured alternative automatically raises the weight of its
# own column until drained, and direction-consistent with the reverse per-pair KL (the
# chain rule of KL(p || q) weights conditional cells by student mass). teacher is the
# historical open-loop weighting q(o) / (1 - q(y_t)), kept as an ablation: it spends
# budget by teacher preference regardless of where the student's error is, which let
# transient mass pile up in the 4-11%-weight tail column (3x teacher tail mass at peak,
# entropy overshoot to 1.0, and the worst@16 val gap that followed).
export L_APD_WEIGHT_SOURCE="${L_APD_WEIGHT_SOURCE:-student}"
# Per-pair discrepancy. reverse_kl sums both outcomes of each pair,
# sum_v r_S(v) log[r_S(v) / r_T(v)], so it is a genuine divergence: bounded below by 0
# and stationary exactly at m = m_T. forward_kl swaps the arguments.
#
# log_ratio keeps only the v = y_t outcome and is kept for ablation only. It is not a
# divergence -- the teacher side becomes an additive stop-gradient constant, leaving a
# margin gradient of 1 - r_S that is always positive and never sees the teacher. It was
# measured at cosine -0.985 against the reverse_kl gradient, and a run of it collapsed
# actor/entropy from 0.66 to 0.04 while actor/l_apd_anchor_kl grew tenfold.
export L_APD_PAIR_DIVERGENCE="${L_APD_PAIR_DIVERGENCE:-reverse_kl}"

run_opd "l-apd-r1-1p5b-justrl-1p5b-src_${L_APD_CANDIDATE_SOURCE}-w_${L_APD_WEIGHT_SOURCE}-tail_${L_APD_TAIL_CANDIDATE}-cmpl_${L_APD_COMPLEMENT_CANDIDATE}-div_${L_APD_PAIR_DIVERGENCE}" \
    "actor_rollout_ref.actor.l_apd.enable=True" \
    "actor_rollout_ref.actor.l_apd.candidate_source=${L_APD_CANDIDATE_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.tail_candidate=${L_APD_TAIL_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.complement_candidate=${L_APD_COMPLEMENT_CANDIDATE}" \
    "actor_rollout_ref.actor.l_apd.normalize_weights=${L_APD_NORMALIZE_WEIGHTS}" \
    "actor_rollout_ref.actor.l_apd.weight_source=${L_APD_WEIGHT_SOURCE}" \
    "actor_rollout_ref.actor.l_apd.pair_divergence=${L_APD_PAIR_DIVERGENCE}" \
    "$@"
