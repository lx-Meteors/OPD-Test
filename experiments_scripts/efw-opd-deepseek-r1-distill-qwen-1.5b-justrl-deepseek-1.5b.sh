#!/usr/bin/env bash

# EFW x OPD baseline channel (三条件蒸馏, primary comparison arm) for
# DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B.
#
# Single-variable comparison against the OPD baseline script: identical
# rollout, teacher scoring, only_stu top-16 residual (3D rm_scores) and PPO
# machinery; the only difference is that token_level_scores are scaled
# per state by the frozen edit field
#
#     w(s) = KL(b || q)(s),
#
# estimated on the same student top-16 ids (b = base = the actor's initial
# weights, served by the ref worker; q = teacher). Teaching mass per state =
# (reachable: on-policy sampling) x (RL edited here: sg(w)) x (not learned
# yet: the baseline reverse-KL residual). At init the student equals b, so the
# effective signal starts at w x residual = w^2. Because this arm shares the
# advantage channel with baseline OPD / Prune-OPD / G-OPD, wins here attribute
# cleanly to the field itself.
#
# The only new compute is one ref designated-ids forward per step (it also
# yields ref_log_prob for free).
#
# Watch: efw/field_* (the field profile), efw/field_frac_low (un-anchored
# surface), efw/b_mass_coverage (candidate-set health; if it sags well below
# ~0.9 the student top-16 no longer covers b), efw/low_field_kl_p_b (drift
# where the loss no longer corrects; the reserved EFW_FLOOR knob exists for
# this, default off), critic/rewards vs the baseline (overall scale shrinks by
# ~E[w]; check actor/grad_norm before touching lr).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-/input0/models}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"

# Baseline-aligned settings, spelled out for the record (common.sh defaults).
# The method needs no group statistics, so N_RESPONSES=1 is fully supported;
# the default stays at the baseline's value for the controlled A/B.
export N_RESPONSES="${N_RESPONSES:-4}"
export TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"

# The one reserved EFW knob (default off): epsilon floor of the field, guarding
# low-field un-anchored drift. Watch efw/low_field_kl_p_b before touching it.
EFW_FLOOR="${EFW_FLOOR:-0.0}"

run_opd "efw-opd-r1-1p5b-justrl-1p5b" \
    "+actor_rollout_ref.rollout.efw.enable=True" \
    "+actor_rollout_ref.rollout.efw.floor=${EFW_FLOOR}" \
    "$@"
