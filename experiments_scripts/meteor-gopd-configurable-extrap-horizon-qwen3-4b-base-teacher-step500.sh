#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-trajectory OPD with a configurable Teacher-Ref extrapolation horizon.
# Accepted examples:
#   GOPD_EXTRAPOLATION_TOKENS=0     # no extrapolation: standard full-response OPD
#   GOPD_EXTRAPOLATION_TOKENS=full  # full-response extrapolation
#   GOPD_EXTRAPOLATION_TOKENS=2k    # first 2048 response tokens
#   GOPD_EXTRAPOLATION_TOKENS=4096  # first 4096 response tokens
raw_extrapolation_tokens="${GOPD_EXTRAPOLATION_TOKENS:-0}"

case "${raw_extrapolation_tokens}" in
    0|none|NONE)
        extrapolation_mode="none"
        extrapolation_tokens=0
        extrapolation_tag="none"
        ;;
    full|FULL)
        extrapolation_mode="gopd"
        extrapolation_tokens=0
        extrapolation_tag="full"
        ;;
    *[kK])
        extrapolation_mode="gopd"
        token_k="${raw_extrapolation_tokens%[kK]}"
        if [[ ! "${token_k}" =~ ^[0-9]+$ ]] || (( token_k <= 0 )); then
            echo "Invalid GOPD_EXTRAPOLATION_TOKENS: ${raw_extrapolation_tokens}" >&2
            echo "Use 0, full, a positive integer, or values such as 2k/4k/6k." >&2
            exit 1
        fi
        extrapolation_tokens=$((token_k * 1024))
        extrapolation_tag="${token_k}k"
        ;;
    *)
        extrapolation_mode="gopd"
        if [[ ! "${raw_extrapolation_tokens}" =~ ^[0-9]+$ ]] || (( raw_extrapolation_tokens <= 0 )); then
            echo "Invalid GOPD_EXTRAPOLATION_TOKENS: ${raw_extrapolation_tokens}" >&2
            echo "Use 0, full, a positive integer, or values such as 2k/4k/6k." >&2
            exit 1
        fi
        extrapolation_tokens="${raw_extrapolation_tokens}"
        extrapolation_tag="${extrapolation_tokens}tok"
        ;;
esac

# OPD_MAX_TOKENS=0 means no OPD truncation. Only the extrapolation residual is
# restricted; positions after the horizon remain standard OPD.
export OPD_MAX_TOKENS=0
export GOPD_EXTRAPOLATION_MAX_TOKENS="${extrapolation_tokens}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-gopd-full-opd-extrap-${extrapolation_tag}-qwen3-4b-base-teacher-step500-train${TOTAL_TRAINING_STEPS}}"

if [[ "${extrapolation_mode}" == "none" ]]; then
    exec "${SCRIPT_DIR}/meteor-opd-qwen3-4b-base-teacher-step500.sh" "$@"
fi

exec "${SCRIPT_DIR}/meteor-gopd-qwen3-4b-base-teacher-step500.sh" "$@"
