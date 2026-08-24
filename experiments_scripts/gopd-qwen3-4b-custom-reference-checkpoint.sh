#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill in a merged Hugging Face model directory, a VERL global_step_* directory,
# or its actor directory. Raw VERL checkpoints are merged automatically.
# Example:
# export REFERENCE_CHECKPOINT_PATH="/ossfs/workspace/code/Prune-OPD/checkpoint/<run>/global_step_50"
export REFERENCE_CHECKPOINT_PATH="${REFERENCE_CHECKPOINT_PATH:-}"
export AUTO_MERGE_REFERENCE_CHECKPOINT="${AUTO_MERGE_REFERENCE_CHECKPOINT:-True}"

if [[ -z "${REFERENCE_CHECKPOINT_PATH}" ]]; then
    echo "REFERENCE_CHECKPOINT_PATH is required." >&2
    echo "Fill it in near the top of $(basename "${BASH_SOURCE[0]}") or export it before launching." >&2
    exit 1
fi

export GOPD_ENABLE=True
export USE_KL=True
export GOPD_LAMBDA="${GOPD_LAMBDA:-1.25}"

# Include the reference run/checkpoint in W&B and checkpoint directory names.
reference_path="${REFERENCE_CHECKPOINT_PATH%/}"
reference_step="$(basename "${reference_path}")"
reference_parent="$(basename "$(dirname "${reference_path}")")"
reference_label="${REFERENCE_RUN_TAG:-${reference_parent}-${reference_step}}"
reference_label="${reference_label//[^[:alnum:]._-]/-}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-gopd-qwen3-4b-step500-teacher-custom-ref-${reference_label}-lambda-${GOPD_LAMBDA}}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/opd-baseline-qwen3-4b-base-qwen3-4b-non-thinking.sh" "$@"
