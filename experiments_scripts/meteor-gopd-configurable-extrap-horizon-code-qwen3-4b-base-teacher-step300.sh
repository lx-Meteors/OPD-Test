#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
export GOPD_DATA_ROOT="${GOPD_DATA_ROOT:-${DATA_ROOT}/G-OPD-Training-Data}"

# Code-domain Teacher. A local model is preferred; otherwise Transformers will
# resolve the supplied Hugging Face repository name.
if [[ -z "${REWARD_MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Code-Step300" ]]; then
        export REWARD_MODEL_PATH="${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Code-Step300"
    else
        export REWARD_MODEL_PATH="Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300"
    fi
fi

# Official G-OPD code-domain data. Downloads are deliberately left to the user
# because the training parquet is about 1.65 GB.
export AUTO_DOWNLOAD_GOPD_DATA=0
export TRAIN_DATASET="${TRAIN_DATASET:-${GOPD_DATA_ROOT}/Eurus/code_train.parquet}"
export TEST_DATASET="${TEST_DATASET:-[\"${GOPD_DATA_ROOT}/Eurus/code_validation.parquet\"]}"
export TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-Eurus-Code}"
export REQUIRE_DEFAULT_MATH_TEST_DATASETS=False
export CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_code/__init__.py}"
export CUSTOM_REWARD_FUNCTION_NAME="${CUSTOM_REWARD_FUNCTION_NAME:-reward_func}"

# Online code validation is pass@1. HumanEval+, MBPP+, and LiveCodeBench should
# be run independently on saved checkpoints with their official evaluators.
export VAL_N_RESPONSES="${VAL_N_RESPONSES:-1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-52}"

extrapolation_setting="${GOPD_EXTRAPOLATION_TOKENS:-0}"
export OPD_RUN_NAME="${OPD_RUN_NAME:-meteor-gopd-code-full-opd-extrap-${extrapolation_setting}-qwen3-4b-base-teacher-code-step300-train${TOTAL_TRAINING_STEPS}}"

exec "${SCRIPT_DIR}/meteor-gopd-configurable-extrap-horizon-qwen3-4b-base-teacher-step500.sh" "$@"
