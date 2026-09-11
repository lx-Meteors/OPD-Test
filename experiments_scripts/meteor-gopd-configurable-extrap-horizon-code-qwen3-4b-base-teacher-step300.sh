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

export RUN_OFFICIAL_CODE_EVAL_AFTER_TRAIN="${RUN_OFFICIAL_CODE_EVAL_AFTER_TRAIN:-True}"
export CODE_EVAL_ROOT="${CODE_EVAL_ROOT:-${REPO_ROOT}/.cache/G-OPD/code_eval}"

if [[ "${DRY_RUN:-0}" == "1" || "${RUN_OFFICIAL_CODE_EVAL_AFTER_TRAIN}" != "True" ]]; then
    exec "${SCRIPT_DIR}/meteor-gopd-configurable-extrap-horizon-qwen3-4b-base-teacher-step500.sh" "$@"
fi

if [[ ! -f "${CODE_EVAL_ROOT}/scripts/run_evalplus.sh" ]]; then
    echo "Official G-OPD code evaluator is missing: ${CODE_EVAL_ROOT}" >&2
    echo "Clone RUCBM/G-OPD and set CODE_EVAL_ROOT before starting training." >&2
    exit 1
fi
if [[ ! -f "${CODE_EVAL_ROOT}/data/HumanEvalPlus.jsonl" || ! -f "${CODE_EVAL_ROOT}/data/MbppPlus.jsonl" ]]; then
    echo "HumanEval+ or MBPP+ data is missing from ${CODE_EVAL_ROOT}/data." >&2
    exit 1
fi
if [[ ! -d "${CODE_EVAL_ROOT}/coding/LiveCodeBench/code_generation_lite" ]]; then
    echo "LiveCodeBench v6 data is missing under ${CODE_EVAL_ROOT}/coding/LiveCodeBench." >&2
    exit 1
fi

"${SCRIPT_DIR}/meteor-gopd-configurable-extrap-horizon-qwen3-4b-base-teacher-step500.sh" "$@"

ckpt_root="${CKPT_ROOT:-${REPO_ROOT}/checkpoint}"
latest_run_dir=""
latest_run_mtime=0
while IFS= read -r -d '' candidate; do
    candidate_mtime="$(stat -c %Y "${candidate}" 2>/dev/null || stat -f %m "${candidate}")"
    if (( candidate_mtime > latest_run_mtime )); then
        latest_run_mtime="${candidate_mtime}"
        latest_run_dir="${candidate}"
    fi
done < <(find "${ckpt_root}" -mindepth 1 -maxdepth 1 -type d -name "${OPD_RUN_NAME}_*" -print0)

if [[ -z "${latest_run_dir}" ]]; then
    echo "Could not find the completed training directory for ${OPD_RUN_NAME}." >&2
    exit 1
fi

latest_step_dir=""
latest_step=-1
while IFS= read -r -d '' candidate; do
    step_name="$(basename "${candidate}")"
    step_number="${step_name#global_step_}"
    if [[ "${step_number}" =~ ^[0-9]+$ ]] && (( step_number > latest_step )); then
        latest_step="${step_number}"
        latest_step_dir="${candidate}"
    fi
done < <(find "${latest_run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0)

if [[ -z "${latest_step_dir}" ]]; then
    echo "No global_step_* checkpoint found under ${latest_run_dir}." >&2
    exit 1
fi

export VERL_ACTOR_CHECKPOINT="${latest_step_dir}/actor"
export CODE_EVAL_RUN_NAME="$(basename "${latest_run_dir}")-step${latest_step}"
exec "${SCRIPT_DIR}/meteor-eval-code-checkpoint.sh"
