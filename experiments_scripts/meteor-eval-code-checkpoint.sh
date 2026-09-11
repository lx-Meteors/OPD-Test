#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

: "${VERL_ACTOR_CHECKPOINT:?Set VERL_ACTOR_CHECKPOINT to a saved global_step_*/actor directory}"

CODE_EVAL_ROOT="${CODE_EVAL_ROOT:-${REPO_ROOT}/.cache/G-OPD/code_eval}"
CODE_EVAL_RUN_NAME="${CODE_EVAL_RUN_NAME:-$(basename "$(dirname "${VERL_ACTOR_CHECKPOINT}")")}"
CODE_EVAL_CUDA_VISIBLE_DEVICES="${CODE_EVAL_CUDA_VISIBLE_DEVICES:-0}"
CODE_EVAL_N_SAMPLES="${CODE_EVAL_N_SAMPLES:-4}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-1.0}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_MAX_TOKENS="${CODE_EVAL_MAX_TOKENS:-16384}"
CODE_EVAL_RESULTS_ROOT="${CODE_EVAL_RESULTS_ROOT:-${REPO_ROOT}/code_eval_results}"

if [[ ! -f "${CODE_EVAL_ROOT}/scripts/run_evalplus.sh" ]]; then
    echo "Missing G-OPD code evaluator: ${CODE_EVAL_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${CODE_EVAL_ROOT}/data/HumanEvalPlus.jsonl" || ! -f "${CODE_EVAL_ROOT}/data/MbppPlus.jsonl" ]]; then
    echo "HumanEval+ or MBPP+ data is missing from ${CODE_EVAL_ROOT}/data." >&2
    exit 1
fi
if [[ ! -d "${CODE_EVAL_ROOT}/coding/LiveCodeBench/code_generation_lite" ]]; then
    echo "LiveCodeBench v6 data is missing." >&2
    exit 1
fi

has_huggingface_weights() {
    local model_dir="$1"
    [[ -f "${model_dir}/config.json" ]] || return 1
    compgen -G "${model_dir}/*.safetensors" >/dev/null \
        || compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null
}

actor_dir="${VERL_ACTOR_CHECKPOINT}"
if has_huggingface_weights "${actor_dir}"; then
    model_path="${actor_dir}"
elif has_huggingface_weights "${actor_dir}/huggingface"; then
    model_path="${actor_dir}/huggingface"
elif has_huggingface_weights "${actor_dir}/merged_huggingface"; then
    model_path="${actor_dir}/merged_huggingface"
else
    if [[ -f "${actor_dir}/fsdp_config.json" ]]; then
        merge_backend="fsdp"
    elif [[ -d "${actor_dir}/dist_ckpt" ]]; then
        merge_backend="megatron"
    else
        echo "Unsupported VERL checkpoint layout: ${actor_dir}" >&2
        exit 1
    fi

    model_path="${actor_dir}/merged_huggingface"
    echo "Merging ${merge_backend} actor checkpoint into ${model_path}"
    PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
        python -m verl.model_merger merge \
        --backend "${merge_backend}" \
        --local_dir "${actor_dir}" \
        --target_dir "${model_path}"
fi

if ! has_huggingface_weights "${model_path}"; then
    echo "Merged model is not loadable: ${model_path}" >&2
    exit 1
fi

eval_work_dir="${CODE_EVAL_RESULTS_ROOT}/${CODE_EVAL_RUN_NAME}"
mkdir -p "${eval_work_dir}"
if [[ ! -e "${eval_work_dir}/code_eval" ]]; then
    ln -s "${CODE_EVAL_ROOT}" "${eval_work_dir}/code_eval"
fi

echo "Evaluating model: ${model_path}"
echo "Evaluation results: ${eval_work_dir}"

pushd "${eval_work_dir}" >/dev/null
CUDA_VISIBLE_DEVICES="${CODE_EVAL_CUDA_VISIBLE_DEVICES}" \
    bash "${CODE_EVAL_ROOT}/scripts/run_evalplus.sh" \
    humaneval "${model_path}" 0 "${CODE_EVAL_TEMPERATURE}" "${CODE_EVAL_TOP_P}" "${CODE_EVAL_N_SAMPLES}"
CUDA_VISIBLE_DEVICES="${CODE_EVAL_CUDA_VISIBLE_DEVICES}" \
    bash "${CODE_EVAL_ROOT}/scripts/run_evalplus.sh" \
    mbpp "${model_path}" 0 "${CODE_EVAL_TEMPERATURE}" "${CODE_EVAL_TOP_P}" "${CODE_EVAL_N_SAMPLES}"
popd >/dev/null

pushd "${eval_work_dir}" >/dev/null
CUDA_VISIBLE_DEVICES="${CODE_EVAL_CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="${CODE_EVAL_ROOT}/coding/LiveCodeBench${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m lcb_runner.runner.main \
    --model Qwen3-4B-NonThinking \
    --local_model_path "${model_path}" \
    --trust_remote_code \
    --scenario codegeneration \
    --release_version v6 \
    --tensor_parallel_size 1 \
    --use_cache \
    --n "${CODE_EVAL_N_SAMPLES}" \
    --temperature "${CODE_EVAL_TEMPERATURE}" \
    --max_tokens "${CODE_EVAL_MAX_TOKENS}" \
    --custom_output_save_name "${CODE_EVAL_RUN_NAME}" \
    --top_p "${CODE_EVAL_TOP_P}" \
    --timeout 60 \
    --evaluate \
    --continue_existing \
    --continue_existing_with_eval
popd >/dev/null

echo "HumanEval+, MBPP+, and LiveCodeBench evaluation completed."
