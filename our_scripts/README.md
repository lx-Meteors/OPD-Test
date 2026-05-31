# OPD Launch Scripts

This directory contains the OPD launch scripts for the three experiment combinations in `idea.md`.

Available scripts:

- `run_opd_combo1_qwen3_1p7b_base_to_qwen3_4b.sh`
- `run_opd_combo2_qwen3_4b_base_to_qwen3_4b.sh`
- `run_opd_combo3_r1_1p5b_to_r1_7b.sh`
- `run_grpo_r1_1p5b_12k.sh`

Usage examples:

```bash
bash our_scripts/run_opd_combo1_qwen3_1p7b_base_to_qwen3_4b.sh
```

Preview the resolved command without launching training:

```bash
DRY_RUN=1 bash our_scripts/run_opd_combo1_qwen3_1p7b_base_to_qwen3_4b.sh
```

Override a few knobs from the shell:

```bash
MAX_RESP_LENGTH=4096 MINI_BATCH_SIZE=32 DRY_RUN=1 \
  bash our_scripts/run_opd_combo3_r1_1p5b_to_justrl_1p5b.sh
```

Pass extra Hydra overrides at the end:

```bash
bash our_scripts/run_opd_combo2_qwen3_4b_base_to_qwen3_4b.sh \
  trainer.test_freq=10 actor_rollout_ref.actor.optim.lr=5e-7
```

GRPO example aligned with the R1 1.5B combo:

```bash
bash our_scripts/run_grpo_r1_1p5b_12k.sh
```

Notes:

- The scripts activate the `opd` conda environment unless it is already active.
- They export `PYTHONPATH=$REPO_ROOT/verl` so `verl.trainer.main_ppo` resolves from this repo.
- They default to `trainer.logger=[console,wandb]` and export `WANDB_API_KEY` plus `WANDB_DIR` from `our_scripts/opd_common.sh`.
- You can override the logger list with `TRACKING_BACKENDS`, for example `TRACKING_BACKENDS='[console,wandb]'`.
- Combo 1 and combo 2 explicitly disable `enable_thinking` because they are non-thinking Qwen3 setups.
- The default train dataset is `datasets/dapo-math-17k.parquet`, matching the current top-level OPD setup.
- GRPO scripts set `REWARD_MODEL_ENABLE=False` and `LOG_PROB_TOP_K=0`, but still reuse the same rule-based reward function path.
