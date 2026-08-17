# Distillation-to-Go (DTG)

DTG 保留标准 Top-K OPD，并增加一个只作用于 student 实际采样 token 的未来信用分配项。它把后续师生分歧视作当前动作的 student-conditioned future cost，而不是直接提高后段 token 的 OPD 权重。

局部差异默认使用 student Top-K 上的 total-variation lower bound：

```text
d_t = 0.5 * sum_k |p_student(k) - p_teacher(k)|
```

每 256 token 聚合成一个 block，并计算之后 blocks 的折扣平均差异。对同一道题的 4 条 rollout 使用 leave-one-out baseline：未来差异小于同组其他回答的动作获得正 advantage，未来差异更大的动作获得负 advantage。

最终目标为：

```text
L = L_OPD + dtg_weight * L_DTG
```

`L_OPD` 仍在 Top-K 候选上计算；`L_DTG` 使用实际 rollout token 的 log-prob，二者没有混用 candidate axis。已有答案正确性 reward 作为 terminal cost 加入 DTG，防止 student 通过错误地提前输出 EOS 来消除未来差异。

## 运行

```bash
nohup bash experiments_scripts/distillation-to-go-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/distillation-to-go.log 2>&1 &
```

默认配置：

- `DTG_WEIGHT=0.05`
- `DTG_BLOCK_SIZE=256`
- `DTG_BLOCK_GAMMA=0.95`
- `DTG_OUTCOME_WEIGHT=0.25`
- `DTG_NORMALIZE_BY_STD=False`
- `DTG_MAX_ABS_ADVANTAGE=0.5`
- `N_RESPONSES=4`

`block_gamma=0.95` 对应约 20 个 block，即约 5120 token 的有效信用传播范围。第一组实验不要同时启用 HI-OPD、GRPO 混合项或其他辅助 loss。

W&B 中重点观察：

- `actor/pg_loss`：原始 OPD loss。
- `actor/dtg_pg_loss`：独立 DTG sampled-token loss。
- `dtg/local_disagreement_mean`：当前 token 的平均师生差异。
- `dtg/future_cost_mean`：加入终局错误成本后的平均未来代价。
- `dtg/advantage_abs_mean`：DTG 实际信用信号强度。
- `dtg/positive_advantage_ratio`：获得正向未来信用的 token 比例。
- `dtg/valid_comparison_ratio`：具有同题存活 peer、能够构造反事实 baseline 的 block 比例。

若 `dtg/advantage_abs_mean` 长期小于 `1e-3`，可把 `DTG_WEIGHT` 提到 `0.1`；若 actor gradient norm 明显高于标准 OPD 或验证在前 20 step 就下降，先把它降到 `0.02`，不要开启 std normalization。
