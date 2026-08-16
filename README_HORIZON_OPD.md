# Horizon-Invariant OPD (HI-OPD)

标准 token-mean OPD 的位置权重由当前 student rollout 的 token 数量决定。若训练后回答变短，深层位置即使仍然很难，也会因为出现得更少而对总 loss 贡献更小。HI-OPD 固定训练初期的位置分布，使后续优化始终在同一个“推理深度测度”下比较。

将回答位置按 `bin_size` 分箱，记预热阶段冻结的 token 质量为 `omega_ref[b]`，当前 batch 的质量为 `omega_cur[b]`。每个位置使用

```text
w[b] = clip((omega_ref[b] / omega_cur[b]) ** alpha, min_weight, max_weight)
```

然后用 `w[b]` 重加权原 OPD advantage。它不是第二个辅助 loss；teacher/student 的 token-level OPD 信号不变，只改变不同推理深度对同一个 OPD loss 的贡献。

## 运行

```bash
nohup bash experiments_scripts/horizon-invariant-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/horizon-invariant-opd.log 2>&1 &
```

默认参数：

- `HORIZON_OPD_BIN_SIZE=1024`
- `HORIZON_OPD_REFERENCE_STEPS=5`
- `HORIZON_OPD_ALPHA=1.0`
- `HORIZON_OPD_MIN_WEIGHT=0.25`
- `HORIZON_OPD_MAX_WEIGHT=3.0`

前 5 个训练 batch 用于估计 reference，权重统一为 1，因此预热阶段退化为标准 OPD。第 6 个 batch 开始校正。脚本强制使用 `seq-mean-token-sum`，并在每个 PPO mini-batch 内归一化权重；权重为 1 时，其尺度与原来的 token-mean OPD 相同。

W&B 中重点观察：

- `horizon_opd/position_mass_tv`：当前位置分布相对初始分布的漂移。
- `horizon_opd/corrected_mass_tv`：校正后的残余漂移，应该低于前者。
- `horizon_opd/current_tail_mass` 与 `reference_tail_mass`：深层 token 质量是否减少。
- `horizon_opd/bin_weight_*`：每个深度区间实际得到的权重。
- `horizon_opd/weight_min`、`weight_max`：是否频繁撞到裁剪边界。

首次实验建议保持默认值并与完全相同 seed/config 的标准 OPD 对照。如果大量深层 bin 长期撞到 `max_weight`，先把 `alpha` 降到 `0.5`，而不是直接扩大裁剪上限。

当前 reference 是运行时状态，恢复 checkpoint 后会重新用前 5 个 batch 校准。因此正式对照实验建议从初始模型开始完整运行。
