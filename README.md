# L-APD：锚点式成对蒸馏（Anchored Pairwise Distillation）

> 本仓库是在 Prune-OPD 框架上实现的 L-APD。框架本身（Prune-OPD / OPD baseline 的说明、
> 安装步骤、其他实验脚本）见 [`README_Prune-OPD.md`](README_Prune-OPD.md)。

L-APD 是在 Prune-OPD 框架内实现的一个**不依赖 reward、advantage、PPO ratio 和 GRPO 分组**的
on-policy 蒸馏目标。student 先在自己的轨迹上采样出真实 token `y_t` 作为锚点，teacher 再告诉
student：`y_t` 相对每个重要候选 `z` 应该排得更高还是更低、以及应当相差多少。

```text
                           候选 z1
                              ↑
                              |
候选 z2  ←── 真实 token y ──→  候选 z3
                              |
                              ↓
                           候选 z4
```

## 1. 目标函数

单个位置的损失为 teacher 加权的成对 Bernoulli KL：

```text
L_t = Σ_{z ≠ y_t}  q̃_t(z) · KL_B( σ(T(y_t) − T(z))  ‖  σ(S(y_t) − S(z)) )

q̃_t(z) = q_t(z) / (1 − q_t(y_t))
```

- `S`、`T` 分别是 student / teacher 的 logit，`σ` 为 sigmoid；
- teacher 候选权重 `q̃` 与 teacher 胜率均为 stop-gradient，只有 student 接收梯度；
- 序列级损失是对有效 response token 求平均（`loss_agg_mode`，默认 `token-mean`）。

**实现上的关键点**：softmax 归一化项在 logit 差里会抵消，即
`T(y) − T(z) = log q(y) − log q(z)`。因此所有成对 margin 都能直接从框架已有的 top-k
log-prob 读出，**不需要传输或重算原始 logits，也不需要额外的 teacher forward**。每次更新
仍然只有一次 student forward，显存与吞吐和 OPD baseline 完全一致。

**候选集与归一化范围**：为了和 OPD baseline 可比，默认候选取 **student top-k**（baseline 的
`top_k_strategy=only_stu` 打分的就是这一组），权重按 teacher 在这些 id 上的概率
`q(z)` 在 **top-k 内部**重归一化，与 baseline 在 K 维上做一次 softmax 的做法一致。注意
权重来源仍是 teacher 概率而非 baseline 的 student 概率，这是 `q̃(z) = q(z)/(1 − q(y_t))`
的定义决定的，改掉就不是 L-APD 了。

另有一个 tail 候选选项（`tail_candidate=True`）：追加一个聚合候选，用 `logsumexp` 承载
候选集之外的概率质量，此时归一化基准变成全词表，等价于严格除以 `1 − q(y_t)`，截断的尾部
也能获得监督。实测该 tail 会拿到约 10% 的权重，因此它是一个会改变目标的选项，默认关闭以
保持与 baseline 的可比性。

## 2. 代码位置

| 文件 | 作用 |
| --- | --- |
| `verl/verl/trainer/ppo/l_apd.py` | 目标函数本体（只依赖 torch，可独立测试） |
| `verl/verl/workers/actor/dp_actor.py` | `update_policy` 中的 L-APD 分支与 `_compute_l_apd_loss` |
| `verl/verl/workers/config/actor.py` | `LAPDConfig` 配置项 |
| `verl/verl/trainer/config/actor/actor.yaml` | `actor_rollout_ref.actor.l_apd.*` 默认值 |
| `verl/verl/trainer/ppo/ray_trainer.py` | 保留 L-APD 需要的 teacher 张量到 actor update |
| `experiments_scripts/l-apd-*.sh` | Table 1 配置的启动脚本 |
| `verl/tests/trainer/ppo/test_l_apd_on_cpu.py` | 单元测试 |

`l_apd.enable=false` 时以上改动全部旁路，OPD / Prune-OPD 原有行为逐字不变。

## 3. 实验配置（与 Table 1 对齐）

| 项目 | 取值 |
| --- | --- |
| Student | DeepSeek-R1-Distill-Qwen-1.5B |
| Teacher | JustRL-DeepSeek-1.5B（冻结） |
| 训练数据 | DAPO-Math-17K |
| 评测 | AIME24 + AIME25 + AMC23，Avg@16 |
| max response length | 12288（validation 31744） |
| rollout number | 4 |
| mini-batch size | 64 |
| log-prob top-k | 16 |
| learning rate | 1e-6 |
| 训练步数 | 203 |
| 采样温度 | student 1.0 / teacher 1.0 |
| 硬件 | 单节点 8 卡 |

## 4. 准备

### 4.1 环境

本机可直接复用已有的 conda 环境（已含 torch 2.6 / vllm 0.8.5 / ray 2.56 / hydra / tensordict）：

```bash
conda activate /openbayes/input/input0/miniconda3/envs/g-opd-verl
cd /input0/yyy/Prune-OPD
```

若要从零搭建，按 [`README_Prune-OPD.md`](README_Prune-OPD.md) 的 Setup 一节创建 `opd` 环境即可。
下文命令里的 `/input0/yyy/Prune-OPD` 是本机的仓库路径，换机器时替换成实际 clone 位置。

### 4.2 数据

仓库自带的 `datasets/` 已包含所需文件，无需额外准备：

```text
datasets/dapo-math-17k.parquet
datasets/test_data/AIME24/test.parquet
datasets/test_data/AIME25/test.parquet
datasets/test_data/AMC23/test.parquet
```

### 4.3 模型

student 与 teacher 必须共享 tokenizer 和词表（L-APD 直接比较同一 token 上的 margin）。
下载到任意目录后用 `MODEL_ROOT` 指向它：

```bash
export MODEL_ROOT=/input0/models

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir "${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"

huggingface-cli download hbx/JustRL-DeepSeek-1.5B \
  --local-dir "${MODEL_ROOT}/JustRL-DeepSeek-1.5B"
```

目录名必须与上面一致，脚本按 `${MODEL_ROOT}/<模型名>` 拼路径；也可以用
`ACTOR_MODEL_PATH` / `REWARD_MODEL_PATH` 直接指定完整路径。

## 5. 启动训练

### 5.1 先做一次 dry run

`DRY_RUN=1` 会走完全部路径校验（student / teacher 权重、训练集、`TEST_DATASET` 里实际列出的
每个评测文件），打印展开后的完整命令后退出，不启动 ray、不占用 GPU：

```bash
cd /input0/yyy/Prune-OPD

DRY_RUN=1 MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

看到 `Required path does not exist: ...` 说明对应模型或数据还没准备好。

### 5.2 正式启动

```bash
cd /input0/yyy/Prune-OPD
conda activate /openbayes/input/input0/miniconda3/envs/g-opd-verl

MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

脚本会自动 `ray stop --force` → `ray start --head` → 启动训练，退出时清理 ray；
想复用已有集群就传 `MANAGE_RAY=0`。日志同时打到终端和
`logs/opd/<experiment_name>.log`。

### 5.3 后台长跑并跟踪日志

```bash
cd /input0/yyy/Prune-OPD

nohup env MODEL_ROOT=/input0/models \
  bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > /dev/null 2>&1 &

# 跟踪最新日志
tail -f "$(ls -t logs/opd/l-apd-*.log | head -1)"

# 每个 step 的指标是一整行（step:N - key:value - ...），抽出其中的 L-APD 字段
tail -f "$(ls -t logs/opd/l-apd-*.log | head -1)" \
  | grep --line-buffered -oE "step:[0-9]+|actor/l_apd_[a-z_]+:[-0-9.e+]+|val-core[^ ]+"
```

### 5.4 开启 W&B

```bash
WANDB_API_KEY=<your-key> WANDB_MODE=online TRACKING_BACKENDS='[console,wandb]' \
MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

### 5.5 对照组：OPD baseline（相同 rollout / teacher / optimizer / token budget / batch）

只有蒸馏 loss 不同，因此必须用同一套评测集，才能和 L-APD 直接比较：

```bash
cd /input0/yyy/Prune-OPD
export DATA_ROOT=/input0/yyy/Prune-OPD/datasets

MODEL_ROOT=/input0/models \
TEST_DATASET="[\"${DATA_ROOT}/test_data/AIME24/test.parquet\",\"${DATA_ROOT}/test_data/AIME25/test.parquet\",\"${DATA_ROOT}/test_data/AMC23/test.parquet\"]" \
bash experiments_scripts/opd-baseline-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

## 6. 常用调整

任何 Hydra override 都可以直接追加在脚本后面：

```bash
MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  trainer.test_freq=10 \
  actor_rollout_ref.actor.optim.lr=5e-7
```

卡数用 `N_GPUS_PER_NODE` 控制（脚本会传给 `trainer.n_gpus_per_node`）：

```bash
N_GPUS_PER_NODE=4 MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

### 6.1 L-APD 自身的开关

| 环境变量 | Hydra key | 默认 | 含义 |
| --- | --- | --- | --- |
| — | `actor_rollout_ref.actor.l_apd.enable` | 脚本内置 `True` | 用 L-APD 替换 policy-gradient 目标 |
| `L_APD_CANDIDATE_SOURCE` | `...l_apd.candidate_source` | `student` | 候选来源：student top-k（与 baseline 对齐）或 teacher top-k |
| `L_APD_TAIL_CANDIDATE` | `...l_apd.tail_candidate` | `False` | 是否追加聚合 tail 候选 |
| `L_APD_NORMALIZE_WEIGHTS` | `...l_apd.normalize_weights` | `True` | 权重按自身和归一化，而不是除以 `1 − q(y_t)` |
| `L_APD_TARGET_LOSS_COEF` | `...l_apd.target_loss_coef` | `0.0` | 额外 `KL_B(q(y)‖p(y))` 项，仅用于消融 |

消融示例：

```bash
# 打开 tail 候选：归一化基准从 top-16 内部变成全词表，尾部也获得监督
L_APD_TAIL_CANDIDATE=True MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 候选改用 teacher top-16
L_APD_CANDIDATE_SOURCE=teacher MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 加上 target loss（idea 文档 §8 的消融）
L_APD_TARGET_LOSS_COEF=1.0 MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

### 6.2 训练/评测规模

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `TOTAL_TRAINING_STEPS` | 203 | 训练步数 |
| `MAX_RESP_LENGTH` | 12288 | 训练最大生成长度 |
| `MAX_VAL_RESP_LENGTH` | 31744 | 评测最大生成长度 |
| `N_RESPONSES` | 4 | 每个 prompt 的 rollout 数 |
| `MINI_BATCH_SIZE` | 64 | mini-batch |
| `LOG_PROB_TOP_K` | 16 | top-k 候选数 |
| `TEST_FREQ` / `SAVE_FREQ` | 20 / 100 | 评测与存档间隔 |
| `VAL_BEFORE_TRAIN` | True | 训练前先跑一次评测作为 step 0 基线 |
| `TEST_DATASET` | AMC23/AIME24/AIME25/HMMT24/HMMT25 | 评测集，会按实际列出的文件逐个校验存在性 |
| `MODEL_ROOT` / `DATA_ROOT` | `/input0/models`、仓库内 `datasets/` | 模型与数据根目录 |
| `CKPT_ROOT` / `LOG_DIR` | `checkpoint/`、`logs/opd` | 输出位置 |
| `N_GPUS_PER_NODE` | 8 | 单节点卡数 |
| `MANAGE_RAY` | 1 | 是否由脚本负责 `ray stop` / `ray start --head`；已有集群时设 0 |

脚本默认评测 5 个测试集共 203 道题，`val_kwargs.n=16` 且生成上限 31744 token，
所以 `VAL_BEFORE_TRAIN=True` 那一轮要生成 3248 条长序列，耗时较长（期间日志静默但
GPU 满载，属正常）。只要 Table 1 的三个集时覆盖掉即可：

```bash
DATA_ROOT=/input0/yyy/Prune-OPD/datasets
TEST_DATASET="[\"${DATA_ROOT}/test_data/AIME24/test.parquet\",\"${DATA_ROOT}/test_data/AIME25/test.parquet\",\"${DATA_ROOT}/test_data/AMC23/test.parquet\"]" \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

## 7. 训练日志里的 L-APD 指标

| 指标 | 含义 |
| --- | --- |
| `actor/pg_loss` | L-APD 目标值（复用了原字段名） |
| `actor/l_apd_bernoulli_kl` | teacher 加权的成对 Bernoulli KL（扣掉 teacher 熵的纯 KL） |
| `actor/l_apd_pairwise_agreement` | student 与 teacher 排序方向一致的加权比例 |
| `actor/l_apd_pairwise_gap` | 加权的 `abs(r_S − r_T)`，student 与 teacher 胜率的差距 |
| `actor/l_apd_teacher_anchor_prob` / `..._student_anchor_prob` | 锚点 token 上的 `q(y_t)` / `p(y_t)` |
| `actor/l_apd_tail_weight` | tail 候选拿到的权重，仅 `tail_candidate=True` 时记录 |
| `actor/l_apd_teacher_tail_prob` / `..._student_tail_prob` | 候选集之外的尾部质量，仅 `tail_candidate=True` 时记录 |
| `actor/l_apd_anchor_in_candidates` | 锚点落在候选 top-k 内的比例 |
| `actor/l_apd_candidate_count` | 参与 loss 的候选数 |
| `actor/l_apd_candidate_weight_sum` | 候选权重和，`normalize_weights=True` 时应约等于 1 |

评测结果仍在 `val-core/*` 下（AIME24 / AIME25 / AMC23 的 Avg@16）。

## 8. 单元测试

```bash
cd /input0/yyy/Prune-OPD/verl
PYTHONPATH=$(pwd) python tests/trainer/ppo/test_l_apd_on_cpu.py
# 或
PYTHONPATH=$(pwd) pytest tests/trainer/ppo/test_l_apd_on_cpu.py -v
```

覆盖：全词表候选时与定义式逐元素相等；autograd 梯度等于解析式
`∂L/∂(S(y)−S(z)) = q̃(z)·(r_S − r_T)`；tail 候选只含一个 token 时与全词表 loss 精确相等；
`p = q` 时梯度为 0；padding 位置无 loss、无 NaN；候选权重的归一化性质。

## 9. 注意事项

- **必须共享词表**。teacher 与 student 的 tokenizer / vocabulary 必须一致，否则单 token
  margin 不可比。DeepSeek-R1-Distill-Qwen-1.5B 与 JustRL-DeepSeek-1.5B 满足该条件。
- **teacher 必须真的更强**。训练前建议先确认 teacher 在同样的 prompt 格式下优于 student，
  否则蒸馏到的只是措辞偏好。`VAL_BEFORE_TRAIN=True` 会给出 student 的 step 0 基线。
- **L-APD 不读 reward**。框架仍会计算 rm_scores / advantage 用于诊断与对照，但 L-APD 的
  梯度完全不经过它们（`update_policy` 在 L-APD 分支下根本不会取 `advantages`）。
- **`TOP_K_STRATEGY` 不影响 L-APD 的候选集**。teacher top-k 的 id 与 log-prob 在任何
  strategy 下都会被计算并传下来，L-APD 直接读 `teacher_top_k_ids` / `teacher_top_k_log_probs`
  与锚点上的 `teacher_log_probs`；该变量只影响 OPD 侧的 reward 构造。候选数由
  `LOG_PROB_TOP_K` 决定。
- **不要打开 `TORCH_NCCL_BLOCKING_WAIT`**。它和 vLLM 的 CUDA graph capture 会死锁：8 个 rank
  全堵在 torch 的 `ProcessGroupNCCL::waitForPendingWorks()` 里，而那个等待循环没有超时，
  表现是显存占满、GPU 利用率 0%、日志停在
  `Waiting for pending NCCL work to finish before starting graph capture` 且永不恢复。
  `common.sh` 已默认置 0 并改用 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`。若确实需要它，
  必须同时加 `actor_rollout_ref.rollout.enforce_eager=True` 绕开 graph capture。
- **恢复训练**：experiment name 带时间戳，所以每次启动都是新目录。要接着上次跑，需显式指定
  `trainer.default_local_dir=<旧 checkpoint 目录> trainer.resume_mode=auto`。
- 单节点默认 8 卡（`trainer.n_gpus_per_node=8`），卡数不同时请追加 override。
