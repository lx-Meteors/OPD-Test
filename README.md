# L-APD：锚点式成对蒸馏（Anchored Pairwise Distillation）

> 本仓库是在 Prune-OPD 框架上实现的 L-APD。框架本身（Prune-OPD / OPD baseline 的说明、
> 安装步骤、其他实验脚本）见 [`README_Prune-OPD.md`](README_Prune-OPD.md)。

L-APD 是在 Prune-OPD 框架内实现的一个**不依赖 reward、advantage、PPO ratio 和 GRPO 分组**的
on-policy 蒸馏目标。student 先在自己的轨迹上采样出真实 token $y_t$ 作为锚点，teacher 再告诉
student：$y_t$ 相对每个重要候选 $z$ 应该排得更高还是更低、以及应当相差多少。

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

记 $p_t$、$q_t$ 为 student / teacher 在位置 $t$ 的分布，$S$、$T$ 为对应的 logit，$\sigma$ 为
sigmoid，二值 KL 为

$$
\mathrm{KL_B}(a  \Vert  b) \;=\; a\log\frac{a}{b} \;+\; (1-a)\log\frac{1-a}{1-b}.
$$

每个 response 位置 $t$，锚点是 student 自己采样出的 token $y_t$，其余候选是

$$
\mathcal{V}_t \;=\; \bigl\lbrace\, z \in \text{student top-}K \;:\; z \neq y_t \,\bigr\rbrace,
\qquad K = 16,\quad \text{通常 } \lvert \mathcal{V}_t \rvert = 15.
$$

**权重是 teacher 概率在这 $K$ 个 token（含目标 token $y_t$）上一起归一化**得到的：

$$
Z_t \;=\; \sum_{v \in \text{top-}K} q_t(v) \;=\; q_t(y_t) + \sum_{z \in \mathcal{V}_t} q_t(z).
$$

损失由 $K-1$ 个成对项和一个目标 token 项组成，各自的乘数就是自己那个 token 的归一化概率：

$$
\boxed{\;
L_t \;=\; \sum_{z \in \mathcal{V}_t} \frac{q_t(z)}{Z_t}\,
\mathrm{KL_B}\bigl(r_S(y_t, z) \Vert r_T(y_t, z)\bigr)
\;+\; \frac{q_t(y_t)}{Z_t}\,
\mathrm{KL_B}\bigl(p_t(y_t) \Vert q_t(y_t)\bigr)
\;}
$$

默认用**逆向**方向(student 在前),由 `l_apd.pair_divergence` 控制,`forward_kl` 是消融项。
两个方向的最优点相同、且在最优点附近一阶一致,差别只在大误差处如何修正(见 §1.3)。

所有乘数加起来恰好为 1（诊断 `actor/l_apd_candidate_weight_sum` 就是在查这一点），目标 token
那一项的乘数记作 $a_t = q_t(y_t)/Z_t$（诊断 `actor/l_apd_anchor_weight`，实测均值约 0.65）。

胜率是 Bradley–Terry 式的归一化对，都有闭式。对候选 $z \in \mathcal{V}_t$：

$$
r_S(y, z) = \sigma\bigl(\log p(y) - \log p(z)\bigr) = \frac{p(y)}{p(y) + p(z)},
\qquad
r_T(y, z) = \sigma\bigl(\log q(y) - \log q(z)\bigr) = \frac{q(y)}{q(y) + q(z)}.
$$

**目标 token 项其实也是一个成对项**，只不过对手是"除 $y_t$ 外的全体 token 打包成一个"
（代码里记作 $\perp$，即 `complement_candidate`）。它的 margin 是 $\log\frac{p(y)}{1-p(y)}$，于是

$$
r_S(y, \perp) = \sigma\Bigl(\log \tfrac{p(y)}{1 - p(y)}\Bigr) = p(y),
\qquad
r_T(y, \perp) = \sigma\Bigl(\log \tfrac{q(y)}{1 - q(y)}\Bigr) = q(y),
$$

代入 $\mathrm{KL_B}(r_S \Vert r_T)$ **恰好等于** $\mathrm{KL_B}\bigl(p_t(y_t) \Vert q_t(y_t)\bigr)$，
也就是 boxed 公式里的最后一项。于是整个 loss 可以合并成一个统一的加权和：把 $\perp$ 当成
一个额外"对手"、它的 teacher 质量取 $q_t(y_t)$，则

$$
L_t \;=\; \sum_{o \in \mathcal{V}_t \cup \lbrace \perp \rbrace} \frac{m_t(o)}{Z_t}\;
\mathrm{KL_B}\bigl(r_S(y_t, o) \Vert r_T(y_t, o)\bigr),
\qquad
m_t(z) = q_t(z),\quad m_t(\perp) = q_t(y_t).
$$

这也是代码的实现方式：目标 token 项作为一列追加进候选张量，归一化、散度、诊断全部复用同一
套逻辑。所以它**不是一个带系数的附加 loss**，没有自由超参。

- 权重 $m_t/Z_t$ 与 teacher 胜率 $r_T$ 均为 stop-gradient，只有 student 接收梯度；
- 序列级损失是对有效 response token 求平均（`loss_agg_mode`，默认 `token-mean`）。

**实现上的两个要点**：

1. 逆向下 student 持有熵项，所以 loss **本身就是 KL**，`actor/l_apd_loss` 会收敛到 0，和
   `actor/l_apd_bernoulli_kl` 相等。正向下代码用 `binary_cross_entropy_with_logits`，算的是
   Bernoulli 交叉熵，比 KL 多一个 teacher 侧熵（student 无关的常数，梯度相同），此时只有
   `actor/l_apd_bernoulli_kl` 是纯 KL。两个方向的 $\mathrm{KL_B}$ 都用 log-sigmoid 写成,
   任意实数 margin 下都有限。
2. softmax 归一化项在 logit 差里会抵消，即 $T(y) - T(z) = \log q(y) - \log q(z)$。因此所有
   成对 margin 都能直接从框架已有的 top-$k$ log-prob 读出，**不需要传输或重算原始 logits，
   也不需要额外的 teacher forward**。每次更新仍然只有一次 student forward，显存与吞吐和 OPD
   baseline 完全一致。

### 1.1 候选集与归一化范围

为了和 OPD baseline 可比，默认候选取 **student top-$k$**（baseline 的
`top_k_strategy=only_stu` 打分的就是这一组），权重来源是 teacher 在这些 id 上的概率
$q(z)$ 而非 baseline 的 student 概率 —— 这是 $\tilde q(z) \propto q(z)$ 的定义决定的，改掉
就不是 L-APD 了。归一化范围是**含锚点的这 16 个 id**（见下），和 baseline 在 $K$ 维上做一次
softmax 的范围逐一对应。

### 1.2 为什么必须有一个聚合对手

成对项只通过 logit 差 $S(y_t) - S(z)$ 依赖 student，所以如果对手只有 top-$k$ 里那些**真实
token**，就存在一个**精确的不变性**：把 $\mathcal{S}_t = \lbrace y_t \rbrace \cup \text{top-}k$
这些 token 的 logit 同时加上同一个常数（等价于把这一组的总质量整体缩放，多出/少掉的部分由
截断掉的尾部吸收），所有成对 margin 不变，**loss 一个字都不变**。于是在成对最优点上只能
推出比例关系

$$
p(y_t) \;=\; q(y_t)\cdot\frac{M_S}{N_S},
\qquad
M_S = \sum_{v \in \mathcal{S}_t} p(v),
\qquad
N_S = \sum_{v \in \mathcal{S}_t} q(v).
$$

其中 $M_S$ 完全自由 —— student 可以把 15 个成对关系全部拟合到完美，同时把大量质量漏进
尾部。`tests/trainer/ppo/test_l_apd_on_cpu.py::test_token_candidates_alone_leave_the_tail_mass_unidentified`
锁住了这个性质；一个数值实验里 student 最终停在 $p_{\text{tail}} = 18.9\%$（teacher 只有
$1.15\%$），top-$k$ 内每个 token 都被统一压低 18%，而 loss 已经收敛。

补一个**聚合对手**就能消掉这个自由方向：只要对手集合覆盖全词表，由
$p(y)/p(c) = q(y)/q(c)$ 对每一块 $c$ 成立、两边求和且都归一到 1，立刻得到
$p(y_t) = q(y_t)$。有两种选法：

| | 聚合对手 $\perp$ | 该项的含义 | 权重（自动定标） |
| --- | --- | --- | --- |
| `complement_candidate=True`（默认） | $y_t$ 的补集，即"除 $y_t$ 外的全体" | $\mathrm{KL_B}(p(y_t) \Vert q(y_t))$，即目标 token 的 loss | $a_t = q(y_t)/Z_t$，实测约 $0.65$ |
| `tail_candidate=True` | top-$k$ 之外的十几万个 token，用 `logsumexp` + `log1mexp` 打包 | 锚点相对整个尾部应排多高 | $q_{\text{tail}}/Z_t$，此时 $Z_t = 1 - q(y_t)$，实测约 $0.10$ |

两者都是**同一个成对构造**在一个聚合对手上的特例，权重都由 teacher 质量自动决定，
所以都没有自由系数。`tail_candidate` 优先级更高；它开启时对手集合
$\lbrace z_1, \dots, z_k, \text{tail} \rbrace$ 构成"非 $y_t$"的真正**划分**，补集对手则与 $z_j$ 有重叠，
是一个让总权重恰好为 1 的启发式。

默认选补集，理由有三条：

1. **归一化范围与 baseline 对齐**。此时
   $Z_t = q(y_t) + \sum_{z \in \mathcal{V}_t} q(z)$ 是 teacher 在
   $\lbrace y_t \rbrace \cup \text{top-}16$ 上的总质量，**含锚点**，所以权重是在 baseline 用的那 16 个 id
   上归一化的；`tail_candidate` 会引入 baseline 没有的第 17 个虚拟对手。展开就是 §1 那个凸
   组合，每个 token 的总权重恒为 1。
2. **顺带修掉了权重归一化的 eps 下限问题**。归一化分母现在至少是 $q(y_t)$，不会塌陷。
   此前 teacher 在采样 token 上几乎确定（$q(y_t) > 1 - 10^{-6}$）的位置会撞上 $10^{-6}$ 下限、
   权重被压到 0、梯度丢失，实测 `candidate_weight_sum` $\approx 0.94$，约 6% 的 token 受影响。
   现在这些位置 $a_t \to 1$，锚点项接管全部权重。
3. **目标在两种模式间自动切换**。$q(y_t) \to 0$（teacher 认为采样 token 很差）时
   $a_t \to 0$，几乎全是排序信号；$q(y_t) \to 1$ 时 $a_t \to 1$，竞争者概率都趋于 0、本来
   无序可排，信号转为锚点水平。

已知代价：$a_t$ 均值约 $0.65$，意味着**多数 token 上锚点项占主导**，排序监督相应被稀释。
一个两位置的数值实验里，$q(y)=0.95$ 那个位置的 $p_{\text{tail}}$ 收到
$8.67\times10^{-3}$（真值 $2.37\times10^{-3}$，偏高 3.7 倍），因为成对项只剩 $0.05$ 的权重、
竞争者之间的相对水平收敛很慢；$p(y_t)$ 本身是准的。这是收敛速度而非渐近性质的问题，
但 203 步训练里速度就是结果，所以 `tail_candidate=True` 值得作为对照跑一次。

顺带一提，框架里没有其它项在约束这个方向：`USE_KL` 默认 `False`，而且 ref-KL 那段代码在
`update_policy` 的非 L-APD 分支里；`entropy_coeff` 也是 0。

### 1.3 梯度与方向的选择

记 $m = S(y_t) - S(o)$ 为 student margin、$m_T = T(y_t) - T(o)$ 为 teacher margin。两个方向对
每个成对 margin 的梯度都有闭式：

$$
\text{逆向（默认）：}\quad
\frac{\partial L_t}{\partial m}
\;=\; \frac{m_t(o)}{Z_t}\,\sigma'(m)\,\bigl(m - m_T\bigr)
$$

$$
\text{正向：}\quad
\frac{\partial L_t}{\partial m}
\;=\; \frac{m_t(o)}{Z_t}\,\bigl(r_S(y_t, o) - r_T(y_t, o)\bigr)
$$

逆向就是**直接对齐 margin**：梯度正比于 $m$ 与 $m_T$ 之差，由 $\sigma'(m)$ 门控。两者都在
$m = m_T$ 处归零，而且在最优点附近**一阶一致**（$\delta = 10^{-3}$ 时两者梯度都等于
$\sigma'(m_T)\delta$，相对差 $10^{-3}$），所以这个选择**只影响大误差如何被修正**：

| $m$（$m_T = -4$，student 越来越自信地站错边） | 逆向 loss | 正向 loss | 逆向梯度 | 正向梯度 |
|---:|---:|---:|---:|---:|
| 0 | 1.33 | 0.60 | 1.0e+0 | 0.482 |
| 8 | 4.01 | 7.77 | 4.0e-3 | 0.982 |
| 30 | **4.018** | 29.37 | **3.2e-12** | 0.982 |

逆向 loss 封顶在 $-\log r_T$，自信站错边的代价有界、梯度按 $\sigma'(m)$ 指数衰减；正向无界，
梯度饱和到 $\pm 1$ 但不消失。这也是唯一需要注意的取舍：代入 $o = \perp$ 时正向的锚点梯度是
$a_t\bigl(p_t(y_t) - q_t(y_t)\bigr)$，逆向多了一个 $p_t(y_t)\bigl(1 - p_t(y_t)\bigr)$ 因子，
在 $p_t(y_t) \to 1$ 时消失 —— 即逆向下 §1.2 的可辨识性在理论上仍成立（最优点唯一），但对
**已经过度自信**的锚点没有拉回力。`L_APD_PAIR_DIVERGENCE=forward_kl` 可以切回正向做对照。
`test_reverse_kl_is_bounded_where_forward_diverges` 锁住了上表的定性行为。

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
| `L_APD_TAIL_CANDIDATE` | `...l_apd.tail_candidate` | `False` | 聚合对手取"top-k 之外"，优先级高于补集 |
| `L_APD_COMPLEMENT_CANDIDATE` | `...l_apd.complement_candidate` | `True` | 聚合对手取"`y_t` 的补集"，即目标 token 的 loss，见 §1.2 |
| `L_APD_NORMALIZE_WEIGHTS` | `...l_apd.normalize_weights` | `True` | 权重按自身和归一化，而不是除以 `1 − q(y_t)` |
| `L_APD_PAIR_DIVERGENCE` | `...l_apd.pair_divergence` | `reverse_kl` | 成对 Bernoulli KL 的方向，见 §1.3。`forward_kl` 为对照 |

> 两个聚合对手**至少要开一个**。全关掉时锚点概率不可辨识（§1.2），只用于复现那个退化情形。

消融示例：

```bash
# KL 方向换成正向：自信站错边的 token 保留满强度梯度，代价是 loss 无界
L_APD_PAIR_DIVERGENCE=forward_kl MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 聚合对手换成 tail：对手集合成为"非 y_t"的真正划分，锚点项不再需要
L_APD_TAIL_CANDIDATE=True MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 候选改用 teacher top-16
L_APD_CANDIDATE_SOURCE=teacher MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 退化对照：只有真实 token 对手，锚点概率不可辨识
L_APD_TAIL_CANDIDATE=False L_APD_COMPLEMENT_CANDIDATE=False MODEL_ROOT=/input0/models \
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
| `actor/l_apd_bernoulli_kl` | teacher 加权的成对 Bernoulli KL（纯 KL，正向下已扣掉 teacher 熵），含聚合对手那一项。逆向下它与 `actor/pg_loss` 相等 |
| `actor/l_apd_pairwise_agreement` | student 与 teacher 排序方向一致的加权比例 |
| `actor/l_apd_pairwise_gap` | 加权的 $\lvert r_S - r_T \rvert$，student 与 teacher 胜率的差距 |
| `actor/l_apd_teacher_anchor_prob` / `..._student_anchor_prob` | 锚点 token 上的 $q(y_t)$ / $p(y_t)$，两者是否收敛到一起是判断锚点项是否起效的主要观测量 |
| `actor/l_apd_anchor_kl` | 锚点项的 Bernoulli KL（方向随 `pair_divergence`），$p(y_t) = q(y_t)$ 时恰好为 0；仅 `complement_candidate` 生效时记录 |
| `actor/l_apd_anchor_weight` | 补集对手拿到的权重 $a_t = q(y_t)/Z_t$（见 §1）；仅 `complement_candidate` 生效时记录 |
| `actor/l_apd_tail_weight` | tail 候选拿到的权重，仅 `tail_candidate=True` 时记录 |
| `actor/l_apd_teacher_tail_prob` / `..._student_tail_prob` | 候选集之外的尾部质量，仅 `tail_candidate=True` 时记录 |
| `actor/l_apd_anchor_in_candidates` | 锚点落在候选 top-k 内的比例 |
| `actor/l_apd_candidate_count` | 参与 loss 的候选数 |
| `actor/l_apd_candidate_weight_sum` | 候选权重和，`normalize_weights=True` 且有聚合对手时应恒等于 1 |

评测结果仍在 `val-core/*` 下（AIME24 / AIME25 / AMC23 的 Avg@16）。

## 8. 单元测试

```bash
cd /input0/yyy/Prune-OPD/verl
PYTHONPATH=$(pwd) python tests/trainer/ppo/test_l_apd_on_cpu.py
# 或
PYTHONPATH=$(pwd) pytest tests/trainer/ppo/test_l_apd_on_cpu.py -v
```

覆盖：两个方向的 autograd 梯度分别等于 §1.3 的两个解析式；逆向 loss 逐元素等于加权的
$\mathrm{KL_B}(r_S \Vert r_T)$ 且与 `bernoulli_kl` 诊断相等；逆向 loss 在自信站错边处有界而
正向发散（锁住 §1.3 那张表的定性行为）；`reverse_kl` 确实是库的默认方向；未知方向名会报错；
$p = q$ 时两个方向梯度都为 0；正向下全词表候选与定义式逐元素相等、tail 候选只含一个 token 时
与全词表 loss 精确相等；padding 位置无 loss、无 NaN；候选权重的归一化性质；§1.2 的可辨识性
（只有真实 token 对手时 loss 对"质量在 top-$k$ 与尾部之间如何分配"完全不变，而两种聚合对手
都能破掉这个不变性）；补集对手那一项的权重逐元素等于 $q(y_t)/Z_t$、且含它的权重和恒为 1。

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
- **两个聚合对手全关掉时权重归一化会撞 eps 下限**。权重按 `raw / max(Σ raw, 1e-6)` 归一化，
  只有真实 token 对手时，teacher 在采样 token 上几乎确定（$q(y_t) > 1 - 10^{-6}$）的位置竞争者
  总质量小于 `1e-6`，下限生效、权重被压到 0、梯度丢失（实测 `candidate_weight_sum ≈ 0.94`，
  约 6% 的 token）。默认配置不受影响：补集对手让分母至少为 $q(y_t)$，tail 对手则由
  `_log1mexp` 的 clamp 兜了一个 `~1e-6` 的地板。
- **不要打开 `TORCH_NCCL_BLOCKING_WAIT`**。它和 vLLM 的 CUDA graph capture 会死锁：8 个 rank
  全堵在 torch 的 `ProcessGroupNCCL::waitForPendingWorks()` 里，而那个等待循环没有超时，
  表现是显存占满、GPU 利用率 0%、日志停在
  `Waiting for pending NCCL work to finish before starting graph capture` 且永不恢复。
  `common.sh` 已默认置 0 并改用 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`。若确实需要它，
  必须同时加 `actor_rollout_ref.rollout.enforce_eager=True` 绕开 graph capture。
- **恢复训练**：experiment name 带时间戳，所以每次启动都是新目录。要接着上次跑，需显式指定
  `trainer.default_local_dir=<旧 checkpoint 目录> trainer.resume_mode=auto`。
- 单节点默认 8 卡（`trainer.n_gpus_per_node=8`），卡数不同时请追加 override。
