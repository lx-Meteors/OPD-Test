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

## 0. 快速开始

```bash
conda activate /openbayes/input/input0/miniconda3/envs/g-opd-verl
cd /input0/yyy/Prune-OPD

# 先做一次路径校验，不启动 ray、不占 GPU
DRY_RUN=1 MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 正式启动，默认就是方法本身（reverse_kl + student 加权 + 尾部块）
MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

> [!WARNING]
> **不要用 `L_APD_PAIR_DIVERGENCE=log_ratio` 跑正式实验。** 它是一个刻意保留的消融，
> 把每对的 KL 砍掉一半，因此不是散度、没有下界，梯度里也没有 teacher。实测它的梯度与
> `reverse_kl` 的余弦是 **−0.985**（几乎精确反向），一条 138 步的 run 把 `actor/entropy`
> 从 0.66 打到 0.04、`actor/l_apd_anchor_kl` 反向涨了 10 倍。详见
> [§1.4](#14-log_ratio只保留一项的消融)。

## 目录

| 节 | 内容 |
| --- | --- |
| [§1 目标函数](#1-目标函数) | 完整损失、候选与权重的两个来源、聚合对手、三种 `pair_divergence` |
| [§2 代码位置](#2-代码位置) | 改动涉及的文件 |
| [§3 实验配置](#3-实验配置与-table-1-对齐) | student / teacher / 数据 / 超参 |
| [§4 准备](#4-准备) | 环境、数据、模型 |
| [§5 启动训练](#5-启动训练) | dry run、前台与后台、W&B、baseline 对照 |
| [§6 常用调整](#6-常用调整) | 全部开关与消融命令 |
| [§7 训练日志里的指标](#7-训练日志里的-l-apd-指标) | 每个 `actor/l_apd_*` 字段的含义 |
| [§8 单元测试](#8-单元测试) | 怎么跑、覆盖了什么 |
| [§9 注意事项](#9-注意事项) | 共享词表、NCCL 死锁、恢复训练等坑 |
| [§10 更新记录](#10-更新记录) | 方法与默认配置的演进史 |

## 1. 目标函数

记 $p_t$、$q_t$ 为 student / teacher 在位置 $t$ 的分布，$S$、$T$ 为对应的 logit，$\sigma$ 为
sigmoid。

每个 response 位置 $t$，锚点是 student 自己采样出的 token $y_t$。具名候选取 **teacher top-$K$**
（剔除 $y_t$ 自己），没被具名的其余全部 token 打包成一个**尾部块** $\tau_t$：

$$
\mathcal{V}_t \;=\; \bigl\lbrace\, z \in \text{teacher top-}K \;:\; z \neq y_t \,\bigr\rbrace,
\qquad
\tau_t \;=\; \mathcal{V} \setminus \bigl(\lbrace y_t\rbrace \cup \mathcal{V}_t\bigr),
\qquad K = 16.
$$

锚点 $y_t$ 通常也落在 top-$K$ 里，会被剔除，所以具名候选数一般是 $\lvert \mathcal{V}_t \rvert = 15$
（诊断 `actor/l_apd_candidate_count` 实测 15.0）。对手集合是**具名候选加尾部块**：

$$
\mathcal{O}_t \;=\; \mathcal{V}_t \cup \lbrace \tau_t \rbrace .
$$

它恰好是"除 $y_t$ 外整个词表"的一个**划分**——每个非锚点 token 要么被具名、要么落在尾部块里，
**恰好出现一次**。

**权重是 student 在真正备选上的条件分布**（stop-gradient）：对手集合的 student 总质量恰好是
$1 - p_t(y_t)$，所以按自身和归一化后

$$
\tilde w_t(o) \;=\; \mathrm{sg}\!\left[\frac{p_t(o)}{\sum_{o' \in \mathcal{O}_t} p_t(o')}\right]
\;=\; \mathrm{sg}\!\left[\frac{p_t(o)}{1 - p_t(y_t)}\right],
\qquad
p_t(\tau_t) = 1 - p_t(y_t) - \sum_{z \in \mathcal{V}_t} p_t(z).
$$

这可以读成一个对决故事：$y_t \sim p_t$ 是 student 落子，挑战者从 student **自己此刻还愿意给质量
的备选**里抽出（$z \neq y_t$，跟自己对决学不到东西——这是分母 $1 - p_t(y_t)$ 的来源），teacher
只负责评判每场对决的胜负，loss 是每场对决代价的期望。**谁攥着质量，谁的对决就被认真打分**，
这使权重成为闭环的：student 在哪里囤积了多余质量（胀起来的尾部、一个错爱的候选），哪一列的
权重就自动变大，直到质量被排干为止。它也是方向一致的搭配——$\mathrm{KL}(p \Vert q)$ 的链式法则
里，条件格子的权重正是 student 的条件质量；逆向的成对 KL 配 student 权重，正如正向 KL 才配
teacher 权重。历史上的 teacher 权重 $q_t(o)/(1-q_t(y_t))$ 保留为消融
（`weight_source=teacher`）：它是开环的，预算按 teacher 偏好预先定死、不看 student 的错误在哪，
实测让暂态质量在权重仅 4–11% 的尾部列里堆到 teacher 的 3 倍、熵冲到 1.0（§10 更新记录）。

一次成对比较只关心"$y_t$ 和 $z$ 谁赢"，所以把两侧分布**限制到 $\lbrace y_t, z\rbrace$ 上再
归一化**，得到两个二元分布：

$$
\tilde p_{y,z}(v) \;=\; \frac{p_t(v)}{p_t(y) + p_t(z)},
\qquad
\tilde q_{y,z}(v) \;=\; \frac{q_t(v)}{q_t(y) + q_t(z)},
\qquad v \in \lbrace y, z\rbrace.
$$

这一对上的散度就是标准的**逆向 KL**（student 在前），按定义展开就是 $\log p/q$ 的求和：

$$
\mathrm{KL}\bigl(\tilde p_{y,z} \,\Vert\, \tilde q_{y,z}\bigr)
\;=\; \sum_{v \in \lbrace y, z\rbrace} \tilde p_{y,z}(v)\,
\log \frac{\tilde p_{y,z}(v)}{\tilde q_{y,z}(v)}.
$$

损失是**单独一项**——所有对决的逆向 KL 按挑战者分布加权求和：

$$
\boxed{\;
L_t \;=\; \sum_{o \in \mathcal{O}_t} \tilde w_t(o)\;
\mathrm{KL}\bigl(\tilde p_{y_t,o} \Vert \tilde q_{y_t,o}\bigr),
\qquad
\tilde w_t(o) = \mathrm{sg}\!\left[\frac{p_t(o)}{1 - p_t(y_t)}\right]
\;}
$$

上式是方法本身。尾部块作为普通对手参与：它那一对的 margin 是
$\log p_t(y_t) - \log p_t(\tau_t)$，实现里用 `logsumexp` + `log1mexp` 从已有的 top-$k$
log-prob 直接算出。两个退化性质说明这个形式没有丢东西：

- **$K = 0$ 时**对手只剩 $\tau_t$，此时 $\tilde p_{y_t,\tau_t}(y_t) = p_t(y_t)$，loss 精确退化为
  目标 token 的校准项 $\mathrm{KL}\bigl((p_t(y_t), 1 - p_t(y_t)) \Vert (q_t(y_t), 1 - q_t(y_t))\bigr)$，
  系数为 1。所以单项形式是"锚点校准 + 排序"的推广，不是删减。
- **锚点的绝对质量被自动钉住**：对手集合覆盖除 $y_t$ 外的全部质量，所有对决打平
  $\iff$ 所有比值 $p(y_t)/p(o)$ 与 teacher 一致 $\iff$ 在划分上 $p = q$，于是
  $p_t(y_t) = q_t(y_t)$ 是**定理**而不是需要单独加项去逼的约束（见 §1.2）。

每对用什么散度由 `l_apd.pair_divergence` 控制，一共三个取值：

| 取值 | 每对的量 | 性质 |
| --- | --- | --- |
| `reverse_kl` | $\mathrm{KL}(\tilde p \Vert \tilde q)$，即上式 | 有下界，$m = m_T$ 处梯度归零。**方法本身** |
| `forward_kl` | $\mathrm{KL}(\tilde q \Vert \tilde p)$，两个参数对调 | 同一个最优点，只改大误差如何修正（§1.3）。消融 |
| `log_ratio` | 只保留 $\log\frac{\tilde p(y_t)}{\tilde q(y_t)}$ 一项 | **不是散度**，无下界，梯度里没有 teacher，实测退化（§1.4）。消融 |

库默认值（`actor.yaml`）和启动脚本默认值都是 `reverse_kl`。

写成对求和的形式，可以看出它和 baseline OPD 的 $\sum_z \hat p(z)\log\frac{p(z)}{q(z)}$ 是同一个模板
——权重是第一个分布自己在该结果上的概率，每个结果配一个对数比值，没有额外的交叉熵项：

$$
\mathrm{KL}\bigl(\tilde p_{y_t,o} \Vert \tilde q_{y_t,o}\bigr)
= \sum_{v \in \{y_t,\,o\}} \tilde p_{y_t,o}(v)\,\log\frac{\tilde p_{y_t,o}(v)}{\tilde q_{y_t,o}(v)}
$$

区别只在求和跑几项：OPD 的支撑是 top-$k$ 共 $k$ 个结果，L-APD 的一对只有 2 个结果。
常见的 $a\log\frac{a}{b} + (1-a)\log\frac{1-a}{1-b}$ 只是上式在两点支撑上的展开，那个 $1-a$
就是对手 $o$ 自己的那份归一化概率 $\tilde p_{y_t,o}(o)$，不是交叉熵的残留。

尾部块作为对手时形式与其它 $K$ 项完全一致——代码里就是把它作为一列追加进候选张量，归一化、
散度、诊断全部复用同一套逻辑，**不是一个带系数的附加 loss**，没有自由超参。所有权重加起来
恰好为 1（诊断 `actor/l_apd_candidate_weight_sum` 就是在查这一点），尾部那一列的权重
$\mathrm{sg}[p_t(\tau_t)/(1 - p_t(y_t))]$ 记在 `actor/l_apd_tail_weight`——它随 student 尾部
质量涨落，尾部一胀权重就变大，这正是闭环排水的入口（teacher 权重消融下该列是
$q_t(\tau_t)/(1-q_t(y_t))$，实测约 $0.02\sim0.14$，不随 student 变化）。

**历史形式（现为消融 `complement_candidate`）。** 早期版本是两项式：具名候选对决加一个
"补集对手" $\perp$（除 $y_t$ 外的**全体** token 打包，含具名候选自己），后者恰好是目标 token
的逆向 KL $\mathrm{KL}\bigl((p(y_t), 1-p(y_t)) \Vert (q(y_t), 1-q(y_t))\bigr)$，权重
$q_t(y_t)/Z_t$，其中 $Z_t = q_t(y_t) + \sum_{z\in\mathcal V_t} q_t(z)$ 是含锚点的归一化子。它的问题是一次精确的**重复计数**：补集里含着全部具名候选，每个候选在 loss 里
出现两次——一次具名对决、一次藏在补集里——而藏着的那次拿走了约 $0.68$ 的权重，多数位置上
loss 在给一个大体已满足的校准约束反复付费，排序监督被稀释。单项形式下每个非锚点 token 恰好
出现一次，这个错误**在公式里写不出来**；而锚点校准并没有丢（见上方 $K = 0$ 退化与 §1.2）。
真实模型测量：换成单项形式后，聚合列的梯度份额从 0.378 降到 0.024，第 3 名以后候选拿到的梯度
份额从 0.173 升到 0.213，40 步蒸馏小验证里留出集全词表 KL 与两项式打平、熵更健康
（0.760 vs 0.746）。

**为什么求和不能省。** $\log p/q$ 只是 KL 的被积项，$\mathrm{KL}$ 是它在 $\tilde p$ 下的期望；
只留 $\log\frac{\tilde p(y)}{\tilde q(y)}$ 那一项、丢掉 $v = z$ 那一项，得到的东西就不是散度，
**没有下界**，最小化它等于无脑压低候选的 $\log p$（实测 loss 一路跌向 $-\infty$、top-16 总质量
被压到 $4\times 10^{-5}$）。OPD baseline 里那个只有 $\log p/q$ 的表达式之所以能用，是因为它是
**reward**（在 `no_grad` 下算，梯度由 policy gradient 从 $\log\pi$ 那边来），不是可微 loss。
成对比较的结果空间只有两个元素，所以这里的期望求和展开**恰好只有两项**，这就是上式的全部内容。

- 权重 $\tilde w_t$（虽由 student 质量算出，但 detach 成常数）与 teacher 侧的 $\tilde q$ 均为
  stop-gradient，梯度只从每对的 student margin 流过：权重决定每场对决多大声，不提供第二条
  梯度通路（否则 loss 可以靠把质量从没排对的列上挪走来作弊，而不是真的去修排序）；
- 序列级损失是对有效 response token 求平均（`loss_agg_mode`，默认 `token-mean`）。

**实现上的两个要点**：

1. 逆向下 student 持有熵项，所以 loss **本身就是 KL**，`actor/l_apd_loss` 会收敛到 0，和
   `actor/l_apd_pair_kl` 相等。正向下代码用 `binary_cross_entropy_with_logits`，算的是交叉熵，
   比 KL 多一个 teacher 侧熵（student 无关的常数，梯度相同），此时只有 `actor/l_apd_pair_kl`
   是纯 KL。两个方向的 $\mathrm{KL}$ 都用 log-sigmoid 写成，任意实数 margin 下都有限。
2. softmax 归一化项在 logit 差里会抵消，即 $T(y) - T(z) = \log q(y) - \log q(z)$。因此所有
   成对 margin 都能直接从框架已有的 top-$k$ log-prob 读出，**不需要传输或重算原始 logits，
   也不需要额外的 teacher forward**。每次更新仍然只有一次 student forward，显存与吞吐和 OPD
   baseline 完全一致。

### 1.1 候选来源与权重来源

这是两个独立的开关，分工明确：**id 决定谁有名字，权重决定谁有音量**。

默认候选取 **teacher top-$k$**（`candidate_source=teacher`）。用 student 自己的 top-$k$
（baseline 的 `top_k_strategy=only_stu` 打分的那一组，`candidate_source=student` 可切回做对照）
有一个结构性盲区：锚点 99.4% 落在 student top-$k$ 内，于是每个对手都是 student 本来就排得
靠前的 token，loss 只能在 student 已偏好的集合内重排，**永远看不到 teacher 想要、但被 student
排出 top-$k$ 之外的 token**——一条 run 里 `pairwise_agreement` 从 step 140 起停在 0.95 不再
上升就是这个原因。teacher top-$k$ 携带着约 1.5% 落在 student top-$k$ 之外的 teacher 质量，
正是双方仍有分歧的地方。

权重则取 **student 条件质量**（`weight_source=student`，§1 的定义）。两个来源搭配后，两类
关键 token 都被正确处理：student 的坏爱好（$p$ 大 $q\approx 0$，不在 teacher top-$k$ 里）落进
student 视角质量很大的尾部块，权重自动放大、被 teacher margin 重锤；teacher 的心头好（$q$ 大
$p\approx 0$）有自己的具名列，指名道姓地拉升——起步权重小，但 $p(o)$ 一涨权重自动跟上。
把 id 也换成 student 会毁掉后者（好 token 被聚合进尾部、信息销毁），把权重换回 teacher
则是开环消融（§1、§10）。

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

| | 聚合对手 | 该项的含义 | 权重（自动定标，加权侧质量） |
| --- | --- | --- | --- |
| `tail_candidate=True`（**默认，方法本身**） | 尾部块 $\tau_t$：top-$k$ 之外的十几万个 token，用 `logsumexp` + `log1mexp` 打包 | 锚点相对整个尾部应排多高 | $\mathrm{sg}[p(\tau_t)/(1 - p(y_t))]$（teacher 消融下 $q(\tau_t)/(1-q(y_t))$，实测约 $0.02\sim0.14$） |
| `complement_candidate=True`（消融） | $y_t$ 的补集 $\perp$，即"除 $y_t$ 外的全体"，**含具名候选自己** | $\mathrm{KL}(p(y_t) \Vert q(y_t))$，即目标 token 的 loss | 加权侧锚点质量归一化后所得（teacher 消融下 $q(y_t)/Z_t$，实测约 $0.65\sim0.68$） |

两者都是**同一个成对构造**在一个聚合对手上的特例，权重都由加权侧质量自动决定，
所以都没有自由系数。区别在对手集合的结构：尾部块让对手集合
$\lbrace z_1, \dots, z_k, \tau_t \rbrace$ 构成"非 $y_t$"的真正**划分**，每个非锚点 token 恰好
出现一次；补集则与具名候选 $z_j$ **重叠**——候选们在 loss 里被数了两次，且藏在补集里的那次
拿走约 0.68 的权重。

选划分（tail）而不选补集，就是 §1 讲过的重复计数问题的解。补集形式曾是默认，它当时的三条
理由现在都有了更好的答案：

1. *"归一化范围与 baseline 的 16 个 id 对齐"*——对齐的代价是重复计数；单项形式的归一化分母
   $1 - q(y_t)$ 是"挑战者必须是真备选"这个定义自动给出的，不是设计选择。
2. *"补集修掉了权重归一化的 eps 下限"*——尾部块同样修掉：`_log1mexp` 给分母兜了 $\sim 10^{-6}$
   的地板，且梯度直通（§9）。
3. *"目标在排序/校准两种模式间自动切换"*——单项形式同样自动切换且不付重复计数的代价：
   $q(y_t) \to 1$ 时具名候选权重趋 0、尾部对决（此时就是锚点校准，见 $K=0$ 退化）接管；
   $q(y_t) \to 0$ 时几乎全是排序信号。

真实模型上换成划分后：聚合列梯度份额 0.378 → 0.024，第 3 名以后候选的梯度份额
0.173 → 0.213，grad_norm 0.95x（无需任何补偿系数），40 步蒸馏小验证里留出集全词表 KL 与
补集形式打平、熵更健康。约 29% 的位置候选集覆盖质量撞 float32 分辨极限
（`actor/l_apd_tail_saturated`），直通估计保证这些位置梯度照常回传、无 NaN（§9）。

顺带一提，框架里没有其它项在约束这个方向：`USE_KL` 默认 `False`，而且 ref-KL 那段代码在
`update_policy` 的非 L-APD 分支里；`entropy_coeff` 也是 0。

### 1.3 梯度与方向的选择

记 $m = S(y_t) - S(o)$ 为 student margin、$m_T = T(y_t) - T(o)$ 为 teacher margin。两个方向对
每个成对 margin 的梯度都有闭式：

逆向（`reverse_kl`）：

$$
\frac{\partial L_t}{\partial m}
\;=\; \tilde w_t(o)\,\sigma'(m)\,\bigl(m - m_T\bigr)
$$

正向（`forward_kl`）：

$$
\frac{\partial L_t}{\partial m}
\;=\; \tilde w_t(o)\,\bigl(\tilde p_{y_t,o}(y_t) - \tilde q_{y_t,o}(y_t)\bigr)
$$

逆向就是**直接对齐 margin**：梯度正比于 $m$ 与 $m_T$ 之差，由 $\sigma'(m)$ 门控。两者都在
$m = m_T$ 处归零，而且在最优点附近**一阶一致**（$\delta = 10^{-3}$ 时两者梯度都等于
$\sigma'(m_T)\delta$，相对差 $10^{-3}$），所以这个选择**只影响大误差如何被修正**：

| $m$（$m_T = -4$，student 越来越自信地站错边） | 逆向 loss | 正向 loss | 逆向梯度 | 正向梯度 |
|---:|---:|---:|---:|---:|
| 0 | 1.33 | 0.60 | 1.0e+0 | 0.482 |
| 8 | 4.01 | 7.77 | 4.0e-3 | 0.982 |
| 30 | **4.018** | 29.37 | **3.2e-12** | 0.982 |

逆向 loss 封顶在 $-\log \tilde q_{y_t,o}(y_t)$，自信站错边的代价有界、梯度按 $\sigma'(m)$ 指数衰减；正向无界，
梯度饱和到 $\pm 1$ 但不消失。这也是唯一需要注意的取舍：$\sigma'(m)$ 因子意味着已分出胜负的
对决（不管方向对错）梯度都很小——这是成对 Bernoulli 几何的内在性质，对**已经过度自信**的
锚点（$p_t(y_t) \to 1$ 时尾部对决的 margin 很大）没有拉回力，虽然 §1.2 的可辨识性在理论上
仍成立（最优点唯一）。若训练中低概率候选出现梯度饥饿，`L_APD_PAIR_DIVERGENCE=forward_kl`
是同一形式下的现成退路（还是 Bernoulli KL，不引入新参数）。
`test_reverse_kl_is_bounded_where_forward_diverges` 锁住了上表的定性行为。

### 1.4 `log_ratio`：只保留一项的消融

`L_APD_PAIR_DIVERGENCE=log_ratio` 把每一对的 $\mathrm{KL}$ 换成只有 $v = y_t$ 那一项的裸对数比值，
即 §1 的式子变成

$$
\tilde L_t
\;=\; \sum_{o \in \mathcal{O}_t} \tilde w_t(o)\,
\log \frac{r_S(y_t,o)}{r_T(y_t,o)},
\qquad r_S = \tilde p_{y_t,o}(y_t),\; r_T = \tilde q_{y_t,o}(y_t)
$$

（历史上出问题的那条 run 用的是补集消融的两项式写法，补集那一列自动退化成锚点的裸对数比值
$\log\frac{p_t(y_t)}{q_t(y_t)}$，因为它的 margin 让两个受限概率恰好是 $p_t(y_t)$ 和 $q_t(y_t)$。）
**这不是散度**，只能作为消融跑：teacher 侧是 stop-gradient 的加性常数，所以它对每个成对
margin 的梯度是

$$
\frac{\partial \tilde L_t}{\partial m} \;=\; \tilde w_t(o)\,\bigl(1 - \sigma(m)\bigr) \;>\; 0
$$

恒为正、永不归零，而且式子里根本没有 $m_T$ —— 完全看不到 teacher。$m_T$ 只改变 loss 的数值，
不改变任何一个梯度分量，teacher 的信息只剩下**权重**和**候选集**两个入口。目标因此单调、无下界。

**锚点固定时**，最小化它就是把 $p_t(y_t)$ 推向 0。200 词表上从 $p = q$ 出发做 300 步 SGD，
loss 跌到 $-405$（仍在跌）、$p(y_t)$ 归零、$\mathrm{KL}(p \Vert q)$ 从 0 涨到 0.75；
`test_log_ratio_has_no_stationary_point` 锁住了这个行为，
`test_log_ratio_matches_the_two_part_bare_form` 锁住上式本身。

**但训练是 on-policy 的，锚点每步都从当前策略重采样，实际动力学恰好相反。** 每步压低刚采到的
token，会让概率质量不断从被采样处逃走、挤进一个越来越窄的集合；一旦集中，采样就总落在那个集合
里，于是 $p_t(y_t)$ 反而变大。400 词表上从 $p = q$ 出发、每步重采样 256 个锚点做 400 步 SGD：

| | $H(p)$ | $E[p(y_t)]$ | $\mathrm{KL}(p \Vert q)$ | loss |
| --- | --- | --- | --- | --- |
| `log_ratio` | 4.755 → **1.030** | 0.034 → **0.486** | 0 → **1.212** | 0 → **+0.32** |
| `reverse_kl` | 4.755 → 4.755 | 0.036 → 0.036 | 0 → 0 | 0 |

`reverse_kl` 从 $p = q$ 出发是精确不动点，`log_ratio` 则熵塌缩、$\mathrm{KL}$ 反向增大，而且 loss
**在涨**而不是发散到 $-\infty$。真实训练完全复现了这一点：一条 138 步的 run 里 `actor/entropy`
0.66 → 0.04、`actor/l_apd_student_anchor_prob` 0.71 → 0.98、`actor/l_apd_anchor_kl` 0.19 → 1.97，
`response_length/clip_ratio` 涨到 0.73（模型进入重复不终止），5 个 benchmark 的 pass@16 在 step 60
见顶 0.564 后掉到 0.513。所以排查时该盯的是**熵和 `anchor_kl`**，不是 `actor/pg_loss`。

同一个量当 **reward** 用是成立的，那就是 OPD baseline：`dp_actor.py` 的 `only_stu` 分支算
`rm_scores = -(S_logp - T_on_S) * w`，在 `no_grad` 下作为 $\nabla \log \pi$ 的系数，本身不被求导，
所以单调性无关。想要"一项版"的可微目标，直接对齐 margin 的 $\tfrac12 (m - m_T)^2$ 才是有下界的选择。

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

不传 `L_APD_PAIR_DIVERGENCE` 就是脚本默认的 `reverse_kl`，即 §1 那个方法本身。

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
| `L_APD_CANDIDATE_SOURCE` | `...l_apd.candidate_source` | `teacher` | 候选来源：teacher top-k（方法本身，§1.1）或 student top-k（对照） |
| `L_APD_TAIL_CANDIDATE` | `...l_apd.tail_candidate` | `True` | 聚合对手取尾部块"top-k 之外"，对手集合成为真划分（方法本身，§1.2），优先级高于补集 |
| `L_APD_COMPLEMENT_CANDIDATE` | `...l_apd.complement_candidate` | `False` | 聚合对手取"`y_t` 的补集"（历史两项式，重复计数，消融），见 §1.2 |
| `L_APD_NORMALIZE_WEIGHTS` | `...l_apd.normalize_weights` | `True` | 权重按自身和归一化；tail 模式下与除以加权侧的 `1 − 锚点质量` 浮点等价 |
| `L_APD_WEIGHT_SOURCE` | `...l_apd.weight_source` | `student` | 权重用谁的质量：`student` 是方法本身（闭环、与逆向成对 KL 方向一致，§1/§1.1），`teacher` 是历史开环加权，保留为消融 |
| `L_APD_PAIR_DIVERGENCE` | `...l_apd.pair_divergence` | `reverse_kl` | 每对用什么散度。`reverse_kl` 是方法本身（§1.3），`forward_kl` 是方向对照，`log_ratio` 是只保留一项的裸对数比值消融、非散度且实测退化（§1.4）。库默认与脚本默认一致 |

> 两个聚合对手**至少要开一个**。全关掉时锚点概率不可辨识（§1.2），只用于复现那个退化情形。

消融示例：

```bash
# KL 方向换成正向：自信站错边的 token 保留满强度梯度，代价是 loss 无界
L_APD_PAIR_DIVERGENCE=forward_kl MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 历史两项式：聚合对手换回补集（重复计数，锚点列吃 ~0.68 权重）
L_APD_TAIL_CANDIDATE=False L_APD_COMPLEMENT_CANDIDATE=True MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 候选改用 student top-16（锚点 99.4% 在内，看不到被 student 丢掉的 token）
L_APD_CANDIDATE_SOURCE=student MODEL_ROOT=/input0/models \
bash experiments_scripts/l-apd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh

# 权重换回 teacher 条件质量（开环消融：预算按 teacher 偏好定死，不看 student 的错在哪）
L_APD_WEIGHT_SOURCE=teacher MODEL_ROOT=/input0/models \
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
| `actor/pg_loss` | L-APD 目标值（复用了原字段名），已乘上 `loss_scale_factor`，所以数值不能直接和 `pair_kl` 对齐。`log_ratio` 下它不是收敛指标——on-policy 时它会**往上涨**而不是往负走（§1.4） |
| `actor/l_apd_pair_kl` | teacher 加权的成对 KL，含聚合对手那一项，**永远是诚实的 KL**（正向下已扣掉 teacher 熵，`log_ratio` 下另算一份逆向 KL）。逆向下它与 `actor/pg_loss` 相等；其余两种方向下它是唯一有界、$p = q$ 时归零的收敛指标 |
| `actor/l_apd_pairwise_agreement` | student 与 teacher 排序方向一致的加权比例 |
| `actor/l_apd_pairwise_gap` | 加权的 $\lvert \tilde p_{y_t,o}(y_t) - \tilde q_{y_t,o}(y_t) \rvert$，两侧成对概率的差距 |
| `actor/l_apd_teacher_anchor_prob` / `..._student_anchor_prob` | 锚点 token 上的 $q(y_t)$ / $p(y_t)$，两者是否收敛到一起是判断锚点项是否起效的主要观测量 |
| `actor/l_apd_tail_weight` | 尾部块拿到的权重（默认配置下记录）。student 加权（默认）下是 $\mathrm{sg}[p(\tau_t)/(1-p(y_t))]$，随 student 尾部质量涨落——它升高说明闭环排水正在加力；teacher 加权消融下是 $q(\tau_t)/(1-q(y_t))$，实测约 $0.02\sim0.14$ |
| `actor/l_apd_teacher_tail_prob` / `..._student_tail_prob` | 候选集之外的尾部质量（默认配置下记录） |
| `actor/l_apd_tail_saturated` | 候选集覆盖质量已到 float32 分辨极限、尾部 margin 被封顶的位置占比（实测约 0.29）。梯度仍照常回传（§9），这个指标只是让被封顶的规模可见 |
| `actor/l_apd_anchor_kl` | 锚点项的 KL（方向随 `pair_divergence`），$p(y_t) = q(y_t)$ 时恰好为 0；仅补集消融（`complement_candidate` 生效）时记录 |
| `actor/l_apd_anchor_weight` | 补集对手拿到的权重（加权侧锚点质量归一化后所得，teacher 加权下即 $q(y_t)/Z_t$）；仅补集消融时记录 |
| `actor/l_apd_anchor_saturated` | $p(y_t) > 1 - 10^{-6}$ 的位置占比，即 float32 已无法分辨 $1 - p(y_t)$、补集 margin 被封顶的那部分；仅补集消融时记录 |
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
$\mathrm{KL}(\tilde p \Vert \tilde q)$ 且与 `pair_kl` 诊断相等；逆向 loss 在自信站错边处有界而
正向发散（锁住 §1.3 那张表的定性行为）；`reverse_kl` 确实是库的默认方向；未知方向名会报错；
$p = q$ 时两个方向梯度都为 0；正向下全词表候选与定义式逐元素相等、tail 候选只含一个 token 时
与全词表 loss 精确相等；padding 位置无 loss、无 NaN；候选权重的归一化性质；§1.2 的可辨识性
（只有真实 token 对手时 loss 对"质量在 top-$k$ 与尾部之间如何分配"完全不变，而两种聚合对手
都能破掉这个不变性）；补集对手那一项的权重逐元素等于 $q(y_t)/Z_t$、且含它的权重和恒为 1；
student 加权确实是库默认、权重逐元素等于 $\mathrm{sg}[p(o)/(1-p(y_t))]$、且是 stop-gradient
（autograd 梯度与"权重视为常数"的解析式逐元素相等）；$K=0$ 时两种加权下 loss 都精确退化为
锚点 Bernoulli KL；未知 `weight_source` 会报错；
`log_ratio` 逐元素等于 §1.4 那个两段式，且它在 $p = q$ 处仍有非零梯度、50 步 SGD 就把 loss
推到 $-10$ 以下并把 $p(y_t)$ 压到 $e^{-10}$ 以下（锁住"它不是散度"这个性质）。

另有三个针对补集列数值饱和区的回归测试（对应 §9 那条）：`test_overconfident_anchor_keeps_its_gradient`
锁住"跨过 $p(y_t) = 1 - 10^{-6}$ 时梯度不出现断崖"——修复前它会在一步之内掉 6 个数量级；
`test_saturated_anchor_is_free_of_nan` 锁住 $p(y_t) = 1$ 时 loss 与梯度仍然有限；
`test_anchor_saturated_reports_the_capped_share` 锁住 `anchor_saturated` 诊断的取值。

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
  只有真实 token 对手时，加权侧在采样 token 上几乎确定（锚点质量 $> 1 - 10^{-6}$）的位置竞争者
  总质量小于 `1e-6`，下限生效、权重被压到 0、梯度丢失（teacher 加权下实测
  `candidate_weight_sum ≈ 0.94`，约 6% 的 token；student 加权下正是 student 最过度自信、
  最该被修正的位置）。默认配置不受影响：tail 对手由 `_log1mexp` 的 clamp 兜了一个 `~1e-6`
  的地板，补集对手让分母至少为锚点质量。
- **聚合对手的 margin 在覆盖质量趋 1 处被封顶，但梯度照常回传**。`_log1mexp` 必须把输入截在
  $-10^{-6}$ 才能让 $\log(1 - e^x)$ 在 $x = 0$ 处有限，而 float32 的 log-prob 本来也只能分辨到
  这个量级。默认配置下这发生在候选集几乎覆盖全部 student 质量的位置（实测约 29%，看
  `actor/l_apd_tail_saturated`；补集消融下对应 $p(y_t) \to 1$，看 `anchor_saturated`）。
  关键是这个截断对 autograd **透明**（straight-through）：如果让它挡住梯度，聚合列会
  在 student 最过度自信的位置——正是 on-policy 蒸馏存在的理由——被静默清零，而且是断崖式的
  （实测跨过阈值一步之内掉 6 个数量级）。放行是安全的，因为成对的 $r(1-r)$ 因子恰好抵消
  $1/(1-p)$ 的爆炸，合成梯度只按 $\log\frac{1}{1-p}$ 增长；封顶后它停在一个有限值而不是跌到 0。
- **不要打开 `TORCH_NCCL_BLOCKING_WAIT`**。它和 vLLM 的 CUDA graph capture 会死锁：8 个 rank
  全堵在 torch 的 `ProcessGroupNCCL::waitForPendingWorks()` 里，而那个等待循环没有超时，
  表现是显存占满、GPU 利用率 0%、日志停在
  `Waiting for pending NCCL work to finish before starting graph capture` 且永不恢复。
  `common.sh` 已默认置 0 并改用 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`。若确实需要它，
  必须同时加 `actor_rollout_ref.rollout.enforce_eager=True` 绕开 graph capture。
- **恢复训练**：experiment name 带时间戳，所以每次启动都是新目录。要接着上次跑，需显式指定
  `trainer.default_local_dir=<旧 checkpoint 目录> trainer.resume_mode=auto`。
- 单节点默认 8 卡（`trainer.n_gpus_per_node=8`），卡数不同时请追加 override。

## 10. 更新记录

### 2026-08-12：权重来源改为 student 条件质量

- **权重从 teacher 条件分布改为 student 条件分布**（`weight_source=student` 成为默认，公式见
  §1：$\tilde w_t(o) = \mathrm{sg}[p_t(o)/(1-p_t(y_t))]$，只换权重的来源，loss 结构、候选集、
  散度方向一字未动，teacher 加权保留为消融 `L_APD_WEIGHT_SOURCE=teacher`）。动机来自对
  teacher 加权 62 步 run 的诊断：它是**开环**的——预算按 teacher 偏好定死，尾部列只拿
  4–11% 的权重，student 往尾部漏质量时得不到额外的拉回力，实测 `student_tail_prob` 冲到
  teacher 的约 3 倍、`actor/entropy` 从 0.66 过冲到 ~1.0（baseline OPD 单调下行），step 60
  验证分落后 baseline。student 加权是**闭环**的：尾部一胀，尾部列权重自动同步放大直到质量
  被排干，过冲被源头抑制；对"student 错爱、teacher 不要"的 token 同理。
- **方向一致性**：$\mathrm{KL}(p \Vert q)$ 的链式法则把条件格子按 *student* 条件质量加权——
  逆向的成对 KL 本就该配 student 权重，teacher 权重是从正向 KL 链式法则里借来的搭配。
- 单元测试新增 5 项：student 加权是库默认、权重逐元素等于 $\mathrm{sg}[p(o)/(1-p(y_t))]$、
  权重严格 stop-gradient（autograd 与"权重视为常数"的解析梯度逐元素相等）、$K=0$ 时两种加权
  都精确退化为锚点 Bernoulli KL、未知 `weight_source` 报错。启动脚本新增
  `L_APD_WEIGHT_SOURCE`，run 名加 `w_<source>` 标签。

### 2026-08-11：单项形式成为方法本身

- **loss 从两项式改为单项式**（`tail_candidate=True` 成为默认，`complement_candidate` 降级为
  消融）。旧两项式的补集对手把具名候选重复计数了一次，且重复的那份拿走约 0.68 的权重，多数
  位置上 loss 在给已满足的锚点校准反复付费。单项式把对手集合改成"具名候选 + 尾部块"的真划分，
  每个非锚点 token 恰好出现一次，权重就是 teacher 在真备选上的条件分布 $q(o)/(1-q(y_t))$，
  零个新超参；$K=0$ 时精确退化回原锚点项，锚点校准由可辨识性定理自动保证（§1、§1.2）。
  真实模型测量：聚合列梯度份额 0.378 → 0.024，第 3 名以后候选的梯度份额 0.173 → 0.213；
  40 步蒸馏小验证中留出集全词表 KL 与两项式打平、熵更健康（0.760 vs 0.746），两者都快于
  OPD 式策略梯度（KL 0.105/0.104 vs 0.119）。
- **候选来源默认 `student` → `teacher`**（§1.1）。student top-k 里 99.4% 含锚点，loss 只能
  在 student 已偏好的集合内重排，看不到被 student 丢出 top-k 的 teacher 高分 token；一条 run
  的 `pairwise_agreement` 从 step 140 起停在 0.95 不再上升即此原因。
- **修复 `_log1mexp` 截断阻断梯度**：截断改为对 autograd 透明（straight-through），聚合对手
  列在 student 过度自信的位置不再被断崖式清零（修复前跨过阈值一步掉 6 个数量级），§9 有
  安全性论证，三个回归测试锁定。
- **`pair_divergence` 默认回到 `reverse_kl`**，`log_ratio` 定性为消融并写明退化机制（§1.4）：
  它不是散度、梯度里没有 teacher，on-policy 下熵塌缩（实测 0.66 → 0.04）、要缩小的 KL 反而
  涨了 4 倍。

### 2026-08-10：初版

- L-APD 初版实现：锚点式成对蒸馏，reverse Bernoulli KL，两项式（具名候选 + 补集对手），
  权重按 teacher 概率在含锚点的 top-16 上归一化。CPU 单元测试、启动脚本、W&B 跟踪。
