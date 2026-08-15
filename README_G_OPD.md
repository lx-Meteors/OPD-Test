# G-OPD reproduction

This implementation follows the objective in [Learning beyond Teacher:
Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125)
and the authors' [official implementation](https://github.com/RUCBM/G-OPD).

For Student `S`, Teacher `T`, and frozen reference `R`, G-OPD optimizes

```text
J(S) = lambda * E_S[log(T / R)] - KL(S || R),
```

whose target distribution is

```text
q_lambda proportional to R * (T / R)^lambda.
```

Equivalently, a state-only normalizer can be omitted and the target log-score
on every shared candidate is

```text
log q_lambda = log R + lambda * (log T - log R).
```

- `lambda=0`: imitate the reference.
- `lambda=1`: exactly recover the repository's standard OPD target.
- `lambda>1`: extrapolate beyond the Teacher; the paper's default is `1.25`.

## Reference model

The official code supports two choices for `R`: the Student initialization in
its default mode, or the Teacher's pre-RL checkpoint for reward correction. The
provided script follows the default and uses the Student initialization. Set
`G_OPD_REFERENCE_MODEL_PATH` to the Teacher's pre-RL base if that is the
experiment you want to reproduce. Student, Teacher, and reference must share a
tokenizer/vocabulary in this Top-K implementation.

## Run

```bash
nohup bash experiments_scripts/g-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/g-opd.log 2>&1 &
```

To verify the reduction to ordinary OPD:

```bash
G_OPD_LAMBDA=1.0 bash experiments_scripts/g-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

This repository's OPD baseline estimates reverse KL over Student Top-K
candidates, so G-OPD uses the same candidate set and weighting for a controlled
comparison. Only the target changes. The paper's objective is unchanged, but
this Top-K estimator is specific to this repository rather than the official
sampled-action training implementation.

W&B receives `g_opd/lambda`, implicit-reward statistics, target shift, and
Student/Teacher-to-target KL diagnostics.
