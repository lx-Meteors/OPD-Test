# Chi2-OPD: safe linear reward extrapolation

Chi2-OPD is a drop-in Top-K OPD target that uses the same Student, Teacher,
and frozen Ref models as G-OPD. It adds no model and no rollout. The Ref is
evaluated on the existing Student Top-K IDs in one forward pass; that same pass
also supplies the sampled-token Ref log-probability, so it is not repeated.

For each Student state and candidate token, define

\[
r(a)=\log T(a)-\log R(a),\qquad
\widetilde r(a)=r(a)-\mathbb E_{a\sim T_K}[r(a)].
\]

The target density ratio is

\[
\frac{q_K(a)}{T_K(a)}=1+\kappa\widetilde r(a).
\]

Here \(T_K\) is the Teacher distribution conditioned on the shared Student
Top-K candidate set. The implementation preserves the Teacher's raw probability
mass on that set, so `kappa=0` is exactly the existing Top-K OPD target rather
than a renormalized approximation.

## Safety rule

With `adaptive_kappa=True`, every state receives the largest value no greater
than the requested `kappa` that keeps

\[
\text{min_density_ratio}\le q_K(a)/T_K(a)\le\text{max_density_ratio}.
\]

The default interval is `[0.1, 3.0]`. This guarantees positive target support
for reverse KL and prevents a noisy Teacher/Ref likelihood ratio from creating
an arbitrarily large target peak. If the requested value is already safe, it is
used unchanged.

## Run

```bash
nohup bash experiments_scripts/chi2-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/chi2-opd.log 2>&1 &
```

The default `kappa=0.25` is the first-order tangent counterpart of ExOPD
`lambda=1.25`. Override it without editing code:

```bash
CHI2_OPD_KAPPA=0.5 \
bash experiments_scripts/chi2-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh
```

For a strict baseline-equivalence check, run with `CHI2_OPD_KAPPA=0`.

## W&B diagnostics

- `chi2_opd/kappa_mean`, `kappa_min`: actual safe extrapolation strength.
- `chi2_opd/kappa_shrink_fraction`: fraction of states whose requested strength
  was reduced by the trust bound.
- `chi2_opd/density_ratio_min`, `density_ratio_max`: observed target/Teacher
  density-ratio range.
- `chi2_opd/implicit_reward_mean`, `implicit_reward_std`: Teacher/Ref reward
  statistics on the candidate set.
- `chi2_opd/target_shift_tv`: total-variation shift from Teacher to the linear
  target.
- `chi2_opd/student_target_kl`: current conditional Student-to-target KL.
- `chi2_opd/teacher_target_kl`: conditional Teacher-to-target KL, measuring the
  extrapolation itself.

The method intentionally requires `top_k_strategy=only_stu` and
`reward_weight_mode=student_p`. These constraints ensure all three models are
compared on identical candidates and the update remains a reverse-KL gradient.
It is intentionally rejected when L-APD, Prune-OPD, or Bridge-OPD is enabled,
so an experiment cannot silently mix objectives and become uninterpretable.
