# Preference-OPD

Preference-OPD separates the ordinary RKL-OPD signal into teacher preference
and teacher confidence components.

For every response position and its Top-K candidate set, it fits

\[
a^*=\arg\min_{a>0}\mathrm{KL}\left(T_C\,\|\,\mathrm{softmax}(a\log S_C)\right)
\]

and constructs a calibrated distribution `Q` that preserves the teacher's
total probability mass on the candidate set. The exact OPD decomposition is

\[
\log T-\log S=(\log T-\log Q)+(\log Q-\log S).
\]

Training uses

\[
A=(\log T-\log Q)+\beta(\log Q-\log S).
\]

- `beta=1`: standard OPD, exactly.
- `beta=0`: preference-only OPD.
- `0<beta<1`: retain part of the teacher confidence signal.

Run the default `beta=0.25` experiment with:

```bash
nohup bash experiments_scripts/preference-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/preference-opd.log 2>&1 &
```

Run the pure preference-only version with:

```bash
PREFERENCE_OPD_CONFIDENCE_BETA=0 \
nohup bash experiments_scripts/preference-opd-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh \
  > meteor_run/preference-opd-beta0.log 2>&1 &
```

The implementation logs `preference_opd/*` diagnostics, including fitted
temperature, temperature-explained KL ratio, Top-1 agreement, preference and
confidence advantage magnitudes, and candidate probability mass.

The calibration is performed on the selected candidate set. With the default
`only_stu` strategy this is Student Top-K; changing the candidate strategy also
changes what “preference” means.
