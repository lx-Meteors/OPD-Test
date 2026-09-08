# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entropy-tempered extrapolation for G-OPD (ET-OPD).

Standard OPD uses the sampled-token reverse-KL advantage

    a_t = log T(y_t) - log S(y_t).

G-OPD adds a frozen extrapolation residual in log space,

    A_t = a_t + (lambda - 1) * (log T(y_t) - log R(y_t)),

whose fixed point is S* ∝ T^lambda R^(1-lambda). The log-space residual is
scale-free: a token with (T, R) = (0.008, 0.001) receives the same residual as
one with (0.8, 0.1), and a sampled EOS at a state where the RL teacher is
certain to continue (T ~ 1e-4, R ~ 0.08) receives a residual of -6 to -17 nats.
On sampled-token, return-free advantages that residual is the only channel
through which extrapolation can change response length, and it is one-sided:
it never rewards stopping (d(EOS) <= 0 at every completion point) and heavily
punishes it at every optional stop.

ET-OPD keeps the alignment term and moves the residual to Box-Cox (power)
space with a per-token exponent set by the teacher's entropy contribution on
the sampled token:

    A_t = a_t + (T(y_t)^alpha_t - R(y_t)^alpha_t) / alpha_t,
    alpha_t = 1 / (e * c_T(y_t)),      c_T(y) = -T(y) * log T(y).

Since -x log x <= 1/e on (0, 1], alpha_t >= 1 by construction with equality at
T = 1/e, the most contested probability, where the residual reduces to the
mass difference T - R. The comparison temperature 1/alpha_t = e * c_T(y_t) is
the token's normalised uncertainty contribution under the teacher: tokens the
teacher is certain about (T -> 1) or ignores (T -> 0) are compared at a very
cold temperature and receive (almost) no extrapolation. The zero set therefore
contains the EOS cliff at optional stops, the tokens neither model emphasises,
and the corridors, while decision tokens (T around 0.3-0.7) are extrapolated
at unit temperature.

Properties:
  * alpha_t >= 1; |residual| <= 1 / alpha_t <= 1.
  * The residual is free of S and sign-consistent with T - R, so the per-state
    fixed point S* ∝ T * exp(residual) / Z is unique with a tilt factor inside
    [1/e, e]; the alignment term keeps every closed-loop property of standard
    OPD (including its own anti-early-stop and confident-error corrections).
  * Only the three sampled-token log-probabilities are needed; no state
    scalar, position, vocabulary statistics or new wires.

The 1/alpha_t factor cannot be absorbed into the learning rate because
alpha_t varies per token; it is the factor that keeps near-certain tokens
(T^alpha stays O(1) for T -> 1) from dominating the residual. The constant e
normalises c_T by its maximum and is not a tunable.

Relative variant (alpha_source="ratio")
--------------------------------------
    alpha_t = c_T(y_t) / c_R(y_t)

The comparison temperature is the RATIO of the token's entropy contribution
under the teacher and under the base (e cancels; no constant). alpha -> 0
(the residual becomes G-OPD's log T - log R) exactly where RL *resolved* the
token's uncertainty relative to the base -- committed to it (T -> 1) or
abandoned it (T -> 0) -- which is where the useful sharpening lives (tail
thinning at confident states, top boosts). alpha -> infinity (residual -> 0)
where RL *created* uncertainty the base did not have -- new hedges, invented
continuations at optional stops, unsettling a base-certain stop -- which is
where the completion damage lives. Equivalent to a token-level
lambda_t = 1 + (lambda - 1) * h_t with h_t in (0, 1]; with
``residual_coef = lambda - 1`` (G-OPD's lambda) the method is G-OPD with an
adaptive exponent and nothing else changed. Numerically the Box-Cox form is
evaluated with expm1 so the alpha -> 0 limit is exact.

Probe metrics (``etopd/*``) follow the num/den convention: raw values, NOT
multiplied by ``loss_scale_factor``; ``*_num`` / ``*_den`` are per-micro-batch
sums, so the mean reduction over micro-batches preserves their ratio exactly.
See ``compute_etopd_probe_metrics`` for the readout map (force decomposition,
teacher-probability bins, same-token counterfactual residuals, termination
channel, beyond-teacher realisation, comparison temperature, depth profile).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ETOPD_C_FLOOR",
    "ALPHA_SOURCES",
    "entropy_contribution",
    "entropy_tempered_alpha",
    "entropy_tempered_residual",
    "compute_etopd_advantages",
    "compute_etopd_probe_metrics",
]

ALPHA_SOURCES = ("teacher", "reference", "ratio")

# Floor for the teacher's entropy contribution before inverting it. A sampled
# token with T == 1 exactly has c_T == 0; the floor makes alpha finite (~4e7),
# at which the residual is (1 - R^alpha) / alpha ~ 0 as intended.
ETOPD_C_FLOOR = 1e-8

_SEG_BOUNDS = (1024, 2048, 4096, 8192)
_T_LOW = 0.1
_T_HIGH = 0.9


def entropy_contribution(log_p: torch.Tensor, c_floor: float = ETOPD_C_FLOOR) -> torch.Tensor:
    """c(p) = -p log p in float32, floored so it can be inverted (c -> 0 at p -> 0 and p -> 1)."""
    lp = log_p.to(torch.float32)
    return (-(torch.exp(lp) * lp)).clamp_min(c_floor)


def entropy_tempered_alpha(
    teacher_log_prob: torch.Tensor,
    fixed_alpha: float = 0.0,
    c_floor: float = ETOPD_C_FLOOR,
    ref_log_prob: torch.Tensor | None = None,
    alpha_source: str = "teacher",
) -> torch.Tensor:
    """Per-token Box-Cox exponent.

    alpha_source:
        "teacher"    alpha = 1 / (e * c_T)      (ET-OPD; >= 1, == 1 at T = 1/e)
        "reference"  alpha = 1 / (e * c_R)      (temperature set by the base model)
        "ratio"      alpha = c_T / c_R          (relative entropy contribution; < 1 where RL
                                                 resolved uncertainty, > 1 where RL created it)

    Args:
        teacher_log_prob: log T(y_t) on the sampled tokens, any float dtype.
        fixed_alpha: if > 0, return a constant exponent instead (ablation:
            1.0 gives the L2 residual T - R; the alpha -> 0 limit is G-OPD).
        c_floor: numerical floor for the entropy contributions.
        ref_log_prob: log R(y_t); required for "reference" and "ratio".
        alpha_source: one of ALPHA_SOURCES.

    Returns:
        float32 tensor of the same shape.
    """
    log_t = teacher_log_prob.to(torch.float32)
    if fixed_alpha > 0:
        return torch.full_like(log_t, float(fixed_alpha))
    if alpha_source not in ALPHA_SOURCES:
        raise ValueError(f"alpha_source must be one of {ALPHA_SOURCES}, got {alpha_source!r}")
    if alpha_source == "teacher":
        return 1.0 / (math.e * entropy_contribution(log_t, c_floor))
    if ref_log_prob is None:
        raise ValueError(f"alpha_source={alpha_source!r} needs ref_log_prob")
    c_r = entropy_contribution(ref_log_prob, c_floor)
    if alpha_source == "reference":
        return 1.0 / (math.e * c_r)
    return entropy_contribution(log_t, c_floor) / c_r


def entropy_tempered_residual(
    teacher_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """(T^alpha - R^alpha) / alpha in float32.

    Two numerically equivalent forms are used depending on the regime:
      * alpha >= 1: (exp(a log T) - exp(a log R)) / a. a * log p <= 0 so nothing overflows; very
        negative arguments underflow to exactly 0, giving residual -> 0 for alpha -> infinity.
      * alpha < 1 (only with alpha_source="ratio"): (expm1(a log T) - expm1(a log R)) / a, which
        is exact in the alpha -> 0 limit where the residual tends to log T - log R and the exp
        form would cancel two numbers near 1.
    """
    log_t = teacher_log_prob.to(torch.float32)
    log_r = ref_log_prob.to(torch.float32)
    alpha = alpha.to(torch.float32)
    exp_form = (torch.exp(alpha * log_t) - torch.exp(alpha * log_r)) / alpha
    expm1_form = (torch.special.expm1(alpha * log_t) - torch.special.expm1(alpha * log_r)) / alpha
    return torch.where(alpha < 1.0, expm1_form, exp_form)


def compute_etopd_advantages(
    teacher_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    fixed_alpha: float = 0.0,
    alpha_source: str = "teacher",
    residual_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (advantages, align, residual, alpha) for the ET-OPD family.

    advantages = align + residual with align = log T - log S (the standard OPD
    sampled-token advantage) and residual = residual_coef * (T^alpha - R^alpha) / alpha.
    With alpha_source="ratio" and residual_coef = lambda - 1 this is G-OPD whose exponent
    adapts per token (G-OPD itself is the alpha -> 0 limit). All tensors are
    (bsz, response_len); advantages is cast back to the dtype of ``old_log_prob``.
    """
    align = teacher_log_prob.to(torch.float32) - old_log_prob.to(torch.float32)
    alpha = entropy_tempered_alpha(
        teacher_log_prob, fixed_alpha=fixed_alpha, ref_log_prob=ref_log_prob, alpha_source=alpha_source
    )
    residual = float(residual_coef) * entropy_tempered_residual(teacher_log_prob, ref_log_prob, alpha)
    advantages = (align + residual).to(old_log_prob.dtype)
    return advantages, align, residual, alpha


def compute_etopd_probe_metrics(
    align: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Probe suite for ET-OPD (raw values, num/den convention, one GPU sync).

    Every ``*_num`` / ``*_den`` pair is a per-micro-batch sum, so the mean
    reduction over micro-batches preserves the ratio exactly; divide the logged
    series in W&B. Names without num/den are already per-token means of the
    micro-batch (fine for reading, slightly biased under dynamic batch sizes).

    Force decomposition
      align_mean, residual_mean, residual_abs_mean, adv_abs_mean
        residual_mean is the net rent of the residual (G-OPD's plateau was
        ~0.025); adv_abs_mean is the distance from the per-token fixed point.
      rent_num / rent_den
        residual on agreement corridors (T > 0.5 and R > 0.5), the part of the
        rent no entropy gate can remove.

    Where the residual lands (teacher-probability bins)
      resabs_num_<bin> / resabs_den, res_num_<bin> / tok_num_<bin>, tok_num_<bin> / tok_den
        bins: tlow T<0.1 | t1 0.1-0.3 | t2 0.3-0.7 (decision) | t3 0.7-0.9 |
        thigh T>0.9, plus tmid = t1+t2+t3 for the pre-registered 3-bin split.
        Pre-registered: resabs share on tlow < 1%, thigh ~10%.

    Counterfactual residuals on the SAME tokens (design claims without extra arms)
      cf_log_*   G-OPD log-space residual d = log T - log R (per unit lambda-1)
      cf_l2_*    alpha == 1 residual T - R
      cf_noe_*   alpha' = e * alpha (the "drop e" variant)
      cf_noinv_* T^alpha - R^alpha (the "drop 1/alpha" variant)
      each with abs_den, num (signed sum), abs_num_tlow / abs_num_thigh /
      abs_num_t2 and eos_num. Offline expectation: cf_log puts ~24% of |d| on
      tlow and -1.5..-4.4 on the sampled EOS; cf_noinv puts >80% on thigh.

    Termination channel
      eos_den, eos_res_num, eos_align_num, eos_logT_num, eos_logR_num, eos_logS_num
        the last valid token of terminated rows is the sampled EOS. Predicted
        eos_res_num / eos_den ~= 0. eos_logT vs eos_logR tracks the EOS cliff
        (teacher continues where the student stopped) and whether stops become
        agreed (eos_logT -> 0) as training proceeds.
      capped_tok_den, capped_res_num, capped_align_num
        full-buffer (runaway) rows; complement = terminated rows.

    Beyond-teacher realisation
      beyond_num_<bin> / tok_num_<bin>, beyond_num / tok_den
        sum of sign(T - R) * (log S - log T): > 0 means the student already sits
        past the teacher in the RL-edit direction. Prediction: grows on t2/t3,
        stays ~0 on tlow/thigh (G-OPD would grow it everywhere incl. tlow).
      opposed_num / opposed_den
        tokens where alignment and residual pull in opposite directions (the
        student has overshot the teacher and the residual holds it there),
        among tokens with |residual| > 1e-4.

    Comparison temperature
      log_alpha_mean, log_alpha_p10/p50/p90, alpha_ge2_frac, alpha_lt1_frac
        alpha >= 2 means the residual is at most half of the mass difference;
        alpha < 1 ("hot", log-like) only occurs with alpha_source="ratio".

    Edit type (the claim behind alpha_source="ratio")
      tok_num_<type>, res_num_<type>, resabs_num_<type>, beyond_num_<type>, cf_log_abs_num_<type>
        types: resolution (c_T < c_R: RL committed to or abandoned the token),
        creation (T > R and c_T > c_R: RL raised a token the base found less contested --
        hedges, invented continuations, rare boosts), uncommit (T < R and c_T > c_R: RL
        made a base-certain token uncertain, e.g. contested stops), tie (c_T == c_R, zero
        residual). The four partition the valid tokens. Offline (74k val tokens) the
        "ratio" residual puts 82% of |f| on resolution, 8% on creation, 10% on uncommit
        (token shares 68 / 9 / 15 / 9%); G-OPD's |d| splits 53 / 39 / 8%. The
        cf_log_abs_num_<type> / cf_log_abs_den columns reproduce that split in-run.
      cf_aT_*  the ET-OPD (alpha = 1/(e c_T)) residual as a same-token counterfactual, so
        the run also logs what the previous arm would have applied.
      cf_ratio_*  the unscaled (coefficient 1) ratio residual; in a "ratio" run the applied
        residual is residual_coef times this.

    Depth profile
      res_num_seg{k}, resabs_num_seg{k}, tok_den_seg{k}
        [0,1k), [1k,2k), [2k,4k), [4k,8k), [8k,+).
    """
    if align.dim() != 2:
        return {}
    a = align.float()
    r = residual.float()
    al = alpha.float()
    log_t = teacher_log_prob.to(torch.float32)
    log_r = ref_log_prob.to(torch.float32)
    log_s = log_t - a
    mask = response_mask.to(torch.float32)
    n_tok = mask.sum().clamp_min(1.0)
    t_prob = torch.exp(log_t)
    r_prob = torch.exp(log_r)
    adv = a + r

    names: list[str] = []
    vals: list[torch.Tensor] = []

    def put(name: str, value: torch.Tensor) -> None:
        names.append(f"etopd/{name}")
        vals.append(value.reshape(()))

    def msum(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        return (x * sel).sum()

    # --- force decomposition ------------------------------------------------------
    put("align_mean", msum(a, mask) / n_tok)
    put("residual_mean", msum(r, mask) / n_tok)
    put("residual_abs_mean", msum(r.abs(), mask) / n_tok)
    put("adv_abs_mean", msum(adv.abs(), mask) / n_tok)
    put("residual_num", msum(r, mask))
    put("resabs_den", msum(r.abs(), mask))
    put("tok_den", mask.sum())
    agree = ((t_prob > 0.5) & (r_prob > 0.5)).float() * mask
    put("rent_num", msum(r, agree))
    put("rent_den", agree.sum())

    # --- comparison temperature ---------------------------------------------------
    log_al = torch.log(al)
    put("log_alpha_mean", msum(log_al, mask) / n_tok)
    put("alpha_ge2_frac", msum((al >= 2.0).float(), mask) / n_tok)
    put("alpha_lt1_frac", msum((al < 1.0).float(), mask) / n_tok)
    valid_log_al = log_al[mask > 0]
    if valid_log_al.numel() > 0:
        qs = torch.quantile(valid_log_al, torch.tensor([0.1, 0.5, 0.9], device=valid_log_al.device))
        put("log_alpha_p10", qs[0])
        put("log_alpha_p50", qs[1])
        put("log_alpha_p90", qs[2])

    # --- teacher-probability bins ---------------------------------------------------
    beyond = torch.sign(t_prob - r_prob) * (log_s - log_t)
    bins = {
        "tlow": (t_prob < _T_LOW).float() * mask,
        "t1": ((t_prob >= _T_LOW) & (t_prob < 0.3)).float() * mask,
        "t2": ((t_prob >= 0.3) & (t_prob <= 0.7)).float() * mask,
        "t3": ((t_prob > 0.7) & (t_prob <= _T_HIGH)).float() * mask,
        "thigh": (t_prob > _T_HIGH).float() * mask,
    }
    bins["tmid"] = bins["t1"] + bins["t2"] + bins["t3"]
    for name, sel in bins.items():
        put(f"tok_num_{name}", sel.sum())
        put(f"resabs_num_{name}", msum(r.abs(), sel))
        put(f"res_num_{name}", msum(r, sel))
        put(f"beyond_num_{name}", msum(beyond, sel))
    put("beyond_num", msum(beyond, mask))
    active = (r.abs() > 1e-4).float() * mask
    put("opposed_den", active.sum())
    put("opposed_num", msum((a * r < 0).float(), active))

    # --- terminal token of terminated rows (the sampled EOS) ------------------------
    lengths = mask.sum(dim=-1)
    resp_len = mask.shape[-1]
    capped_row = (lengths >= resp_len).float()
    terminated_row = (1.0 - capped_row) * (lengths > 0).float()
    last_idx = (lengths - 1).clamp_min(0).long()
    positions = torch.arange(resp_len, device=mask.device)
    last_pos = (positions.unsqueeze(0) == last_idx.unsqueeze(-1)).float() * mask
    eos_sel = last_pos * terminated_row.unsqueeze(-1)
    put("eos_den", eos_sel.sum())
    put("eos_res_num", msum(r, eos_sel))
    put("eos_align_num", msum(a, eos_sel))
    put("eos_logT_num", msum(log_t, eos_sel))
    put("eos_logR_num", msum(log_r, eos_sel))
    put("eos_logS_num", msum(log_s, eos_sel))

    # --- runaway (full-buffer) rows -----------------------------------------------
    capped_sel = mask * capped_row.unsqueeze(-1)
    put("capped_tok_den", capped_sel.sum())
    put("capped_res_num", msum(r, capped_sel))
    put("capped_align_num", msum(a, capped_sel))

    # --- edit type: did RL resolve or create uncertainty on this token? -------------
    c_t = entropy_contribution(log_t)
    c_r = entropy_contribution(log_r)
    d_log = log_t - log_r
    types = {
        "resolution": (c_t < c_r).float() * mask,
        "creation": ((t_prob > r_prob) & (c_t > c_r)).float() * mask,
        "uncommit": ((t_prob < r_prob) & (c_t > c_r)).float() * mask,
        "tie": (c_t == c_r).float() * mask,  # T == R (zero residual) and both-saturated tokens
    }
    for name, sel in types.items():
        put(f"tok_num_{name}", sel.sum())
        put(f"res_num_{name}", msum(r, sel))
        put(f"resabs_num_{name}", msum(r.abs(), sel))
        put(f"beyond_num_{name}", msum(beyond, sel))
        put(f"cf_log_abs_num_{name}", msum(d_log.abs(), sel))

    # --- counterfactual residuals on the same tokens ------------------------------
    adaptive_alpha = entropy_tempered_alpha(log_t)
    cf = {
        "cf_log": d_log,
        "cf_l2": t_prob - r_prob,
        "cf_noe": entropy_tempered_residual(log_t, log_r, adaptive_alpha * math.e),
        "cf_noinv": torch.exp(adaptive_alpha * log_t) - torch.exp(adaptive_alpha * log_r),
        "cf_aT": entropy_tempered_residual(log_t, log_r, adaptive_alpha),
        "cf_ratio": entropy_tempered_residual(log_t, log_r, c_t / c_r),
    }
    for name, x in cf.items():
        put(f"{name}_abs_den", msum(x.abs(), mask))
        put(f"{name}_num", msum(x, mask))
        put(f"{name}_abs_num_tlow", msum(x.abs(), bins["tlow"]))
        put(f"{name}_abs_num_t2", msum(x.abs(), bins["t2"]))
        put(f"{name}_abs_num_thigh", msum(x.abs(), bins["thigh"]))
        put(f"{name}_eos_num", msum(x, eos_sel))

    # --- depth segments -----------------------------------------------------------
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, resp_len + 1)):
        seg = ((positions >= lo) & (positions < hi)).float().unsqueeze(0) * mask
        put(f"tok_den_seg{k}", seg.sum())
        put(f"res_num_seg{k}", msum(r, seg))
        put(f"resabs_num_seg{k}", msum(r.abs(), seg))
        lo = hi

    values = torch.stack(vals).tolist()  # single device sync
    return dict(zip(names, values, strict=True))
