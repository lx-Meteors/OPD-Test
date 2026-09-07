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

Probe metrics (``etopd/*``) follow the num/den convention: raw values, NOT
multiplied by ``loss_scale_factor``; ``*_num`` / ``*_den`` are per-micro-batch
sums, so the mean reduction over micro-batches preserves their ratio exactly.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ETOPD_C_FLOOR",
    "entropy_tempered_alpha",
    "entropy_tempered_residual",
    "compute_etopd_advantages",
    "compute_etopd_probe_metrics",
]

# Floor for the teacher's entropy contribution before inverting it. A sampled
# token with T == 1 exactly has c_T == 0; the floor makes alpha finite (~4e7),
# at which the residual is (1 - R^alpha) / alpha ~ 0 as intended.
ETOPD_C_FLOOR = 1e-8

_SEG_BOUNDS = (1024, 2048, 4096, 8192)
_T_LOW = 0.1
_T_HIGH = 0.9


def entropy_tempered_alpha(
    teacher_log_prob: torch.Tensor,
    fixed_alpha: float = 0.0,
    c_floor: float = ETOPD_C_FLOOR,
) -> torch.Tensor:
    """Per-token Box-Cox exponent alpha = 1 / (e * c_T), c_T = -T log T.

    Args:
        teacher_log_prob: log T(y_t) on the sampled tokens, any float dtype.
        fixed_alpha: if > 0, return a constant exponent instead (ablation:
            1.0 gives the L2 residual T - R; the alpha -> 0 limit is G-OPD).
        c_floor: numerical floor for c_T.

    Returns:
        float32 tensor of the same shape, >= 1 in the adaptive case.
    """
    log_t = teacher_log_prob.to(torch.float32)
    if fixed_alpha > 0:
        return torch.full_like(log_t, float(fixed_alpha))
    c_t = (-(torch.exp(log_t) * log_t)).clamp_min(c_floor)
    return 1.0 / (math.e * c_t)


def entropy_tempered_residual(
    teacher_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """(T^alpha - R^alpha) / alpha computed in float32 as exp(alpha * log p).

    alpha * log p is <= 0 so exp never overflows; very negative arguments
    underflow to exactly 0, which is the intended limit for T -> 0.
    """
    log_t = teacher_log_prob.to(torch.float32)
    log_r = ref_log_prob.to(torch.float32)
    alpha = alpha.to(torch.float32)
    return (torch.exp(alpha * log_t) - torch.exp(alpha * log_r)) / alpha


def compute_etopd_advantages(
    teacher_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    fixed_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (advantages, align, residual, alpha) for the ET-OPD advantage.

    advantages = align + residual with align = log T - log S (the standard
    OPD sampled-token advantage) and residual = (T^alpha - R^alpha) / alpha.
    All tensors are (bsz, response_len); advantages is cast back to the dtype
    of ``old_log_prob``.
    """
    align = teacher_log_prob.to(torch.float32) - old_log_prob.to(torch.float32)
    alpha = entropy_tempered_alpha(teacher_log_prob, fixed_alpha=fixed_alpha)
    residual = entropy_tempered_residual(teacher_log_prob, ref_log_prob, alpha)
    advantages = (align + residual).to(old_log_prob.dtype)
    return advantages, align, residual, alpha


def compute_etopd_probe_metrics(
    align: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Probe suite for ET-OPD (raw values, num/den convention).

    Readouts (divide the logged series in W&B):
      * etopd/align_mean, residual_mean, residual_abs_mean: force decomposition
        of the advantage (residual_mean is the net "rent" of the residual).
      * etopd/log_alpha_mean, alpha_ge2_frac: how cold the comparison is and
        the share of tokens the residual has effectively switched off.
      * resabs_num_t{low,mid,high} / resabs_den: share of |residual| landing on
        tokens with T < 0.1, 0.1 <= T <= 0.9, T > 0.9 (expected: ~1%, most,
        ~10%). tok_num_t* / tok_den give the corresponding token shares.
      * eos_res_num / eos_den, eos_align_num / eos_den: residual and alignment
        on the last valid token of terminated rows (the sampled EOS). The
        pre-registered prediction is eos_res_num / eos_den ~= 0.
      * capped_res_num / capped_tok_den vs its complement
        (residual_num - capped_res_num) / (tok_den - capped_tok_den): residual
        on full-buffer (runaway) rows vs terminated rows.
      * res_num_seg{k} / tok_den_seg{k}, resabs_num_seg{k} / tok_den_seg{k}:
        depth profile over [0,1k), [1k,2k), [2k,4k), [4k,8k), [8k,+).
    """
    if align.dim() != 2:
        return {}
    a = align.float()
    r = residual.float()
    al = alpha.float()
    mask = response_mask.to(torch.float32)
    n_tok = mask.sum().clamp_min(1.0)
    t_prob = torch.exp(teacher_log_prob.to(torch.float32))

    out: dict[str, float] = {}

    def full_mean(x: torch.Tensor) -> float:
        return ((x * mask).sum() / n_tok).item()

    out["etopd/align_mean"] = full_mean(a)
    out["etopd/residual_mean"] = full_mean(r)
    out["etopd/residual_abs_mean"] = full_mean(r.abs())
    out["etopd/log_alpha_mean"] = full_mean(torch.log(al))
    out["etopd/alpha_ge2_frac"] = full_mean((al >= 2.0).float())

    # --- where does the residual land, by teacher probability -----------------
    low = (t_prob < _T_LOW).float() * mask
    high = (t_prob > _T_HIGH).float() * mask
    mid = mask - low - high
    out["etopd/resabs_den"] = (r.abs() * mask).sum().item()
    out["etopd/residual_num"] = (r * mask).sum().item()
    out["etopd/tok_den"] = mask.sum().item()
    for name, sel in (("tlow", low), ("tmid", mid), ("thigh", high)):
        out[f"etopd/resabs_num_{name}"] = (r.abs() * sel).sum().item()
        out[f"etopd/tok_num_{name}"] = sel.sum().item()

    # --- terminal token of terminated rows (the sampled EOS) -------------------
    lengths = mask.sum(dim=-1)
    resp_len = mask.shape[-1]
    capped_row = (lengths >= resp_len).float()
    terminated_row = (1.0 - capped_row) * (lengths > 0).float()
    last_idx = (lengths - 1).clamp_min(0).long()
    positions = torch.arange(resp_len, device=mask.device)
    last_pos = (positions.unsqueeze(0) == last_idx.unsqueeze(-1)).float() * mask
    eos_sel = last_pos * terminated_row.unsqueeze(-1)
    out["etopd/eos_den"] = eos_sel.sum().item()
    out["etopd/eos_res_num"] = (r * eos_sel).sum().item()
    out["etopd/eos_align_num"] = (a * eos_sel).sum().item()

    # --- runaway (full-buffer) rows --------------------------------------------
    capped_sel = mask * capped_row.unsqueeze(-1)
    out["etopd/capped_tok_den"] = capped_sel.sum().item()
    out["etopd/capped_res_num"] = (r * capped_sel).sum().item()

    # --- depth segments --------------------------------------------------------
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, resp_len + 1)):
        seg = ((positions >= lo) & (positions < hi)).float().unsqueeze(0) * mask
        out[f"etopd/tok_den_seg{k}"] = seg.sum().item()
        out[f"etopd/res_num_seg{k}"] = (r * seg).sum().item()
        out[f"etopd/resabs_num_seg{k}"] = (r.abs() * seg).sum().item()
        lo = hi

    return out
