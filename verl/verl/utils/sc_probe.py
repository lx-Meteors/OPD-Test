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
"""Online probes for the SC-ratio OPD advantage adv = a * (1 + w).

One readout per pre-registered claim:

  * align_mean            E[a] = -KL(S||T): the debt-repayment progress bar
                          (starts ~ -0.2, rises to 0; rent is impossible, so it
                          must never settle at a positive plateau).
  * bonus_mean            E[w*a]: the net bonus force. Contrast with G-OPD's
                          frozen-tilt plateau (+0.025) / first4k (+0.019): this
                          series has no positive-plateau failure mode.
  * weight_mean/clamp_frac  bonus budget and firewall bite. Expected shape:
                          clamp_frac ~2% at start, rises during the length
                          explosion (loop states), drifts toward ~50% at
                          convergence as the bonus retires (harmless: the
                          unclamped half's w is ~0 there too).
  * seg{k} num/den        depth profile [0,2k)/[2k,4k)/[4k,8k)/[8k,+): the
                          force amplification 1 + wabsa_num/absa_num should
                          reproduce the offline prediction (~1.23/1.24/1.19/1.13
                          vs G-OPD's flat 1.25) and anneal to 1; absa_num/tok_den
                          is the debt ladder collapsing over training.
  * capped rows           runaway starving and the ledger flip: weight and
                          (signed) bonus mass flowing to full-buffer rows, plus
                          the teacher-SC collapse (offline: ~14.5 capped-deep
                          vs ~22.5 normal) that drives it.
  * terminal              executed-stop probe: a and w on the last valid token
                          of terminated rows (the anti-early-stop channel).

Conventions (inherited from the earlier probe suite's hardening):
  * float32 throughout: bf16 summation over 16k-token rows loses percent-level
    accuracy in num/den comparisons.
  * Raw values, deliberately NOT multiplied by loss_scale_factor.
  * *_num / *_den are per-micro sums; the metric pipeline mean-reduces over
    micro-batches, which preserves sum ratios exactly. Divide the two logged
    series in W&B to recover the true ratio (per-micro ratios can degenerate
    when a micro-batch holds a single sequence).
  * Rows whose valid length fills the whole response buffer are treated as
    capped (runaway proxy); zero-length rows are guarded out of the terminal
    probe.
"""

from __future__ import annotations

import torch

_SEG_BOUNDS = (2048, 4096, 8192)

__all__ = ["compute_sc_probe_metrics"]


@torch.no_grad()
def compute_sc_probe_metrics(
    align_adv: torch.Tensor,
    sc_weight: torch.Tensor,
    sc_student: torch.Tensor,
    sc_teacher: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Probes for the SC-ratio advantage. All inputs are (bsz, response_len).

    Args:
        align_adv: alignment debt a = logT - logS on the sampled token.
        sc_weight: the applied bonus weight w = clamp(1 - SC_S/SC_T, 0, 1).
        sc_student: student self-certainty SC_S per position.
        sc_teacher: teacher self-certainty SC_T per position.
        response_mask: valid-token mask.
    """
    if align_adv.dim() != 2 or sc_weight.dim() != 2:
        return {}

    a = align_adv.float()
    w = sc_weight.float()
    sc_s = sc_student.float()
    sc_t = sc_teacher.float()
    mask = response_mask.to(torch.float32)
    n_tok = mask.sum().clamp_min(1.0)
    bonus = w * a
    out: dict[str, float] = {}

    def full_mean(x: torch.Tensor) -> float:
        return ((x * mask).sum() / n_tok).item()

    out["sc_probe/align_mean"] = full_mean(a)
    out["sc_probe/align_abs_mean"] = full_mean(a.abs())  # remaining token-debt mass
    out["sc_probe/bonus_mean"] = full_mean(bonus)
    out["sc_probe/weight_mean"] = full_mean(w)
    out["sc_probe/clamp_frac"] = full_mean((w <= 0).float())
    out["sc_probe/sc_student_mean"] = full_mean(sc_s)
    out["sc_probe/sc_teacher_mean"] = full_mean(sc_t)

    # --- depth segments (num/den pairs) --------------------------------------
    positions = torch.arange(mask.shape[-1], device=mask.device)
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, mask.shape[-1] + 1)):
        seg = ((positions >= lo) & (positions < hi)).float().unsqueeze(0) * mask
        out[f"sc_probe/tok_den_seg{k}"] = seg.sum().item()
        out[f"sc_probe/absa_num_seg{k}"] = (a.abs() * seg).sum().item()
        out[f"sc_probe/wabsa_num_seg{k}"] = (w * a.abs() * seg).sum().item()
        out[f"sc_probe/weight_num_seg{k}"] = (w * seg).sum().item()
        out[f"sc_probe/clamp_num_seg{k}"] = ((w <= 0).float() * seg).sum().item()
        lo = hi

    # --- trajectory ledger: capped (runaway proxy) rows ----------------------
    lengths = mask.sum(dim=-1)
    capped_row = (lengths >= mask.shape[-1]).float()
    capped_col = capped_row.unsqueeze(-1)
    out["sc_probe/capped_seq_frac"] = capped_row.mean().item()
    out["sc_probe/tok_total_den"] = n_tok.item()
    out["sc_probe/tok_capped_den"] = (capped_col * mask).sum().item()
    out["sc_probe/weight_capped_num"] = (w * capped_col * mask).sum().item()
    out["sc_probe/bonus_capped_num"] = (bonus * capped_col * mask).sum().item()
    out["sc_probe/sct_capped_num"] = (sc_t * capped_col * mask).sum().item()

    # --- executed-stop probe: last valid token of terminated rows ------------
    terminated_row = 1.0 - capped_row
    last_idx = (lengths.long() - 1).clamp_min(0)
    row_idx = torch.arange(mask.shape[0], device=mask.device)
    terminal_mask = torch.zeros_like(mask)
    terminal_mask[row_idx, last_idx] = terminated_row
    # Guard zero-length rows: their clamped last_idx points at padding.
    terminal_mask = terminal_mask * mask
    out["sc_probe/terminal_cnt_den"] = terminal_mask.sum().item()
    out["sc_probe/terminal_a_num"] = (a * terminal_mask).sum().item()
    out["sc_probe/terminal_w_num"] = (w * terminal_mask).sum().item()

    return out
