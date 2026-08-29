from __future__ import annotations

import torch

# Depth segmentation follows the Prune-OPD 4096-token segment convention and the
# first4k boundary: [0, 2048), [2048, 4096), [4096, 8192), [8192, inf).
_SEG_BOUNDS = (2048, 4096, 8192)


@torch.no_grad()
def compute_gopd_probe_metrics(
    align_adv: torch.Tensor,
    tilt_adv: torch.Tensor,
    cf_tilt_adv: torch.Tensor,
    d_raw: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Online probes for the demolition/supplement (chai/bu) decomposition and the
    rent -> length -> runaway causal chain, computed on the sampled-token G-OPD path.

    Args:
        align_adv: (bsz, resp_len) alignment advantage logT - logS (positive = supplement
            owed, negative = demolition due). This is the RKL term of the advantage.
        tilt_adv: (bsz, resp_len) extrapolation term actually applied, i.e.
            (lambda - 1) * residual after the position mask and/or debt gate.
        cf_tilt_adv: (bsz, resp_len) counterfactual ungated extrapolation
            (lambda - 1) * d_raw, what plain full-trajectory G-OPD would apply on the
            same rollouts. actual - counterfactual = what this arm's gate withheld.
        d_raw: (bsz, resp_len) raw extrapolation contrast logT - logR before any gating.
        response_mask: (bsz, resp_len) valid-token mask. A row whose valid length fills
            the whole response buffer is treated as capped (runaway proxy).

    Conventions:
        * Values are raw - NOT multiplied by loss_scale_factor.
        * Metrics named *_num / *_den are sums/counts per micro-batch. The metric
          pipeline mean-reduces over micro-batches, which preserves sum ratios
          exactly: divide the two logged series to recover the true ratio. This is
          required because micro-batches can hold a single sequence, where per-micro
          ratios degenerate to 0/1 indicators.
    """
    if align_adv.dim() != 2 or tilt_adv.dim() != 2 or d_raw.dim() != 2:
        return {}

    # float32 throughout: inputs may arrive in bf16, whose ~8 mantissa bits lose
    # percent-level accuracy when summing over 16k tokens (the ledger num/den
    # comparisons need better than that).
    align_adv = align_adv.float()
    tilt_adv = tilt_adv.float()
    cf_tilt_adv = cf_tilt_adv.float()
    d_raw = d_raw.float()
    mask = response_mask.to(align_adv.dtype)
    n_tok = mask.sum().clamp_min(1.0)
    out: dict[str, float] = {}

    def full_mean(x: torch.Tensor) -> float:
        return ((x * mask).sum() / n_tok).item()

    # --- demolition/supplement decomposition of the total advantage ------------
    adv = align_adv + tilt_adv
    out["gopd_probe/force_supp_mean"] = full_mean(adv.clamp_min(0.0))
    out["gopd_probe/force_demo_mean"] = full_mean(adv.clamp_max(0.0))
    out["gopd_probe/align_mean"] = full_mean(align_adv)
    out["gopd_probe/align_abs_mean"] = full_mean(align_adv.abs())  # remaining imitation debt
    out["gopd_probe/tilt_mean"] = full_mean(tilt_adv)
    out["gopd_probe/cf_tilt_mean"] = full_mean(cf_tilt_adv)

    # --- quadrant occupancy: sign(align) x sign(d_raw) -------------------------
    # q1: owed + RL-favored (legitimate supplement)   q2: owed + RL-suppressed
    # q3: overpaid + RL-favored (the toxic buyout quadrant)  q4: overpaid + RL-suppressed
    owed = align_adv > 0
    dpos = d_raw > 0
    for name, q in (
        ("q1", owed & dpos),
        ("q2", owed & ~dpos),
        ("q3", ~owed & dpos),
        ("q4", ~owed & ~dpos),
    ):
        out[f"gopd_probe/quad_frac_{name}"] = full_mean(q.to(mask.dtype))

    # --- buyout probe: extrapolation actively fighting demolition --------------
    # Exact rates come from the num/den pairs (divide the logged series):
    #   buyout rate = buyout_tok_den / tok_total_den, mean size = tilt_num / tok_den.
    buyout = ((align_adv < 0) & (tilt_adv > 0)).to(mask.dtype) * mask
    cf_buyout = ((align_adv < 0) & (cf_tilt_adv > 0)).to(mask.dtype) * mask
    out["gopd_probe/buyout_tilt_num"] = (tilt_adv.clamp_min(0.0) * buyout).sum().item()
    out["gopd_probe/buyout_tok_den"] = buyout.sum().item()
    out["gopd_probe/cf_buyout_tok_num"] = cf_buyout.sum().item()

    # --- granted supplement: qualified positive tilt actually paid --------------
    # align >= 0 complements buyout's align < 0, so positive tilt partitions
    # exactly: postilt_total_den == buyout_tilt_num + supp_granted_tilt_num.
    granted = ((align_adv >= 0) & (tilt_adv > 0)).to(mask.dtype) * mask
    out["gopd_probe/supp_granted_tilt_num"] = (tilt_adv.clamp_min(0.0) * granted).sum().item()
    out["gopd_probe/supp_granted_tok_den"] = granted.sum().item()

    # --- depth-segmented probes (num/den pairs) ---------------------------------
    positions = torch.arange(mask.shape[-1], device=mask.device)
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, mask.shape[-1] + 1)):
        seg = ((positions >= lo) & (positions < hi)).to(mask.dtype).unsqueeze(0) * mask
        out[f"gopd_probe/tok_den_seg{k}"] = seg.sum().item()
        out[f"gopd_probe/tilt_num_seg{k}"] = (tilt_adv * seg).sum().item()
        out[f"gopd_probe/cf_tilt_num_seg{k}"] = (cf_tilt_adv * seg).sum().item()
        out[f"gopd_probe/dstrong_num_seg{k}"] = ((d_raw > 0.5).to(mask.dtype) * seg).sum().item()
        out[f"gopd_probe/buyout_num_seg{k}"] = (buyout * seg).sum().item()
        lo = hi

    # --- trajectory ledger: does positive tilt mass flow to capped (runaway) rows?
    lengths = mask.sum(dim=-1)
    capped_row = (lengths >= mask.shape[-1]).to(mask.dtype)
    capped_col = capped_row.unsqueeze(-1)
    pos_tilt = tilt_adv.clamp_min(0.0) * mask
    cf_pos_tilt = cf_tilt_adv.clamp_min(0.0) * mask
    out["gopd_probe/capped_seq_frac"] = capped_row.mean().item()
    out["gopd_probe/tok_capped_num"] = (capped_col * mask).sum().item()
    out["gopd_probe/tok_total_den"] = n_tok.item()
    out["gopd_probe/postilt_capped_num"] = (pos_tilt * capped_col).sum().item()
    out["gopd_probe/postilt_total_den"] = pos_tilt.sum().item()
    out["gopd_probe/cf_postilt_capped_num"] = (cf_pos_tilt * capped_col).sum().item()
    out["gopd_probe/cf_postilt_total_den"] = cf_pos_tilt.sum().item()

    # --- executed-stop probe: the last valid token of a terminated row is its EOS
    terminated_row = 1.0 - capped_row
    last_idx = (lengths.long() - 1).clamp_min(0)
    row_idx = torch.arange(mask.shape[0], device=mask.device)
    terminal_mask = torch.zeros_like(mask)
    terminal_mask[row_idx, last_idx] = terminated_row
    # Guard zero-length rows: their clamped last_idx points at padding.
    terminal_mask = terminal_mask * mask
    out["gopd_probe/terminal_cnt_den"] = terminal_mask.sum().item()
    out["gopd_probe/terminal_tilt_num"] = (tilt_adv * terminal_mask).sum().item()
    out["gopd_probe/terminal_align_num"] = (align_adv * terminal_mask).sum().item()

    return out
