from __future__ import annotations

import torch

# Depth segmentation follows the Prune-OPD 4096-token segment convention and the
# first4k boundary: [0, 2048), [2048, 4096), [4096, 8192), [8192, inf).
_SEG_BOUNDS = (2048, 4096, 8192)


@torch.no_grad()
def compute_gopd_probe_metrics(
    align_adv: torch.Tensor,
    tilt_adv: torch.Tensor,
    d_raw: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Online probes for the demolition/supplement (chai/bu) decomposition and the
    rent -> length -> runaway causal chain, computed on the sampled-token G-OPD path.

    Args:
        align_adv: (bsz, resp_len) alignment advantage logT - logS (positive = supplement
            owed, negative = demolition due). This is the RKL term of the advantage.
        tilt_adv: (bsz, resp_len) the extrapolation term actually applied to the
            advantage, i.e. (lambda - 1) * residual after any position mask or debt gate.
        d_raw: (bsz, resp_len) raw extrapolation contrast logT - logR before any
            gating (signal probe).
        response_mask: (bsz, resp_len) valid-token mask. A row whose valid length fills
            the whole response buffer is treated as capped (runaway proxy).

    Values are emitted raw (NOT multiplied by loss_scale_factor) so they can be read
    directly from W&B. Reduction over micro-batches is a plain mean.
    """
    if align_adv.dim() != 2 or tilt_adv.dim() != 2 or d_raw.dim() != 2:
        return {}

    mask = response_mask.to(align_adv.dtype)
    n_tok = mask.sum().clamp_min(1.0)
    out: dict[str, float] = {}

    def masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        denom = m.sum().clamp_min(1.0)
        return ((x * m).sum() / denom).item()

    # --- demolition/supplement decomposition of the total advantage ------------
    adv = align_adv + tilt_adv
    out["gopd_probe/force_supp_mean"] = masked_mean(adv.clamp_min(0.0), mask)
    out["gopd_probe/force_demo_mean"] = masked_mean(adv.clamp_max(0.0), mask)
    out["gopd_probe/align_mean"] = masked_mean(align_adv, mask)
    out["gopd_probe/tilt_mean"] = masked_mean(tilt_adv, mask)

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
        out[f"gopd_probe/quad_frac_{name}"] = masked_mean(q.to(mask.dtype), mask)

    # --- buyout probe: extrapolation actively fighting demolition --------------
    buyout = (align_adv < 0) & (tilt_adv > 0)
    buyout_f = buyout.to(mask.dtype)
    out["gopd_probe/buyout_frac"] = masked_mean(buyout_f, mask)
    out["gopd_probe/buyout_mean_tilt"] = masked_mean(tilt_adv.clamp_min(0.0), buyout_f * mask)

    # --- depth-segmented probes -------------------------------------------------
    positions = torch.arange(mask.shape[-1], device=mask.device)
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, mask.shape[-1] + 1)):
        seg = ((positions >= lo) & (positions < hi)).to(mask.dtype).unsqueeze(0) * mask
        out[f"gopd_probe/tilt_mean_seg{k}"] = masked_mean(tilt_adv, seg)
        out[f"gopd_probe/d_strong_frac_seg{k}"] = masked_mean((d_raw > 0.5).to(mask.dtype), seg)
        out[f"gopd_probe/buyout_frac_seg{k}"] = masked_mean(buyout_f, seg)
        lo = hi

    # --- trajectory ledger: does positive tilt mass flow to capped (runaway) rows?
    lengths = mask.sum(dim=-1)
    capped_row = (lengths >= mask.shape[-1]).to(mask.dtype)
    capped_tok = capped_row.unsqueeze(-1) * mask
    pos_tilt = tilt_adv.clamp_min(0.0) * mask
    total_pos = pos_tilt.sum().clamp_min(1e-8)
    out["gopd_probe/capped_seq_frac"] = capped_row.mean().item()
    out["gopd_probe/token_share_capped"] = (capped_tok.sum() / n_tok).item()
    out["gopd_probe/tilt_possum_share_capped"] = ((pos_tilt * capped_row.unsqueeze(-1)).sum() / total_pos).item()

    # --- executed-stop probe: the last valid token of a terminated row is its EOS
    terminated_row = 1.0 - capped_row
    last_idx = (lengths.long() - 1).clamp_min(0)
    row_idx = torch.arange(mask.shape[0], device=mask.device)
    terminal_mask = torch.zeros_like(mask)
    terminal_mask[row_idx, last_idx] = terminated_row
    out["gopd_probe/tilt_terminal_mean"] = masked_mean(tilt_adv, terminal_mask)
    out["gopd_probe/align_terminal_mean"] = masked_mean(align_adv, terminal_mask)

    return out
