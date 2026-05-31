from __future__ import annotations

import math
from typing import Any

import torch


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def apply_prune_opd_to_scores(
    rm_scores: torch.Tensor,
    overlap_mask: torch.Tensor | None,
    response_mask: torch.Tensor | None,
    config: Any,
    responses: torch.Tensor | None = None,
    teacher_top_k_ids: torch.Tensor | None = None,
    teacher_top_k_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply prune-opd weights to OPD token-level scores.

    prune-opd builds a monotone non-increasing token weight from a teacher
    reliability signal. Every violation decays all future OPD weights by a
    fixed step.
    """

    enabled = bool(_cfg_get(config, "enable", False))
    if not enabled:
        return rm_scores, {}

    metric = _cfg_get(config, "metric", "overlap_ratio")

    if response_mask is None:
        raise ValueError("prune-opd requires response_mask, but it is missing from the OPD batch.")
    if rm_scores.dim() != 3:
        raise ValueError(f"prune-opd expects 3D rm_scores for top-k OPD, got shape {tuple(rm_scores.shape)}.")

    threshold = float(_cfg_get(config, "threshold", 0.9))
    w_drop = float(_cfg_get(config, "w_drop", 0.02))
    w_base = float(_cfg_get(config, "w_base", 0.0))
    positive_eps = float(_cfg_get(config, "positive_eps", 1e-8))
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"prune-opd threshold must be in (0, 1], got {threshold}.")
    if not math.isfinite(w_drop) or w_drop < 0.0:
        raise ValueError(f"prune-opd w_drop must be finite and non-negative, got {w_drop}.")
    if not math.isfinite(w_base) or w_base < 0.0:
        raise ValueError(f"prune-opd w_base must be finite and non-negative, got {w_base}.")

    device = rm_scores.device
    response_mask = response_mask.to(device)

    valid_mask = response_mask.bool()
    aux_tensors = {}

    if metric == "overlap_ratio":
        if overlap_mask is None:
            raise ValueError("prune-opd requires overlap_mask, but it is missing from the OPD batch.")

        overlap_mask = overlap_mask.to(device)
        overlap_ratio = overlap_mask.float().mean(dim=-1)
        bad_event = (overlap_ratio < threshold) & valid_mask
        aux_tensors["prune_opd_overlap_ratio"] = overlap_ratio
    elif metric == "teacher_top_p_accept":
        if responses is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires responses.")
        if teacher_top_k_ids is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires teacher_top_k_ids.")
        if teacher_top_k_log_probs is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires teacher_top_k_log_probs.")

        responses = responses.to(device)
        teacher_top_k_ids = teacher_top_k_ids.to(device)
        teacher_top_k_log_probs = teacher_top_k_log_probs.to(device)

        teacher_top_k_probs = teacher_top_k_log_probs.float().exp()
        teacher_top_k_cum_probs = teacher_top_k_probs.cumsum(dim=-1)
        crossed = teacher_top_k_cum_probs >= threshold
        has_crossed = crossed.any(dim=-1, keepdim=True)
        first_cross_idx = crossed.to(torch.long).argmax(dim=-1, keepdim=True)
        rank_idx = torch.arange(teacher_top_k_ids.size(-1), device=device).view(1, 1, -1)
        top_p_mask = rank_idx <= first_cross_idx
        accept_mask = torch.where(has_crossed, top_p_mask, torch.ones_like(top_p_mask, dtype=torch.bool))

        response_in_accept_set = ((teacher_top_k_ids == responses.unsqueeze(-1)) & accept_mask).any(dim=-1)
        bad_event = (~response_in_accept_set) & valid_mask
        aux_tensors["prune_opd_teacher_top_p_accept"] = response_in_accept_set.to(torch.float32)
    else:
        raise NotImplementedError(f"Unsupported prune-opd metric: {metric}")

    cumulative_bad_event = bad_event.to(rm_scores.dtype).cumsum(dim=-1)
    weights = 1.0 - w_drop * cumulative_bad_event
    weights = torch.clamp(weights, min=0.0, max=1.0)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    loss_weights = weights + w_base
    loss_weights = torch.where(valid_mask, loss_weights, torch.zeros_like(loss_weights))

    weighted_rm_scores = rm_scores * loss_weights.unsqueeze(-1)
    effective_response_length = (valid_mask & (weights > positive_eps)).sum(dim=-1).to(torch.float32)

    aux_tensors["prune_opd_weights"] = weights
    aux_tensors["prune_opd_loss_weights"] = loss_weights
    aux_tensors["prune_opd_bad_event"] = bad_event.to(torch.float32)
    aux_tensors["prune_opd_effective_response_length"] = effective_response_length
    return weighted_rm_scores, aux_tensors
