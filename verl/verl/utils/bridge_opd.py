# Copyright 2026 Bridge-OPD authors
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
"""Teacher-student prefix-distribution bridge for on-policy distillation."""

from __future__ import annotations

import math
from collections import defaultdict
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


def _group_indices(index: Any, batch_size: int) -> list[list[int]]:
    if index is None:
        raise ValueError("Bridge-OPD requires the rollout uid for grouping responses from the same prompt.")

    if isinstance(index, torch.Tensor):
        values = index.detach().cpu().tolist()
    elif hasattr(index, "tolist"):
        values = index.tolist()
    else:
        values = list(index)

    if len(values) != batch_size:
        raise ValueError(f"Bridge-OPD received {len(values)} uids for a batch of size {batch_size}.")

    groups: dict[Any, list[int]] = defaultdict(list)
    for row, uid in enumerate(values):
        # NumPy can return a nested one-element list for object arrays.
        if isinstance(uid, list):
            uid = tuple(uid)
        groups[uid].append(row)
    return list(groups.values())


def _normalized_weights_and_ess(
    prefix_scores: torch.Tensor,
    active: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean-one exponential weights and ESS independently at each position."""

    active_count = active.sum(dim=0).to(prefix_scores.dtype)
    scaled = prefix_scores * beta.unsqueeze(0)
    scaled = torch.where(active, scaled, torch.full_like(scaled, -torch.inf))
    max_scaled = scaled.max(dim=0).values
    max_scaled = torch.where(active_count > 0, max_scaled, torch.zeros_like(max_scaled))

    raw_weights = torch.where(active, torch.exp(scaled - max_scaled.unsqueeze(0)), torch.zeros_like(scaled))
    weight_sum = raw_weights.sum(dim=0)
    squared_sum = raw_weights.square().sum(dim=0)

    normalized = raw_weights * active_count.unsqueeze(0) / weight_sum.clamp_min(torch.finfo(raw_weights.dtype).tiny)
    ess = weight_sum.square() / squared_sum.clamp_min(torch.finfo(raw_weights.dtype).tiny)
    ess = torch.where(active_count > 0, ess, torch.zeros_like(ess))
    return normalized, ess


@torch.no_grad()
def compute_bridge_opd_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the detached Bridge-OPD state weights.

    For state ``s_t = (x, y_<t)``, the unnormalized bridge weight is

    ``exp(beta * sum_{j<t}(log pi_T(y_j|s_j) - log pi_S(y_j|s_j)))``.

    We normalize across responses from the same prompt at each position, so
    active responses have mean weight one. The current token is deliberately
    excluded: its likelihood ratio describes the transition out of ``s_t``,
    not the probability of reaching ``s_t``.
    """

    if student_log_probs.dim() != 2 or teacher_log_probs.dim() != 2:
        raise ValueError("Bridge-OPD expects 2D sampled-token log-probability tensors [batch, response_length].")
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "Bridge-OPD teacher/student sampled-token log-probabilities must have identical shapes, got "
            f"{tuple(student_log_probs.shape)} and {tuple(teacher_log_probs.shape)}."
        )
    if response_mask.shape != student_log_probs.shape:
        raise ValueError(
            "Bridge-OPD response_mask has shape "
            f"{tuple(response_mask.shape)}, expected {tuple(student_log_probs.shape)}."
        )

    beta = float(_cfg_get(config, "beta", 0.2))
    adaptive_beta = bool(_cfg_get(config, "adaptive_beta", True))
    min_ess_ratio = float(_cfg_get(config, "min_ess_ratio", 0.5))
    beta_search_steps = int(_cfg_get(config, "beta_search_steps", 12))

    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"Bridge-OPD beta must be finite and in [0, 1], got {beta}.")
    if not math.isfinite(min_ess_ratio) or not 0.0 < min_ess_ratio <= 1.0:
        raise ValueError(f"Bridge-OPD min_ess_ratio must be in (0, 1], got {min_ess_ratio}.")
    if beta_search_steps < 1:
        raise ValueError(f"Bridge-OPD beta_search_steps must be positive, got {beta_search_steps}.")

    device = student_log_probs.device
    active = response_mask.to(device).bool()
    student_log_probs = student_log_probs.detach().to(device=device, dtype=torch.float32)
    teacher_log_probs = teacher_log_probs.detach().to(device=device, dtype=torch.float32)

    valid_log_probs = torch.isfinite(student_log_probs) & torch.isfinite(teacher_log_probs)
    if not torch.all(valid_log_probs | ~active):
        raise ValueError("Bridge-OPD found a non-finite teacher/student log-probability on an active response token.")

    token_log_ratio = torch.where(active, teacher_log_probs - student_log_probs, torch.zeros_like(student_log_probs))
    # Shift the cumulative sum right by one: state s_t contains y_<t, never y_t.
    prefix_log_ratio = token_log_ratio.cumsum(dim=-1) - token_log_ratio

    weights = torch.zeros_like(prefix_log_ratio)
    beta_used = torch.zeros_like(prefix_log_ratio)
    ess = torch.zeros_like(prefix_log_ratio)

    for rows in _group_indices(index, student_log_probs.shape[0]):
        row_index = torch.as_tensor(rows, device=device, dtype=torch.long)
        group_scores = prefix_log_ratio.index_select(0, row_index)
        group_active = active.index_select(0, row_index)
        active_count = group_active.sum(dim=0).to(group_scores.dtype)
        valid_position = active_count > 0

        target_beta = torch.full_like(active_count, beta)
        chosen_beta = target_beta

        if adaptive_beta and beta > 0.0:
            _, target_ess = _normalized_weights_and_ess(group_scores, group_active, target_beta)
            required_ess = min_ess_ratio * active_count
            needs_shrink = valid_position & (target_ess < required_ess)

            low = torch.zeros_like(target_beta)
            high = target_beta.clone()
            for _ in range(beta_search_steps):
                middle = (low + high) * 0.5
                _, middle_ess = _normalized_weights_and_ess(group_scores, group_active, middle)
                feasible = middle_ess >= required_ess
                low = torch.where(feasible, middle, low)
                high = torch.where(feasible, high, middle)
            chosen_beta = torch.where(needs_shrink, low, target_beta)

        group_weights, group_ess = _normalized_weights_and_ess(group_scores, group_active, chosen_beta)
        weights.index_copy_(0, row_index, group_weights)
        beta_used.index_copy_(0, row_index, chosen_beta.unsqueeze(0).expand_as(group_scores) * group_active)
        ess.index_copy_(0, row_index, group_ess.unsqueeze(0).expand_as(group_scores) * group_active)

    aux = {
        "bridge_opd_weights": weights,
        "bridge_opd_beta": beta_used,
        "bridge_opd_ess": ess,
        "bridge_opd_prefix_log_ratio": prefix_log_ratio * active,
    }
    return weights, aux


@torch.no_grad()
def apply_bridge_opd_to_scores(
    rm_scores: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multiply an existing 2D or top-k 3D OPD score by Bridge-OPD state weights."""

    weights, aux = compute_bridge_opd_weights(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        index=index,
        config=config,
    )

    if rm_scores.shape[:2] != weights.shape:
        raise ValueError(
            f"Bridge-OPD rm_scores starts with shape {tuple(rm_scores.shape[:2])}, expected {tuple(weights.shape)}."
        )
    if rm_scores.dim() == 2:
        weighted_scores = rm_scores * weights.to(device=rm_scores.device, dtype=rm_scores.dtype)
    elif rm_scores.dim() == 3:
        weighted_scores = rm_scores * weights.to(device=rm_scores.device, dtype=rm_scores.dtype).unsqueeze(-1)
    else:
        raise ValueError(f"Bridge-OPD supports 2D or 3D rm_scores, got shape {tuple(rm_scores.shape)}.")
    return weighted_scores, aux


def bridge_opd_metrics(aux: dict[str, torch.Tensor], response_mask: torch.Tensor) -> dict[str, float]:
    """Summarize Bridge-OPD tensors for the normal trainer/W&B metric path."""

    valid = response_mask.to(aux["bridge_opd_weights"].device).bool()
    if not valid.any():
        return {}

    weights = aux["bridge_opd_weights"][valid].float()
    beta = aux["bridge_opd_beta"][valid].float()
    ess = aux["bridge_opd_ess"][valid].float()
    prefix_log_ratio = aux["bridge_opd_prefix_log_ratio"][valid].float()
    return {
        "bridge_opd/weight_mean": weights.mean().item(),
        "bridge_opd/weight_std": weights.std(unbiased=False).item(),
        "bridge_opd/weight_min": weights.min().item(),
        "bridge_opd/weight_max": weights.max().item(),
        "bridge_opd/beta_mean": beta.mean().item(),
        "bridge_opd/beta_min": beta.min().item(),
        "bridge_opd/ess_mean": ess.mean().item(),
        "bridge_opd/ess_min": ess.min().item(),
        "bridge_opd/prefix_log_ratio_mean": prefix_log_ratio.mean().item(),
        "bridge_opd/prefix_log_ratio_std": prefix_log_ratio.std(unbiased=False).item(),
    }
