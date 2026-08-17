# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _sequence_scores(true_reward_score: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
    if true_reward_score is None:
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    scores = true_reward_score.detach().float().to(device)
    if scores.shape[0] != batch_size:
        raise ValueError(
            f"DTG true_reward_score batch dimension must match response_mask, got {scores.shape[0]} and {batch_size}."
        )
    if scores.ndim > 1:
        scores = scores.reshape(batch_size, -1).sum(dim=-1)
    return scores.clamp(0.0, 1.0)


def _group_inverse(group_ids: Sequence[Any], batch_size: int, device: torch.device) -> tuple[torch.Tensor, int]:
    if len(group_ids) != batch_size:
        raise ValueError(f"DTG needs one group id per response, got {len(group_ids)} ids for {batch_size} responses.")

    group_to_index: dict[Any, int] = {}
    inverse = []
    for group_id in group_ids:
        key = group_id.item() if hasattr(group_id, "item") else group_id
        if key not in group_to_index:
            group_to_index[key] = len(group_to_index)
        inverse.append(group_to_index[key])
    return torch.tensor(inverse, device=device, dtype=torch.long), len(group_to_index)


@torch.no_grad()
def compute_distillation_to_go_advantages(
    *,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any],
    config: Any,
    true_reward_score: torch.Tensor | None = None,
    student_top_k_log_probs: torch.Tensor | None = None,
    teacher_on_student_log_probs: torch.Tensor | None = None,
    token_level_rewards: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute student-conditioned future-distillation advantages.

    The local discrepancy is a bounded Top-K total-variation lower bound when
    both student and teacher log probabilities are available.  The discrepancy
    is averaged in blocks, propagated backwards as a discounted future mean,
    and converted from a cost into a leave-one-out reward advantage among
    responses generated for the same prompt.

    The returned tensor is two-dimensional because it supervises the sampled
    rollout token.  It must be optimized as a separate policy loss instead of
    being broadcast over the candidates in the regular Top-K OPD objective.
    """

    if response_mask.ndim != 2:
        raise ValueError(f"DTG expects response_mask [batch, sequence], got {tuple(response_mask.shape)}.")

    block_size = int(_cfg_get(config, "block_size", 256))
    block_gamma = float(_cfg_get(config, "block_gamma", 0.95))
    outcome_weight = float(_cfg_get(config, "outcome_weight", 0.25))
    normalize_by_std = bool(_cfg_get(config, "normalize_by_std", False))
    max_abs_advantage = float(_cfg_get(config, "max_abs_advantage", 0.5))
    eps = float(_cfg_get(config, "eps", 1e-6))

    if block_size <= 0:
        raise ValueError(f"DTG block_size must be positive, got {block_size}.")
    if not math.isfinite(block_gamma) or not 0.0 <= block_gamma <= 1.0:
        raise ValueError(f"DTG block_gamma must be in [0, 1], got {block_gamma}.")
    if not math.isfinite(outcome_weight) or outcome_weight < 0.0:
        raise ValueError(f"DTG outcome_weight must be non-negative, got {outcome_weight}.")
    if not math.isfinite(max_abs_advantage) or max_abs_advantage <= 0.0:
        raise ValueError(f"DTG max_abs_advantage must be positive, got {max_abs_advantage}.")

    mask = response_mask.detach().float()
    batch_size, sequence_length = mask.shape
    if mask.sum().item() <= 0:
        raise ValueError("DTG received a batch with no valid response tokens.")

    if student_top_k_log_probs is not None or teacher_on_student_log_probs is not None:
        if student_top_k_log_probs is None or teacher_on_student_log_probs is None:
            raise ValueError("DTG requires both student and teacher Top-K log probabilities when either is provided.")
        if student_top_k_log_probs.shape != teacher_on_student_log_probs.shape:
            raise ValueError(
                "DTG student and teacher Top-K log probabilities must have the same shape, got "
                f"{tuple(student_top_k_log_probs.shape)} and {tuple(teacher_on_student_log_probs.shape)}."
            )
        if student_top_k_log_probs.shape[:2] != response_mask.shape:
            raise ValueError(
                "DTG Top-K log probabilities must start with response_mask dimensions, got "
                f"{tuple(student_top_k_log_probs.shape[:2])} and {tuple(response_mask.shape)}."
            )
        student_probs = student_top_k_log_probs.detach().float().exp()
        teacher_probs = teacher_on_student_log_probs.detach().float().exp()
        local_disagreement = 0.5 * (student_probs - teacher_probs).abs().sum(dim=-1)
        disagreement_source = "topk_tv"
    elif token_level_rewards is not None:
        rewards = token_level_rewards.detach().float()
        if rewards.shape[:2] != response_mask.shape:
            raise ValueError(
                "DTG token_level_rewards must start with response_mask dimensions, got "
                f"{tuple(rewards.shape[:2])} and {tuple(response_mask.shape)}."
            )
        local_disagreement = rewards.abs().sum(dim=-1) if rewards.ndim == 3 else rewards.abs()
        disagreement_source = "reward_magnitude"
    else:
        raise ValueError("DTG needs Top-K log probabilities or token_level_rewards to measure future disagreement.")

    local_disagreement = local_disagreement.to(mask.device) * mask
    num_blocks = math.ceil(sequence_length / block_size)
    padded_length = num_blocks * block_size
    pad_size = padded_length - sequence_length
    padded_mask = F.pad(mask, (0, pad_size))
    padded_disagreement = F.pad(local_disagreement, (0, pad_size))
    block_mask = padded_mask.reshape(batch_size, num_blocks, block_size)
    block_counts = block_mask.sum(dim=-1)
    active_blocks = block_counts > 0
    block_costs = (padded_disagreement.reshape(batch_size, num_blocks, block_size) * block_mask).sum(dim=-1)
    block_costs = block_costs / block_counts.clamp_min(1.0)

    # Exponentially discounted *mean* of later blocks.  Using a mean rather
    # than an unnormalized sum prevents long responses from receiving larger
    # costs solely because more tokens remain.
    future_disagreement = torch.zeros_like(block_costs)
    future_numerator = torch.zeros(batch_size, device=mask.device)
    future_denominator = torch.zeros(batch_size, device=mask.device)
    for block_index in range(num_blocks - 1, -1, -1):
        future_disagreement[:, block_index] = future_numerator / future_denominator.clamp_min(eps)
        is_active = active_blocks[:, block_index]
        next_numerator = block_costs[:, block_index] + block_gamma * future_numerator
        next_denominator = torch.ones_like(future_denominator) + block_gamma * future_denominator
        future_numerator = torch.where(is_active, next_numerator, future_numerator)
        future_denominator = torch.where(is_active, next_denominator, future_denominator)

    outcome_scores = _sequence_scores(true_reward_score, batch_size, mask.device)
    future_costs = future_disagreement + outcome_weight * (1.0 - outcome_scores).unsqueeze(-1)
    future_costs = future_costs * active_blocks

    inverse, num_groups = _group_inverse(group_ids, batch_size, mask.device)
    scatter_index = inverse.unsqueeze(-1).expand(-1, num_blocks)
    group_cost_sums = torch.zeros(num_groups, num_blocks, device=mask.device)
    group_counts = torch.zeros(num_groups, num_blocks, device=mask.device)
    group_cost_sums.scatter_add_(0, scatter_index, future_costs)
    group_counts.scatter_add_(0, scatter_index, active_blocks.float())

    peer_counts = group_counts[inverse] - 1.0
    peer_cost_sums = group_cost_sums[inverse] - future_costs
    peer_baseline = peer_cost_sums / peer_counts.clamp_min(1.0)
    valid_comparisons = active_blocks & (peer_counts > 0)

    # The PPO loss maximizes an advantage, while future disagreement is a
    # cost.  Responses with lower future cost than their siblings therefore
    # receive positive advantages.
    raw_block_advantages = (peer_baseline - future_costs) * valid_comparisons
    valid_raw_advantages = raw_block_advantages[valid_comparisons]
    raw_std = valid_raw_advantages.std(unbiased=False) if valid_raw_advantages.numel() > 0 else torch.tensor(0.0)
    block_advantages = raw_block_advantages
    if normalize_by_std and valid_raw_advantages.numel() > 0:
        block_advantages = block_advantages / raw_std.clamp_min(eps)
    block_advantages = block_advantages.clamp(-max_abs_advantage, max_abs_advantage)

    token_advantages = block_advantages.repeat_interleave(block_size, dim=-1)[:, :sequence_length]
    token_advantages = token_advantages * mask
    valid_token_advantages = token_advantages[mask.bool()]
    valid_local_disagreement = local_disagreement[mask.bool()]
    group_sizes = torch.bincount(inverse, minlength=num_groups).float()
    active_future_costs = future_costs[active_blocks]

    metrics = {
        "dtg/enabled": 1.0,
        "dtg/disagreement_source_topk_tv": float(disagreement_source == "topk_tv"),
        "dtg/local_disagreement_mean": float(valid_local_disagreement.mean().item()),
        "dtg/local_disagreement_max": float(valid_local_disagreement.max().item()),
        "dtg/future_cost_mean": float(active_future_costs.mean().item()),
        "dtg/raw_advantage_std": float(raw_std.item()),
        "dtg/advantage_abs_mean": float(valid_token_advantages.abs().mean().item()),
        "dtg/advantage_abs_max": float(valid_token_advantages.abs().max().item()),
        "dtg/positive_advantage_ratio": float((valid_token_advantages > 0).float().mean().item()),
        "dtg/outcome_success_mean": float(outcome_scores.mean().item()),
        "dtg/group_size_min": float(group_sizes.min().item()),
        "dtg/group_size_max": float(group_sizes.max().item()),
        "dtg/valid_comparison_ratio": float(valid_comparisons.float().mean().item()),
    }
    return token_advantages.float(), metrics
