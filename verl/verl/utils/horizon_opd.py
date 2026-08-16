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


class HorizonInvariantOPDTracker:
    """Build OPD loss weights under a frozen response-horizon measure.

    Standard token-mean OPD implicitly weights each horizon bin by its current
    token frequency. Since that frequency changes with the student policy, the
    objective can appear to improve merely because responses become shorter.
    This tracker estimates a reference bin distribution during the first few
    updates, freezes it, and importance-corrects later batches back to that
    fixed distribution.
    """

    def __init__(self, config: Any, max_response_length: int):
        self.enabled = bool(_cfg_get(config, "enable", False))
        self.bin_size = int(_cfg_get(config, "bin_size", 1024))
        self.reference_steps = int(_cfg_get(config, "reference_steps", 5))
        self.alpha = float(_cfg_get(config, "alpha", 1.0))
        self.min_weight = float(_cfg_get(config, "min_weight", 0.25))
        self.max_weight = float(_cfg_get(config, "max_weight", 3.0))
        self.eps = float(_cfg_get(config, "eps", 1e-8))

        if max_response_length <= 0:
            raise ValueError(f"max_response_length must be positive, got {max_response_length}.")
        if self.bin_size <= 0:
            raise ValueError(f"horizon-opd bin_size must be positive, got {self.bin_size}.")
        if self.reference_steps <= 0:
            raise ValueError(f"horizon-opd reference_steps must be positive, got {self.reference_steps}.")
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"horizon-opd alpha must be in [0, 1], got {self.alpha}.")
        if (
            not math.isfinite(self.min_weight)
            or not math.isfinite(self.max_weight)
            or self.min_weight <= 0.0
            or self.max_weight < self.min_weight
        ):
            raise ValueError(
                f"horizon-opd requires 0 < min_weight <= max_weight, got {self.min_weight} and {self.max_weight}."
            )

        self.max_response_length = int(max_response_length)
        self.num_bins = math.ceil(self.max_response_length / self.bin_size)
        self.steps_seen = 0
        self._reference_accumulator = torch.zeros(self.num_bins, dtype=torch.float64)
        self.reference_mass: torch.Tensor | None = None

    @torch.no_grad()
    def compute(self, response_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if response_mask.ndim != 2:
            raise ValueError(
                f"horizon-opd expects response_mask with shape [batch, sequence], got {tuple(response_mask.shape)}."
            )
        if response_mask.shape[-1] > self.max_response_length:
            raise ValueError(
                "horizon-opd response length exceeds configured maximum: "
                f"{response_mask.shape[-1]} > {self.max_response_length}."
            )

        device = response_mask.device
        mask = response_mask.detach().float()
        total_tokens = mask.sum()
        if total_tokens.item() <= 0:
            raise ValueError("horizon-opd received a batch with no valid response tokens.")

        sequence_length = mask.shape[-1]
        position_bins = torch.arange(sequence_length, device=device) // self.bin_size
        token_counts = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        token_counts.scatter_add_(0, position_bins, mask.sum(dim=0))
        current_mass = token_counts / total_tokens

        self.steps_seen += 1
        if self.steps_seen <= self.reference_steps:
            self._reference_accumulator += current_mass.double().cpu()
            self.reference_mass = self._reference_accumulator / self.steps_seen
            bin_weights = torch.ones_like(current_mass)
            reference_ready = self.steps_seen == self.reference_steps
            correction_active = False
        else:
            if self.reference_mass is None:
                raise RuntimeError("horizon-opd reference distribution was not initialized.")
            reference_mass = self.reference_mass.to(device=device, dtype=current_mass.dtype)
            raw_ratio = reference_mass / current_mass.clamp_min(self.eps)
            bin_weights = raw_ratio.pow(self.alpha).clamp(min=self.min_weight, max=self.max_weight)
            bin_weights = torch.where(current_mass > self.eps, bin_weights, torch.zeros_like(bin_weights))
            reference_ready = True
            correction_active = True

        token_weights = bin_weights[position_bins].unsqueeze(0).expand_as(mask)
        token_weights = token_weights * mask

        reference_mass = self.reference_mass.to(device=device, dtype=current_mass.dtype)
        corrected_mass = current_mass * bin_weights
        corrected_mass = corrected_mass / corrected_mass.sum().clamp_min(self.eps)
        position_mass_tv = 0.5 * (current_mass - reference_mass).abs().sum()
        corrected_mass_tv = 0.5 * (corrected_mass - reference_mass).abs().sum()

        valid_weights = token_weights[mask.bool()]
        tail_start = max(0, (3 * self.num_bins) // 4)
        metrics = {
            "horizon_opd/reference_ready": float(reference_ready),
            "horizon_opd/correction_active": float(correction_active),
            "horizon_opd/reference_steps_seen": float(min(self.steps_seen, self.reference_steps)),
            "horizon_opd/current_mean_response_length": float(mask.sum(dim=-1).mean().item()),
            "horizon_opd/position_mass_tv": float(position_mass_tv.item()),
            "horizon_opd/corrected_mass_tv": float(corrected_mass_tv.item()),
            "horizon_opd/weight_mean": float(valid_weights.mean().item()),
            "horizon_opd/weight_min": float(valid_weights.min().item()),
            "horizon_opd/weight_max": float(valid_weights.max().item()),
            "horizon_opd/current_tail_mass": float(current_mass[tail_start:].sum().item()),
            "horizon_opd/reference_tail_mass": float(reference_mass[tail_start:].sum().item()),
        }
        for bin_index in range(self.num_bins):
            start = bin_index * self.bin_size
            end = min((bin_index + 1) * self.bin_size, self.max_response_length)
            suffix = f"{start}_{end}"
            metrics[f"horizon_opd/current_mass_{suffix}"] = float(current_mass[bin_index].item())
            metrics[f"horizon_opd/reference_mass_{suffix}"] = float(reference_mass[bin_index].item())
            metrics[f"horizon_opd/corrected_mass_{suffix}"] = float(corrected_mass[bin_index].item())
            metrics[f"horizon_opd/bin_weight_{suffix}"] = float(bin_weights[bin_index].item())

        # Response masks are commonly boolean or integer tensors.  Keep the
        # importance weights floating point so fractional ratios survive when
        # the tensor is attached to the training batch.
        return token_weights.float(), metrics


@torch.no_grad()
def normalize_horizon_weights_for_loss(
    horizon_weights: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize detached horizon weights without changing their relative values.

    ``seq-mean-token-sum`` is the exact aggregation for HI-OPD. Setting the
    total weight to the mini-batch size keeps the loss on the scale of a
    per-sequence mean and makes uniform weights equivalent to token-mean OPD.
    Other aggregation modes retain their usual scale by giving valid tokens a
    mean weight of one.
    """

    if horizon_weights.shape != response_mask.shape:
        raise ValueError(
            "horizon-opd weights must match response_mask, got "
            f"{tuple(horizon_weights.shape)} and {tuple(response_mask.shape)}."
        )

    mask = response_mask.detach().to(horizon_weights.dtype)
    weights = horizon_weights.detach() * mask
    weight_sum = weights.sum()
    if weight_sum.item() <= eps:
        raise ValueError("horizon-opd weights have zero mass on valid response tokens.")

    if loss_agg_mode == "seq-mean-token-sum":
        target_sum = float(response_mask.shape[0])
    else:
        target_sum = mask.sum()
    return weights * (target_sum / weight_sum.clamp_min(eps))
