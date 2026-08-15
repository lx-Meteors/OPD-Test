# Copyright 2026 G-OPD reproduction authors
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
"""Generalized On-Policy Distillation (G-OPD) target construction.

G-OPD interprets the teacher-to-reference log-probability ratio as a
dense implicit reward and scales it by ``lambda``.  Its optimal target is

    q_lambda proportional to R * (T / R) ** lambda,

or, up to a state-dependent normalization constant,

    log q_lambda = log R + lambda * (log T - log R).

The state-dependent normalizer does not affect the policy-gradient
objective.  Keeping raw candidate log-probabilities also guarantees that
``lambda=1`` is exactly the repository's standard Student-Top-K OPD.
"""

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


def get_g_opd_lambda(config: Any) -> float:
    """Return and validate the G-OPD reward scale."""

    lambda_value = float(_cfg_get(config, "lambda", 1.25))
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError(f"G-OPD lambda must be finite and non-negative, got {lambda_value}.")
    return lambda_value


def validate_g_opd_config(config: Any) -> None:
    """Validate scalar G-OPD settings before workers are launched."""

    get_g_opd_lambda(config)


def _validate_inputs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    if student_log_probs.dim() != 3:
        raise ValueError(
            "G-OPD expects top-k log-probabilities with shape [batch, response_length, K], "
            f"got {tuple(student_log_probs.shape)}."
        )
    if teacher_log_probs.shape != student_log_probs.shape or reference_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "G-OPD student, teacher, and reference log-probabilities must have identical shapes; got "
            f"{tuple(student_log_probs.shape)}, {tuple(teacher_log_probs.shape)}, and "
            f"{tuple(reference_log_probs.shape)}."
        )
    if response_mask.shape != student_log_probs.shape[:2]:
        raise ValueError(
            f"G-OPD response_mask has shape {tuple(response_mask.shape)}, "
            f"expected {tuple(student_log_probs.shape[:2])}."
        )

    for name, log_probs in (
        ("student", student_log_probs),
        ("teacher", teacher_log_probs),
        ("reference", reference_log_probs),
    ):
        active = response_mask.to(device=log_probs.device).bool().unsqueeze(-1)
        if not torch.all(torch.isfinite(log_probs) | ~active):
            raise ValueError(f"G-OPD found a non-finite {name} log-probability at an active response position.")


@torch.no_grad()
def compute_g_opd_target_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct the official G-OPD exponential target on shared IDs."""

    _validate_inputs(student_log_probs, teacher_log_probs, reference_log_probs, response_mask)
    lambda_value = get_g_opd_lambda(config)

    device = student_log_probs.device
    active = response_mask.to(device=device).bool()
    active_3d = active.unsqueeze(-1)

    student_lp = student_log_probs.detach().to(device=device, dtype=torch.float32)
    teacher_lp = teacher_log_probs.detach().to(device=device, dtype=torch.float32)
    reference_lp = reference_log_probs.detach().to(device=device, dtype=torch.float32)
    student_lp = torch.where(active_3d, student_lp, torch.zeros_like(student_lp))
    teacher_lp = torch.where(active_3d, teacher_lp, torch.zeros_like(teacher_lp))
    reference_lp = torch.where(active_3d, reference_lp, torch.zeros_like(reference_lp))

    implicit_reward = teacher_lp - reference_lp
    target_lp = reference_lp + lambda_value * implicit_reward

    student_cond_lp = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)
    teacher_cond_lp = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
    target_cond_lp = target_lp - torch.logsumexp(target_lp, dim=-1, keepdim=True)
    student_cond_p = student_cond_lp.exp()
    teacher_cond_p = teacher_cond_lp.exp()
    target_cond_p = target_cond_lp.exp()

    reward_mean = (student_cond_p * implicit_reward).sum(dim=-1)
    reward_variance = (student_cond_p * (implicit_reward - reward_mean.unsqueeze(-1)).square()).sum(dim=-1)
    target_shift_tv = 0.5 * (target_cond_p - teacher_cond_p).abs().sum(dim=-1)
    student_target_kl = (student_cond_p * (student_cond_lp - target_cond_lp)).sum(dim=-1)
    teacher_target_kl = (teacher_cond_p * (teacher_cond_lp - target_cond_lp)).sum(dim=-1)
    active_float = active.to(torch.float32)

    aux = {
        "g_opd_lambda": torch.full_like(active_float, lambda_value) * active_float,
        "g_opd_implicit_reward_mean": reward_mean * active_float,
        "g_opd_implicit_reward_std": reward_variance.clamp_min(0.0).sqrt() * active_float,
        "g_opd_target_shift_tv": target_shift_tv * active_float,
        "g_opd_student_target_kl": student_target_kl * active_float,
        "g_opd_teacher_target_kl": teacher_target_kl * active_float,
    }
    return target_lp.to(dtype=teacher_log_probs.dtype), aux


@torch.no_grad()
def compute_g_opd_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return detached Student-Top-K G-OPD policy-gradient scores."""

    target_lp, aux = compute_g_opd_target_log_probs(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        reference_log_probs=reference_log_probs,
        response_mask=response_mask,
        config=config,
    )
    active = response_mask.to(device=student_log_probs.device).bool().unsqueeze(-1)
    student_lp_raw = student_log_probs.detach().to(torch.float32)
    student_lp = torch.where(active, student_lp_raw, torch.zeros_like(student_lp_raw))
    student_weights = torch.softmax(student_lp, dim=-1)
    scores = -(student_lp - target_lp.detach().to(torch.float32)) * student_weights
    scores = torch.where(active, scores, torch.zeros_like(scores))
    aux["g_opd_target_log_probs"] = target_lp
    return scores.to(dtype=student_log_probs.dtype), aux


def g_opd_metrics(aux: dict[str, torch.Tensor], response_mask: torch.Tensor) -> dict[str, float]:
    """Summarize G-OPD diagnostics for the trainer/W&B metric path."""

    required = {
        "g_opd_lambda",
        "g_opd_implicit_reward_mean",
        "g_opd_implicit_reward_std",
        "g_opd_target_shift_tv",
        "g_opd_student_target_kl",
        "g_opd_teacher_target_kl",
    }
    missing = required.difference(aux)
    if missing:
        raise ValueError(f"G-OPD diagnostics are missing keys: {sorted(missing)}")

    valid = response_mask.to(aux["g_opd_lambda"].device).bool()
    if not valid.any():
        return {}

    def values(key: str) -> torch.Tensor:
        return aux[key][valid].detach().float()

    return {
        "g_opd/lambda": values("g_opd_lambda").mean().item(),
        "g_opd/implicit_reward_mean": values("g_opd_implicit_reward_mean").mean().item(),
        "g_opd/implicit_reward_std": values("g_opd_implicit_reward_std").mean().item(),
        "g_opd/target_shift_tv": values("g_opd_target_shift_tv").mean().item(),
        "g_opd/student_target_kl": values("g_opd_student_target_kl").mean().item(),
        "g_opd/teacher_target_kl": values("g_opd_teacher_target_kl").mean().item(),
    }
