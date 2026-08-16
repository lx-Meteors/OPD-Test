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


def _masked_log_softmax(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_values = torch.where(valid_mask, values, torch.full_like(values, -torch.inf))
    log_normalizer = torch.logsumexp(masked_values, dim=-1, keepdim=True)
    has_valid = valid_mask.any(dim=-1, keepdim=True)
    log_normalizer = torch.where(has_valid, log_normalizer, torch.zeros_like(log_normalizer))
    return torch.where(valid_mask, values - log_normalizer, torch.zeros_like(values))


@torch.no_grad()
def fit_candidate_temperature(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    newton_steps: int = 6,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit teacher conditionals with a temperature-scaled student distribution.

    The fit is performed independently for every leading position over the last
    (candidate) dimension:

        a* = argmin_a KL(T_C || softmax(a * log S_C)).

    Returns ``a*``, calibrated conditional log probabilities, the fraction of
    the original conditional KL explained by temperature, and top-1 agreement.
    All returned tensors except the calibrated log probabilities omit the last
    candidate dimension.
    """

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "preference-opd requires matching student/teacher shapes, got "
            f"{tuple(student_log_probs.shape)} and {tuple(teacher_log_probs.shape)}."
        )
    if valid_mask.shape != student_log_probs.shape:
        raise ValueError(
            "preference-opd requires valid_mask to match log-prob shapes, got "
            f"{tuple(valid_mask.shape)} and {tuple(student_log_probs.shape)}."
        )
    if student_log_probs.ndim < 1:
        raise ValueError("preference-opd requires a candidate dimension.")
    if not math.isfinite(min_scale) or not math.isfinite(max_scale) or min_scale <= 0 or max_scale < min_scale:
        raise ValueError(f"preference-opd requires 0 < min_scale <= max_scale, got {min_scale} and {max_scale}.")
    if newton_steps < 0:
        raise ValueError(f"preference-opd newton_steps must be non-negative, got {newton_steps}.")

    student = student_log_probs.float()
    teacher = teacher_log_probs.float()
    valid = valid_mask.bool()
    valid_float = valid.to(student.dtype)
    valid_count = valid_float.sum(dim=-1, keepdim=True)
    has_multiple = valid_count > 1

    student_cond = _masked_log_softmax(student, valid)
    teacher_cond = _masked_log_softmax(teacher, valid)
    teacher_probs = torch.where(valid, teacher_cond.exp(), torch.zeros_like(teacher_cond))

    # A centered least-squares projection gives a much better Newton starting
    # point than a=1 when the teacher is substantially sharper or flatter.
    denom_count = valid_count.clamp_min(1.0)
    student_mean = (student * valid_float).sum(dim=-1, keepdim=True) / denom_count
    teacher_mean = (teacher * valid_float).sum(dim=-1, keepdim=True) / denom_count
    student_centered = torch.where(valid, student - student_mean, torch.zeros_like(student))
    teacher_centered = torch.where(valid, teacher - teacher_mean, torch.zeros_like(teacher))
    projection_denom = student_centered.square().sum(dim=-1, keepdim=True)
    projection_num = (student_centered * teacher_centered).sum(dim=-1, keepdim=True)
    scale = projection_num / projection_denom.clamp_min(eps)
    scale = scale.clamp(min=min_scale, max=max_scale)
    scale = torch.where(has_multiple & (projection_denom > eps), scale, torch.ones_like(scale))

    teacher_student_mean = (teacher_probs * student).sum(dim=-1, keepdim=True)
    for _ in range(newton_steps):
        calibrated_cond = _masked_log_softmax(scale * student, valid)
        calibrated_probs = torch.where(valid, calibrated_cond.exp(), torch.zeros_like(calibrated_cond))
        calibrated_student_mean = (calibrated_probs * student).sum(dim=-1, keepdim=True)
        gradient = calibrated_student_mean - teacher_student_mean
        variance = (calibrated_probs * (student - calibrated_student_mean).square()).sum(dim=-1, keepdim=True)
        candidate_scale = (scale - gradient / variance.clamp_min(eps)).clamp(min=min_scale, max=max_scale)
        scale = torch.where(has_multiple & (variance > eps), candidate_scale, scale)

    calibrated_cond = _masked_log_softmax(scale * student, valid)

    baseline_kl = (teacher_probs * (teacher_cond - student_cond)).sum(dim=-1)
    residual_kl = (teacher_probs * (teacher_cond - calibrated_cond)).sum(dim=-1)
    explained_ratio = torch.where(
        baseline_kl > eps,
        (1.0 - residual_kl / baseline_kl.clamp_min(eps)).clamp(min=0.0, max=1.0),
        torch.zeros_like(baseline_kl),
    )

    masked_student = torch.where(valid, student, torch.full_like(student, -torch.inf))
    masked_teacher = torch.where(valid, teacher, torch.full_like(teacher, -torch.inf))
    top1_agreement = (masked_student.argmax(dim=-1) == masked_teacher.argmax(dim=-1)).to(student.dtype)
    top1_agreement = torch.where(valid.any(dim=-1), top1_agreement, torch.zeros_like(top1_agreement))

    return (
        scale.squeeze(-1),
        calibrated_cond,
        explained_ratio,
        top1_agreement,
    )


@torch.no_grad()
def apply_preference_opd_to_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    reward_weights: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decompose OPD into preference and confidence transfer.

    For calibrated log probabilities Q, the ordinary OPD advantage decomposes
    exactly as

        log T - log S = (log T - log Q) + (log Q - log S).

    ``confidence_beta`` controls how much of the second term is retained. A
    value of 1 exactly recovers the input OPD score; 0 keeps only preference
    changes that cannot be explained by temperature on the candidate set.
    """

    enabled = bool(_cfg_get(config, "enable", False))
    standard_scores = (teacher_log_probs - student_log_probs) * reward_weights
    standard_scores = torch.where(valid_mask, standard_scores, torch.zeros_like(standard_scores))
    if not enabled:
        return standard_scores, {}

    confidence_beta = float(_cfg_get(config, "confidence_beta", 0.25))
    min_scale = float(_cfg_get(config, "min_scale", 0.1))
    max_scale = float(_cfg_get(config, "max_scale", 10.0))
    newton_steps = int(_cfg_get(config, "newton_steps", 6))
    eps = float(_cfg_get(config, "eps", 1e-8))
    if not math.isfinite(confidence_beta) or not 0.0 <= confidence_beta <= 1.0:
        raise ValueError(f"preference-opd confidence_beta must be in [0, 1], got {confidence_beta}.")

    scale, calibrated_cond, explained_ratio, top1_agreement = fit_candidate_temperature(
        student_log_probs,
        teacher_log_probs,
        valid_mask,
        min_scale=min_scale,
        max_scale=max_scale,
        newton_steps=newton_steps,
        eps=eps,
    )

    work_dtype = torch.float32
    student = student_log_probs.to(work_dtype)
    teacher = teacher_log_probs.to(work_dtype)
    valid = valid_mask.bool()
    weights = reward_weights.to(work_dtype)
    calibrated_cond = calibrated_cond.to(work_dtype)

    # Preserve the teacher's total mass on the selected candidate set. The
    # temperature fit therefore explains changes of confidence within C while
    # preference_adv measures only the conditional geometry left over.
    masked_teacher = torch.where(valid, teacher, torch.full_like(teacher, -torch.inf))
    teacher_log_mass = torch.logsumexp(masked_teacher, dim=-1, keepdim=True)
    teacher_log_mass = torch.where(
        valid.any(dim=-1, keepdim=True), teacher_log_mass, torch.zeros_like(teacher_log_mass)
    )
    calibrated_log_probs = calibrated_cond + teacher_log_mass

    preference_adv = torch.where(valid, teacher - calibrated_log_probs, torch.zeros_like(teacher))
    confidence_adv = torch.where(valid, calibrated_log_probs - student, torch.zeros_like(student))
    standard_adv = torch.where(valid, teacher - student, torch.zeros_like(student))

    if confidence_beta == 1.0:
        # Keep the baseline path bit-for-bit identical instead of relying on
        # cancellation through the calibrated distribution.
        scores = standard_scores
    else:
        combined_adv = preference_adv + confidence_beta * confidence_adv
        scores = torch.where(valid, combined_adv * weights, torch.zeros_like(combined_adv))
        scores = scores.to(standard_scores.dtype)

    valid_float = valid.to(work_dtype)
    valid_count = valid_float.sum(dim=-1).clamp_min(1.0)

    def candidate_abs_mean(value: torch.Tensor) -> torch.Tensor:
        return (value.abs() * valid_float).sum(dim=-1) / valid_count

    preference_abs = candidate_abs_mean(preference_adv)
    confidence_abs = candidate_abs_mean(confidence_adv)
    standard_abs = candidate_abs_mean(standard_adv)
    preference_fraction = torch.where(
        standard_abs > eps, preference_abs / standard_abs.clamp_min(eps), torch.zeros_like(preference_abs)
    )

    masked_student = torch.where(valid, student, torch.full_like(student, -torch.inf))
    student_log_mass = torch.logsumexp(masked_student, dim=-1)
    student_log_mass = torch.where(valid.any(dim=-1), student_log_mass, torch.zeros_like(student_log_mass))

    aux_tensors = {
        "preference_opd_temperature": scale.float(),
        "preference_opd_explained_ratio": explained_ratio.float(),
        "preference_opd_top1_agreement": top1_agreement.float(),
        "preference_opd_preference_abs": preference_abs.float(),
        "preference_opd_confidence_abs": confidence_abs.float(),
        "preference_opd_standard_abs": standard_abs.float(),
        "preference_opd_preference_fraction": preference_fraction.float(),
        "preference_opd_student_candidate_mass": student_log_mass.exp().float(),
        "preference_opd_teacher_candidate_mass": teacher_log_mass.squeeze(-1).exp().float(),
        "preference_opd_valid_candidate_count": valid_float.sum(dim=-1).float(),
    }
    return scores, aux_tensors
