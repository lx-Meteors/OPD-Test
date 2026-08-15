# Copyright 2026 Chi2-OPD authors
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
"""Pearson-chi-squared tangent extrapolation for on-policy distillation.

For a teacher ``T`` and its pre-RL reference ``R``, define the implicit
reward ``r = log(T / R)``.  ExOPD obtains an exponentially tilted target

    q_exp proportional to T * exp(kappa * r).

Chi2-OPD instead solves a Pearson-chi-squared trust-region problem around
the teacher.  On a candidate set its closed-form density ratio is

    q_chi2 / T = 1 + kappa * (r - E_T[r]).

This is exactly the first-order (tangent) extrapolation of the exponential
target.  The implementation uses an analytic per-state trust radius so the
density ratio stays in a configurable positive interval.  Candidate mass is
kept equal to the teacher candidate mass; therefore kappa=0 is *exactly* the
existing top-k OPD target, even when the candidate set is truncated.
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


def _validate_config(config: Any) -> tuple[float, bool, float, float]:
    kappa = float(_cfg_get(config, "kappa", 0.25))
    adaptive_kappa = bool(_cfg_get(config, "adaptive_kappa", True))
    min_density_ratio = float(_cfg_get(config, "min_density_ratio", 0.1))
    max_density_ratio = float(_cfg_get(config, "max_density_ratio", 3.0))

    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"Chi2-OPD kappa must be finite and non-negative, got {kappa}.")
    if not math.isfinite(min_density_ratio) or not 0.0 < min_density_ratio <= 1.0:
        raise ValueError(f"Chi2-OPD min_density_ratio must be finite and in (0, 1], got {min_density_ratio}.")
    if not math.isfinite(max_density_ratio) or max_density_ratio < 1.0:
        raise ValueError(f"Chi2-OPD max_density_ratio must be finite and at least 1, got {max_density_ratio}.")
    return kappa, adaptive_kappa, min_density_ratio, max_density_ratio


def validate_chi2_opd_config(config: Any) -> None:
    """Validate scalar Chi2-OPD settings before workers are launched."""

    _validate_config(config)


def _check_inputs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    if student_log_probs.dim() != 3:
        raise ValueError(
            "Chi2-OPD expects top-k log-probabilities with shape [batch, response_length, K], "
            f"got {tuple(student_log_probs.shape)}."
        )
    if teacher_log_probs.shape != student_log_probs.shape or reference_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "Chi2-OPD student, teacher, and reference log-probabilities must have identical shapes; got "
            f"{tuple(student_log_probs.shape)}, {tuple(teacher_log_probs.shape)}, and "
            f"{tuple(reference_log_probs.shape)}."
        )
    if response_mask.shape != student_log_probs.shape[:2]:
        raise ValueError(
            f"Chi2-OPD response_mask has shape {tuple(response_mask.shape)}, "
            f"expected {tuple(student_log_probs.shape[:2])}."
        )

    for name, log_probs in (
        ("student", student_log_probs),
        ("teacher", teacher_log_probs),
        ("reference", reference_log_probs),
    ):
        active = response_mask.to(device=log_probs.device).bool().unsqueeze(-1)
        if not torch.all(torch.isfinite(log_probs) | ~active):
            raise ValueError(f"Chi2-OPD found a non-finite {name} log-probability at an active response position.")


@torch.no_grad()
def compute_chi2_target_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct the safe linear-extrapolation target on a shared candidate set.

    All three distributions are evaluated on the same token IDs.  Expectations
    are computed under the teacher distribution conditioned on that candidate
    set.  The final target retains the teacher's *raw* probability mass on the
    set, which makes ``kappa=0`` exactly identical to standard top-k OPD.
    """

    _check_inputs(student_log_probs, teacher_log_probs, reference_log_probs, response_mask)
    kappa, adaptive_kappa, min_density_ratio, max_density_ratio = _validate_config(config)

    device = student_log_probs.device
    active = response_mask.to(device=device).bool()
    student_lp = student_log_probs.detach().to(device=device, dtype=torch.float32)
    teacher_lp = teacher_log_probs.detach().to(device=device, dtype=torch.float32)
    reference_lp = reference_log_probs.detach().to(device=device, dtype=torch.float32)

    # Some generation backends use non-finite sentinels in padded positions.
    # They are semantically irrelevant, but NaN * 0 would still contaminate
    # the detached reward tensor, so replace them before any arithmetic.
    active_3d = active.unsqueeze(-1)
    student_lp = torch.where(active_3d, student_lp, torch.zeros_like(student_lp))
    teacher_lp = torch.where(active_3d, teacher_lp, torch.zeros_like(teacher_lp))
    reference_lp = torch.where(active_3d, reference_lp, torch.zeros_like(reference_lp))

    # Conditioning on the candidate set is necessary for a well-defined finite
    # top-k objective.  Additive log-normalization constants disappear after
    # centering the implicit reward.
    teacher_cond_lp = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
    teacher_cond_p = teacher_cond_lp.exp()
    student_cond_lp = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)
    student_cond_p = student_cond_lp.exp()

    implicit_reward = teacher_lp - reference_lp
    reward_mean = (teacher_cond_p * implicit_reward).sum(dim=-1, keepdim=True)
    centered_reward = implicit_reward - reward_mean

    requested_kappa = torch.full_like(reward_mean, kappa)
    kappa_used = requested_kappa.clone()

    if adaptive_kappa and kappa > 0.0:
        min_centered = centered_reward.min(dim=-1, keepdim=True).values
        max_centered = centered_reward.max(dim=-1, keepdim=True).values
        infinity = torch.full_like(kappa_used, torch.inf)

        lower_bound = torch.where(
            min_centered < 0.0,
            (1.0 - min_density_ratio) / (-min_centered).clamp_min(torch.finfo(torch.float32).tiny),
            infinity,
        )
        upper_bound = torch.where(
            max_centered > 0.0,
            (max_density_ratio - 1.0) / max_centered.clamp_min(torch.finfo(torch.float32).tiny),
            infinity,
        )
        kappa_used = torch.minimum(kappa_used, torch.minimum(lower_bound, upper_bound))

    density_ratio_unclipped = 1.0 + kappa_used * centered_reward
    density_ratio = density_ratio_unclipped.clamp(min=min_density_ratio, max=max_density_ratio)

    # Clamping is normally a numerical no-op under adaptive_kappa.  Renormalize
    # nevertheless so E_T[q/T] == 1 exactly, including non-adaptive ablations.
    density_normalizer = (teacher_cond_p * density_ratio).sum(dim=-1, keepdim=True)
    density_ratio = density_ratio / density_normalizer.clamp_min(torch.finfo(torch.float32).tiny)

    # Preserve the teacher's raw mass on the candidate set.  Equivalently,
    # target_lp = teacher_lp + log(q_cond / teacher_cond).
    target_lp = teacher_lp + density_ratio.log()

    target_cond_p = teacher_cond_p * density_ratio
    target_cond_p = target_cond_p / target_cond_p.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    target_cond_lp = target_cond_p.clamp_min(torch.finfo(torch.float32).tiny).log()

    active_float = active.to(torch.float32)
    reward_variance = (teacher_cond_p * centered_reward.square()).sum(dim=-1)
    target_shift_tv = 0.5 * (target_cond_p - teacher_cond_p).abs().sum(dim=-1)
    student_target_kl = (student_cond_p * (student_cond_lp - target_cond_lp)).sum(dim=-1)
    teacher_target_kl = (teacher_cond_p * (teacher_cond_lp - target_cond_lp)).sum(dim=-1)
    density_min = density_ratio.min(dim=-1).values
    density_max = density_ratio.max(dim=-1).values
    shrink = (kappa_used < requested_kappa - 1e-7).to(torch.float32).squeeze(-1)

    aux = {
        "chi2_opd_kappa": kappa_used.squeeze(-1) * active_float,
        "chi2_opd_kappa_shrunk": shrink * active_float,
        "chi2_opd_density_min": density_min * active_float,
        "chi2_opd_density_max": density_max * active_float,
        "chi2_opd_reward_mean": reward_mean.squeeze(-1) * active_float,
        "chi2_opd_reward_std": reward_variance.clamp_min(0.0).sqrt() * active_float,
        "chi2_opd_target_shift_tv": target_shift_tv * active_float,
        "chi2_opd_student_target_kl": student_target_kl * active_float,
        "chi2_opd_teacher_target_kl": teacher_target_kl * active_float,
    }
    return target_lp.to(dtype=teacher_log_probs.dtype), aux


@torch.no_grad()
def compute_chi2_opd_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return detached reverse-KL policy-gradient scores and diagnostics."""

    target_lp, aux = compute_chi2_target_log_probs(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        reference_log_probs=reference_log_probs,
        response_mask=response_mask,
        config=config,
    )
    active = response_mask.to(device=student_log_probs.device).bool().unsqueeze(-1)
    student_lp_raw = student_log_probs.detach().to(torch.float32)
    student_lp = torch.where(active, student_lp_raw, torch.zeros_like(student_lp_raw))
    target_lp_float = target_lp.detach().to(torch.float32)
    student_weights = torch.softmax(student_lp, dim=-1)
    scores = -(student_lp - target_lp_float) * student_weights
    scores = torch.where(active, scores, torch.zeros_like(scores))
    aux["chi2_opd_target_log_probs"] = target_lp
    return scores.to(dtype=student_log_probs.dtype), aux


def chi2_opd_metrics(aux: dict[str, torch.Tensor], response_mask: torch.Tensor) -> dict[str, float]:
    """Summarize Chi2-OPD tensors for the trainer/W&B metric path."""

    required = {
        "chi2_opd_kappa",
        "chi2_opd_kappa_shrunk",
        "chi2_opd_density_min",
        "chi2_opd_density_max",
        "chi2_opd_reward_mean",
        "chi2_opd_reward_std",
        "chi2_opd_target_shift_tv",
        "chi2_opd_student_target_kl",
        "chi2_opd_teacher_target_kl",
    }
    missing = required.difference(aux)
    if missing:
        raise ValueError(f"Chi2-OPD diagnostics are missing keys: {sorted(missing)}")

    valid = response_mask.to(aux["chi2_opd_kappa"].device).bool()
    if not valid.any():
        return {}

    def values(key: str) -> torch.Tensor:
        return aux[key][valid].detach().float()

    return {
        "chi2_opd/kappa_mean": values("chi2_opd_kappa").mean().item(),
        "chi2_opd/kappa_min": values("chi2_opd_kappa").min().item(),
        "chi2_opd/kappa_shrink_fraction": values("chi2_opd_kappa_shrunk").mean().item(),
        "chi2_opd/density_ratio_min": values("chi2_opd_density_min").min().item(),
        "chi2_opd/density_ratio_max": values("chi2_opd_density_max").max().item(),
        "chi2_opd/implicit_reward_mean": values("chi2_opd_reward_mean").mean().item(),
        "chi2_opd/implicit_reward_std": values("chi2_opd_reward_std").mean().item(),
        "chi2_opd/target_shift_tv": values("chi2_opd_target_shift_tv").mean().item(),
        "chi2_opd/student_target_kl": values("chi2_opd_student_target_kl").mean().item(),
        "chi2_opd/teacher_target_kl": values("chi2_opd_teacher_target_kl").mean().item(),
    }
