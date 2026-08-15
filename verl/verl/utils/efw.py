"""EFW-OPD: Edit-Field Weighted on-policy distillation (三条件蒸馏).

Objective:

.. math::

    \\mathcal{L}(\\theta)=\\mathbb{E}_{\\tau\\sim p_\\theta}\\Bigl[\\sum_t
    \\mathrm{sg}\\bigl(w(s_t)\\bigr)\\cdot
    \\mathrm{KL}\\bigl(p_\\theta(\\cdot\\mid s_t)\\,\\Vert\\,q(\\cdot\\mid s_t)\\bigr)\\Bigr],
    \\qquad w(s)=\\mathrm{KL}\\bigl(b\\,\\Vert\\,q\\bigr)(s)

with student ``p_theta``, frozen teacher ``q`` (the RL-trained model) and frozen
base ``b`` (the teacher's RL starting point == the student's initial weights,
served by the ref worker). The three factors carry the three conditions:
on-policy sampling ("the student reaches this state"), the frozen edit field
``w`` ("RL edited this state, by this much") and the residual KL ("not learned
yet"; it self-extinguishes as ``p -> q``).

This module only computes the edit field and multiplies it into the existing
distillation scores. The residual estimation machinery (the ``only_stu`` top-k
``rm_scores``) is untouched, so ``sg(w)`` is automatic: scores are constants by
the time they reach the policy-gradient update.

Field estimator (student top-16): the candidate set is the student's own
``T_k(p_theta)``, which the rollout pipeline already ships around, and

.. math::

    \\hat w(s) = \\sum_{z\\in T_k(p_\\theta)} b(z\\mid s)\\,
    \\bigl(\\log b - \\log q\\bigr)(z\\mid s)

clamped at zero (the exact KL is nonnegative; only the truncation can dip
below). At init ``p_theta = b`` so ``T_k(p_theta) = T_k(b)`` and the b-mass
coverage of the candidate set is >99%, which makes the estimate near-exact
(measured on step-0 trajectories against the full-vocabulary field: median
relative error 1.6%, r = 0.9955). During training the candidate set follows the
student; ``efw/b_mass_coverage`` monitors that drift.
"""

from __future__ import annotations

import torch

__all__ = ["compute_edit_field", "apply_efw_field_to_scores"]

# Below this field value a state counts as "un-anchored": the loss no longer
# pulls the student toward the teacher there. Only used for monitoring.
_LOW_FIELD_THRESHOLD = 1.0e-2


def compute_edit_field(
    ref_on_candidates_log_probs: torch.Tensor,
    teacher_on_candidates_log_probs: torch.Tensor,
    floor: float = 0.0,
) -> torch.Tensor:
    """Estimate the frozen edit field ``w(s) = KL(b || q)(s)`` on a candidate set.

    Args:
        ref_on_candidates_log_probs: ``(bs, response_length, k)`` base ``log b(z|s)``
            on the candidate ids (true log-probs, not renormalized over the set).
        teacher_on_candidates_log_probs: ``(bs, response_length, k)`` teacher
            ``log q(z|s)`` on the same candidate ids.
        floor: optional epsilon floor for the field (the one reserved knob of the
            method, default off). A positive floor keeps a minimal pull toward the
            teacher even at ``w == 0`` states, guarding against un-anchored drift.

    Returns:
        ``(bs, response_length)`` nonnegative field values.
    """
    b_lp = ref_on_candidates_log_probs.float()
    q_lp = teacher_on_candidates_log_probs.float()
    field = (torch.exp(b_lp) * (b_lp - q_lp)).sum(dim=-1)
    field = field.clamp_min(0.0)
    if floor > 0.0:
        field = field.clamp_min(float(floor))
    return field


def apply_efw_field_to_scores(
    scores: torch.Tensor,
    ref_on_student_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    student_top_k_log_probs: torch.Tensor | None = None,
    floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Scale distillation scores by the edit field and report field diagnostics.

    Args:
        scores: ``(bs, response_length, k)`` (per-candidate ``rm_scores``) or
            ``(bs, response_length)`` distillation scores; the residual factor.
        ref_on_student_log_probs: base log-probs on the student top-k ids.
        teacher_on_student_log_probs: teacher log-probs on the same ids.
        response_mask: ``(bs, response_length)`` valid-token mask.
        student_top_k_log_probs: optional student log-probs on the same ids; only
            feeds the low-field drift diagnostic ``efw/low_field_kl_p_b``.
        floor: see :func:`compute_edit_field`.

    Returns:
        weighted_scores: same shape as ``scores``.
        field: ``(bs, response_length)`` the edit field.
        metrics: scalar diagnostics (``efw/*``).
    """
    field = compute_edit_field(ref_on_student_log_probs, teacher_on_student_log_probs, floor=floor)
    field = field.to(scores.dtype)

    if scores.dim() == 3:
        weighted_scores = scores * field.unsqueeze(-1)
    elif scores.dim() == 2:
        weighted_scores = scores * field
    else:
        raise ValueError(f"efw expects 2D or 3D distillation scores, got shape {tuple(scores.shape)}.")

    with torch.no_grad():
        valid = response_mask.bool()
        field_valid = field[valid].float()
        low_field = field < _LOW_FIELD_THRESHOLD

        metrics: dict[str, float] = {}
        if field_valid.numel() > 0:
            quantiles = torch.quantile(field_valid, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
            metrics["efw/field_mean"] = field_valid.mean().item()
            metrics["efw/field_p50"] = quantiles[0].item()
            metrics["efw/field_p90"] = quantiles[1].item()
            metrics["efw/field_p99"] = quantiles[2].item()
            metrics["efw/field_max"] = field_valid.max().item()
            metrics["efw/field_frac_low"] = (field_valid < _LOW_FIELD_THRESHOLD).float().mean().item()

        # Candidate-set quality: how much of b's mass the student top-k still covers.
        # ~1.0 at init (student == base); a drop means the field estimate degrades.
        b_mass = torch.exp(ref_on_student_log_probs.float()).sum(dim=-1)
        if valid.any():
            metrics["efw/b_mass_coverage"] = b_mass[valid].mean().item()

        # Un-anchored drift monitor: truncated KL(p || b) restricted to low-field
        # states, where the loss no longer corrects the student.
        if student_top_k_log_probs is not None:
            p_lp = student_top_k_log_probs.float()
            kl_p_b = (torch.exp(p_lp) * (p_lp - ref_on_student_log_probs.float())).sum(dim=-1).clamp_min(0.0)
            low_valid = valid & low_field
            if low_valid.any():
                metrics["efw/low_field_kl_p_b"] = kl_p_b[low_valid].mean().item()

    return weighted_scores, field, metrics
