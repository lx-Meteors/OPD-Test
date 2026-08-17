from __future__ import annotations

import torch


def compute_tri_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Signed triangular-discrimination cell rewards.

    Replaces the baseline per-cell reward -p * (log p - log q) with

        r_c = -sign(p_c - q_c) * (p_c - q_c)^2 / (p_c + q_c)

    where p, q are the student/teacher probabilities renormalized over the
    valid cells so both sum to 1 on the same support. Motivation (validated
    offline on rollout trajectories):

    * The baseline reward vanishes on cells the student barely ranks
      (p ~ 0) even when the teacher puts high mass there, so at confident
      mistakes it suppresses the student's wrong choice but never routes
      probability mass into the teacher's choice; TRI is symmetric in
      over-/under-weighting and sees both sides.
    * Each cell contribution is bounded in [-1, 1] (token sum bounded by 2),
      so no heavy-tailed reward spikes.

    Args:
        student_log_probs: (batch, seq, K) student log probs on the cells.
        teacher_log_probs: (batch, seq, K) teacher log probs on the same cells.
        valid_mask: (batch, seq, K) bool mask of cells that participate.

    Returns:
        (batch, seq, K) rewards, zero on invalid cells, dtype of the input.
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))

    # softmax over log probs == renormalization of the probabilities
    p = torch.softmax(s_lp, dim=-1)
    q = torch.softmax(t_lp, dim=-1)

    diff = p - q
    scores = -torch.sign(diff) * diff.square() / (p + q).clamp_min(1e-12)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)
