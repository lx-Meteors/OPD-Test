from __future__ import annotations

import torch

_LAM_MAX = 64.0
_BISECT_ITERS = 50


def _cell_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dim; -inf cells contribute exactly zero."""
    return -(log_probs.exp() * log_probs.clamp_min(-30.0)).sum(-1)


def compute_rkl_dt_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Baseline reverse-KL cell rewards against a one-sided entropy-detempered teacher.

    The reward formula is the baseline's own  r_c = -p_c * (log p_c - log q_c);
    the single change is the target. Per position, with p and q renormalized
    over the valid cells:

        if H(q) > H(p):  q~ = q^lam / Z(lam)  with lam >= 1 solving H(q~) = H(p)
        else:            q~ = q                       (teacher kept as-is)

        r_c = -p_c * (log p_c - log q~_c)

    Why (validated offline on rollout trajectories):

    * On student-sampled contexts the teacher's extra softness is mostly
      off-manifold epistemic noise, not "many valid continuations". Matching
      it (or reverse-KL against it) keeps ordering the student to spread out;
      the symmetric-TRI run showed the resulting entropy runaway. Tempering
      is rank-preserving (log q~ = lam * log q - log Z), so the teacher's
      preferences pass through at the student's own contrast level, and the
      "sharpen up" edits (teacher sharper than student) pass untouched.
    * At confidently-wrong positions the sharpened q~ makes the wrong top's
      log-ratio large, so the suppression cascade (re-computed each step)
      hands the position to the teacher's choice: 6/6 confidently-wrong and
      ~99%/88% of rank-2/rank-3+ flips fixed in iterated one-step probes.

    lam is found by bisection on [1, 64] (entropy is monotone in lam);
    50 iterations of (batch, seq)-vectorized tensor ops, negligible cost.

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

    # renormalize both distributions on the shared cell support
    s = torch.log_softmax(s_lp, dim=-1)
    t = torch.log_softmax(t_lp, dim=-1)

    h_p = _cell_entropy(s)
    h_q = _cell_entropy(t)
    need = h_q > h_p + 1e-4

    lo = torch.ones_like(h_p)
    hi = torch.full_like(h_p, _LAM_MAX)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        t_mid = torch.log_softmax(
            torch.where(valid_mask, t * mid.unsqueeze(-1), torch.full_like(t, neg_inf)), dim=-1
        )
        too_soft = _cell_entropy(t_mid) > h_p
        lo = torch.where(too_soft, mid, lo)
        hi = torch.where(too_soft, hi, mid)

    lam = torch.where(need, 0.5 * (lo + hi), torch.ones_like(h_p))
    t_dt = torch.log_softmax(
        torch.where(valid_mask, t * lam.unsqueeze(-1), torch.full_like(t, neg_inf)), dim=-1
    )

    scores = -s.exp() * (s - t_dt)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)
