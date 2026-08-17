from __future__ import annotations

import torch

_LOG2_MU_MIN = -20.0
_BISECT_ITERS = 60


def _cell_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dim; -inf cells contribute exactly zero."""
    return -(log_probs.exp() * log_probs.clamp_min(-30.0)).sum(-1)


def compute_rkl_sdt_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Reverse-KL cell rewards with one-sided student-side detempering (sdt_gain).

    The teacher q is never touched. Per position, with p and q renormalized
    over the valid cells:

        if H(q) > H(p):  p~ = p^mu / Z(mu)  with mu in (0, 1] solving H(p~) = H(q)
        else:            p~ = p, mu = 1     (student kept as-is)

        r_c = -(1/mu) * p~_c * (log p~_c - log q_c)

    This is the plain reverse-KL field evaluated at the entropy-matched point
    of the student's own temperature family; the 1/mu factor is the Jacobian
    of the tempering map (d log p~ / d log p = mu), so the field is the pull-back
    of the rKL gradient at p~ onto the actual policy p.

    Why (validated offline on rollout trajectories):

    * Matching the raw soft teacher on student-sampled contexts feeds
      off-manifold epistemic noise back as "spread out" orders (the symmetric
      TRI run's entropy runaway). Evaluating rKL at the entropy-matched p~
      removes the temperature channel from the student side while leaving the
      teacher's content untouched: no chase (softened-teacher drift cut ~5x
      vs baseline), and if H(p) ever exceeds H(q) the field reverts to plain
      rKL, which sharpens - runaway is structurally impossible.
    * The 1/mu gain grows exactly where the student is much sharper than the
      teacher (confidently-wrong positions), restoring one-step deep-error
      force that plain student-tempering loses; unlike teacher-sharpening
      there is no q^lam tail collapse, so the field stays at teacher scale
      (max |r| ~ log(1/q_min)) even under a crisis-sharpened student policy.

    mu is found by bisection on log2(mu) in [-20, 0] (entropy is monotone in
    mu); 60 iterations of (batch, seq)-vectorized tensor ops, negligible cost.

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

    # bisection on log2(mu): mu < 1 softens the student, entropy decreases in mu
    lo = torch.full_like(h_p, _LOG2_MU_MIN)
    hi = torch.zeros_like(h_p)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        s_mid = torch.log_softmax(
            torch.where(valid_mask, s * mid.exp2().unsqueeze(-1), torch.full_like(s, neg_inf)), dim=-1
        )
        too_soft = _cell_entropy(s_mid) > h_q
        lo = torch.where(too_soft, mid, lo)
        hi = torch.where(too_soft, hi, mid)

    mu = torch.where(need, (0.5 * (lo + hi)).exp2(), torch.ones_like(h_p))
    s_dt = torch.log_softmax(
        torch.where(valid_mask, s * mu.unsqueeze(-1), torch.full_like(s, neg_inf)), dim=-1
    )

    scores = -(s_dt.exp() * (s_dt - t)) / mu.unsqueeze(-1)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)
