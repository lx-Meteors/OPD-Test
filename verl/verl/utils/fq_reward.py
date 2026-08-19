from __future__ import annotations

import torch

_EPS = 1e-6


def compute_fq_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    return_aux: bool = False,
):
    """Fiber-quotient cell rewards by z-score matching.

        r = z(log q) - z(log p)

    where z(.) centers and unit-scales over the valid cells. Tempering acts
    on log probs as the affine group a*logq + b (a = 1/T > 0), and the
    z-score is the complete invariant of that action, so subtracting
    z-scores IS distillation on the quotient "distribution modulo
    temperature". No bisection, no regression, no projection - two moments
    per distribution, which is the minimum any temperature-quotient method
    can use.

    Consequences:
    * q = p^(1/T)/Z  =>  z(log q) = z(log p)  =>  r = 0 exactly, any T,
      two-sided, no entropy gate.
    * r = 0  iff  p lies on the teacher's tempering orbit {q^mu/Z}: the
      force self-terminates on structure match at the student's own level
      (offline: the FS cascade stops at H = 1.70 with the teacher at 1.89,
      alternative mass landing exactly on the teacher's structural share),
      so no persistent softening/sharpening channel exists (echo severed).
    * sum_c r_c = 0 (no off-support leak); <r, log p> = 0 on the fiber and
      O(||structure||^2) off it - a transport transient that decays to
      zero, not a level order. (Forcing <r, log p> = 0 everywhere is
      possible, r = z_q - corr * z_p, but it creates a dead point at
      anti-parallel teachers and starves deep repairs; conservation on the
      fiber plus transport off it is the correct resolution.)
    * No p-weighted dilution anywhere: confidently-wrong repairs get the
      correct cell's full z-gap (force share ~4% -> ~19%, top-1 flips in a
      few steps and entropy comes back down after the flip); true forks and
      content alternatives transmit at full scale (share ~10% -> ~50%).
    * Bounded: |r| <= 2*(K-1)/sqrt(K) (~7.5 at K=16), junk teacher evals
      stay finite by the z-score bound.

    Args:
        student_log_probs: (batch, seq, K) student log probs on the cells.
        teacher_log_probs: (batch, seq, K) teacher log probs on the same cells.
        valid_mask: (batch, seq, K) bool mask of cells that participate.
        return_aux: if True, also return per-position diagnostics.

    Returns:
        (batch, seq, K) rewards, zero on invalid cells, dtype of the input.
        With return_aux: dict with fq_level_slope (fitted level gap
        1 - corr * sd_q / sd_p; pure fog gives 1 - 1/T) and fq_level_r2
        (share of ||log p - log q||^2 explained by temperature alone:
        1 on fog, ~0 on pure structure).
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))

    # renormalize both distributions on the shared cell support
    s = torch.log_softmax(s_lp, dim=-1)
    t = torch.log_softmax(t_lp, dim=-1)

    m = valid_mask.to(s.dtype)
    n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def moments(lp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.where(valid_mask, lp, torch.zeros_like(lp))
        vc = (v - v.sum(dim=-1, keepdim=True) / n) * m
        sd = (vc.square().sum(dim=-1, keepdim=True) / n).sqrt()
        return vc, sd.clamp_min(_EPS)

    sc, s_sd = moments(s)
    tc, t_sd = moments(t)
    scores = tc / t_sd - sc / s_sd

    scores = torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)
    if not return_aux:
        return scores

    corr = (sc * tc).sum(dim=-1) / (n.squeeze(-1) * (s_sd * t_sd).squeeze(-1))
    sp, sq = s_sd.squeeze(-1), t_sd.squeeze(-1)
    ell = 1.0 - corr * sq / sp
    r2 = (sp - corr * sq).square() / (sp.square() - 2.0 * corr * sp * sq + sq.square()).clamp_min(_EPS)
    aux = {
        "fq_level_slope": torch.nan_to_num(ell, nan=0.0).to(student_log_probs.dtype),
        "fq_level_r2": torch.nan_to_num(r2.clamp(0.0, 1.0), nan=0.0).to(student_log_probs.dtype),
    }
    return scores, aux
