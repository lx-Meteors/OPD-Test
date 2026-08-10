# Copyright 2026 Prune-OPD authors
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
"""L-APD: Anchored Pairwise Distillation.

L-APD is a reward-free on-policy distillation objective. At every response
position the student's own sampled token ``y`` acts as an anchor, and the frozen
teacher states how ``y`` should be ranked against each important competitor
``z``:

.. math::

    L_t = \\sum_{z \\ne y_t} \\tilde q_t(z)
          \\mathrm{KL}_{\\mathrm B}(r_T(y_t, z) \\| r_S(y_t, z))

with Bradley-Terry style soft win rates ``r(y, z) = sigmoid(logit(y) - logit(z))``
and teacher candidate weights ``\\tilde q_t(z) = q_t(z) / (1 - q_t(y_t))``.

Because the candidate set is a truncated top-k, the token candidates alone do not
cover the vocabulary, and an objective built only from them cannot identify
``p_t(y_t)``: it is invariant to moving mass between ``{y_t} + candidates`` and the
truncated tail. One aggregated candidate restores coverage, and there are two
choices for it (see ``tail_candidate`` / ``complement_candidate``). Both enter as
ordinary candidates, weighted by their own teacher mass, so neither introduces a
free coefficient.

Because a softmax normalizer cancels inside a logit difference, all pairwise
margins can be computed from log-probabilities alone, so no raw logits have to
be materialized or communicated:
``logit(y) - logit(z) = log p(y) - log p(z)``.

Only the student receives gradients: teacher candidate weights and teacher win
rates enter as constants.

This module deliberately depends on ``torch`` only, so the objective can be
tested in isolation from the training stack.
"""

import torch
import torch.nn.functional as F

__all__ = ["compute_l_apd_token_loss", "l_apd_batch_keys"]

# Candidate ids together with the teacher log-probs evaluated on them.
_CANDIDATE_SOURCES = {
    "teacher": ("teacher_top_k_ids", "teacher_top_k_log_probs"),
    "student": ("student_top_k_ids", "teacher_on_student_log_probs"),
}


def l_apd_batch_keys(candidate_source: str = "teacher") -> tuple[str, ...]:
    """Batch keys that L-APD needs to survive until the actor update.

    Returns the candidate id key, the teacher log-probs on those ids, and the
    teacher log-probs on the anchor tokens.
    """
    if candidate_source not in _CANDIDATE_SOURCES:
        raise ValueError(
            f"Unknown l_apd.candidate_source: {candidate_source}. Expected one of {sorted(_CANDIDATE_SOURCES)}."
        )
    return _CANDIDATE_SOURCES[candidate_source] + ("teacher_log_probs",)

# Bernoulli probabilities are clamped away from {0, 1} before taking logs so
# that the reported KL/entropy diagnostics stay finite.
_PROB_EPS = 1.0e-7
# ``log(1 - exp(x))`` is only meaningful for x < 0; this bounds the aggregate
# tail probability from below by roughly 1e-6.
_LOG1MEXP_MAX = -1.0e-6


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``."""
    x = x.clamp(max=_LOG1MEXP_MAX)
    # Both branches stay finite everywhere on x <= _LOG1MEXP_MAX, which keeps
    # ``torch.where`` from propagating NaNs into the backward pass.
    return torch.where(x > -0.6931471805599453, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def _bernoulli_entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(min=_PROB_EPS, max=1.0 - _PROB_EPS)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def compute_l_apd_token_loss(
    *,
    student_anchor_log_probs: torch.Tensor,
    student_candidate_log_probs: torch.Tensor,
    teacher_anchor_log_probs: torch.Tensor,
    teacher_candidate_log_probs: torch.Tensor,
    candidate_mask: torch.Tensor,
    response_mask: torch.Tensor,
    tail_candidate: bool = True,
    complement_candidate: bool = True,
    normalize_weights: bool = True,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the per-token L-APD loss.

    Args:
        student_anchor_log_probs: ``(bs, response_length)`` student ``log p(y_t)``,
            differentiable.
        student_candidate_log_probs: ``(bs, response_length, k)`` student
            ``log p(z)`` on the candidate ids, differentiable.
        teacher_anchor_log_probs: ``(bs, response_length)`` teacher ``log q(y_t)``.
        teacher_candidate_log_probs: ``(bs, response_length, k)`` teacher
            ``log q(z)`` on the same candidate ids.
        candidate_mask: ``(bs, response_length, k)`` mask selecting candidates
            that take part in the loss, i.e. ``z != y_t`` at valid positions.
        response_mask: ``(bs, response_length)`` mask of valid response tokens.
        tail_candidate: append one aggregated candidate carrying the probability mass
            outside the candidate set. Its opponents then form a genuine partition of
            the non-anchor vocabulary, so the tail is supervised too.
        complement_candidate: append one aggregated candidate standing for everything
            except the anchor, i.e. the anchor term ``KL_B(q(y) || p(y))`` written in
            pairwise form. Ignored when ``tail_candidate`` is set, which already covers
            the vocabulary.

            One of the two is required. Token candidates alone constrain the student
            only through the logit differences ``S(y) - S(z)``, which leaves one
            direction free: scaling the mass of ``{y} + candidates`` up or down, with
            the slack absorbed by the truncated tail, does not change the loss at all.
            Either aggregated candidate closes that gap, because matching the teacher
            ratio on a set of cells that covers the whole vocabulary forces
            ``p(y) = q(y)`` once both sides are normalized.
        normalize_weights: divide the candidate weights by their own sum instead
            of by ``1 - q(y_t)``. Only meaningful together with an aggregated
            candidate; with ``tail_candidate=True`` the two are equivalent up to
            floating point error.
        eps: floor for the weight normalizer.

    Returns:
        token_loss: ``(bs, response_length)`` loss, zero outside ``response_mask``.
        diagnostics: dict of detached ``(bs, response_length)`` tensors.
    """
    student_anchor = student_anchor_log_probs.float()
    student_candidates = student_candidate_log_probs.float()
    teacher_anchor = teacher_anchor_log_probs.float().detach()
    teacher_candidates = teacher_candidate_log_probs.float().detach()

    valid = candidate_mask.bool() & response_mask.unsqueeze(-1).bool()

    # Star-shaped pairwise problem centered on the anchor token.
    pair_logits = student_anchor.unsqueeze(-1) - student_candidates
    teacher_margins = teacher_anchor.unsqueeze(-1) - teacher_candidates
    raw_weights = torch.exp(teacher_candidates) * valid

    if tail_candidate:
        neg_inf = torch.finfo(torch.float32).min
        covered_student = torch.cat(
            [student_anchor.unsqueeze(-1), student_candidates.masked_fill(~valid, neg_inf)], dim=-1
        )
        covered_teacher = torch.cat(
            [teacher_anchor.unsqueeze(-1), teacher_candidates.masked_fill(~valid, neg_inf)], dim=-1
        )
        student_tail = _log1mexp(torch.logsumexp(covered_student, dim=-1))
        teacher_tail = _log1mexp(torch.logsumexp(covered_teacher, dim=-1))

        pair_logits = torch.cat([pair_logits, (student_anchor - student_tail).unsqueeze(-1)], dim=-1)
        teacher_margins = torch.cat([teacher_margins, (teacher_anchor - teacher_tail).unsqueeze(-1)], dim=-1)
        raw_weights = torch.cat(
            [raw_weights, torch.exp(teacher_tail).unsqueeze(-1) * response_mask.unsqueeze(-1)], dim=-1
        )
    elif complement_candidate:
        # The coarsest opponent: everything except the anchor, aggregated. Its margin
        # log(p(y) / (1 - p(y))) makes the teacher win rate exactly q(y), so this pair
        # is KL_B(q(y) || p(y)) -- the anchor term -- expressed in the same pairwise
        # form as every other candidate, and weighted by q(y) like every other candidate.
        student_complement = _log1mexp(student_anchor)
        teacher_complement = _log1mexp(teacher_anchor)

        pair_logits = torch.cat([pair_logits, (student_anchor - student_complement).unsqueeze(-1)], dim=-1)
        teacher_margins = torch.cat(
            [teacher_margins, (teacher_anchor - teacher_complement).unsqueeze(-1)], dim=-1
        )
        raw_weights = torch.cat(
            [raw_weights, torch.exp(teacher_anchor).unsqueeze(-1) * response_mask.unsqueeze(-1)], dim=-1
        )

    teacher_pair_prob = torch.sigmoid(teacher_margins)

    if normalize_weights:
        normalizer = raw_weights.sum(dim=-1, keepdim=True)
    else:
        normalizer = (1.0 - torch.exp(teacher_anchor)).unsqueeze(-1)
    weights = (raw_weights / normalizer.clamp(min=eps)).detach()

    pair_loss = F.binary_cross_entropy_with_logits(pair_logits, teacher_pair_prob, reduction="none")
    token_loss = (weights * pair_loss).sum(dim=-1) * response_mask

    with torch.no_grad():
        student_pair_prob = torch.sigmoid(pair_logits)
        pairwise_kl = (weights * (pair_loss - _bernoulli_entropy(teacher_pair_prob))).sum(dim=-1)
        agreement = ((student_pair_prob - 0.5) * (teacher_pair_prob - 0.5) > 0).float()
        diagnostics = {
            "bernoulli_kl": pairwise_kl,
            "pairwise_agreement": (weights * agreement).sum(dim=-1),
            "pairwise_gap": (weights * (student_pair_prob - teacher_pair_prob).abs()).sum(dim=-1),
            "teacher_anchor_prob": torch.exp(teacher_anchor),
            "student_anchor_prob": torch.exp(student_anchor),
            "anchor_in_candidates": 1.0 - candidate_mask.float().prod(dim=-1),
            "candidate_count": valid.float().sum(dim=-1),
            # Teacher mass covered by the candidate set: 1 when the tail is modelled.
            "candidate_weight_sum": weights.sum(dim=-1),
        }
        if tail_candidate:
            diagnostics["tail_weight"] = weights[..., -1]
            diagnostics["teacher_tail_prob"] = torch.exp(teacher_tail)
            diagnostics["student_tail_prob"] = torch.exp(student_tail)
        elif complement_candidate:
            diagnostics["anchor_weight"] = weights[..., -1]
            # KL_B(q(y) || p(y)), so it reaches 0 exactly when p(y) == q(y).
            diagnostics["anchor_kl"] = pair_loss[..., -1] - _bernoulli_entropy(teacher_pair_prob[..., -1])

    return token_loss, diagnostics
