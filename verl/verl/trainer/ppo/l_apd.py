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

    L_t = \\sum_{z \\ne y_t} w_t(z)\\,
          \\mathrm{KL}(\\tilde p_{y_t, z} \\| \\tilde q_{y_t, z})

A single comparison only asks which of ``y_t`` and ``z`` wins, so both sides are
restricted to ``{y_t, z}`` and renormalized there,
``p~(v) = p(v) / (p(y_t) + p(z))``, and each pair is scored with the plain
reverse KL between those two-outcome distributions, i.e. the usual
``sum_v p~(v) log[p~(v) / q~(v)]`` over the two outcomes. Dropping either
outcome would leave a bare log-ratio, which is not a divergence and is unbounded
below.

The mixture weights over the opponents default to the *student's* conditional
mass, ``w_t(o) = sg[p_t(o) / (1 - p_t(y_t))]``, evaluated on the same candidate
ids and detached: weights decide how loudly each duel is scored, never where
gradients flow. Student weighting is the direction-consistent companion of the
reverse per-pair KL -- the chain rule of ``KL(p || q)`` over a partition weights
the conditional cells by the student's conditional mass -- and it is
closed-loop: surplus student mass (a bloated tail, a wrongly favoured
alternative) raises the weight of exactly that column until it is drained.
``weight_source="teacher"`` keeps the historical open-loop weighting
``q_t(o) / (1 - q_t(y_t))`` as an ablation; it allocates budget by the
teacher's preference regardless of where the student's error actually is.

``pair_divergence`` picks the direction and defaults to the reverse one above.
``log_ratio`` additionally offers the ablation that keeps only the ``v = y_t``
outcome of that sum, i.e. the bare ``log[p~(y_t) / q~(y_t)]``. That one is not a
divergence: it is monotone in the student margin, so the teacher cancels out of
the gradient and the objective is unbounded below. It is the quantity the OPD
baseline uses as a *reward* under ``no_grad``, where monotonicity is harmless
because the gradient comes from the policy-gradient term instead.

Because the candidate set is a truncated top-k, the token candidates alone do not
cover the vocabulary, and an objective built only from them cannot identify
``p_t(y_t)``: it is invariant to moving mass between ``{y_t} + candidates`` and the
truncated tail. The tail block (``tail_candidate``, the default) restores coverage
as one more ordinary opponent carrying the probability mass outside the candidate
set, which makes the opponents a true partition of the non-anchor vocabulary: every
non-anchor token takes part in exactly one pair, and the weights are the weighting
side's distribution over genuine alternatives (the student's, by default). With
zero token candidates the loss reduces exactly to the anchor calibration term
``KL((p(y), 1-p(y)) || (q(y), 1-q(y)))``. The complement opponent
(``complement_candidate``) is the historical variant that aggregated *everything*
except the anchor instead; it counts the named candidates a second time inside the
aggregate, and is kept as an ablation. Both enter as ordinary candidates, weighted
by their own mass on the weighting side, so neither introduces a free coefficient.

Because a softmax normalizer cancels inside a logit difference, every restricted
pair probability is just ``p~(y) = sigmoid(log p(y) - log p(z))``, so all pairs
can be scored from log-probabilities alone and no raw logits have to be
materialized or communicated.

Only the student's pair margins receive gradients: the mixture weights
(whichever side supplies them) and the teacher side of every pair enter as
constants.

This module deliberately depends on ``torch`` only, so the objective can be
tested in isolation from the training stack.
"""

import torch
import torch.nn.functional as F

__all__ = ["compute_l_apd_token_loss", "l_apd_batch_keys"]

_PAIR_DIVERGENCES = (
    "reverse_kl",
    "forward_kl",
    "log_ratio",
    "partition_kl",
    "odds_kl",
    "jeffreys",
    "order_gated_kl",
)

# Whose probabilities set the per-pair mixture weights.
_WEIGHT_SOURCES = ("student", "teacher")

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

# Pair probabilities are clamped away from {0, 1} before taking logs so that the
# reported KL/entropy diagnostics stay finite.
_PROB_EPS = 1.0e-7
# ``log(1 - exp(x))`` is only meaningful for x < 0; this bounds the aggregate
# tail probability from below by roughly 1e-6.
# float32 log-probs stop resolving ``1 - exp(x)`` just below this, so the forward
# value is capped here to keep ``log(1 - exp(x))`` finite at x = 0.
_LOG1MEXP_MAX = -1.0e-6


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``.

    The cap is transparent to autograd. Letting it block the gradient would zero out
    the aggregate-opponent column exactly where the student is most overconfident --
    the case on-policy distillation exists to correct -- and it would do so as a
    cliff rather than a decay. Passing the gradient through is safe because the pair
    ``r (1 - r)`` factor cancels the ``1 / (1 - exp(x))`` blow-up, leaving a
    composite gradient that only grows like ``log[1 / (1 - exp(x))]``.
    """
    capped = x.clamp(max=_LOG1MEXP_MAX)
    x = capped.detach() + (x - x.detach())
    # Both branches stay finite everywhere on x <= _LOG1MEXP_MAX, which keeps
    # ``torch.where`` from propagating NaNs into the backward pass.
    return torch.where(x > -0.6931471805599453, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def _order_gated_kl_token_loss(
    student_anchor: torch.Tensor,
    student_candidates: torch.Tensor,
    teacher_anchor: torch.Tensor,
    teacher_candidates: torch.Tensor,
    valid: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Order-gated bidirectional KL over the pure student top-k cells.

    Baseline-mechanics form: the differentiable object is ``sg[w_c Delta_c] * log p_c``
    per cell, i.e. a frozen per-cell k1 coefficient carried by the student log-prob --
    exactly the OPD baseline's PPO-wings force at ratio ~= 1. ``Delta_c`` is the raw
    log-ratio ``log p_c - log q_c`` (the baseline's ``kl_val``) and the weight vector
    interpolates between the baseline's own two weight modes,

        ``w = (1 - lambda) p~ + lambda q~``,

    with ``p~`` / ``q~`` the student / teacher probabilities renormalized over the
    cells (``student_p`` / ``teacher_p`` in baseline terms). ``lambda`` is the top-1
    anchored order gap: the student's strongest cell is the reference, and

        ``lambda = max_j |sigma(m_j) - sigma(m_j^T)|``

    over the remaining cells, detached. At ``lambda = 0`` the loss reproduces the
    baseline's reverse-KL scoring cell by cell (student_p weights); at ``lambda = 1``
    it is the baseline's teacher_p (forward-KL) variant, whose teacher-mass weights
    keep pulling cells the student has starved (``p~ ~ 0`` kills the reverse weight,
    the classic dead zone). Both directions share the fixed point ``p_c = q_c`` on
    raw masses, so the gate reroutes the optimization path, never the destination.

    The cells are the candidates exactly as passed (the sampled token is *not*
    deduplicated out): with ``candidate_source="student"`` this is the baseline's
    ``student_top_k_ids`` set verbatim. The anchor tensors only feed diagnostics.
    """
    neg_inf = torch.finfo(torch.float32).min
    masked_student = student_candidates.masked_fill(~valid, neg_inf)
    masked_teacher = teacher_candidates.masked_fill(~valid, neg_inf)

    with torch.no_grad():
        # Order gate: reference = the student's top-1 cell, the token reverse KL
        # sharpens toward. Any candidate whose win-probability against that mode
        # differs between the two models raises lambda toward coverage.
        top1 = masked_student.argmax(dim=-1, keepdim=True)
        student_margins = masked_student.gather(-1, top1) - masked_student
        teacher_margins = masked_teacher.gather(-1, top1) - masked_teacher
        order_gap = (torch.sigmoid(student_margins) - torch.sigmoid(teacher_margins)).abs()
        order_gap = torch.where(valid, order_gap, torch.zeros_like(order_gap))
        order_lambda = order_gap.amax(dim=-1)

        # The baseline's two weight modes over the same cells, blended by the gate.
        student_weights = torch.softmax(masked_student, dim=-1)
        teacher_weights = torch.softmax(masked_teacher, dim=-1)
        delta = student_candidates - teacher_candidates
        gate = order_lambda.unsqueeze(-1)
        blended_weights = (1.0 - gate) * student_weights + gate * teacher_weights
        coefficients = torch.where(valid, blended_weights * delta, torch.zeros_like(delta))

    token_loss = (coefficients * student_candidates).sum(dim=-1) * response_mask

    with torch.no_grad():
        zeros = torch.zeros_like(delta)
        diagnostics = {
            "order_lambda": order_lambda,
            # Same objects the baseline logs as rewards, so the dashboards compare 1:1.
            "rev_kl_est": torch.where(valid, student_weights * delta, zeros).sum(dim=-1),
            "fwd_kl_est": torch.where(valid, teacher_weights * (-delta), zeros).sum(dim=-1),
            "mode_agreement": (masked_student.argmax(dim=-1) == masked_teacher.argmax(dim=-1)).float(),
            "student_covered_prob": (torch.exp(student_candidates) * valid).sum(dim=-1),
            "teacher_covered_prob": (torch.exp(teacher_candidates) * valid).sum(dim=-1),
            "teacher_anchor_prob": torch.exp(teacher_anchor),
            "student_anchor_prob": torch.exp(student_anchor),
            "candidate_count": valid.float().sum(dim=-1),
        }
    return token_loss, diagnostics


def _reverse_pair_kl(student_margins: torch.Tensor, teacher_margins: torch.Tensor) -> torch.Tensor:
    """``sum_v p~(v) log[p~(v) / q~(v)]`` over the two outcomes of each pair.

    Written from log-sigmoids so it stays finite for every real margin: the
    ``p~ log p~`` terms vanish at the saturated ends because ``p~`` decays
    exponentially while its log only grows linearly.
    """
    student_prob = torch.sigmoid(student_margins)
    return student_prob * (F.logsigmoid(student_margins) - F.logsigmoid(teacher_margins)) + (
        1.0 - student_prob
    ) * (F.logsigmoid(-student_margins) - F.logsigmoid(-teacher_margins))


def _pair_entropy(prob: torch.Tensor) -> torch.Tensor:
    """Entropy of the two-outcome distribution ``(prob, 1 - prob)``."""
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
    complement_candidate: bool = False,
    normalize_weights: bool = True,
    weight_source: str = "student",
    pair_divergence: str = "jeffreys",
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
            outside the candidate set (the default). The opponents then form a genuine
            partition of the non-anchor vocabulary -- every non-anchor token appears in
            exactly one pair -- and with ``normalize_weights`` the weights are exactly
            the teacher's distribution over genuine alternatives, ``q(o) / (1 - q(y_t))``.
            With zero token candidates the loss reduces to the anchor term
            ``KL((p(y), 1 - p(y)) || (q(y), 1 - q(y)))``.
        complement_candidate: append one aggregated candidate standing for everything
            except the anchor, i.e. the anchor term -- the KL between the two-outcome
            distributions ``(p(y), 1 - p(y))`` and ``(q(y), 1 - q(y))`` -- written in
            the same pairwise form as every other candidate. Ablation only: the
            complement contains the named candidates a second time, so they are counted
            twice, and this column takes ``q(y_t) / Z`` (~0.68 measured) of the weight.
            Ignored when ``tail_candidate`` is set, which already covers the vocabulary.

            One of the two is required. Token candidates alone constrain the student
            only through the logit differences ``S(y) - S(z)``, which leaves one
            direction free: scaling the mass of ``{y} + candidates`` up or down, with
            the slack absorbed by the truncated tail, does not change the loss at all.
            Either aggregated candidate closes that gap, because matching the teacher
            ratio on a set of cells that covers the whole vocabulary forces
            ``p(y) = q(y)`` once both sides are normalized.
        normalize_weights: divide the candidate weights by their own sum instead
            of by the weighting side's ``1 - anchor mass``. Only meaningful
            together with an aggregated candidate; with ``tail_candidate=True``
            the two are equivalent up to floating point error.
        weight_source: whose probabilities set the mixture weights. ``student``
            (the default) uses the detached student conditional mass
            ``sg[p(o) / (1 - p(y_t))]``: closed-loop, because surplus student
            mass raises the weight of its own column until it is drained, and
            direction-consistent with the reverse per-pair KL, whose chain rule
            weights conditional cells by student mass. ``teacher`` keeps the
            historical ``q(o) / (1 - q(y_t))`` as an ablation: open-loop, its
            budget follows the teacher's preference even where the student
            already agrees, which measurably lets transient mass pile up in the
            weakly weighted tail column (3x the teacher's tail mass at peak).
        pair_divergence: which direction each pair is scored with. Both vanish
            exactly at ``p~ == q~`` and agree to first order around it, so this only
            changes how large disagreements are treated.

            ``reverse_kl`` uses ``KL(p~ || q~)``, whose margin gradient is
            ``sigmoid'(m) * (m - m_T)``: direct margin matching, but bounded in the
            student margin, so a confidently misranked pair contributes at most
            ``-log q~(y)`` and its gradient decays like ``sigmoid'(m)``.

            ``forward_kl`` uses ``KL(q~ || p~)``, whose margin gradient is
            ``p~(y) - q~(y)``: unbounded loss, and the gradient saturates at magnitude
            1 instead of vanishing, so confident misrankings keep full pull.

            ``log_ratio`` keeps only the ``v = y_t`` outcome of the reverse sum, i.e.
            ``log[p~(y) / q~(y)]``, which on the complement column is
            ``log[p(y) / q(y)]``. Ablation only: its margin gradient is ``1 - p~(y)``,
            positive whatever the teacher says, so it has no stationary point and is
            unbounded below. At a fixed anchor it drives ``p(y_t)`` to zero, but under
            on-policy resampling the dynamics invert -- pushing down whatever was just
            sampled makes the mass flee into an ever narrower set, so entropy collapses
            and ``p(y_t)`` rises instead. A run of it took ``actor/entropy`` from 0.66
            to 0.04 while the honest ``pair_kl`` grew fourfold.

            ``partition_kl`` drops the pairwise wrapper altogether: one categorical
            reverse KL between the two distributions coarsened onto the partition
            ``{y_t} + valid candidates + tail`` (requires ``tail_candidate``; the
            cells cover the vocabulary, so both coarsened distributions sum to one
            exactly). Explicit weights and the sigmoid geometry disappear from the
            loss: the per-cell gradient is ``p(c) * (Delta_c - E_p[Delta])`` with
            ``Delta = log[p(c) / q(c)]`` -- the student-mass weighting emerges from
            the KL itself (exactly, no stop-gradient approximation), and there is no
            ``sigmoid'(m)`` gate, so confidently wrong anchors keep a full-strength
            corrective gradient (the pairwise forms gate exactly those to ~zero).
            ``weight_source`` and ``normalize_weights`` do not affect this loss;
            weights are still computed for diagnostics continuity.

            ``odds_kl`` keeps one term per duel but drops the within-duel
            normalization (the sigmoid) entirely: each duel compares the two
            models' *odds* ``rho = p(c) / p(y_t)`` with the generalized KL
            ``rho log(rho / rho_T) - rho + rho_T``, which folds to
            ``rho * phi(u)`` with ``phi(u) = e^u - u - 1`` and duel gap
            ``u = m - m_T`` (also the exact KL between Poisson laws with these
            intensities). The token loss is ``sg[p(y_t)]`` times the sum over
            duels, so the margin gradient is exactly ``p(c) * (m - m_T)``: a
            single mass factor, no ``sigmoid'`` gate, bounded by ``|m - m_T|``.
            Explicit weights and ``normalize_weights`` do not affect this loss
            either; ``sg[p(y_t)] * sum_c GKL`` upper-bounds the partition KL
            (tight at ``p(y_t) = q(y_t)``).

            ``order_gated_kl`` leaves the pairwise geometry entirely and scores the
            candidate cells the way the OPD baseline does: per-cell coefficient
            ``sg[w_c * (log p_c - log q_c)]`` carried by the differentiable
            ``log p_c`` (the baseline's PPO-wings force at ratio ~= 1), with the
            weight vector interpolating between the baseline's student_p and
            teacher_p modes, ``w = (1 - lambda) p~ + lambda q~``. ``lambda`` is the
            top-1 anchored order gap ``max_j |sigma(m_j) - sigma(m_j^T)|``
            (detached): order agreement => baseline reverse-KL scoring, order
            disagreement => teacher-mass (forward-KL) weights that rescue starved
            cells. Requires ``tail_candidate=False`` and
            ``complement_candidate=False``; ``candidate_mask`` should keep the
            sampled token so the cells are the pure student top-k (the caller
            handles this). ``weight_source`` and ``normalize_weights`` are ignored.
            The anchor tensors feed diagnostics only.

            ``jeffreys`` symmetrizes the per-duel Bernoulli KL: for two-point
            distributions the sum of both directions folds to the closed form
            ``(sigma(m) - sigma(m_T)) * (m - m_T)`` -- win-probability gap times
            margin gap, both factors sharing sign, so each duel is a genuine
            divergence with the same zero as ``reverse_kl``. Its margin gradient
            ``sigmoid'(m) (m - m_T) + (sigma(m) - sigma(m_T))`` keeps the reverse
            geometry near the optimum but adds the never-vanishing forward pull,
            so duels the student has confidently decided the wrong way (where
            ``sigmoid'`` alone is exponentially dead -- the measured late-phase
            stall: agreement flat at 0.955, pair_kl floored and rebounding on
            data waves, grad_norm 2x under baseline) retain up to full-strength
            gradient. Weights apply as in ``reverse_kl``.
        eps: floor for the weight normalizer.

    Returns:
        token_loss: ``(bs, response_length)`` loss, zero outside ``response_mask``.
        diagnostics: dict of detached ``(bs, response_length)`` tensors.
    """
    if pair_divergence not in _PAIR_DIVERGENCES:
        raise ValueError(
            f"Unknown l_apd.pair_divergence: {pair_divergence}. Expected one of {list(_PAIR_DIVERGENCES)}."
        )
    if weight_source not in _WEIGHT_SOURCES:
        raise ValueError(
            f"Unknown l_apd.weight_source: {weight_source}. Expected one of {list(_WEIGHT_SOURCES)}."
        )
    if pair_divergence == "partition_kl" and not tail_candidate:
        raise ValueError(
            "l_apd.pair_divergence='partition_kl' requires tail_candidate=True: the categorical KL "
            "is defined on the partition {anchor} + candidates + tail, which must cover the vocabulary."
        )
    if pair_divergence == "order_gated_kl" and (tail_candidate or complement_candidate):
        raise ValueError(
            "l_apd.pair_divergence='order_gated_kl' operates on the pure student top-k cells "
            "(baseline-aligned set); set tail_candidate=False and complement_candidate=False."
        )

    student_anchor = student_anchor_log_probs.float()
    student_candidates = student_candidate_log_probs.float()
    teacher_anchor = teacher_anchor_log_probs.float().detach()
    teacher_candidates = teacher_candidate_log_probs.float().detach()

    valid = candidate_mask.bool() & response_mask.unsqueeze(-1).bool()

    if pair_divergence == "order_gated_kl":
        # Self-contained: no anchor-star machinery, no aggregate opponents, and
        # ``weight_source`` / ``normalize_weights`` do not apply (the weights are
        # the lambda-blend of the two baseline modes by construction).
        return _order_gated_kl_token_loss(
            student_anchor=student_anchor,
            student_candidates=student_candidates,
            teacher_anchor=teacher_anchor,
            teacher_candidates=teacher_candidates,
            valid=valid,
            response_mask=response_mask,
        )

    # The weighting side. Student log-probs are detached here so the weights are
    # per-step constants: they set how loudly each duel is scored, never a second
    # gradient path.
    if weight_source == "student":
        weight_anchor = student_anchor.detach()
        weight_candidates = student_candidates.detach()
    else:
        weight_anchor = teacher_anchor
        weight_candidates = teacher_candidates

    # Star-shaped pairwise problem centered on the anchor token.
    pair_logits = student_anchor.unsqueeze(-1) - student_candidates
    teacher_margins = teacher_anchor.unsqueeze(-1) - teacher_candidates
    raw_weights = torch.exp(weight_candidates) * valid

    if tail_candidate:
        neg_inf = torch.finfo(torch.float32).min
        covered_student = torch.cat(
            [student_anchor.unsqueeze(-1), student_candidates.masked_fill(~valid, neg_inf)], dim=-1
        )
        covered_teacher = torch.cat(
            [teacher_anchor.unsqueeze(-1), teacher_candidates.masked_fill(~valid, neg_inf)], dim=-1
        )
        covered_student_mass = torch.logsumexp(covered_student, dim=-1)
        student_tail = _log1mexp(covered_student_mass)
        teacher_tail = _log1mexp(torch.logsumexp(covered_teacher, dim=-1))

        weight_tail = student_tail.detach() if weight_source == "student" else teacher_tail

        pair_logits = torch.cat([pair_logits, (student_anchor - student_tail).unsqueeze(-1)], dim=-1)
        teacher_margins = torch.cat([teacher_margins, (teacher_anchor - teacher_tail).unsqueeze(-1)], dim=-1)
        raw_weights = torch.cat(
            [raw_weights, torch.exp(weight_tail).unsqueeze(-1) * response_mask.unsqueeze(-1)], dim=-1
        )
    elif complement_candidate:
        # The coarsest opponent: everything except the anchor, aggregated. Its margin
        # log(p(y) / (1 - p(y))) makes the restricted pair probabilities exactly p(y) and
        # q(y), so this pair is the anchor term -- KL between (p(y), 1 - p(y)) and
        # (q(y), 1 - q(y)) -- in the same form as every other candidate, and weighted by
        # the weighting side's anchor mass like every other candidate.
        student_complement = _log1mexp(student_anchor)
        teacher_complement = _log1mexp(teacher_anchor)

        pair_logits = torch.cat([pair_logits, (student_anchor - student_complement).unsqueeze(-1)], dim=-1)
        teacher_margins = torch.cat(
            [teacher_margins, (teacher_anchor - teacher_complement).unsqueeze(-1)], dim=-1
        )
        raw_weights = torch.cat(
            [raw_weights, torch.exp(weight_anchor).unsqueeze(-1) * response_mask.unsqueeze(-1)], dim=-1
        )

    teacher_pair_prob = torch.sigmoid(teacher_margins)

    if normalize_weights:
        normalizer = raw_weights.sum(dim=-1, keepdim=True)
    else:
        normalizer = (1.0 - torch.exp(weight_anchor)).unsqueeze(-1)
    weights = (raw_weights / normalizer.clamp(min=eps)).detach()

    if pair_divergence == "reverse_kl":
        pair_loss = _reverse_pair_kl(pair_logits, teacher_margins)
        # The student owns the entropy term here, so the loss already is the KL.
        pair_kl = pair_loss
    elif pair_divergence == "jeffreys":
        # Symmetrized Bernoulli KL in its two-point closed form: win-probability
        # gap times margin gap. No logs of small numbers, and the forward half of
        # the gradient never vanishes on decided-but-wrong duels.
        pair_loss = (torch.sigmoid(pair_logits) - teacher_pair_prob) * (pair_logits - teacher_margins)
        pair_kl = _reverse_pair_kl(pair_logits.detach(), teacher_margins)
    elif pair_divergence == "forward_kl":
        pair_loss = F.binary_cross_entropy_with_logits(pair_logits, teacher_pair_prob, reduction="none")
        # Cross-entropy; the teacher-side entropy is a stop-gradient constant.
        pair_kl = pair_loss - _pair_entropy(teacher_pair_prob)
    elif pair_divergence == "partition_kl":
        # One categorical reverse KL over the partition cells. Every cell's term is
        # p(c) * (log p(c) - log q(c)); the cells sum to one on both sides by
        # construction (tail = 1 - covered mass), so this is the exact coarse-grained
        # KL(p || q) -- no explicit weights, no sigmoid geometry. The pairwise
        # Bernoulli KL is still reported through ``pair_kl`` for monitoring
        # continuity with the pairwise runs.
        candidate_terms = torch.exp(student_candidates) * (student_candidates - teacher_candidates)
        candidate_terms = torch.where(valid, candidate_terms, torch.zeros_like(candidate_terms))
        partition_kl = (
            torch.exp(student_anchor) * (student_anchor - teacher_anchor)
            + candidate_terms.sum(dim=-1)
            + torch.exp(student_tail) * (student_tail - teacher_tail)
        )
        pair_loss = None
        pair_kl = _reverse_pair_kl(pair_logits.detach(), teacher_margins)
    elif pair_divergence == "odds_kl":
        # One sigma-free generalized KL per duel on the anchored odds. Fused stable
        # evaluation of rho * phi(u) = e^{-m_T} - e^{-m} (u + 1) with u = m - m_T
        # (phi(u) = e^u - u - 1 alone can overflow; this form never exponentiates u).
        # Each duel's term is >= 0 with equality iff m == m_T. The sg[p(y_t)]
        # prefactor converts odds to mass units: the margin gradient becomes
        # exactly p(c) * (m - m_T).
        duel_gap = pair_logits - teacher_margins
        duel_terms = torch.exp(-teacher_margins) - torch.exp(-pair_logits) * (duel_gap + 1.0)
        duel_mask = valid.float()
        if tail_candidate or complement_candidate:
            duel_mask = torch.cat([duel_mask, response_mask.unsqueeze(-1)], dim=-1)
        odds_kl = torch.exp(student_anchor).detach() * (duel_terms * duel_mask).sum(dim=-1)
        pair_loss = None
        pair_kl = _reverse_pair_kl(pair_logits.detach(), teacher_margins)
    else:
        # Only the v = y_t outcome of the reverse KL. On the aggregated complement
        # column its margin makes the restricted probabilities exactly p(y_t) and
        # q(y_t), so that column is log[p(y_t) / q(y_t)]. Monotone in the margin, hence
        # unbounded below and with no stationary point: the teacher term is an additive
        # stop-gradient constant, so teacher information survives only in the weights
        # and the candidate set. pair_kl keeps reporting the honest KL.
        pair_loss = F.logsigmoid(pair_logits) - F.logsigmoid(teacher_margins)
        pair_kl = _reverse_pair_kl(pair_logits.detach(), teacher_margins)

    if pair_divergence == "partition_kl":
        token_loss = partition_kl * response_mask
    elif pair_divergence == "odds_kl":
        token_loss = odds_kl * response_mask
    else:
        token_loss = (weights * pair_loss).sum(dim=-1) * response_mask

    with torch.no_grad():
        student_pair_prob = torch.sigmoid(pair_logits)
        pairwise_kl = (weights * pair_kl).sum(dim=-1)
        agreement = ((student_pair_prob - 0.5) * (teacher_pair_prob - 0.5) > 0).float()
        diagnostics = {
            "pair_kl": pairwise_kl,
            "pairwise_agreement": (weights * agreement).sum(dim=-1),
            "pairwise_gap": (weights * (student_pair_prob - teacher_pair_prob).abs()).sum(dim=-1),
            "teacher_anchor_prob": torch.exp(teacher_anchor),
            "student_anchor_prob": torch.exp(student_anchor),
            "anchor_in_candidates": 1.0 - candidate_mask.float().prod(dim=-1),
            "candidate_count": valid.float().sum(dim=-1),
            # Weighting-side mass covered by the candidate set: 1 when normalized.
            "candidate_weight_sum": weights.sum(dim=-1),
        }
        if tail_candidate:
            diagnostics["tail_weight"] = weights[..., -1]
            diagnostics["teacher_tail_prob"] = torch.exp(teacher_tail)
            diagnostics["student_tail_prob"] = torch.exp(student_tail)
            # Share of positions where float32 can no longer resolve the tail mass, so
            # the aggregate-opponent margin is capped instead of exact.
            diagnostics["tail_saturated"] = (covered_student_mass > _LOG1MEXP_MAX).float()
        if pair_divergence == "partition_kl":
            diagnostics["partition_kl"] = partition_kl * response_mask
        elif pair_divergence == "odds_kl":
            diagnostics["odds_kl"] = odds_kl * response_mask
        elif complement_candidate:
            diagnostics["anchor_weight"] = weights[..., -1]
            diagnostics["anchor_saturated"] = (student_anchor > _LOG1MEXP_MAX).float()
            # KL between (p(y), 1 - p(y)) and (q(y), 1 - q(y)), so it reaches 0 exactly
            # when p(y) == q(y) whichever direction the pairs are scored with.
            diagnostics["anchor_kl"] = pair_kl[..., -1]

    return token_loss, diagnostics
