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
"""Unit tests for the L-APD objective."""

import math

import torch

from verl.trainer.ppo.l_apd import compute_l_apd_token_loss


def _reference_full_vocab_loss(student_logits, teacher_logits, anchors, response_mask):
    """Brute-force L-APD over the full vocabulary, following the definition.

    L_t = sum_{z != y_t} q~_t(z) * BCE(sigmoid(S(y_t) - S(z)), sigmoid(T(y_t) - T(z)))
    with q~_t(z) = q_t(z) / (1 - q_t(y_t)).
    """
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, vocab = student_logits.shape

    token_loss = torch.zeros(bs, seq_len, dtype=torch.float64)
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                continue
            y = int(anchors[b, t])
            teacher_anchor_prob = teacher_log_probs[b, t, y].exp()
            total = 0.0
            for z in range(vocab):
                if z == y:
                    continue
                weight = teacher_log_probs[b, t, z].exp() / (1.0 - teacher_anchor_prob)
                teacher_win = torch.sigmoid(teacher_logits[b, t, y] - teacher_logits[b, t, z])
                student_margin = student_logits[b, t, y] - student_logits[b, t, z]
                bce = torch.nn.functional.binary_cross_entropy_with_logits(student_margin, teacher_win)
                total = total + weight * bce
            token_loss[b, t] = total
    return token_loss


def _make_batch(vocab=8, bs=2, seq_len=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(bs, seq_len, vocab, generator=generator, dtype=torch.float64)
    teacher_logits = torch.randn(bs, seq_len, vocab, generator=generator, dtype=torch.float64) * 1.5
    anchors = torch.randint(vocab, (bs, seq_len), generator=generator)
    response_mask = torch.ones(bs, seq_len)
    response_mask[-1, -1] = 0
    return student_logits, teacher_logits, anchors, response_mask


def _candidates_from_teacher(teacher_logits, k):
    return torch.topk(teacher_logits, k=k, dim=-1).indices


def _call_loss(student_logits, teacher_logits, anchors, response_mask, candidate_ids, **kwargs):
    # The closed forms most tests below compare against are written for the forward
    # direction and the historical teacher weighting, so ask for those explicitly
    # rather than riding on the library defaults.
    kwargs.setdefault("pair_divergence", "forward_kl")
    kwargs.setdefault("weight_source", "teacher")
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    return compute_l_apd_token_loss(
        student_anchor_log_probs=student_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        student_candidate_log_probs=student_log_probs.gather(-1, candidate_ids),
        teacher_anchor_log_probs=teacher_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        teacher_candidate_log_probs=teacher_log_probs.gather(-1, candidate_ids),
        candidate_mask=candidate_ids != anchors.unsqueeze(-1),
        response_mask=response_mask,
        **kwargs,
    )


def test_matches_full_vocab_definition():
    """With the full vocabulary as candidates, the loss matches the definition."""
    vocab = 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab)
    candidate_ids = torch.arange(vocab).expand(*anchors.shape, vocab)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
    )
    expected = _reference_full_vocab_loss(student_logits, teacher_logits, anchors, response_mask)

    torch.testing.assert_close(token_loss.double(), expected, atol=1e-5, rtol=1e-4)


def test_tail_candidate_recovers_the_truncated_token():
    """One aggregated tail candidate that holds a single token is exactly that token."""
    vocab = 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=3)

    # Keep every token but one as an explicit candidate, so the tail is a single token.
    keep = [z for z in range(vocab) if z != vocab - 1]
    truncated_ids = torch.tensor(keep).expand(*anchors.shape, len(keep))
    anchors = torch.full_like(anchors, 0)  # anchor is inside the candidate list

    with_tail, _ = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, truncated_ids, tail_candidate=True
    )
    full = _reference_full_vocab_loss(student_logits, teacher_logits, anchors, response_mask)

    torch.testing.assert_close(with_tail.double(), full, atol=1e-5, rtol=1e-4)


def test_gradient_matches_analytical_margin_gradient():
    """dL/d(S(y) - S(z)) = q~(z) * (p~(y, z) - q~(y, z))."""
    vocab = 12
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=7)
    student_logits = student_logits.float().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=4)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits.float(),
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
    )
    token_loss.sum().backward()

    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    anchor_teacher = teacher_log_probs.gather(-1, anchors.unsqueeze(-1))
    candidate_teacher = teacher_log_probs.gather(-1, candidate_ids)
    valid = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    weights = candidate_teacher.exp() * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    student_win = torch.sigmoid(student_log_probs.gather(-1, anchors.unsqueeze(-1)) - student_log_probs.gather(-1, candidate_ids))
    teacher_win = torch.sigmoid(anchor_teacher - candidate_teacher)
    margin_grad = weights * (student_win - teacher_win)

    # A pair only touches its own two logits, since the softmax normalizer
    # cancels inside a margin.
    expected = torch.zeros_like(student_logits)
    expected.scatter_add_(-1, candidate_ids, -margin_grad)
    expected.scatter_add_(-1, anchors.unsqueeze(-1), margin_grad.sum(dim=-1, keepdim=True))

    torch.testing.assert_close(student_logits.grad, expected, atol=1e-5, rtol=1e-3)


def test_student_matching_teacher_is_a_stationary_point():
    vocab = 16
    _, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=11)
    student_logits = teacher_logits.clone().float().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=5)

    token_loss, diagnostics = _call_loss(
        student_logits, teacher_logits.float(), anchors, response_mask, candidate_ids
    )
    token_loss.sum().backward()

    assert student_logits.grad.abs().max() < 1e-5
    assert diagnostics["pair_kl"].max() < 1e-5
    torch.testing.assert_close(
        diagnostics["pairwise_agreement"][response_mask.bool()],
        torch.ones(int(response_mask.sum())),
        atol=1e-5,
        rtol=1e-5,
    )


def test_masked_positions_are_free_of_loss_and_nan():
    vocab = 10
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=13)
    student_logits = student_logits.float().requires_grad_(True)
    response_mask = torch.zeros_like(response_mask)
    response_mask[0, 0] = 1
    candidate_ids = _candidates_from_teacher(teacher_logits, k=3)

    token_loss, _ = _call_loss(student_logits, teacher_logits.float(), anchors, response_mask, candidate_ids)
    token_loss.sum().backward()

    assert torch.isfinite(token_loss).all()
    assert (token_loss[response_mask == 0] == 0).all()
    assert torch.isfinite(student_logits.grad).all()


def test_weights_are_a_distribution_over_candidates():
    """The tail candidate restores the teacher mass that top-k truncation drops."""
    vocab = 32
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=17)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=4)
    valid_positions = response_mask.bool()

    _, with_tail = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, normalize_weights=False
    )
    _, without_tail = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
        normalize_weights=False,
    )
    _, renormalized = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
    )

    torch.testing.assert_close(
        with_tail["candidate_weight_sum"][valid_positions],
        torch.ones(int(response_mask.sum())),
        atol=1e-5,
        rtol=1e-5,
    )
    assert (without_tail["candidate_weight_sum"][valid_positions] < 1.0).all()
    assert (with_tail["tail_weight"][valid_positions] > 0).all()
    torch.testing.assert_close(
        renormalized["candidate_weight_sum"][valid_positions],
        torch.ones(int(response_mask.sum())),
        atol=1e-5,
        rtol=1e-5,
    )


def test_token_candidates_alone_leave_the_tail_mass_unidentified():
    """Token candidates alone only constrain logit differences.

    Scaling the mass of ``{anchor} + candidates`` up or down, with the slack absorbed
    by the truncated tail, leaves every ``S(y) - S(z)`` untouched and so cannot change
    the loss. Either aggregated candidate removes that freedom, which is why one of
    them is required.
    """
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=3)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    # Move mass from the candidate set into the tail without touching any pairwise margin.
    shifted = student_logits.clone()
    in_set = torch.zeros_like(student_logits, dtype=torch.bool)
    in_set.scatter_(-1, candidate_ids, True)
    in_set.scatter_(-1, anchors.unsqueeze(-1), True)
    shifted[in_set] -= 2.0

    for tail_candidate, complement_candidate, should_differ in [
        (False, False, False),  # the unidentified direction
        (False, True, True),  # complement candidate removes it
        (True, False, True),  # tail candidate removes it
    ]:
        kwargs = dict(tail_candidate=tail_candidate, complement_candidate=complement_candidate)
        base, _ = _call_loss(student_logits, teacher_logits, anchors, response_mask, candidate_ids, **kwargs)
        moved, _ = _call_loss(shifted, teacher_logits, anchors, response_mask, candidate_ids, **kwargs)
        if should_differ:
            assert (moved - base).abs().max() > 1e-3, kwargs
        else:
            torch.testing.assert_close(moved, base, atol=1e-5, rtol=1e-4)


def test_complement_candidate_is_the_anchor_kl():
    """The complement pair reproduces the anchor KL and carries its ``q(y_t)`` weight."""
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=4)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)
    kwargs = dict(tail_candidate=False, complement_candidate=True)

    _, mismatched = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, **kwargs
    )
    _, matched = _call_loss(teacher_logits, teacher_logits, anchors, response_mask, candidate_ids, **kwargs)

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    p_anchor = student_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1).exp()
    q_anchor = teacher_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1).exp()
    expected_kl = q_anchor * (q_anchor / p_anchor).log() + (1 - q_anchor) * ((1 - q_anchor) / (1 - p_anchor)).log()

    torch.testing.assert_close(
        mismatched["anchor_kl"].double() * response_mask, expected_kl * response_mask, atol=1e-5, rtol=1e-4
    )
    assert (matched["anchor_kl"] * response_mask).abs().max() < 1e-5

    # Weight of the complement candidate is q(y_t) renormalized over anchor + candidates.
    q_candidates = teacher_log_probs.gather(-1, candidate_ids).exp()
    kept = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    expected_weight = q_anchor / (q_anchor + (q_candidates * kept).sum(-1))
    torch.testing.assert_close(
        mismatched["anchor_weight"].double() * response_mask,
        expected_weight * response_mask,
        atol=1e-5,
        rtol=1e-4,
    )


def test_matches_the_documented_two_part_form():
    """The loss equals the explicit form documented in the README.

    ``L_t = sum_z [q(z)/Z] H(q~, p~) + [q(y)/Z] H(q(y), p(y))`` with
    ``Z = sum over the top-k ids including the anchor``, i.e. the weights are the
    teacher probabilities renormalized over all k candidates rather than over the
    k - 1 non-anchor ones.
    """
    vocab, k = 64, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=11)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=True,
    )

    p = torch.log_softmax(student_logits, dim=-1)
    q = torch.log_softmax(teacher_logits, dim=-1)
    p_anchor = p.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    q_anchor = q.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    p_cand, q_cand = p.gather(-1, candidate_ids), q.gather(-1, candidate_ids)
    kept = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()

    normalizer = q_anchor.exp() + (q_cand.exp() * kept).sum(-1)

    def pair_ce(target, prob):
        return -(target * prob.log() + (1 - target) * (1 - prob).log())

    r_s = torch.sigmoid(p_anchor.unsqueeze(-1) - p_cand)
    r_t = torch.sigmoid(q_anchor.unsqueeze(-1) - q_cand)
    pairwise = ((q_cand.exp() / normalizer.unsqueeze(-1)) * pair_ce(r_t, r_s) * kept).sum(-1)
    anchor = (q_anchor.exp() / normalizer) * pair_ce(q_anchor.exp(), p_anchor.exp())

    torch.testing.assert_close(
        token_loss.double(), (pairwise + anchor) * response_mask, atol=1e-5, rtol=1e-4
    )


def test_weights_including_the_complement_sum_to_one():
    """With the complement candidate the whole loss is a convex combination."""
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=5)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    _, diagnostics = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=True,
    )

    valid_positions = response_mask.bool()
    torch.testing.assert_close(
        diagnostics["candidate_weight_sum"][valid_positions],
        torch.ones(int(response_mask.sum())),
        atol=1e-5,
        rtol=1e-5,
    )


def test_jeffreys_is_the_default_divergence():
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=23)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    explicit, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        pair_divergence="jeffreys",
        weight_source="student",
    )
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    default, _ = compute_l_apd_token_loss(
        student_anchor_log_probs=student_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        student_candidate_log_probs=student_log_probs.gather(-1, candidate_ids),
        teacher_anchor_log_probs=teacher_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        teacher_candidate_log_probs=teacher_log_probs.gather(-1, candidate_ids),
        candidate_mask=candidate_ids != anchors.unsqueeze(-1),
        response_mask=response_mask,
    )

    torch.testing.assert_close(default, explicit)


def test_reverse_kl_matches_the_kl_definition():
    """The reverse loss is ``sum_o w(o) KL(p~(o) || q~(o))``, an honest KL."""
    vocab, k = 64, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=29)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, diagnostics = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=True,
        pair_divergence="reverse_kl",
    )

    p = torch.log_softmax(student_logits, dim=-1)
    q = torch.log_softmax(teacher_logits, dim=-1)
    p_anchor = p.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    q_anchor = q.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    p_cand, q_cand = p.gather(-1, candidate_ids), q.gather(-1, candidate_ids)
    kept = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    normalizer = q_anchor.exp() + (q_cand.exp() * kept).sum(-1)

    def reverse_kl(a, b):
        return a * (a.log() - b.log()) + (1 - a) * ((1 - a).log() - (1 - b).log())

    pairwise = (
        (q_cand.exp() / normalizer.unsqueeze(-1))
        * reverse_kl(torch.sigmoid(p_anchor.unsqueeze(-1) - p_cand), torch.sigmoid(q_anchor.unsqueeze(-1) - q_cand))
        * kept
    ).sum(-1)
    anchor = (q_anchor.exp() / normalizer) * reverse_kl(p_anchor.exp(), q_anchor.exp())

    torch.testing.assert_close(
        token_loss.double(), (pairwise + anchor) * response_mask, atol=1e-5, rtol=1e-4
    )
    # Reverse scores the pairs with the KL itself, so the reported loss is the KL.
    torch.testing.assert_close(
        diagnostics["pair_kl"].double() * response_mask,
        (pairwise + anchor) * response_mask,
        atol=1e-5,
        rtol=1e-4,
    )


def test_reverse_kl_margin_gradient_closed_form():
    """dL/d(S(y) - S(z)) = w(z) * sigmoid'(m) * (m - m_T) for the reverse direction."""
    vocab = 12
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=31)
    student_logits = student_logits.float().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=4)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits.float(),
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
        pair_divergence="reverse_kl",
    )
    token_loss.sum().backward()

    p = torch.log_softmax(student_logits.detach(), dim=-1)
    q = torch.log_softmax(teacher_logits.float(), dim=-1)
    q_anchor, q_cand = q.gather(-1, anchors.unsqueeze(-1)), q.gather(-1, candidate_ids)
    valid = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    weights = q_cand.exp() * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    margin = p.gather(-1, anchors.unsqueeze(-1)) - p.gather(-1, candidate_ids)
    teacher_margin = q_anchor - q_cand
    student_win = torch.sigmoid(margin)
    margin_grad = weights * student_win * (1 - student_win) * (margin - teacher_margin)

    expected = torch.zeros_like(student_logits)
    expected.scatter_add_(-1, candidate_ids, -margin_grad)
    expected.scatter_add_(-1, anchors.unsqueeze(-1), margin_grad.sum(dim=-1, keepdim=True))

    torch.testing.assert_close(student_logits.grad, expected, atol=1e-5, rtol=1e-3)


def test_both_directions_vanish_on_a_perfect_student():
    """Both directions share the same optimum, so both are zero at p == q."""
    vocab = 16
    _, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=37)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=5)

    for direction in ("reverse_kl", "forward_kl"):
        student_logits = teacher_logits.clone().float().requires_grad_(True)
        token_loss, diagnostics = _call_loss(
            student_logits,
            teacher_logits.float(),
            anchors,
            response_mask,
            candidate_ids,
            pair_divergence=direction,
        )
        token_loss.sum().backward()

        assert student_logits.grad.abs().max() < 1e-5, direction
        assert diagnostics["pair_kl"].max() < 1e-5, direction


def test_reverse_kl_is_bounded_where_forward_diverges():
    """A confidently misranked pair costs at most -log q~(y) under the reverse direction.

    This is the substantive behavioural difference between the two: the reverse loss
    saturates on confident disagreement and its gradient decays, while the forward
    loss grows without bound and keeps a gradient of magnitude up to 1.
    """
    teacher_margin = -4.0
    losses, grads = {}, {}
    for direction in ("reverse_kl", "forward_kl"):
        losses[direction], grads[direction] = [], []
        for student_margin in (8.0, 30.0):
            # A single pair: anchor plus one candidate, teacher margin fixed.
            student_anchor = torch.zeros(1, 1, requires_grad=True)
            student_cand = torch.full((1, 1, 1), -student_margin)
            token_loss, _ = compute_l_apd_token_loss(
                student_anchor_log_probs=student_anchor,
                student_candidate_log_probs=student_cand,
                teacher_anchor_log_probs=torch.zeros(1, 1),
                teacher_candidate_log_probs=torch.full((1, 1, 1), -teacher_margin),
                candidate_mask=torch.ones(1, 1, 1, dtype=torch.bool),
                response_mask=torch.ones(1, 1),
                tail_candidate=False,
                complement_candidate=False,
                # Teacher weighting: these toy log-probs are unnormalized, and the
                # student side would put the lone pair below the eps weight floor.
                weight_source="teacher",
                pair_divergence=direction,
            )
            token_loss.sum().backward()
            losses[direction].append(token_loss.item())
            grads[direction].append(student_anchor.grad.abs().item())

    import math

    ceiling = -math.log(torch.sigmoid(torch.tensor(teacher_margin)).item())
    assert max(losses["reverse_kl"]) < ceiling + 1e-3
    assert losses["forward_kl"][1] > 2 * losses["forward_kl"][0]
    assert grads["reverse_kl"][1] < 1e-6
    assert grads["forward_kl"][1] > 0.9


def test_log_ratio_matches_the_two_part_bare_form():
    """``log_ratio`` is the weighted bare log-ratio plus the target-token log-ratio.

    ``L_t = sum_z [q(z)/Z] log[r_S(y,z) / r_T(y,z)] + [q(y)/Z] log[p(y) / q(y)]``, i.e.
    the reverse KL with the ``v = z`` outcome of every pair dropped. The complement
    column collapses to the plain anchor log-ratio because its margin makes the
    restricted probabilities exactly ``p(y)`` and ``q(y)``.
    """
    vocab, k = 64, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=53)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=True,
        pair_divergence="log_ratio",
    )

    p = torch.log_softmax(student_logits, dim=-1)
    q = torch.log_softmax(teacher_logits, dim=-1)
    p_anchor = p.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    q_anchor = q.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    p_cand, q_cand = p.gather(-1, candidate_ids), q.gather(-1, candidate_ids)
    kept = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    normalizer = q_anchor.exp() + (q_cand.exp() * kept).sum(-1)

    r_s = torch.sigmoid(p_anchor.unsqueeze(-1) - p_cand)
    r_t = torch.sigmoid(q_anchor.unsqueeze(-1) - q_cand)
    pairwise = ((q_cand.exp() / normalizer.unsqueeze(-1)) * (r_s / r_t).log() * kept).sum(-1)
    anchor = (q_anchor.exp() / normalizer) * (p_anchor - q_anchor)

    torch.testing.assert_close(
        token_loss.double(), (pairwise + anchor) * response_mask, atol=1e-5, rtol=1e-4
    )


def test_log_ratio_has_no_stationary_point():
    """Dropping the ``v = z`` outcome leaves a monotone objective, not a divergence.

    The teacher term becomes an additive stop-gradient constant, so at ``p == q`` the
    honest KL is zero while the loss still pulls at full strength, and plain gradient
    descent walks the loss down without bound, taking ``p(y_t)`` to zero with it.
    """
    vocab = 16
    _, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=59)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=5)

    student_logits = teacher_logits.clone().float().requires_grad_(True)
    token_loss, diagnostics = _call_loss(
        student_logits,
        teacher_logits.float(),
        anchors,
        response_mask,
        candidate_ids,
        pair_divergence="log_ratio",
    )
    token_loss.sum().backward()

    assert diagnostics["pair_kl"].max() < 1e-5
    assert student_logits.grad.abs().max() > 1e-2

    optimizer = torch.optim.SGD([student_logits], lr=1.0)
    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = _call_loss(
            student_logits,
            teacher_logits.float(),
            anchors,
            response_mask,
            candidate_ids,
            pair_divergence="log_ratio",
        )
        loss.sum().backward()
        optimizer.step()

    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    anchor_log_probs = student_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1)
    assert loss.sum().item() < -10.0
    assert anchor_log_probs[response_mask.bool()].max().item() < -10.0


def _overconfident_anchor(student_anchor_prob, teacher_anchor_prob=0.5):
    """One position with a single competitor, so the complement pair is exact.

    The competitor absorbs the whole remaining student mass, which makes
    ``r_S(y_t, competitor)`` equal to ``p(y_t)`` just like the complement column and
    keeps the log-probs self-consistent as ``p(y_t)`` approaches 1.
    """
    remainder = max(1.0 - student_anchor_prob, torch.finfo(torch.float64).tiny)
    anchor = torch.tensor([[math.log(student_anchor_prob)]], dtype=torch.float64, requires_grad=True)
    loss, diagnostics = compute_l_apd_token_loss(
        student_anchor_log_probs=anchor,
        student_candidate_log_probs=torch.tensor([[[math.log(remainder)]]], dtype=torch.float64),
        teacher_anchor_log_probs=torch.tensor([[math.log(teacher_anchor_prob)]], dtype=torch.float64),
        teacher_candidate_log_probs=torch.tensor([[[math.log(1.0 - teacher_anchor_prob)]]], dtype=torch.float64),
        candidate_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        response_mask=torch.ones(1, 1),
        tail_candidate=False,
        complement_candidate=True,
        weight_source="teacher",
        pair_divergence="reverse_kl",
    )
    loss.sum().backward()
    return loss.item(), anchor.grad.item(), diagnostics


def test_overconfident_anchor_keeps_its_gradient():
    """No gradient cliff where float32 stops resolving ``1 - p(y_t)``.

    ``_log1mexp`` has to cap its input so that ``log(1 - exp(x))`` stays finite at
    ``x = 0``, and that cap lands exactly where the student is most overconfident --
    the case on-policy distillation exists to correct. Blocking the gradient there
    used to drop it by six orders of magnitude in one step, so the cap has to stay
    transparent to autograd.

    The plateau is legitimate rather than a second truncation: the composite gradient
    is ``w(perp) p(y_t) (m - m_T)``, because the pair's ``r (1 - r)`` factor cancels
    the ``1 / (1 - p(y_t))`` blow-up, so it only grows like ``log[1 / (1 - p(y_t))]``.
    """
    resolved = _overconfident_anchor(1.0 - 1e-6)
    just_capped = _overconfident_anchor(1.0 - 1e-7)
    deeply_capped = _overconfident_anchor(1.0 - 1e-12)

    # The teacher is far less confident, so descent has to push p(y_t) back down.
    assert resolved[1] > 1.0
    torch.testing.assert_close(just_capped[1], resolved[1], atol=0.0, rtol=1e-6)
    torch.testing.assert_close(deeply_capped[1], resolved[1], atol=0.0, rtol=1e-6)


def test_saturated_anchor_is_free_of_nan():
    """``p(y_t) == 1`` is reachable in float32 and must not poison the backward pass."""
    loss, gradient, _ = _overconfident_anchor(1.0)
    assert math.isfinite(loss)
    assert math.isfinite(gradient)
    assert gradient > 1.0


def test_anchor_saturated_reports_the_capped_share():
    """The diagnostic makes the capped region visible instead of silently absorbed."""
    _, _, resolved = _overconfident_anchor(0.9)
    _, _, capped = _overconfident_anchor(1.0 - 1e-9)
    assert resolved["anchor_saturated"].item() == 0.0
    assert capped["anchor_saturated"].item() == 1.0


def test_student_weighting_is_the_default():
    """The library default weights every duel by the student's conditional mass."""
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=61)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    default, _ = compute_l_apd_token_loss(
        student_anchor_log_probs=student_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        student_candidate_log_probs=student_log_probs.gather(-1, candidate_ids),
        teacher_anchor_log_probs=teacher_log_probs.gather(-1, anchors.unsqueeze(-1)).squeeze(-1),
        teacher_candidate_log_probs=teacher_log_probs.gather(-1, candidate_ids),
        candidate_mask=candidate_ids != anchors.unsqueeze(-1),
        response_mask=response_mask,
    )
    explicit, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        pair_divergence="jeffreys",
        weight_source="student",
    )
    teacher_weighted, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        pair_divergence="jeffreys",
        weight_source="teacher",
    )

    torch.testing.assert_close(default, explicit)
    assert (default - teacher_weighted).abs().max() > 1e-4


def test_student_weights_equal_student_conditional_mass():
    """Weights are ``sg[p(o) / (1 - p(y_t))]``; the tail column follows the student tail."""
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=67)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)
    valid_positions = response_mask.bool()

    _, diagnostics = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        weight_source="student",
    )

    p = torch.log_softmax(student_logits, dim=-1)
    p_anchor = p.gather(-1, anchors.unsqueeze(-1)).squeeze(-1).exp()
    kept = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    p_named = (p.gather(-1, candidate_ids).exp() * kept).sum(-1)
    expected_tail_weight = (1.0 - p_anchor - p_named) / (1.0 - p_anchor)

    torch.testing.assert_close(
        diagnostics["tail_weight"][valid_positions].double(),
        expected_tail_weight[valid_positions],
        atol=1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        diagnostics["candidate_weight_sum"][valid_positions],
        torch.ones(int(response_mask.sum())),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_weights_are_stop_gradient():
    """The student weights scale each duel but never open a second gradient path.

    The analytical margin gradient below treats the weights as constants; if the
    weights leaked gradient, the ``p``-dependent normalizer alone would break it.
    """
    vocab = 12
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=71)
    student_logits = student_logits.float().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=4)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits.float(),
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=False,
        complement_candidate=False,
        weight_source="student",
    )
    token_loss.sum().backward()

    p = torch.log_softmax(student_logits.detach(), dim=-1)
    q = torch.log_softmax(teacher_logits.float(), dim=-1)
    valid = (candidate_ids != anchors.unsqueeze(-1)) & response_mask.unsqueeze(-1).bool()
    weights = p.gather(-1, candidate_ids).exp() * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    student_win = torch.sigmoid(p.gather(-1, anchors.unsqueeze(-1)) - p.gather(-1, candidate_ids))
    teacher_win = torch.sigmoid(q.gather(-1, anchors.unsqueeze(-1)) - q.gather(-1, candidate_ids))
    margin_grad = weights * (student_win - teacher_win)

    expected = torch.zeros_like(student_logits)
    expected.scatter_add_(-1, candidate_ids, -margin_grad)
    expected.scatter_add_(-1, anchors.unsqueeze(-1), margin_grad.sum(dim=-1, keepdim=True))

    torch.testing.assert_close(student_logits.grad, expected, atol=1e-5, rtol=1e-3)


def test_k0_reduces_to_anchor_bernoulli_kl_under_both_weightings():
    """With zero token candidates the tail duel is the anchor calibration term.

    Under the default (jeffreys) that term is the symmetrized anchor Bernoulli KL,
    whose closed form is (p - q) times the logit gap; pinning reverse_kl recovers
    the one-direction anchor KL.
    """
    bs, seq_len = 2, 3
    generator = torch.Generator().manual_seed(73)
    student_anchor = torch.log(torch.rand(bs, seq_len, generator=generator) * 0.8 + 0.1)
    teacher_anchor = torch.log(torch.rand(bs, seq_len, generator=generator) * 0.8 + 0.1)
    response_mask = torch.ones(bs, seq_len)

    p, q = student_anchor.exp(), teacher_anchor.exp()
    reverse_bernoulli = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    jeffreys_bernoulli = (p - q) * ((p / (1 - p)).log() - (q / (1 - q)).log())

    def k0_loss(**kwargs):
        token_loss, _ = compute_l_apd_token_loss(
            student_anchor_log_probs=student_anchor,
            student_candidate_log_probs=torch.zeros(bs, seq_len, 0),
            teacher_anchor_log_probs=teacher_anchor,
            teacher_candidate_log_probs=torch.zeros(bs, seq_len, 0),
            candidate_mask=torch.zeros(bs, seq_len, 0, dtype=torch.bool),
            response_mask=response_mask,
            **kwargs,
        )
        return token_loss

    for weight_source in ("student", "teacher"):
        torch.testing.assert_close(
            k0_loss(weight_source=weight_source), jeffreys_bernoulli, atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            k0_loss(weight_source=weight_source, pair_divergence="reverse_kl"),
            reverse_bernoulli,
            atol=1e-5,
            rtol=1e-4,
        )


def test_unknown_weight_source_is_rejected():
    vocab, k = 16, 4
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=43)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    try:
        _call_loss(
            student_logits, teacher_logits, anchors, response_mask, candidate_ids, weight_source="mixed"
        )
    except ValueError as error:
        assert "weight_source" in str(error)
    else:
        raise AssertionError("an unknown weight_source should raise")


def test_unknown_pair_divergence_is_rejected():
    vocab, k = 16, 4
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=41)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    try:
        _call_loss(
            student_logits, teacher_logits, anchors, response_mask, candidate_ids, pair_divergence="js"
        )
    except ValueError as error:
        assert "pair_divergence" in str(error)
    else:
        raise AssertionError("an unknown pair_divergence should raise")


def _reference_partition_kl(student_logits, teacher_logits, anchors, candidate_ids, response_mask):
    """Brute-force coarse KL over the partition {anchor} + valid candidates + tail."""
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, _ = student_logits.shape

    token_loss = torch.zeros(bs, seq_len, dtype=torch.float64)
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                continue
            y = int(anchors[b, t])
            named = {int(z) for z in candidate_ids[b, t] if int(z) != y}
            p_cells = [student_log_probs[b, t, y].exp()]
            q_cells = [teacher_log_probs[b, t, y].exp()]
            for z in sorted(named):
                p_cells.append(student_log_probs[b, t, z].exp())
                q_cells.append(teacher_log_probs[b, t, z].exp())
            p_cells.append(1.0 - sum(p_cells))
            q_cells.append(1.0 - sum(q_cells))
            total = 0.0
            for p_c, q_c in zip(p_cells, q_cells):
                total = total + p_c * (torch.log(p_c) - torch.log(q_c))
            token_loss[b, t] = total
    return token_loss


def test_partition_kl_matches_coarse_kl_definition():
    """partition_kl is the categorical KL between the two coarsened distributions."""
    vocab, k = 12, 5
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=11)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, diagnostics = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="partition_kl",
        weight_source="student",
    )
    expected = _reference_partition_kl(student_logits, teacher_logits, anchors, candidate_ids, response_mask)

    torch.testing.assert_close(token_loss.double(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(diagnostics["partition_kl"].double(), expected, atol=1e-5, rtol=1e-4)


def test_partition_kl_gradient_is_cell_delta_minus_loss():
    """Autograd matches the analytic gradient p(v) * (Delta_cell(v) - L_t) per logit."""
    vocab, k = 10, 4
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=17)
    student_logits = student_logits.clone().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="partition_kl",
    )
    token_loss.sum().backward()

    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, _ = student_logits.shape
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                torch.testing.assert_close(
                    student_logits.grad[b, t], torch.zeros(vocab, dtype=torch.float64)
                )
                continue
            y = int(anchors[b, t])
            named = {int(z) for z in candidate_ids[b, t] if int(z) != y}
            tail = [v for v in range(vocab) if v != y and v not in named]
            p_tail = student_log_probs[b, t, tail].exp().sum()
            q_tail = teacher_log_probs[b, t, tail].exp().sum()
            delta_tail = p_tail.log() - q_tail.log()

            loss_t = float(token_loss[b, t])
            for v in range(vocab):
                p_v = student_log_probs[b, t, v].exp()
                if v == y or v in named:
                    delta = student_log_probs[b, t, v] - teacher_log_probs[b, t, v]
                else:
                    delta = delta_tail
                expected = p_v * (delta - loss_t)
                torch.testing.assert_close(
                    student_logits.grad[b, t, v].double(), expected.double(), atol=1e-4, rtol=1e-3
                )


def test_partition_kl_k0_reduces_to_anchor_bernoulli_kl():
    """With zero token candidates the partition is {y, tail}: the anchor Bernoulli KL."""
    bs, seq_len = 2, 3
    generator = torch.Generator().manual_seed(29)
    student_anchor = torch.log(torch.rand(bs, seq_len, generator=generator) * 0.8 + 0.1)
    teacher_anchor = torch.log(torch.rand(bs, seq_len, generator=generator) * 0.8 + 0.1)
    response_mask = torch.ones(bs, seq_len)

    p, q = student_anchor.exp(), teacher_anchor.exp()
    bernoulli = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()

    token_loss, _ = compute_l_apd_token_loss(
        student_anchor_log_probs=student_anchor,
        student_candidate_log_probs=torch.zeros(bs, seq_len, 0),
        teacher_anchor_log_probs=teacher_anchor,
        teacher_candidate_log_probs=torch.zeros(bs, seq_len, 0),
        candidate_mask=torch.zeros(bs, seq_len, 0, dtype=torch.bool),
        response_mask=response_mask,
        pair_divergence="partition_kl",
    )
    torch.testing.assert_close(token_loss, bernoulli, atol=1e-5, rtol=1e-4)


def test_partition_kl_is_zero_with_zero_gradient_at_p_equals_q():
    """The fixed point is p = q: loss and gradient both vanish there."""
    vocab, k = 12, 5
    student_logits, _, anchors, response_mask = _make_batch(vocab=vocab, seed=31)
    student_logits = student_logits.clone().requires_grad_(True)
    teacher_logits = student_logits.detach().clone()
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="partition_kl",
    )
    assert token_loss.abs().max() < 1e-6
    token_loss.sum().backward()
    assert student_logits.grad.abs().max() < 1e-5


def test_partition_kl_ungates_the_dropped_teacher_token():
    """The teacher-wanted token the student dropped escapes the pairwise double gate.

    Scenario: student holds p(y) ~ 0.7 on its anchor and has demoted the token the
    teacher actually wants to p ~ 0.002. In the pairwise form that duel receives
    weight p(o) AND the sigmoid'(m) gate, so its gradient is exponentially dead.
    The partition form keeps a single mass factor p(o), which is ~two orders of
    magnitude more. The anchor's own corrective gradient also grows, but only
    moderately: the p(1-p) mass factor is intrinsic to the reverse direction and
    stays in both forms.
    """
    vocab = 8
    bs, seq_len = 1, 1
    # Log-prob-shaped logits: p = (0.7, 0.002, 0.0497 x 6), q = (0.1, 0.5, 0.0667 x 6).
    student_logits = torch.full((bs, seq_len, vocab), math.log(0.0497), dtype=torch.float64)
    student_logits[0, 0, 0] = math.log(0.7)
    student_logits[0, 0, 1] = math.log(0.002)
    teacher_logits = torch.full((bs, seq_len, vocab), math.log(0.0667), dtype=torch.float64)
    teacher_logits[0, 0, 0] = math.log(0.1)
    teacher_logits[0, 0, 1] = math.log(0.5)
    anchors = torch.zeros(bs, seq_len, dtype=torch.long)
    response_mask = torch.ones(bs, seq_len)
    # Teacher top-4 contains the anchor (q(y) = 0.1 is rank 2), exercising dedup.
    candidate_ids = _candidates_from_teacher(teacher_logits, 4)

    grads = {}
    for divergence in ("partition_kl", "reverse_kl"):
        logits = student_logits.clone().requires_grad_(True)
        token_loss, _ = _call_loss(
            logits,
            teacher_logits,
            anchors,
            response_mask,
            candidate_ids,
            tail_candidate=True,
            pair_divergence=divergence,
            weight_source="student",
        )
        token_loss.sum().backward()
        grads[divergence] = logits.grad[0, 0].clone()

    dropped_partition = float(grads["partition_kl"][1])
    dropped_pairwise = float(grads["reverse_kl"][1])
    # Both want to raise the dropped token (negative gradient) ...
    assert dropped_partition < 0 and dropped_pairwise < 0, grads
    # ... but the pairwise double gate suppresses it by ~two orders of magnitude.
    assert abs(dropped_partition) > 20.0 * abs(dropped_pairwise), (dropped_partition, dropped_pairwise)

    anchor_partition = float(grads["partition_kl"][0])
    anchor_pairwise = float(grads["reverse_kl"][0])
    # Both push the wrongly confident anchor down; partition moderately harder.
    assert anchor_partition > 0 and anchor_pairwise > 0, grads
    assert anchor_partition > 1.2 * anchor_pairwise, (anchor_partition, anchor_pairwise)


def test_partition_kl_requires_tail_candidate():
    vocab, k = 12, 4
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=37)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    try:
        _call_loss(
            student_logits,
            teacher_logits,
            anchors,
            response_mask,
            candidate_ids,
            tail_candidate=False,
            complement_candidate=False,
            pair_divergence="partition_kl",
        )
    except ValueError as error:
        assert "partition_kl" in str(error)
    else:
        raise AssertionError("partition_kl without tail_candidate should raise")


def _reference_odds_kl(student_logits, teacher_logits, anchors, candidate_ids, response_mask):
    """Brute-force sg[p(y)] * sum over duels of GKL(rho || rho_T), tail duel included."""
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, _ = student_logits.shape

    token_loss = torch.zeros(bs, seq_len, dtype=torch.float64)
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                continue
            y = int(anchors[b, t])
            named = {int(z) for z in candidate_ids[b, t] if int(z) != y}
            p_y = student_log_probs[b, t, y].exp()
            q_y = teacher_log_probs[b, t, y].exp()
            duels = []
            for z in sorted(named):
                duels.append(
                    (student_log_probs[b, t, z].exp() / p_y, teacher_log_probs[b, t, z].exp() / q_y)
                )
            p_named = sum(student_log_probs[b, t, z].exp() for z in named)
            q_named = sum(teacher_log_probs[b, t, z].exp() for z in named)
            duels.append(((1.0 - p_y - p_named) / p_y, (1.0 - q_y - q_named) / q_y))
            total = 0.0
            for rho, rho_t in duels:
                total = total + rho * (torch.log(rho) - torch.log(rho_t)) - rho + rho_t
            token_loss[b, t] = p_y * total
    return token_loss


def test_odds_kl_matches_duel_definition():
    """odds_kl is sg[p(y)] times the summed generalized KL between anchored odds."""
    vocab, k = 12, 5
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=41)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, diagnostics = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="odds_kl",
        weight_source="student",
    )
    expected = _reference_odds_kl(student_logits, teacher_logits, anchors, candidate_ids, response_mask)

    torch.testing.assert_close(token_loss.double(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(diagnostics["odds_kl"].double(), expected, atol=1e-5, rtol=1e-4)
    # Every duel is a divergence and sg[p(y)] >= 0, so the loss is nonnegative.
    assert (token_loss >= -1e-9).all()


def test_odds_kl_gradient_is_mass_times_gap():
    """Logit gradients: anchor sums the duel votes, every other token gets -p(v) * u(duel of v).

    u_c = m_c - m_c^T is the duel gap; named candidates carry their own duel, tail
    tokens share the tail duel's gap. The input-gradients sum to zero, so the
    softmax normalizer contributes nothing and the logit gradient equals the
    log-prob gradient.
    """
    vocab, k = 10, 4
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=43)
    student_logits = student_logits.clone().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="odds_kl",
    )
    token_loss.sum().backward()

    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, _ = student_logits.shape
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                torch.testing.assert_close(
                    student_logits.grad[b, t], torch.zeros(vocab, dtype=torch.float64)
                )
                continue
            y = int(anchors[b, t])
            named = {int(z) for z in candidate_ids[b, t] if int(z) != y}
            tail = [v for v in range(vocab) if v != y and v not in named]
            delta = student_log_probs[b, t] - teacher_log_probs[b, t]
            p_tail = student_log_probs[b, t, tail].exp().sum()
            q_tail = teacher_log_probs[b, t, tail].exp().sum()
            delta_tail = p_tail.log() - q_tail.log()

            expected = torch.zeros(vocab, dtype=torch.float64)
            anchor_grad = 0.0
            for z in named:
                u = delta[y] - delta[z]
                p_z = student_log_probs[b, t, z].exp()
                expected[z] = -p_z * u
                anchor_grad = anchor_grad + p_z * u
            u_tail = delta[y] - delta_tail
            for v in tail:
                expected[v] = -student_log_probs[b, t, v].exp() * u_tail
            anchor_grad = anchor_grad + p_tail * u_tail
            expected[y] = anchor_grad

            torch.testing.assert_close(student_logits.grad[b, t], expected, atol=1e-6, rtol=1e-5)


def test_odds_kl_is_zero_with_zero_gradient_at_p_equals_q():
    vocab, k = 9, 4
    student_logits, _, anchors, response_mask = _make_batch(vocab=vocab, seed=47)
    student_logits = student_logits.clone().requires_grad_(True)
    teacher_logits = student_logits.detach().clone()
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="odds_kl",
    )
    token_loss.sum().backward()

    torch.testing.assert_close(token_loss, torch.zeros_like(token_loss), atol=1e-8, rtol=0)
    torch.testing.assert_close(
        student_logits.grad, torch.zeros_like(student_logits.grad), atol=1e-7, rtol=0
    )


def test_odds_kl_ungates_the_dropped_teacher_token():
    """Same scenario as the partition test: the dropped teacher-wanted token escapes
    the pairwise double gate with a single mass factor (~100x), and the anchor's
    corrective gradient grows through the duel votes."""
    vocab = 8
    bs, seq_len = 1, 1
    student_logits = torch.full((bs, seq_len, vocab), math.log(0.0497), dtype=torch.float64)
    student_logits[0, 0, 0] = math.log(0.7)
    student_logits[0, 0, 1] = math.log(0.002)
    teacher_logits = torch.full((bs, seq_len, vocab), math.log(0.0667), dtype=torch.float64)
    teacher_logits[0, 0, 0] = math.log(0.1)
    teacher_logits[0, 0, 1] = math.log(0.5)
    anchors = torch.zeros(bs, seq_len, dtype=torch.long)
    response_mask = torch.ones(bs, seq_len)
    candidate_ids = _candidates_from_teacher(teacher_logits, 4)

    grads = {}
    for divergence in ("odds_kl", "reverse_kl"):
        logits = student_logits.clone().requires_grad_(True)
        token_loss, _ = _call_loss(
            logits,
            teacher_logits,
            anchors,
            response_mask,
            candidate_ids,
            tail_candidate=True,
            pair_divergence=divergence,
            weight_source="student",
        )
        token_loss.sum().backward()
        grads[divergence] = logits.grad[0, 0].clone()

    dropped_odds = float(grads["odds_kl"][1])
    dropped_pairwise = float(grads["reverse_kl"][1])
    assert dropped_odds < 0 and dropped_pairwise < 0, grads
    assert abs(dropped_odds) > 20.0 * abs(dropped_pairwise), (dropped_odds, dropped_pairwise)

    anchor_odds = float(grads["odds_kl"][0])
    anchor_pairwise = float(grads["reverse_kl"][0])
    assert anchor_odds > 0 and anchor_pairwise > 0, grads
    assert anchor_odds > 1.5 * anchor_pairwise, (anchor_odds, anchor_pairwise)


def test_odds_kl_upper_bounds_partition_kl():
    """sg[p(y)] * sum GKL(rho || rho_T) >= coarse KL(p_bar || q_bar) (log x <= x - 1)."""
    vocab, k = 12, 5
    for seed in (53, 59, 61):
        student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=seed)
        candidate_ids = _candidates_from_teacher(teacher_logits, k)
        common = dict(tail_candidate=True, weight_source="student")
        odds, _ = _call_loss(
            student_logits, teacher_logits, anchors, response_mask, candidate_ids,
            pair_divergence="odds_kl", **common,
        )
        partition, _ = _call_loss(
            student_logits, teacher_logits, anchors, response_mask, candidate_ids,
            pair_divergence="partition_kl", **common,
        )
        assert (odds - partition >= -1e-7).all(), (odds - partition).min()


def test_odds_kl_k0_reduces_to_anchor_odds_gkl():
    """With no named candidates the loss is one tail duel:
    sg[p(y)] * GKL((1-p(y))/p(y) || (1-q(y))/q(y))."""
    vocab = 10
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=67)
    candidate_ids = anchors.unsqueeze(-1)  # every column deduplicates against the anchor

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="odds_kl",
    )

    p_y = torch.log_softmax(student_logits, dim=-1).gather(-1, anchors.unsqueeze(-1)).squeeze(-1).exp()
    q_y = torch.log_softmax(teacher_logits, dim=-1).gather(-1, anchors.unsqueeze(-1)).squeeze(-1).exp()
    rho = (1.0 - p_y) / p_y
    rho_t = (1.0 - q_y) / q_y
    expected = p_y.detach() * (rho * (rho.log() - rho_t.log()) - rho + rho_t) * response_mask

    torch.testing.assert_close(token_loss.double(), expected.double(), atol=1e-6, rtol=1e-5)


def _reference_jeffreys(student_logits, teacher_logits, anchors, candidate_ids, response_mask):
    """Brute-force sum of both Bernoulli KL directions per duel, student-mass weighted."""
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    bs, seq_len, _ = student_logits.shape

    def bernoulli_kl(p, q):
        return p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))

    token_loss = torch.zeros(bs, seq_len, dtype=torch.float64)
    for b in range(bs):
        for t in range(seq_len):
            if response_mask[b, t] == 0:
                continue
            y = int(anchors[b, t])
            named = {int(z) for z in candidate_ids[b, t] if int(z) != y}
            p_y = student_log_probs[b, t, y].exp()
            q_y = teacher_log_probs[b, t, y].exp()
            p_named = sum(student_log_probs[b, t, z].exp() for z in named)
            q_named = sum(teacher_log_probs[b, t, z].exp() for z in named)
            opponents = [(student_log_probs[b, t, z].exp(), teacher_log_probs[b, t, z].exp()) for z in sorted(named)]
            opponents.append((1.0 - p_y - p_named, 1.0 - q_y - q_named))
            weights = torch.tensor([p_o for p_o, _ in opponents], dtype=torch.float64)
            weights = weights / weights.sum()
            total = 0.0
            for w, (p_o, q_o) in zip(weights, opponents):
                r_s = p_y / (p_y + p_o)
                r_t = q_y / (q_y + q_o)
                total = total + w * (bernoulli_kl(r_s, r_t) + bernoulli_kl(r_t, r_s))
            token_loss[b, t] = total
    return token_loss


def test_jeffreys_matches_sum_of_both_kl_directions():
    """(sigma(m) - sigma(m_T)) * (m - m_T) is exactly KL(p||q) + KL(q||p) per duel."""
    vocab, k = 12, 5
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=71)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="jeffreys",
        weight_source="student",
    )
    expected = _reference_jeffreys(student_logits, teacher_logits, anchors, candidate_ids, response_mask)

    torch.testing.assert_close(token_loss.double(), expected, atol=1e-5, rtol=1e-4)
    # Both factors share sign, so every duel term and hence the loss is nonnegative.
    assert (token_loss >= -1e-9).all()


def test_jeffreys_is_zero_with_zero_gradient_at_p_equals_q():
    vocab, k = 9, 4
    student_logits, _, anchors, response_mask = _make_batch(vocab=vocab, seed=73)
    student_logits = student_logits.clone().requires_grad_(True)
    teacher_logits = student_logits.detach().clone()
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    token_loss, _ = _call_loss(
        student_logits,
        teacher_logits,
        anchors,
        response_mask,
        candidate_ids,
        tail_candidate=True,
        pair_divergence="jeffreys",
        weight_source="student",
    )
    token_loss.sum().backward()

    torch.testing.assert_close(token_loss, torch.zeros_like(token_loss), atol=1e-8, rtol=0)
    torch.testing.assert_close(
        student_logits.grad, torch.zeros_like(student_logits.grad), atol=1e-7, rtol=0
    )


def test_jeffreys_ungates_the_dropped_teacher_token():
    """The forward half of the Jeffreys gradient survives where sigma' is dead."""
    vocab = 8
    bs, seq_len = 1, 1
    student_logits = torch.full((bs, seq_len, vocab), math.log(0.0497), dtype=torch.float64)
    student_logits[0, 0, 0] = math.log(0.7)
    student_logits[0, 0, 1] = math.log(0.002)
    teacher_logits = torch.full((bs, seq_len, vocab), math.log(0.0667), dtype=torch.float64)
    teacher_logits[0, 0, 0] = math.log(0.1)
    teacher_logits[0, 0, 1] = math.log(0.5)
    anchors = torch.zeros(bs, seq_len, dtype=torch.long)
    response_mask = torch.ones(bs, seq_len)
    candidate_ids = _candidates_from_teacher(teacher_logits, 4)

    grads = {}
    for divergence in ("jeffreys", "reverse_kl"):
        logits = student_logits.clone().requires_grad_(True)
        token_loss, _ = _call_loss(
            logits,
            teacher_logits,
            anchors,
            response_mask,
            candidate_ids,
            tail_candidate=True,
            pair_divergence=divergence,
            weight_source="student",
        )
        token_loss.sum().backward()
        grads[divergence] = logits.grad[0, 0].clone()

    dropped_jeffreys = float(grads["jeffreys"][1])
    dropped_pairwise = float(grads["reverse_kl"][1])
    assert dropped_jeffreys < 0 and dropped_pairwise < 0, grads
    assert abs(dropped_jeffreys) > 10.0 * abs(dropped_pairwise), (dropped_jeffreys, dropped_pairwise)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
