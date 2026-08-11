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
    # direction, so ask for it explicitly rather than riding on the library default.
    kwargs.setdefault("pair_divergence", "forward_kl")
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


def test_reverse_kl_is_the_default_direction():
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=23)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    explicit, _ = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, pair_divergence="reverse_kl"
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
