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
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, tail_candidate=False
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
    """dL/d(S(y) - S(z)) = q~(z) * (r_S(y, z) - r_T(y, z))."""
    vocab = 12
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=7)
    student_logits = student_logits.float().requires_grad_(True)
    candidate_ids = _candidates_from_teacher(teacher_logits, k=4)

    token_loss, _ = _call_loss(
        student_logits, teacher_logits.float(), anchors, response_mask, candidate_ids, tail_candidate=False
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
    assert diagnostics["bernoulli_kl"].max() < 1e-5
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
        normalize_weights=False,
    )
    _, renormalized = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, tail_candidate=False
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


def test_truncated_pairs_alone_leave_the_tail_mass_unidentified():
    """Without a tail candidate the pairwise terms only constrain logit differences.

    Scaling the mass of ``{anchor} + candidates`` up or down, with the slack absorbed
    by the truncated tail, leaves every ``S(y) - S(z)`` untouched and so cannot change
    the loss. The anchor term is what removes that freedom, which is why
    ``target_loss_coef`` must be non-zero whenever ``tail_candidate`` is off.
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

    for tail_candidate, target_loss_coef, should_differ in [
        (False, 0.0, False),  # the unidentified direction
        (False, 1.0, True),  # anchor term removes it
        (True, 0.0, True),  # tail candidate removes it
    ]:
        kwargs = dict(tail_candidate=tail_candidate, target_loss_coef=target_loss_coef)
        base, _ = _call_loss(student_logits, teacher_logits, anchors, response_mask, candidate_ids, **kwargs)
        moved, _ = _call_loss(shifted, teacher_logits, anchors, response_mask, candidate_ids, **kwargs)
        if should_differ:
            assert (moved - base).abs().max() > 1e-3, (tail_candidate, target_loss_coef)
        else:
            torch.testing.assert_close(moved, base, atol=1e-5, rtol=1e-4)


def test_anchor_term_is_minimized_when_anchor_probs_match():
    """The reported anchor KL vanishes exactly when ``p(y_t) == q(y_t)``."""
    vocab, k = 32, 8
    student_logits, teacher_logits, anchors, response_mask = _make_batch(vocab=vocab, seed=4)
    candidate_ids = _candidates_from_teacher(teacher_logits, k)

    _, mismatched = _call_loss(
        student_logits, teacher_logits, anchors, response_mask, candidate_ids, target_loss_coef=1.0
    )
    _, matched = _call_loss(
        teacher_logits, teacher_logits, anchors, response_mask, candidate_ids, target_loss_coef=1.0
    )

    assert (matched["anchor_kl"] * response_mask).abs().max() < 1e-5
    assert (mismatched["anchor_kl"] * response_mask).max() > 1e-3
    assert (mismatched["anchor_kl"] * response_mask).min() >= -1e-5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
