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
"""Unit tests for the EFW edit field (w = KL(b||q) on the student top-k)."""

import torch

from verl.utils.efw import apply_efw_field_to_scores, compute_edit_field

TOP_K = 16


def _make_sharp_models(vocab=64, bs=2, seq_len=5, seed=0):
    """Peaked base/teacher distributions so top-16 of b covers >99% of b's mass.

    Deterministic decaying logits (permuted per position) guarantee the coverage
    regardless of the torch RNG; small noise varies the per-candidate edits.
    """
    generator = torch.Generator().manual_seed(seed)
    decay = -0.6 * torch.arange(vocab, dtype=torch.float32)
    b_logits = torch.empty(bs, seq_len, vocab)
    for i in range(bs):
        for t in range(seq_len):
            perm = torch.randperm(vocab, generator=generator)
            b_logits[i, t] = decay[perm.argsort()]
    b_logits = b_logits + 0.2 * torch.randn(bs, seq_len, vocab, generator=generator)
    q_logits = b_logits + 0.8 * torch.randn(bs, seq_len, vocab, generator=generator)
    b_log_probs = torch.log_softmax(b_logits, dim=-1)
    q_log_probs = torch.log_softmax(q_logits, dim=-1)
    return b_log_probs, q_log_probs


def _gather_on_candidates(log_probs, candidate_ids):
    return log_probs.gather(dim=-1, index=candidate_ids)


def test_field_matches_full_vocab_kl_when_coverage_high():
    b_log_probs, q_log_probs = _make_sharp_models()

    exact = (b_log_probs.exp() * (b_log_probs - q_log_probs)).sum(dim=-1)

    # At init the student equals b, so its top-k is b's top-k.
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    coverage = _gather_on_candidates(b_log_probs, candidate_ids).exp().sum(dim=-1)
    assert coverage.min() > 0.99, "test setup should give near-total b-mass coverage"

    estimated = compute_edit_field(
        _gather_on_candidates(b_log_probs, candidate_ids),
        _gather_on_candidates(q_log_probs, candidate_ids),
    )

    assert estimated.shape == exact.shape
    rel_err = (estimated - exact).abs() / exact.clamp_min(1e-6)
    assert rel_err.max() < 0.05, f"max relative error {rel_err.max():.4f} too large under >99% coverage"


def test_field_nonnegative_under_truncation():
    # b spreads mass widely; q concentrates extra mass exactly on b's top-k, so the
    # truncated sum over those candidates is negative while the exact KL is not.
    vocab = 64
    b_log_probs = torch.log_softmax(torch.zeros(1, 1, vocab), dim=-1)
    q_logits = torch.zeros(1, 1, vocab)
    q_logits[..., :TOP_K] = 2.0
    q_log_probs = torch.log_softmax(q_logits, dim=-1)

    candidate_ids = torch.arange(TOP_K).view(1, 1, -1)
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)
    q_on_c = _gather_on_candidates(q_log_probs, candidate_ids)

    raw = (b_on_c.exp() * (b_on_c - q_on_c)).sum(dim=-1)
    assert raw.item() < 0.0, "test setup should make the truncated sum negative"

    field = compute_edit_field(b_on_c, q_on_c)
    assert field.item() == 0.0


def test_floor_lifts_zero_field():
    b_log_probs, _ = _make_sharp_models()
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)

    # q == b: the exact field is identically zero.
    field = compute_edit_field(b_on_c, b_on_c)
    assert torch.all(field == 0.0)

    floored = compute_edit_field(b_on_c, b_on_c, floor=0.05)
    assert torch.all(floored == 0.05)


def test_apply_scales_scores_per_position():
    b_log_probs, q_log_probs = _make_sharp_models(bs=3, seq_len=7)
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)
    q_on_c = _gather_on_candidates(q_log_probs, candidate_ids)

    generator = torch.Generator().manual_seed(1)
    scores = torch.randn(3, 7, TOP_K, generator=generator)
    response_mask = torch.ones(3, 7)
    response_mask[-1, -2:] = 0.0

    weighted, field, metrics = apply_efw_field_to_scores(
        scores=scores,
        ref_on_student_log_probs=b_on_c,
        teacher_on_student_log_probs=q_on_c,
        response_mask=response_mask,
        student_top_k_log_probs=b_on_c,
        floor=0.0,
    )

    torch.testing.assert_close(weighted, scores * field.unsqueeze(-1))
    assert torch.all(field >= 0.0)

    # Zero field must kill the teaching signal entirely at that state.
    zero_field_weighted, zero_field, _ = apply_efw_field_to_scores(
        scores=scores,
        ref_on_student_log_probs=b_on_c,
        teacher_on_student_log_probs=b_on_c,
        response_mask=response_mask,
    )
    assert torch.all(zero_field == 0.0)
    assert torch.all(zero_field_weighted == 0.0)

    for key in (
        "efw/field_mean",
        "efw/field_p50",
        "efw/field_p90",
        "efw/field_p99",
        "efw/field_max",
        "efw/field_frac_low",
        "efw/b_mass_coverage",
    ):
        assert key in metrics, f"missing metric {key}"

    expected_coverage = b_on_c.exp().sum(dim=-1)[response_mask.bool()].mean().item()
    assert abs(metrics["efw/b_mass_coverage"] - expected_coverage) < 1e-6
    assert expected_coverage <= 1.0 + 1e-6


def test_metrics_respect_response_mask():
    b_log_probs, q_log_probs = _make_sharp_models(bs=1, seq_len=4)
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)
    q_on_c = _gather_on_candidates(q_log_probs, candidate_ids)

    # Corrupt the last (masked) position with an absurd teacher stream; the field
    # there explodes, but masked positions must not leak into the metrics.
    q_on_c_corrupt = q_on_c.clone()
    q_on_c_corrupt[0, -1, :] = -50.0
    response_mask = torch.ones(1, 4)
    response_mask[0, -1] = 0.0

    scores = torch.ones(1, 4, TOP_K)
    _, _, metrics_clean = apply_efw_field_to_scores(
        scores=scores,
        ref_on_student_log_probs=b_on_c,
        teacher_on_student_log_probs=q_on_c,
        response_mask=response_mask,
    )
    _, _, metrics_corrupt = apply_efw_field_to_scores(
        scores=scores,
        ref_on_student_log_probs=b_on_c,
        teacher_on_student_log_probs=q_on_c_corrupt,
        response_mask=response_mask,
    )
    assert abs(metrics_clean["efw/field_max"] - metrics_corrupt["efw/field_max"]) < 1e-6


def test_efw_scales_order_gated_loss_gradients():
    """The dp_actor composition: token_loss <- field * token_loss (og-kl flow).

    The field is a frozen per-state constant, so the per-position gradient of the
    weighted loss must be exactly field(s_t) times the unweighted gradient.
    """
    from verl.trainer.ppo.l_apd import compute_l_apd_token_loss

    b_log_probs, q_log_probs = _make_sharp_models(bs=2, seq_len=3)
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)
    q_on_c = _gather_on_candidates(q_log_probs, candidate_ids)
    field = compute_edit_field(b_on_c, q_on_c)
    assert (field > 0).any(), "test setup should produce a nontrivial field"

    # Student mid-way between b and q so the og-kl loss and its gradients are nonzero.
    student_log_probs = torch.log_softmax(0.5 * (b_log_probs + q_log_probs), dim=-1)
    anchor = student_log_probs[..., 0]
    response_mask = torch.ones(2, 3)
    candidate_mask = torch.ones(2, 3, TOP_K, dtype=torch.bool)

    def _token_loss(leaf):
        token_loss, _ = compute_l_apd_token_loss(
            student_anchor_log_probs=anchor,
            student_candidate_log_probs=leaf,
            teacher_anchor_log_probs=q_on_c[..., 0],
            teacher_candidate_log_probs=q_on_c,
            candidate_mask=candidate_mask,
            response_mask=response_mask,
            tail_candidate=False,
            complement_candidate=False,
            pair_divergence="order_gated_kl",
        )
        return token_loss

    student_on_c = _gather_on_candidates(student_log_probs, candidate_ids)

    leaf_plain = student_on_c.clone().requires_grad_(True)
    _token_loss(leaf_plain).sum().backward()

    leaf_weighted = student_on_c.clone().requires_grad_(True)
    (_token_loss(leaf_weighted) * field).sum().backward()

    torch.testing.assert_close(leaf_weighted.grad, leaf_plain.grad * field.unsqueeze(-1))
    assert leaf_plain.grad.abs().sum() > 0, "unweighted loss should have nonzero gradients"


def test_2d_scores_path():
    b_log_probs, q_log_probs = _make_sharp_models(bs=1, seq_len=4)
    candidate_ids = b_log_probs.topk(TOP_K, dim=-1).indices
    b_on_c = _gather_on_candidates(b_log_probs, candidate_ids)
    q_on_c = _gather_on_candidates(q_log_probs, candidate_ids)

    scores = torch.ones(1, 4)
    weighted, field, _ = apply_efw_field_to_scores(
        scores=scores,
        ref_on_student_log_probs=b_on_c,
        teacher_on_student_log_probs=q_on_c,
        response_mask=torch.ones(1, 4),
    )
    torch.testing.assert_close(weighted, field)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all EFW cpu tests passed")
