"""Acceptance tests for the fkl mode: r = q - p, the exact logit gradient of
-KL(q||p) on the shared support.

Criteria (lesion-derived, plus the exactness/convexity properties that make
this the KL-family general solution):

    1. exactness: r equals the autograd gradient of -KL(q||p) w.r.t. the
       student logits; zero-sum per position; |r| <= 1.
    2. lesion triage embedded: confident-wrong gets the 2-cell surgery
       (full missing mass on the correction cell), teacher alternatives get
       the full mass difference, healthy tokens get ~0 - no classifier.
    3. bridge endpoint: on r^lam = (p^(1-lam) q^lam - p)/lam the lesion
       forces are monotone increasing in lam (lam=0 is the baseline force).
    4. convex flow: KL(q||p) decreases monotonically along the field, the
       CW cascade flips top-1, and the student never sharpens beyond the
       teacher (no new confident-wrongs).
    5. boundedness: finite on junk evals and masks, exact zeros off-mask.
"""

import torch

from verl.utils.detemper_reward import compute_fkl_scores

K = 16


def _support(head):
    head = torch.tensor(head, dtype=torch.float64)
    ntail = K - head.numel()
    w = torch.tensor([0.72**i for i in range(ntail)], dtype=torch.float64)
    tail = (1.0 - head.sum()) * w / w.sum()
    return torch.cat([head, tail]).log().float().view(1, 1, K)


def test_exact_gradient_zero_sum_bounded():
    torch.manual_seed(0)
    z = torch.randn(4, 4, K, requires_grad=True)
    t = torch.log_softmax(torch.randn(4, 4, K) * 2, -1)
    q = t.exp()
    kl = (q * (t - torch.log_softmax(z, -1))).sum()
    kl.backward()
    r = compute_fkl_scores(z.detach(), t, torch.ones(4, 4, K, dtype=torch.bool))
    assert (r - (-z.grad)).abs().max().item() < 1e-5  # r = -dKL/dz exactly
    assert r.sum(-1).abs().max().item() < 1e-5  # conservative transport
    assert r.abs().max().item() <= 1.0 + 1e-6


def test_cw_two_cell_surgery_and_flip():
    s = _support([0.94, 0.03])
    t = _support([0.12, 0.62])
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    r = compute_fkl_scores(s, t, mask)
    p, q = torch.log_softmax(s, -1).exp(), torch.log_softmax(t, -1).exp()
    assert abs(r[0, 0, 1].item() - (q - p)[0, 0, 1].item()) < 1e-6
    assert r[0, 0, 1].item() > 0.5  # full missing mass, not p-diluted
    assert r[0, 0, 0].item() < -0.5
    assert r[0, 0, 2:].abs().max().item() < 0.1  # junk cells nearly silent
    z = s.clone()
    for _ in range(40):
        z = z + 2.0 * compute_fkl_scores(z, t, mask).float()
    pf = torch.log_softmax(z, -1).exp()
    assert pf[0, 0, 1] > pf[0, 0, 0]  # flipped
    assert pf.max().item() < q.max().item() + 0.05  # never sharper than teacher


def test_structure_full_mass_difference():
    s = _support([0.90, 0.01])
    t = _support([0.65, 0.22])
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    r = compute_fkl_scores(s, t, mask)
    p, q = torch.log_softmax(s, -1).exp(), torch.log_softmax(t, -1).exp()
    assert abs(r[0, 0, 1].item() - (q[0, 0, 1] - p[0, 0, 1]).item()) < 1e-6
    assert r[0, 0, 1].item() > 0.15  # baseline gives ~+0.03 here


def test_healthy_token_silent():
    torch.manual_seed(1)
    s = torch.randn(4, 4, K) * 2
    r = compute_fkl_scores(s, s + torch.randn_like(s) * 0.01, torch.ones(4, 4, K, dtype=torch.bool))
    assert r.abs().max().item() < 0.02  # p ~ q -> ~0, no classifier needed


def test_bridge_endpoint_monotone():
    s = torch.log_softmax(_support([0.94, 0.03]), -1)
    t = torch.log_softmax(_support([0.12, 0.62]), -1)
    p, q = s.exp(), t.exp()
    prev = -1e9
    for lam in (0.25, 0.5, 0.75, 1.0):
        promo = ((p.pow(1 - lam) * q.pow(lam) - p) / lam)[0, 0, 1].item()
        assert promo > prev
        prev = promo
    base = (-p * (s - t))[0, 0, 1].item()
    fkl = compute_fkl_scores(s, t, torch.ones(1, 1, K, dtype=torch.bool))[0, 0, 1].item()
    assert fkl > 3 * base  # endpoint dominates the baseline force on the lesion cell


def test_convex_descent():
    torch.manual_seed(2)
    s = torch.randn(8, 4, K) * 3
    t = torch.log_softmax(torch.randn(8, 4, K) * 2, -1)
    mask = torch.ones(8, 4, K, dtype=torch.bool)
    q = t.exp()
    z = s.clone()
    prev = None
    for _ in range(30):
        kl = (q * (t - torch.log_softmax(z, -1))).sum(-1)
        if prev is not None:
            assert (kl <= prev + 1e-6).all()  # monotone: convex gradient flow
        prev = kl
        z = z + 1.0 * compute_fkl_scores(z, t, mask).float()


def test_bounded_finite_and_mask_respecting():
    torch.manual_seed(3)
    s = torch.randn(8, 8, K) * 4
    t = torch.randn(8, 8, K) * 4
    t[..., -1] = -35.0  # junk teacher eval: bounded response, no log blow-up
    mask = torch.rand(8, 8, K) > 0.4
    mask[..., :2] = True
    r = compute_fkl_scores(s, t, mask)
    assert torch.isfinite(r).all()
    assert r.abs().max().item() <= 1.0 + 1e-6
    assert r[~mask].abs().max().item() == 0.0
