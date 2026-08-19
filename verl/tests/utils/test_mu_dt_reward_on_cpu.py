"""Acceptance tests for mu_dt: r = -p * (logp - logq~), q~ = q^(1/sqrt(mu)) / Z,
mu = KL(q||u)/KL(p||u) = (logK - H(q))/(logK - H(p)), clamp-free.

Lesion-derived criteria:

    1. radius match is a contraction: on tempered pairs the target's entropy
       lands between H(q) and H(p) and NEVER crosses the student's radius
       (the 1/mu exponent crosses on 18-20% of real fog positions - the
       reversed-chase channel; the sqrt never does);
    2. level chasing damped two-sided: force shrinks vs baseline whether the
       teacher is softer or sharper (rkl_dt's one-sided gate is gone);
    3. confident-wrong boosted: at CW the teacher is right-but-softer
       (mu < 1), the rule sharpens the target, promotion AND suppression
       strengthen vs baseline;
    4. no false kills: true forks survive the re-radiusing;
    5. baseline recovered exactly at H(p) = H(q) (mu = 1) - strict
       generalization of baseline OPD;
    6. no clamps: finite on junk evals, near-uniform teachers, masks,
       singleton supports.
"""

import torch

from verl.utils.detemper_reward import compute_mu_dt_scores

K = 16


def _support(head):
    head = torch.tensor(head, dtype=torch.float64)
    ntail = K - head.numel()
    w = torch.tensor([0.72**i for i in range(ntail)], dtype=torch.float64)
    tail = (1.0 - head.sum()) * w / w.sum()
    return torch.cat([head, tail]).log().float().view(1, 1, K)


def _ent(lp):
    return -(lp.exp() * lp.clamp_min(-30)).sum(-1)


def _base(s, t):
    sN = torch.log_softmax(s, -1)
    tN = torch.log_softmax(t, -1)
    return -sN.exp() * (sN - tN)


def test_reduces_to_baseline_at_radius_match():
    # q = permutation of p => H(q) = H(p) => mu = 1 => q~ = q exactly
    torch.manual_seed(0)
    s = torch.log_softmax(torch.randn(6, 5, K) * 3, -1)
    perm = torch.randperm(K)
    t = s[..., perm]
    mask = torch.ones_like(s, dtype=torch.bool)
    r = compute_mu_dt_scores(s, t, mask).float()
    assert (r - _base(s, t)).abs().max().item() < 1e-5


def test_fog_damped_two_sided_and_radius_never_crossed():
    torch.manual_seed(1)
    s = torch.log_softmax(torch.randn(64, 4, K) * 3, -1)
    mask = torch.ones_like(s, dtype=torch.bool)
    for temps, max_ratio in ((torch.rand(64, 4, 1) * 2.0 + 1.4, 0.6),   # teacher softer
                             (torch.rand(64, 4, 1) * 0.3 + 0.55, 0.95)):  # teacher sharper
        t = torch.log_softmax(s / temps, -1)
        r_mu = compute_mu_dt_scores(s, t, mask).float().abs().sum()
        r_b = _base(s, t).abs().sum()
        assert r_mu < max_ratio * r_b, f"{r_mu / r_b}"
        # recover the target entropy and check the contraction property
        log_k = torch.log(torch.tensor(float(K)))
        n_p = (log_k - _ent(s)).clamp_min(0.0)
        n_q = (log_k - _ent(t)).clamp_min(1e-6)
        t_mudt = torch.log_softmax((n_p / n_q).sqrt().unsqueeze(-1) * t, -1)
        gap_q = _ent(t) - _ent(s)
        gap_t = _ent(t_mudt) - _ent(s)
        assert (gap_t * gap_q >= -1e-4).all()  # never crosses the student's radius
        assert (gap_t.abs() <= gap_q.abs() + 1e-4).all()  # strictly closer


def test_confidently_wrong_boosted():
    # teacher right-but-softer (the measured CW anatomy: teacher confidence
    # 0.55-0.70, H(q) - H(p) ~ +0.5-1.3 nats): mu < 1 sharpens the target,
    # so promotion AND suppression strengthen vs baseline
    s = _support([0.947, 0.047])
    t = _support([0.10, 0.60]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    r = compute_mu_dt_scores(s, t, mask).float()
    rb = _base(s, t)
    assert r[0, 0, 1].item() > rb[0, 0, 1].item() > 0  # promotion up
    assert r[0, 0, 0].item() < rb[0, 0, 0].item() < 0  # suppression up
    z = s.clone()
    for _ in range(40):
        z = z + compute_mu_dt_scores(z, t, mask).float()
    p = torch.log_softmax(z, -1).exp()
    assert p[0, 0, 1].item() > p[0, 0, 0].item()  # top-1 flipped


def test_true_fork_survives():
    s = _support([0.85, 0.03, 0.04])
    t = _support([0.50, 0.02, 0.40]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    z = s.clone()
    for _ in range(300):
        z = z + 0.1 * compute_mu_dt_scores(z, t, mask).float()
    p = torch.log_softmax(z, -1).exp()
    assert p[0, 0, 2].item() > 0.30  # the fork branch is alive (audit: 0.38)
    assert p[0, 0, 0].item() > p[0, 0, 2].item()  # teacher's mild preference kept


def test_healthy_untouched():
    # same top-1, same radius, small structural noise: force ~ baseline force,
    # and both are small - the correction is exactly off (mu = 1)
    torch.manual_seed(2)
    s = torch.log_softmax(torch.randn(16, 4, K) * 2.5, -1)
    t = torch.log_softmax(s + torch.randn_like(s) * 0.05, -1)
    mask = torch.ones_like(s, dtype=torch.bool)
    r = compute_mu_dt_scores(s, t, mask).float()
    rb = _base(s, t)
    assert (r.abs().sum() / rb.abs().sum()).item() < 1.1


def test_masked_support_and_variable_k():
    # logK must be the log of the VALID cell count: tempered pair restricted
    # to a 10-cell support still gets damped vs baseline on that support
    torch.manual_seed(3)
    s = torch.randn(8, 4, K) * 3
    mask = torch.zeros(8, 4, K, dtype=torch.bool)
    mask[..., :10] = True
    neg_inf = torch.finfo(torch.float32).min
    sm = torch.where(mask, s, torch.full_like(s, neg_inf))
    sN = torch.log_softmax(sm, -1)
    t = torch.where(mask, sN / 1.8, torch.full_like(s, neg_inf))
    r = compute_mu_dt_scores(sm, t, mask).float()
    tN = torch.log_softmax(t, -1)
    rb = -sN.exp() * (sN - tN)
    rb = torch.where(mask, rb, torch.zeros_like(rb))
    assert r[~mask].abs().max().item() == 0.0
    assert r.abs().sum() < 0.6 * rb.abs().sum()


def test_bounded_finite_degenerate():
    torch.manual_seed(4)
    s = torch.randn(8, 8, K) * 4
    t = torch.randn(8, 8, K) * 4
    t[..., -1] = -35.0  # junk teacher eval
    mask = torch.rand(8, 8, K) > 0.4
    mask[..., :2] = True
    r = compute_mu_dt_scores(s, t, mask)
    assert torch.isfinite(r).all()
    assert r[~mask].abs().max().item() == 0.0
    # near-uniform teacher (the theoretical singularity, KL(q||u) -> _EPS):
    # finite without any clamp
    r_u = compute_mu_dt_scores(
        _support([0.9, 0.05]), torch.randn(1, 1, K) * 0.01, torch.ones(1, 1, K, dtype=torch.bool)
    )
    assert torch.isfinite(r_u).all()
    # near-uniform student: beta -> 0, target -> uniform, force bounded by logK
    r_d = compute_mu_dt_scores(
        torch.zeros(1, 1, K), _support([0.9, 0.05]), torch.ones(1, 1, K, dtype=torch.bool)
    )
    assert torch.isfinite(r_d).all()
    assert r_d.abs().max().item() < 3.0
    # singleton support: zero force, no NaN from 0 * (-inf)
    mask1 = torch.zeros(1, 1, K, dtype=torch.bool)
    mask1[..., 0] = True
    r_1 = compute_mu_dt_scores(torch.randn(1, 1, K), torch.randn(1, 1, K), mask1)
    assert torch.isfinite(r_1).all()
    assert r_1.abs().max().item() < 1e-6
