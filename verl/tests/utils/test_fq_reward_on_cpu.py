"""Acceptance tests for the fiber-quotient reward (fq): r = z(logq) - z(logp).

Each test encodes one clause of the lesion-derived acceptance criteria:

    1. no level chasing: zero force on pure temperature fog (any T,
       two-sided), fixed set = the teacher's tempering orbit, level leak
       off-fiber is second order in the structure size (transient, not an
       order);
    2. structure transmission: correction/alternative cells receive their
       full z-gap, not a p-diluted trickle;
    3. no false kills: true semantic forks pass at full scale (no entropy
       gates, no one-sided filters);
    4. boundedness / degeneracy: finite, bounded, mask-respecting.
"""

import torch

from verl.utils.fq_reward import compute_fq_scores

K = 16


def _support(head):
    head = torch.tensor(head, dtype=torch.float64)
    ntail = K - head.numel()
    w = torch.tensor([0.72**i for i in range(ntail)], dtype=torch.float64)
    tail = (1.0 - head.sum()) * w / w.sum()
    return torch.cat([head, tail]).log().float().view(1, 1, K)


def _baseline(s_raw, t_raw, mask):
    neg_inf = torch.finfo(torch.float32).min
    s = torch.log_softmax(torch.where(mask, s_raw.float(), torch.full_like(s_raw, neg_inf)), -1)
    t = torch.log_softmax(torch.where(mask, t_raw.float(), torch.full_like(t_raw, neg_inf)), -1)
    return -s.exp() * (s - t)


def test_fiber_nullification_on_fog():
    torch.manual_seed(0)
    b, seq = 8, 4
    s = torch.randn(b, seq, K) * 3
    mask = torch.ones(b, seq, K, dtype=torch.bool)
    temps = torch.rand(b, seq, 1) * 4.8 + 0.2  # T in [0.2, 5], both sides of 1
    t = torch.log_softmax(torch.log_softmax(s, -1) / temps, -1)
    scores, aux = compute_fq_scores(s, t, mask, return_aux=True)
    assert scores.abs().max().item() < 1e-4
    assert (aux["fq_level_slope"] - (1.0 - 1.0 / temps.squeeze(-1))).abs().max().item() < 1e-3
    assert (aux["fq_level_r2"] - 1.0).abs().max().item() < 1e-4


def test_fixed_set_is_teacher_orbit():
    torch.manual_seed(1)
    t = torch.randn(4, 4, K) * 3
    mask = torch.ones(4, 4, K, dtype=torch.bool)
    mus = torch.rand(4, 4, 1) * 3.0 + 0.3
    s = torch.log_softmax(torch.log_softmax(t, -1) * mus, -1)  # p = q^mu / Z
    scores = compute_fq_scores(s, t, mask)
    assert scores.abs().max().item() < 1e-4


def test_level_leak_is_second_order():
    s = _support([0.72, 0.05, 0.03])
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    logp = torch.log_softmax(s, -1)
    fog = torch.log_softmax(s / 1.3, -1)
    leaks = []
    for eps in (0.1, 0.2):
        t = fog.clone()
        t[0, 0, 6] += eps
        t[0, 0, 0] -= eps
        r = compute_fq_scores(s, t, mask).float()
        assert r.sum(-1).abs().max().item() < 1e-4  # no off-support leak
        leaks.append((r * logp).sum().abs().item())
    assert 3.0 < leaks[1] / leaks[0] < 5.5  # <r, logp> scales as eps^2: transient, not an order


def test_confidently_wrong_gets_full_z_gap():
    s = _support([0.94, 0.03])
    t = _support([0.10, 0.60]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    scores = compute_fq_scores(s, t, mask)
    base = _baseline(s, t, mask)
    share = lambda r: (r[0, 0, 1].abs() / r.abs().sum()).item()
    assert scores[0, 0, 1].item() > 0.6  # strong repair force on the correct cell
    assert share(scores) > 3 * share(base)  # un-starved vs p-diluted baseline


def test_confidently_wrong_cascade_flips_top1():
    s = _support([0.94, 0.03])
    t = _support([0.10, 0.60]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    z = s.clone()
    entropies = []
    for _ in range(60):
        a = compute_fq_scores(z, t, mask)
        lp = torch.log_softmax(z, -1)
        entropies.append(-(lp.exp() * lp.clamp_min(-30)).sum().item())
        z = z + 0.3 * a.float()
    p = torch.log_softmax(z, -1).exp()
    assert p[0, 0, 1].item() > p[0, 0, 0].item()  # top-1 flipped to teacher's choice
    assert entropies[-1] < max(entropies)  # entropy comes back down: transport, not level chase


def test_true_fork_transmitted_full_scale():
    s = _support([0.93, 0.02, 0.01])
    t = _support([0.52, 0.40, 0.01]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    scores, aux = compute_fq_scores(s, t, mask, return_aux=True)
    base = _baseline(s, t, mask)
    share = lambda r: (r[0, 0, 1].abs() / r.abs().sum()).item()
    assert aux["fq_level_r2"].item() < 0.5  # fork not explained away as temperature
    assert scores[0, 0, 1].item() > 0.8  # full-scale transmission
    assert share(scores) > 3 * share(base)


def test_self_termination_at_student_level():
    s = _support([0.72, 0.05, 0.03])
    q = _support([0.60, 0.05, 0.03]).exp().squeeze()
    q[6] += 0.21 - q[6]
    q = q / q.sum()
    t = torch.log_softmax(q.log().view(1, 1, K) / 1.15, -1)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    teacher_h = -(t.exp() * t).sum().item()
    z = s.clone()
    for _ in range(80):
        z = z + 0.3 * compute_fq_scores(z, t, mask).float()
    a_final = compute_fq_scores(z, t, mask)
    lp = torch.log_softmax(z, -1)
    h_final = -(lp.exp() * lp.clamp_min(-30)).sum().item()
    assert a_final.abs().max().item() < 1e-2  # force self-terminates (echo severed)
    assert h_final < teacher_h - 0.1  # ... at the student's level, not the teacher's


def test_bounded_finite_and_mask_respecting():
    torch.manual_seed(3)
    s = torch.randn(8, 8, K) * 4
    t = torch.randn(8, 8, K) * 4
    t[..., -1] = -35.0  # junk teacher eval outlier
    mask = torch.rand(8, 8, K) > 0.4
    mask[..., :2] = True
    scores = compute_fq_scores(s, t, mask)
    assert torch.isfinite(scores).all()
    assert scores.abs().max().item() < 2 * (K - 1) / K**0.5 + 1e-3  # z-score bound
    assert scores[~mask].abs().max().item() == 0.0
    # degenerate student (uniform on support): no fiber, finite force
    scores_u = compute_fq_scores(torch.zeros(1, 1, K), torch.randn(1, 1, K), torch.ones(1, 1, K, dtype=torch.bool))
    assert torch.isfinite(scores_u).all()
