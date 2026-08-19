"""Acceptance tests for rkl_cdt: r = -p * (logp - logq~), logq~ = lsm(sigma_pw * z_w(logq)),
with the polar decomposition taken in the student-mass (Fisher) metric w = p.

Lesion-derived criteria, adapted to the -p*(.) force form:

    1. no level chasing: exact zero on pure temperature fog, BOTH directions
       (no one-sided entropy gate), fixed set = the teacher's tempering orbit;
       AND the real-support variant - head is a pure temper but the deep
       massless tails are two models' unrelated junk (this is what broke the
       unweighted polar on real val trajectories);
    2. confident-wrong: dt-style funnel (negative force everywhere except the
       teacher's choice region) and the iterated field flips top-1, with the
       entropy transient coming back down;
    3. no false kills: ties are z-invariant and partial forks survive the
       re-leveling (no entropy squeeze), the cascade restores the fork;
    4. boundedness / degeneracy: no clamps needed, finite on junk evals,
       near-uniform teachers, masks.
"""

import torch

from verl.utils.detemper_reward import compute_rkl_cdt_scores

K = 16


def _support(head):
    head = torch.tensor(head, dtype=torch.float64)
    ntail = K - head.numel()
    w = torch.tensor([0.72**i for i in range(ntail)], dtype=torch.float64)
    tail = (1.0 - head.sum()) * w / w.sum()
    return torch.cat([head, tail]).log().float().view(1, 1, K)


def test_fog_nullified_two_sided():
    torch.manual_seed(0)
    s = torch.randn(8, 4, K) * 3
    mask = torch.ones(8, 4, K, dtype=torch.bool)
    temps = torch.rand(8, 4, 1) * 4.8 + 0.2  # includes T < 1: teacher SHARPER
    t = torch.log_softmax(torch.log_softmax(s, -1) / temps, -1)
    scores = compute_rkl_cdt_scores(s, t, mask)
    assert scores.abs().max().item() < 1e-4  # rkl_dt leaves ~0.9 on the T<1 half


def test_fixed_set_is_teacher_orbit():
    torch.manual_seed(1)
    t = torch.randn(4, 4, K) * 3
    mask = torch.ones(4, 4, K, dtype=torch.bool)
    mus = torch.rand(4, 4, 1) * 3.0 + 0.3
    s = torch.log_softmax(torch.log_softmax(t, -1) * mus, -1)  # p = q^mu / Z
    scores = compute_rkl_cdt_scores(s, t, mask)
    assert scores.abs().max().item() < 1e-4


def test_polar_identity():
    # r = -sigma_pw * p * (z_pw - z_qw) - c * p, c = log-partition gap,
    # with the moments taken under the student's own mass w = p
    torch.manual_seed(2)
    s_raw = torch.randn(4, 4, K) * 3
    t_raw = torch.randn(4, 4, K) * 3
    mask = torch.ones(4, 4, K, dtype=torch.bool)
    s = torch.log_softmax(s_raw, -1)
    t = torch.log_softmax(t_raw, -1)
    w = s.exp()

    def polar(lp):
        vc = lp - (w * lp).sum(-1, keepdim=True)
        sd = (w * vc.square()).sum(-1, keepdim=True).sqrt()
        return vc / sd, sd

    z_p, sigma_p = polar(s)
    z_q, _ = polar(t)
    t_cdt = torch.log_softmax(sigma_p * z_q, -1)
    c = (s - t_cdt - sigma_p * (z_p - z_q)).mean(-1, keepdim=True)
    rhs = -s.exp() * (sigma_p * (z_p - z_q) + c)
    lhs = compute_rkl_cdt_scores(s_raw, t_raw, mask).float()
    assert (lhs - rhs).abs().max().item() < 1e-4


def test_fog_with_junk_tails_nullified():
    # Real-support failure mode of the unweighted polar: the HEAD is a pure
    # temper of the student, but the deep massless tails are two models'
    # unrelated junk (real top-16 supports span 12-18 nats). The level must
    # be read under the student's mass, not the counting measure.
    torch.manual_seed(4)
    n_head, n_tail = 4, K - 4
    head_p = torch.tensor([0.62, 0.20, 0.10, 0.06]) * (1.0 - 1e-4)
    tail_p = torch.tensor([0.60**i for i in range(n_tail)])
    tail_p = tail_p / tail_p.sum() * 1e-4
    s = torch.cat([head_p, tail_p]).log().view(1, 1, K).repeat(8, 1, 1)
    s = s + torch.randn_like(s) * 0.05
    s = torch.log_softmax(s, -1)
    head_q = torch.log_softmax(s[..., :n_head] / 1.35, -1).exp() * (1.0 - 3e-4)
    junk = torch.rand(8, 1, n_tail) * 8.0 + 8.0  # teacher tail: -8..-16, unrelated
    tail_q = torch.softmax(-junk, -1) * 3e-4
    t = torch.cat([head_q, tail_q], -1).log()
    mask = torch.ones(8, 1, K, dtype=torch.bool)
    r_cdt = compute_rkl_cdt_scores(s, t, mask).abs().sum(-1).mean().item()
    sN = torch.log_softmax(s, -1)
    tN = torch.log_softmax(t, -1)
    r_base = (sN.exp() * (sN - tN)).abs().sum(-1).mean().item()
    assert r_cdt < 0.05, f"junk-tail fog force {r_cdt}"
    assert r_cdt < 0.12 * r_base, f"cdt {r_cdt} vs baseline {r_base}"


def test_confidently_wrong_funnel_and_flip():
    s = _support([0.94, 0.03])
    t = _support([0.10, 0.60]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    r = compute_rkl_cdt_scores(s, t, mask)
    assert r[0, 0, 0].item() < -1.0  # strong suppression of the wrong top-1
    assert r[0, 0, 1].item() > 0.0  # teacher's choice is promoted
    z = s.clone()
    entropies = []
    for _ in range(30):
        lp = torch.log_softmax(z, -1)
        entropies.append(-(lp.exp() * lp.clamp_min(-30)).sum().item())
        z = z + compute_rkl_cdt_scores(z, t, mask).float()
    p = torch.log_softmax(z, -1).exp()
    assert p[0, 0, 1].item() > p[0, 0, 0].item()  # top-1 flipped
    assert entropies[-1] < max(entropies) - 0.1  # transport transient, not level chase


def test_partial_fork_survives_releveling():
    s = _support([0.93, 0.02, 0.01])
    t = _support([0.52, 0.40, 0.01]).exp().squeeze().log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    z = s.clone()
    for _ in range(60):
        z = z + compute_rkl_cdt_scores(z, t, mask).float()
    p = torch.log_softmax(z, -1).exp()
    # entropy-matched dt would squeeze the fork cell to ~0.04; cdt restores it
    assert p[0, 0, 1].item() > 0.3
    assert p[0, 0, 0].item() > p[0, 0, 1].item()  # teacher's mild preference kept


def test_tie_is_invariant():
    s = _support([0.90, 0.04])
    q = _support([0.45, 0.45]).exp().squeeze()
    t = q.log().view(1, 1, K)
    mask = torch.ones(1, 1, K, dtype=torch.bool)
    z = s.clone()
    for _ in range(60):
        z = z + compute_rkl_cdt_scores(z, t, mask).float()
    p = torch.log_softmax(z, -1).exp()
    assert abs(p[0, 0, 0].item() - p[0, 0, 1].item()) < 0.05  # the 50/50 fork is reproduced


def test_bounded_finite_and_mask_respecting():
    torch.manual_seed(3)
    s = torch.randn(8, 8, K) * 4
    t = torch.randn(8, 8, K) * 4
    t[..., -1] = -35.0  # junk teacher eval
    mask = torch.rand(8, 8, K) > 0.4
    mask[..., :2] = True
    r = compute_rkl_cdt_scores(s, t, mask)
    assert torch.isfinite(r).all()
    assert r[~mask].abs().max().item() == 0.0
    # near-uniform teacher: the implicit lambda = sigma_p/sigma_q explodes,
    # but sigma_p * z_q stays bounded - no clamp needed
    r_u = compute_rkl_cdt_scores(_support([0.9, 0.05]), torch.randn(1, 1, K) * 0.01, torch.ones(1, 1, K, dtype=torch.bool))
    assert torch.isfinite(r_u).all()
    r_d = compute_rkl_cdt_scores(torch.zeros(1, 1, K), torch.randn(1, 1, K), torch.ones(1, 1, K, dtype=torch.bool))
    assert torch.isfinite(r_d).all()
