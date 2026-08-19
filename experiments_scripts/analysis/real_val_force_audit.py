"""Real-trajectory force audit of rkl_cdt vs baseline / rkl_dt.

Data (all pre-existing caches from the lesion dissection, no GPU needed):
  - Prune-OPD/analysis_probe/val_force_cells_BC.pt : top-16 cells (student p,
    teacher q) on REAL baseline-run val trajectories, steps 0..200 (testbed B)
    and rkl_sdt collapse-regime trajectories (testbed C).
  - Prune-OPD/analysis_probe/val_force_cells_A.pt  : l-apd@100 evolved student
    on its own step-100 val dump (testbed A, most on-policy-like).
  - Prune-OPD/analysis_probe/cells_cache.pt        : 12 step-0 val trajectories
    WITH cell token ids -> token-level gallery with decoded text.

Position taxonomy = the updated lesion definitions:
  FS  : top-1 agree & H(q) - H(p) > 0.5        (false-softness candidates)
  fog : FS & RMS(z_p - z_q) small               (level gap only, no structure)
  CW  : p_max > 0.9 & top-1 disagree            (confident-wrong)

Per arm (baseline / rkl_dt / rkl_cdt, all from the shipped module) we measure
where the force actually lands on real tokens, at equal update budgets.
"""

import importlib.util
import sys
from pathlib import Path

import torch

OLD = Path("/input0/yyy/Prune-OPD")
NEW = Path("/input0/yyy/Prune-OPD-new")
spec = importlib.util.spec_from_file_location(
    "dtr", NEW / "verl/verl/utils/detemper_reward.py")
dtr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtr)

torch.set_grad_enabled(False)


def base_scores(s, t, m):
    return -s.exp() * (s - t)


ARMS = {
    "baseline": base_scores,
    "rkl_dt": dtr.compute_rkl_dt_scores,
    "rkl_cdt": dtr.compute_rkl_cdt_scores,
    "fkl": dtr.compute_fkl_scores,
}


def ent(lp):
    return -(lp.exp() * lp.clamp_min(-30)).sum(-1)


def zscore(lp, w):
    """Polar structure in the student-mass (Fisher) metric - matches the module."""
    u = lp - (w * lp).sum(-1, keepdim=True)
    sd = (w * u.square()).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return u / sd


def norm(lp):
    return torch.log_softmax(lp.float(), -1)


def kl_rows(la, lb):
    return (la.exp() * (la - lb)).sum(-1)


def calibrate(lps, r, target):
    lo, hi = 0.0, 1.0
    while kl_rows(torch.log_softmax(lps + hi * r, -1), lps).mean() < target:
        hi *= 2
        if hi > 1e6:
            break
    for _ in range(50):
        mid = (lo + hi) / 2
        if kl_rows(torch.log_softmax(lps + mid * r, -1), lps).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def taxonomy(S, T):
    Hp, Hq = ent(S), ent(T)
    agree = S.argmax(-1) == T.argmax(-1)
    fs = agree & (Hq - Hp > 0.5)
    cw = (~agree) & (S.exp().max(-1).values > 0.9)
    w = S.exp()
    zres = ((zscore(S, w) - zscore(T, w)).square() * w).sum(-1).sqrt()
    fog = fs & (zres < 0.35)
    return dict(Hp=Hp, Hq=Hq, agree=agree, fs=fs, cw=cw, fog=fog, zres=zres)


def arm_fields(S, T):
    m = torch.ones_like(S, dtype=torch.bool)
    return {nm: f(S.unsqueeze(0), T.unsqueeze(0), m.unsqueeze(0)).squeeze(0).float()
            for nm, f in ARMS.items()}


def class_shares(r, tax):
    tot = r.abs().sum()
    out = {}
    for cname in ("fs", "cw"):
        out[cname] = (r[tax[cname]].abs().sum() / tot * 100).item()
    out["fog"] = (r[tax["fog"]].abs().sum() / tot * 100).item()
    return out


def fs_split(r, S, T, tax):
    """On structured FS positions: force banked on the teacher's alternative
    cells vs pure softening of the student's top-1."""
    m = tax["fs"] & ~tax["fog"]
    if m.sum() == 0:
        return dict(alt=0.0, top=0.0, n=0)
    p, q = S.exp(), T.exp()
    stop = S.argmax(-1)
    alt_cells = (q > p + 0.02)
    alt_cells.scatter_(1, stop.unsqueeze(-1), False)
    alt = (r * alt_cells)[m].sum(-1).mean().item()
    top = r[m].gather(-1, stop[m].unsqueeze(-1)).mean().item()
    return dict(alt=alt, top=top, n=int(m.sum()))


def cw_metrics(r, S, T, tax, arm_fn):
    m = tax["cw"]
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    qtop = T.argmax(-1)
    share = (r[m].gather(-1, qtop[m].unsqueeze(-1)).abs().sum() / r[m].abs().sum() * 100).item()
    nontgt = r[m].clone()
    nontgt.scatter_(1, qtop[m].unsqueeze(-1), 0.0)
    funnel = ((nontgt <= 1e-6).float().sum(-1) / 15.0).mean().item() * 100
    cur, Tm = S[m], T[m]
    mask3 = torch.ones_like(cur, dtype=torch.bool).unsqueeze(0)
    flips = {}
    for stepi in range(20):
        rr = arm_fn(cur.unsqueeze(0), Tm.unsqueeze(0), mask3).squeeze(0).float()
        eta = calibrate(cur, rr, 2e-3)
        cur = torch.log_softmax(cur + eta * rr, -1)
        if stepi + 1 in (1, 20):
            flips[stepi + 1] = (cur.argmax(-1) == qtop[m]).float().mean().item() * 100
    return dict(n=n, share=share, funnel=funnel, f1=flips[1], f20=flips[20])


def collateral(S, T, tax, arm_fn):
    """agree & sharp positions: does 20-step evolution create new CWs?"""
    m = tax["agree"] & (S.exp().max(-1).values > 0.9)
    if m.sum() == 0:
        return 0.0
    idx = m.nonzero().flatten()
    if len(idx) > 4000:
        idx = idx[torch.randperm(len(idx))[:4000]]
    cur, Tm = S[idx], T[idx]
    qtop = Tm.argmax(-1)
    mask3 = torch.ones_like(cur, dtype=torch.bool).unsqueeze(0)
    for _ in range(20):
        rr = arm_fn(cur.unsqueeze(0), Tm.unsqueeze(0), mask3).squeeze(0).float()
        eta = calibrate(cur, rr, 2e-3)
        cur = torch.log_softmax(cur + eta * rr, -1)
    p = cur.exp()
    return ((p.max(-1).values > 0.9) & (cur.argmax(-1) != qtop)).float().mean().item() * 100


def fog_dh(S, T, tax, arm_fn):
    m = tax["fog"]
    if m.sum() == 0:
        return 0.0, 0.0
    mask3 = torch.ones_like(S[m], dtype=torch.bool).unsqueeze(0)
    rr = arm_fn(S[m].unsqueeze(0), T[m].unsqueeze(0), mask3).squeeze(0).float()
    frc = rr.abs().sum(-1).mean().item()
    eta = calibrate(S[m], rr, 2e-3) if frc > 1e-8 else 0.0
    dh = (ent(torch.log_softmax(S[m] + eta * rr, -1)) - ent(S[m])).mean().item()
    return frc, dh


# ---------------------------------------------------------------- load caches
recs = torch.load(OLD / "analysis_probe/val_force_cells_BC.pt", map_location="cpu", weights_only=False)
recs += torch.load(OLD / "analysis_probe/val_force_cells_A.pt", map_location="cpu", weights_only=False)
for r in recs:
    r["S"] = norm(r["LPS"])
    r["T"] = norm(r["LPT"])

print("=" * 104)
print("[0] 测床校验: 真实 val 轨迹上的病灶统计 (baseline 力场) vs 你解剖出的数字")
print("=" * 104)
pool_by = {}
for r in recs:
    pool_by.setdefault(r["testbed"], []).append(r)
for tb, rs in pool_by.items():
    S = torch.cat([r["S"] for r in rs])
    T = torch.cat([r["T"] for r in rs])
    tax = taxonomy(S, T)
    rb = arm_fields(S, T)["baseline"]
    cwsh = (rb[tax["cw"]].gather(-1, T.argmax(-1)[tax["cw"]].unsqueeze(-1)).abs().sum()
            / rb.abs().sum() * 100).item() if tax["cw"].sum() else 0.0
    print(f" [{tb}] N={S.shape[0]:6d} | H(p)={ent(S).mean():.3f} H(q)={ent(T).mean():.3f} | "
          f"FS {1000 * tax['fs'].float().mean():.1f}/1k (fog {1000 * tax['fog'].float().mean():.1f}) | "
          f"CW {1000 * tax['cw'].float().mean():.2f}/1k | baseline CW修正格预算份额 {cwsh:.2f}%")
w_fs = [1000 * taxonomy(r["S"], r["T"])["fs"].float().mean().item() for r in recs if r["testbed"] == "B" and r["score"] == 0]
c_fs = [1000 * taxonomy(r["S"], r["T"])["fs"].float().mean().item() for r in recs if r["testbed"] == "B" and r["score"] == 1]
if w_fs and c_fs:
    print(f" B 错误轨迹 FS 密度 / 正确轨迹 = {sum(w_fs) / len(w_fs):.1f} / {sum(c_fs) / len(c_fs):.1f} "
          f"= {sum(w_fs) / len(w_fs) / max(sum(c_fs) / len(c_fs), 1e-9):.2f}x  (病灶: 1.7x)")

print()
print("=" * 104)
print("[1] 力都花在哪了: 全预算类别份额 + 各病灶类的力度  (三臂, 等预算, 按测床)")
print("=" * 104)
for tb, rs in pool_by.items():
    S = torch.cat([r["S"] for r in rs])
    T = torch.cat([r["T"] for r in rs])
    tax = taxonomy(S, T)
    print(f" --- 测床 {tb} ---")
    for nm, fn in ARMS.items():
        r = arm_fields(S, T)[nm]
        sh = class_shares(r, tax)
        fsx = fs_split(r, S, T, tax)
        fogf, fogd = fog_dh(S, T, tax, fn)
        cwm = cw_metrics(r, S, T, tax, fn)
        col = collateral(S, T, tax, fn)
        cw_str = (f"CW: 份额{cwm['share']:5.1f}% 漏斗{cwm['funnel']:4.0f}% flip@1={cwm['f1']:4.1f}% "
                  f"@20={cwm['f20']:5.1f}%" if cwm["n"] else "CW: n=0")
        print(f"  [{nm:8s}] 预算份额 FS={sh['fs']:4.1f}%(fog {sh['fog']:4.2f}%) CW={sh['cw']:4.2f}% | "
              f"fog力={fogf:6.4f} dH={fogd:+.4f} | FS结构: alt={fsx['alt']:+.4f} top1={fsx['top']:+.4f} | "
              f"{cw_str} | 新CW@20={col:.2f}%")
print(" 判据: fog 力->0; FS 里 top-1 软化(负值)缩水而 alt 保持; CW 漏斗高、flip 快; 新 CW = 0")

print()
print("=" * 104)
print("[2] 具体轨迹逐条看 (B=baseline 训练各步 / C=rkl_sdt 崩溃段 / A=进化后学生@step100)")
print("=" * 104)
picks = []
for r in recs:
    key = (r["testbed"], r["step"], int(r["score"]))
    if key in ((("B", 0, 0)), ("B", 0, 1), ("B", 100, 0), ("B", 100, 1), ("B", 200, 0), ("B", 200, 1),
               ("C", 200, 0), ("C", 200, 1), ("A", 100, 0), ("A", 100, 1)):
        if key not in [p[0] for p in picks]:
            picks.append((key, r))
print(f" {'轨迹':24s} {'H(p)':>5} {'H(q)':>5} {'FS/1k':>6} {'CW/1k':>6} | force: {'base':>6} {'cdt':>6} | "
      f"fogF: {'base':>6} {'cdt':>6} | CW修复@20: {'base':>5} {'cdt':>5}")
for key, r in picks:
    S, T = r["S"], r["T"]
    tax = taxonomy(S, T)
    fields = arm_fields(S, T)
    fb, fc = fields["baseline"], fields["rkl_cdt"]
    fogb, _ = fog_dh(S, T, tax, ARMS["baseline"])
    fogc, _ = fog_dh(S, T, tax, ARMS["rkl_cdt"])
    cwb = cw_metrics(fb, S, T, tax, ARMS["baseline"])
    cwc = cw_metrics(fc, S, T, tax, ARMS["rkl_cdt"])
    tag = f"{key[0]} step{key[1]} {'对' if key[2] else '错'} n={S.shape[0]}"
    f20b = f"{cwb['f20']:5.1f}" if cwb["n"] else "  n/a"
    f20c = f"{cwc['f20']:5.1f}" if cwc["n"] else "  n/a"
    print(f" {tag:24s} {ent(S).mean():5.2f} {ent(T).mean():5.2f} {1000 * tax['fs'].float().mean():6.1f} "
          f"{1000 * tax['cw'].float().mean():6.2f} | {fb.abs().sum(-1).mean():6.3f} {fc.abs().sum(-1).mean():6.3f} | "
          f"{fogb:6.4f} {fogc:6.4f} | {f20b} {f20c}")
print(" force = 全 token 平均每位置力预算; fogF = fog 位置上的力 (有效性: cdt 应显著低于 base 且 CW 修复不慢)")
