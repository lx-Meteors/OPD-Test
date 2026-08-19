"""Token-level gallery: where does the rkl_cdt force land on real val tokens?

Data: Prune-OPD/analysis_probe/cells_cache.pt - 12 real step-0 val
trajectories (4 correct / 4 wrong / 4 unfinished), top-16 cells WITH token
ids, student = DeepSeek-R1-Distill-Qwen-1.5B, teacher = JustRL-DeepSeek-1.5B.

For each lesion class we decode concrete positions - context tail, the cell
tokens, p/q/q~ and the per-cell force under baseline / rkl_dt / rkl_cdt -
plus a per-trajectory rollup.
"""

import importlib.util
import sys
from pathlib import Path

import torch

OLD = Path("/input0/yyy/Prune-OPD")
sys.path.insert(0, str(OLD / "analysis_probe"))
spec = importlib.util.spec_from_file_location(
    "dtr", "/input0/yyy/Prune-OPD-new/verl/verl/utils/detemper_reward.py")
dtr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtr)
torch.set_grad_enabled(False)

from transformers import AutoTokenizer  # noqa: E402

from quotient_kl_probe import STUDENT, load_samples  # noqa: E402

d0 = torch.load(OLD / "analysis_probe/cells_cache.pt", map_location="cpu", weights_only=False)
S = torch.log_softmax(d0["LPS"].float(), -1)
T = torch.log_softmax(d0["LPT"].float(), -1)
CID = d0["CID"]
N = S.shape[0]
tok = AutoTokenizer.from_pretrained(STUDENT)
samples = load_samples(tok)
lens = [len(s["ids_out"]) for s in samples]
assert sum(lens) == N, f"provenance mismatch {sum(lens)} vs {N}"
seq_idx = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
pos_idx = torch.cat([torch.arange(length) for length in lens])
print(f"provenance OK: {len(samples)} trajectories, {N} positions")

p, q = S.exp(), T.exp()


def ent(lp):
    return -(lp.exp() * lp.clamp_min(-30)).sum(-1)


def zw(lp, w):
    u = lp - (w * lp).sum(-1, keepdim=True)
    return u / (w * u.square()).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)


Hp, Hq = ent(S), ent(T)
agree = S.argmax(-1) == T.argmax(-1)
fs = agree & (Hq - Hp > 0.5)
cw = (~agree) & (p.max(-1).values > 0.9)
zres = ((zw(S, p) - zw(T, p)).square() * p).sum(-1).sqrt()
fog = fs & (zres < 0.20)

mask_all = torch.ones_like(S, dtype=torch.bool)


def field(fn, s, t):
    return fn(s.unsqueeze(0), t.unsqueeze(0), torch.ones_like(s, dtype=torch.bool).unsqueeze(0)).squeeze(0).float()


def base_fn(s, t, m):
    sn = torch.log_softmax(torch.where(m, s, torch.full_like(s, -1e30)), -1)
    tn = torch.log_softmax(torch.where(m, t, torch.full_like(t, -1e30)), -1)
    return -sn.exp() * (sn - tn)


R = {
    "base": field(base_fn, S, T),
    "dt": field(dtr.compute_rkl_dt_scores, S, T),
    "cdt": field(dtr.compute_rkl_cdt_scores, S, T),
    "fkl": field(dtr.compute_fkl_scores, S, T),
}

# cdt's re-leveled target for display
w = p
sig_p = (w * (S - (w * S).sum(-1, keepdim=True)).square()).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
QT = torch.log_softmax(sig_p * zw(T, w), -1)

kl_rows = lambda a, b: (a.exp() * (a - b)).sum(-1)


def flip20(idx, fn):
    cur, Tm = S[idx], T[idx]
    qtop = Tm.argmax(-1)
    for _ in range(20):
        r = field(fn, cur, Tm)
        lo, hi = 0.0, 1.0
        while kl_rows(torch.log_softmax(cur + hi * r, -1), cur).mean() < 2e-3:
            hi *= 2
            if hi > 1e6:
                break
        for _ in range(40):
            mid = (lo + hi) / 2
            if kl_rows(torch.log_softmax(cur + mid * r, -1), cur).mean() < 2e-3:
                lo = mid
            else:
                hi = mid
        cur = torch.log_softmax(cur + 0.5 * (lo + hi) * r, -1)
    return cur.argmax(-1) == qtop


def show(t, title, cells=None):
    i, j = int(seq_idx[t]), int(pos_idx[t])
    s = samples[i]
    ctx = tok.decode((s["ids_in"] + s["ids_out"])[: len(s["ids_in"]) + j][-40:])[-110:].replace("\n", "\\n")
    print(f"\n--- {title} | 轨迹{i}({s['group']}) pos{j} | H(p)={Hp[t]:.2f} H(q)={Hq[t]:.2f} zres={zres[t]:.2f}")
    print(f"  ctx: ...{ctx!r}")
    print(f"  {'token':<18}{'p':>7}{'q':>7}{'q~cdt':>7} | {'r_base':>8}{'r_dt':>8}{'r_cdt':>8}{'r_fkl':>8}")
    order = S[t].argsort(descending=True)
    shown = cells if cells is not None else order[:4].tolist() + [int(T[t].argmax())]
    seen = set()
    for c in shown:
        if c in seen:
            continue
        seen.add(c)
        mark = " <-t.top" if c == int(T[t].argmax()) else (" <-s.top" if c == int(S[t].argmax()) else "")
        print(f"  {tok.decode([int(CID[t, c])])!r:<18}{p[t, c]:>7.3f}{q[t, c]:>7.3f}{QT[t, c].exp():>7.3f} | "
              f"{R['base'][t, c]:>+8.3f}{R['dt'][t, c]:>+8.3f}{R['cdt'][t, c]:>+8.3f}{R['fkl'][t, c]:>+8.3f}{mark}")


print("\n" + "=" * 100)
print("[G1] CW 画廊: 自信错位置 (p_max>0.9, top-1 不一致) - 漏斗是否指向正确 token")
print("=" * 100)
cw_idx = cw.nonzero().flatten()
f20 = {nm: flip20(cw_idx, fn) for nm, fn in
       (("base", base_fn), ("dt", dtr.compute_rkl_dt_scores), ("cdt", dtr.compute_rkl_cdt_scores))}
print(f"CW 总数 {len(cw_idx)} | 等预算20步修复: base {int(f20['base'].sum())} dt {int(f20['dt'].sum())} cdt {int(f20['cdt'].sum())}")
qconf = q.gather(-1, T.argmax(-1, keepdim=True)).squeeze(-1)
pick = cw_idx[qconf[cw_idx].argsort(descending=True)[:4]]
for t in pick.tolist():
    show(t, "CW")

print("\n" + "=" * 100)
print("[G2] FS 内容分歧画廊: teacher 在具体替代 token 上存质量 (0.15-0.35) - 力是否指向那里")
print("=" * 100)
alt_q = q.clone()
alt_q.scatter_(1, S.argmax(-1, keepdim=True), 0.0)
amax = alt_q.max(-1).values
content = fs & (~fog) & (amax > 0.15) & (amax < 0.35)
c_idx = content.nonzero().flatten()
pick = c_idx[amax[c_idx].argsort(descending=True)[:3]]
for t in pick.tolist():
    cells = S[t].argsort(descending=True)[:3].tolist() + [int(alt_q[t].argmax())]
    show(t, "FS-content", cells)

print("\n" + "=" * 100)
print("[G2b] 语义/记号分叉画廊: teacher 近平分 (alt>0.4) - dt 误杀 vs cdt 保留")
print("=" * 100)
forkm = fs & (amax > 0.40)
f_idx = forkm.nonzero().flatten()
pick = f_idx[amax[f_idx].argsort(descending=True)[:3]]
for t in pick.tolist():
    cells = S[t].argsort(descending=True)[:3].tolist() + [int(alt_q[t].argmax())]
    show(t, "fork", cells)

print("\n" + "=" * 100)
print("[G3] 纯雾画廊: 结构接近匹配 (zres 最小的 FS) 只差温度 - cdt 力应趋 0, baseline 仍在推软")
print("=" * 100)
fs_idx = fs.nonzero().flatten()
pick = fs_idx[zres[fs_idx].argsort()[:3]]
for t in pick.tolist():
    show(t, "fog")
if fog.any():
    rb = R["base"][fog].abs().sum(-1)
    rc = R["cdt"][fog].abs().sum(-1)
    print(f"\n雾类(zres<0.2)合计 n={int(fog.sum())}: sum|r| base {rb.mean():.3f} / cdt {rc.mean():.3f}"
          f"  (x{(rb.mean() / rc.mean().clamp_min(1e-9)):.1f})")

print("\n" + "=" * 100)
print("[G4] 每条轨迹 rollup")
print("=" * 100)
print(f" {'轨迹':<16}{'n':>6}{'H(p)':>6}{'FS/1k':>7}{'fog/1k':>7}{'CW/1k':>6} | 全力: {'base':>6}{'cdt':>6} | fog力: {'base':>6}{'cdt':>6}")
off = 0
for i, s in enumerate(samples):
    sl = slice(off, off + lens[i])
    off += lens[i]
    m_fs, m_fog, m_cw = fs[sl], fog[sl], cw[sl]
    fb, fc = R["base"][sl].abs().sum(-1), R["cdt"][sl].abs().sum(-1)
    fogb = fb[m_fog].mean().item() if m_fog.any() else float("nan")
    fogc = fc[m_fog].mean().item() if m_fog.any() else float("nan")
    print(f" {i}:{s['group']:<13}{lens[i]:>6}{Hp[sl].mean():>6.2f}{1000 * m_fs.float().mean():>7.1f}"
          f"{1000 * m_fog.float().mean():>7.1f}{1000 * m_cw.float().mean():>6.2f} | "
          f"{fb.mean():>6.3f}{fc.mean():>6.3f} | {fogb:>6.3f}{fogc:>6.3f}")
