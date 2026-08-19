"""Lesion-level simulation: does rkl_cdt resolve P1' (softness chasing +
structure under-transmission) and P2 (confident-wrong, zero net repair)?

Populations are built to match the token-level anatomy from the trajectory
dissection: FS composition 73% content-divergence / 9.5% semantic-fork /
13.3% thin-tail fog / 4.6% tokenizer-variant, step-0 entropy gap ~ +0.175,
CW anatomy p_max~0.94, correction target = student's rank-2, teacher
confidence 0.55-0.70. The learning rate is calibrated so the BASELINE
reproduces the observed absorption speed (0.175 -> ~0 in 20 steps), then the
same rate is used for every arm.

Blocks:
  [1] P1'(a): fog force, entropy pinning, and the echo PROBE - after the
      student converges, the teacher is moved along its temperature fiber
      (retreat tau>1 / advance tau<1); if the force reignites, the
      level-chasing loop is closed; if not, the echo channel is severed.
  [2] P1'(b): step-1 force share on the designated alternative cell and the
      terminal mass it receives, per FS class.
  [3] P2 x shared budget: FS teachers retreat continuously (context
      degradation on wrong trajectories); a density-weighted global budget
      couples FS and CW. Metric: does the FS channel quench (freeing budget
      for repair) and do CW positions flip in bounded steps?
  [4] sdt warning: correct-but-soft control with a SHARPER teacher - no
      downward level chase, no new CW creation.

Run: python experiments_scripts/analysis/lesion_sim_rkl_cdt.py  (CPU, ~30 s)
"""
import importlib.util
import os

import torch

torch.manual_seed(0)
_here = os.path.dirname(os.path.abspath(__file__))
_mod = os.path.join(_here, "..", "..", "verl", "verl", "utils", "detemper_reward.py")
spec = importlib.util.spec_from_file_location("dtr", os.path.normpath(_mod))
dtr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtr)
cdt, dt = dtr.compute_rkl_cdt_scores, dtr.compute_rkl_dt_scores
NEG = torch.finfo(torch.float32).min
K = 16


def base(s_raw, t_raw, mask):
    s = torch.log_softmax(torch.where(mask, s_raw.float(), torch.full_like(s_raw, NEG)), -1)
    t = torch.log_softmax(torch.where(mask, t_raw.float(), torch.full_like(t_raw, NEG)), -1)
    return -s.exp() * (s - t)


ARMS = {"baseline": base, "rkl_dt": dt, "rkl_cdt": cdt}


def mk(head_list):
    n = head_list[0][1].shape[0]
    p = torch.zeros(n, K, dtype=torch.float64)
    used = torch.zeros(n, dtype=torch.float64)
    cells = []
    for c, v in head_list:
        p[:, c] = v
        used += v
        cells.append(c)
    tail = [c for c in range(K) if c not in cells]
    w = torch.tensor([0.72**i for i in range(len(tail))], dtype=torch.float64)
    p[:, tail] = (1.0 - used).unsqueeze(-1) * (w / w.sum()).unsqueeze(0)
    return p.clamp_min(1e-12).log().float().unsqueeze(1)


def H(logits):
    lp = torch.log_softmax(logits, -1)
    return -(lp.exp() * lp.clamp_min(-30)).sum(-1)


U = lambda n, a, b: torch.rand(n, dtype=torch.float64) * (b - a) + a
noise = lambda x, s=0.05: x + torch.randn_like(x) * s
ALT = 5

Nc, Nf, Nt, Nv = 1460, 190, 266, 92  # 73% / 9.5% / 13.3% / 4.6%
s_c = mk([(0, U(Nc, .58, .74)), (ALT, U(Nc, .02, .05))])
q_c = noise(torch.log_softmax(mk([(0, U(Nc, .52, .64)), (ALT, U(Nc, .15, .25))]) / (torch.rand(Nc, 1, 1) * .06 + 1.02), -1))
s_f = mk([(0, U(Nf, .80, .90)), (ALT, U(Nf, .02, .05))])
q_f = noise(mk([(0, U(Nf, .45, .55)), (ALT, U(Nf, .35, .45))]))
s_t = mk([(0, U(Nt, .55, .75))])
q_t = noise(torch.log_softmax(s_t / (torch.rand(Nt, 1, 1) * .10 + 1.08), -1))
s_v = mk([(0, U(Nv, .55, .75)), (1, U(Nv, .02, .04))])
q_v = noise(mk([(0, U(Nv, .32, .42)), (1, U(Nv, .26, .36))]))
s_fs = torch.cat([s_c, s_f, s_t, s_v])
q_fs = torch.cat([q_c, q_f, q_t, q_v])
Nfs = s_fs.shape[0]
mfs = torch.ones_like(s_fs, dtype=torch.bool)
Hs0, Hq0 = H(s_fs).mean().item(), H(q_fs).mean().item()
print(f"FS 群体 N={Nfs} (73/9.5/13.3/4.6%): H_p={Hs0:.3f} H_q={Hq0:.3f} gap={Hq0 - Hs0:+.3f} (病灶 +0.175)")

best = None
for eta in (.04, .07, .10, .14, .20):
    z = s_fs.clone()
    for _ in range(20):
        z = z + eta * base(z, q_fs, mfs)
    cl = (H(z).mean().item() - Hs0) / (Hq0 - Hs0)
    if best is None or abs(cl - .97) < abs(best[1] - .97):
        best = (eta, cl)
ETA = best[0]
print(f"校准 eta={ETA} (baseline 20 步吸收 {best[1] * 100:.0f}% 熵差; 病灶实测 0.175->0.006)\n")

print("=" * 102)
print("[1] P1'(a) 追平与回声 — 静态收敛终点 + 回声探针 (teacher 沿温度纤维退让/前进, 力是否重燃)")
print("=" * 102)
exact_fog = torch.log_softmax(s_t / 1.25, -1)
conv = {}
for nm, f in ARMS.items():
    z = s_fs.clone()
    dh_fog = []
    for step in range(300):
        r = f(z, q_fs, mfs).float()
        if step < 20:
            fog = slice(Nc + Nf, Nc + Nf + Nt)
            dh_fog.append((H((z + ETA * r)[fog]).mean() - H(z[fog]).mean()).item())
        z = z + ETA * r
    conv[nm] = z
    fog_x = f(s_t, exact_fog, torch.ones_like(s_t, dtype=torch.bool)).abs().sum(-1).mean().item()
    print(f" [{nm:8s}] 终态 |H_s-H_q|={abs(H(z).mean().item() - Hq0):.3f}  雾类前20步 dH/步={sum(dh_fog) / 20:+.4f}  纯温度雾 |r|={fog_x:.1e}")
print("   -> 判据: 雾类 dH>0 即在追雾; |H_s-H_q|~0 即水平被钉死")
print("\n 回声探针: 学生收敛后, teacher 退让 (tau=1.15) 或前进 (tau=0.87):")
for nm, f in ARMS.items():
    z = conv[nm]
    r0 = f(z, q_fs, mfs).float().abs().sum(-1).mean().item()
    out = f" [{nm:8s}] 收敛残力={r0:.4f}"
    for tag, tau in (("退让", 1.15), ("前进", 0.87)):
        qq = torch.log_softmax(q_fs / tau, -1)
        rr = f(z, qq, mfs).float()
        zz = z.clone()
        for _ in range(15):
            zz = zz + ETA * f(zz, qq, mfs).float()
        out += f" | {tag}: 重燃|r|={rr.abs().sum(-1).mean():.4f}, 15步dH={H(zz).mean() - H(z).mean():+.3f}"
    print(out)
print("   -> 判据: 重燃|r|=残力且 dH~0 = 回声通道被切断; baseline 双向追, rkl_dt 只挡退让侧 (单侧门)")

print()
print("=" * 102)
print("[2] P1'(b) 结构传输 — 指定替代格的力份额与终态到位率")
print("=" * 102)
for pname, (sp, tp, cell) in {"content(73%)": (s_c, q_c, ALT), "fork(9.5%)": (s_f, q_f, ALT), "variant(4.6%)": (s_v, q_v, 1)}.items():
    m_ = torch.ones_like(sp, dtype=torch.bool)
    qsh = torch.log_softmax(tp, -1).exp()[..., cell].mean().item()
    line = f" {pname:13s} [teacher 份额 {qsh:.3f}]"
    for nm, f in ARMS.items():
        r = f(sp, tp, m_).float()
        share = (r[..., cell].abs().sum() / r.abs().sum()).item() * 100
        z = sp.clone()
        for _ in range(300):
            z = z + ETA * f(z, tp, m_).float()
        pe = torch.log_softmax(z, -1).exp()[..., cell].mean().item()
        line += f" | {nm} 份额@1={share:4.1f}% 终态p={pe:.3f}"
    print(line)
print("   -> 判据: 结构格要拿到力并最终到位 (病灶: '学会该软, 没学会往哪软'); 注意 rkl_dt 在 fork 的份额塌缩")

print()
print("=" * 102)
print("[3] P2 自信错 x 共享预算 — FS teacher 沿纤维持续退让 (无上限), 密度加权全局预算")
print("=" * 102)
Ncw = 400
s_cw = mk([(0, U(Ncw, .92, .96)), (1, U(Ncw, .02, .04))])
q_cw = noise(mk([(0, U(Ncw, .10, .20)), (1, U(Ncw, .55, .70))]))
w = torch.cat([torch.ones(Nfs), torch.full((Ncw,), 0.02)])  # CW 稀疏 (~0.65/1k), FS 密集
for nm, f in ARMS.items():
    r1 = f(s_cw, q_cw, torch.ones_like(s_cw, dtype=torch.bool)).float()
    sh1 = (r1[..., 1].abs().sum() / r1.abs().sum()).item() * 100
    z = torch.cat([s_fs, s_cw])
    mall = torch.ones_like(z, dtype=torch.bool)
    B0 = None
    flipped = torch.full((Ncw,), 999)
    cum_fs = 0.0
    fs_at = {}
    for step in range(200):
        tau = 1.0 + 0.004 * step
        qt = torch.cat([torch.log_softmax(q_fs / tau, -1), q_cw])
        r = f(z, qt, mall).float()
        fs_now = r[:Nfs].abs().sum(-1).mean().item()
        cum_fs += fs_now
        if step + 1 in (1, 50, 100, 200):
            fs_at[step + 1] = fs_now
        tot = ((w * r.abs().sum(-1).squeeze(1)).sum() / w.sum()).item()
        if B0 is None:
            B0 = tot
        z = z + (ETA * B0 / max(tot, 1e-9)) * r
        p = torch.log_softmax(z[Nfs:], -1).exp().squeeze(1)
        newly = (p[:, 1] > p[:, 0]) & (flipped == 999)
        flipped[newly] = step + 1
    fl = lambda t: (flipped <= t).float().mean().item() * 100
    print(f" [{nm:8s}] 单步corr份额={sh1:4.1f}% flip@1=0.0% | FS均力 @1={fs_at[1]:.3f} @50={fs_at[50]:.3f} @100={fs_at[100]:.3f} @200={fs_at[200]:.3f}"
          f" | FS累计燃烧={cum_fs:5.1f} | CW修复@100={fl(100):.0f}% (中位 {flipped.float().median().item():.0f} 步)")
print("   -> 判据: FS 力要熄火 (不追退让); FS 全程燃烧越低, 共享梯度里留给 CW 修复的预算越多")

print()
print("=" * 102)
print("[4] 反例警示 (sdt 过度锐化造新 CW) — 学生对但偏软, teacher 更尖")
print("=" * 102)
Ncs = 800
s_cs = mk([(0, U(Ncs, .45, .65))])
q_cs = noise(torch.log_softmax(s_cs * (torch.rand(Ncs, 1, 1) * .6 + 1.5), -1))
m_ = torch.ones_like(s_cs, dtype=torch.bool)
tt1 = torch.log_softmax(q_cs, -1).argmax(-1).squeeze(1)
for nm, f in ARMS.items():
    z = s_cs.clone()
    for _ in range(50):
        z = z + ETA * f(z, q_cs, m_).float()
    p = torch.log_softmax(z, -1).exp().squeeze(1)
    ncw = ((p.max(-1).values > .9) & (p.argmax(-1) != tt1)).float().mean().item() * 100
    print(f" [{nm:8s}] H {H(s_cs).mean():.3f} -> {H(z).mean():.3f} (teacher {H(q_cs).mean():.3f}) | 新增CW {ncw:.2f}% | top-1 保持 {(p.argmax(-1) == tt1).float().mean() * 100:.0f}%")
print("   -> 判据: 不向下追水平 (H 不塌向 teacher), 零新增 CW  [sdt 实录: CW 密度 1.0->2.0/1k]")
