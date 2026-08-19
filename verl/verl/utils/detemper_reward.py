from __future__ import annotations

import torch

_LAM_MAX = 64.0
_BISECT_ITERS = 50
_EPS = 1e-6


def _cell_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dim; -inf cells contribute exactly zero."""
    return -(log_probs.exp() * log_probs.clamp_min(-30.0)).sum(-1)


def compute_rkl_cdt_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Baseline reverse-KL rewards against a contrast-detempered teacher (cdt).

    Same force form as rkl_dt - the baseline's own r = -p * (log p - log q~) -
    with the entropy bisection replaced by a closed form. Write the log probs
    in polar form (level x structure) UNDER THE STUDENT'S OWN MASS, i.e. the
    Fisher / L2(p) metric on the cells:

        mean_w(v)  = sum_c w_c v_c,          w = p on the valid cells
        sigma_w(v) = sqrt(mean_w((v - mean_w(v))^2))    (level)
        z_w(v)     = (v - mean_w(v)) / sigma_w(v)       (structure)

        log q~ = log_softmax(sigma_w(log p) * z_w(log q))
        r      = -p * (log p - log q~)

    q~ is identically q^(sigma_pw/sigma_qw) / Z - the teacher's structure at
    the student's contrast level - without ever forming the exponent.

    Why weighted moments (real-val audit, 2026-08): on real top-16 supports
    log p spans ~12-18 nats and dead-tail cells (p < 1e-3) carry ~a quarter
    of the UNWEIGHTED variance, so the unweighted sigma measures tail depth,
    not head softness, and the re-leveling gain sigma_p/sigma_q is junk-
    driven (measured on B-testbed val trajectories: fog-position force 0.50
    ~ baseline 0.48, i.e. no fix at all). Under w = p the level is the head
    contrast (sigma median 0.57 vs 3.3 unweighted on the same tokens). This
    is also the principled metric: logit perturbations act on the policy in
    the Fisher geometry, so the sampling measure - not the counting measure
    of an arbitrary top-k support - is the right inner product.

    Why this replaces the bisection (lesion P1'a, level chasing): tempering
    is linear in log space, so on the fiber {q^lam} the entropy H(q^lam) is
    a transcendental coordinate (hence rkl_dt's root-finding) while any
    fixed-weight spread is linear, sigma_w(lam log q) = lam sigma_w(log q) -
    the matching equation solves itself, lam* = sigma_pw / sigma_qw. And
    w = p does not move when the teacher is tempered, so every closed-form
    property is exact:

      * orbit invariance: q and q^lam give the same q~ for any lam > 0 -
        zero force on pure temperature fog in BOTH directions (the one-sided
        "if H(q) > H(p)" gate of rkl_dt is gone), echo channel severed;
      * level conservation: sigma_w(log q~) = sigma_w(log p) exactly;
      * idempotence; fixed set r = 0 iff z_w(log p) = z_w(log q), i.e. the
        force self-terminates on structure match at the student's own level;
      * ties are preserved and partial forks survive the re-leveling (no
        entropy squeeze; the cascade restores the fork at the student's
        level, vs ~0.04 left of a 0.52/0.40 fork after rkl_dt's sharpening).

    Lesion P2 (confident-wrong, promotion starved by the p-weight): kept the
    rkl_dt mechanism - at a CW position the student is sharp, so the
    re-leveled teacher is sharp on the teacher's choice; every cell except
    that choice gets negative force and the iterated field flips top-1 with
    the entropy transient coming back down. Single-step promotion force
    remains p-diluted - the price of the -p*(.) form.

    No gates, no clamps, no iterations; junk teacher evals sit on low-w
    cells and are further damped by the p-weight in r. Cost: two weighted
    moment reductions + one log_softmax.

    Args:
        student_log_probs: (batch, seq, K) student log probs on the cells.
        teacher_log_probs: (batch, seq, K) teacher log probs on the same cells.
        valid_mask: (batch, seq, K) bool mask of cells that participate.

    Returns:
        (batch, seq, K) rewards, zero on invalid cells, dtype of the input.
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))

    # renormalize both distributions on the shared cell support
    s = torch.log_softmax(s_lp, dim=-1)
    t = torch.log_softmax(t_lp, dim=-1)

    m = valid_mask.to(s.dtype)
    w = s.exp() * m  # student mass = Fisher weights; already ~normalized on the support
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    def polar(lp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.where(valid_mask, lp, torch.zeros_like(lp))
        vc = (v - (w * v).sum(dim=-1, keepdim=True)) * m
        sd = (w * vc.square()).sum(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
        return vc / sd, sd

    z_p, sigma_p = polar(s)
    z_q, _ = polar(t)

    t_cdt = torch.log_softmax(torch.where(valid_mask, sigma_p * z_q, torch.full_like(t, neg_inf)), dim=-1)

    scores = -s.exp() * (s - t_cdt)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)


def compute_fkl_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward-KL transport rewards: r = q - p on the shared support.

    The exact logit gradient of -KL(q || p): with p = softmax(z),
    KL(q||p) = LSE(z) - <q, z> + const, hence dKL/dz_c = p_c - q_c and

        r = q - p.

    One formula, no thresholds, no gates, no target transforms, no
    iterations. The lesion triage is embedded in the mass difference:

    * confident-wrong (P2): the correction cell holds q ~ 0.6 while
      p ~ 0.03, so it receives the full missing mass (measured on real val
      CW tokens: +0.60 vs +0.11 under the baseline reverse-KL force, whose
      p-weight starves exactly the cells the student does not yet believe);
      the wrong top-1 gets -(p - q); junk cells get ~0 - the 2-cell surgery
      the CW anatomy prescribes (correction target = student's rank-2).
    * structure transmission (P1'b): a teacher alternative with q = 0.22,
      p ~ 0.01 receives +0.15 - the mass difference itself - instead of the
      baseline's +0.05 (p-diluted).
    * healthy tokens: p ~ q gives r ~ 0. No explicit classifier: lesion
      tokens ARE the tokens where |q - p| is large.

    Endpoint optimality (verified on 274k real val tokens): on the
    geometric bridge between the two KL forces,
    r^lam = (p^(1-lam) q^lam - p)/lam (lam->0 recovers the baseline force
    -p*(logp - logq), lam=1 is this one), both lesion forces are monotone
    increasing in lam (CW promotion +0.105 -> +0.597, FS alternative
    +0.050 -> +0.152) while the pure-fog force is first-order
    lam-invariant: the lesion optimum inside the KL family is this
    endpoint.

    Properties: sum_c r_c = 0 per position (conservative transport - mass
    released from over-confident cells lands exactly on the teacher's named
    cells); |r| <= 1 (junk teacher evals cannot amplify - no unbounded
    log-ratio terms, the rkl_sdt collapse channel); KL(q||p) is CONVEX in
    the student logits, so the per-position field is a convex gradient
    flow: globally convergent, self-terminating, and it cannot sharpen the
    student beyond the teacher (equal-budget CW overshoot halves vs the
    baseline, 21.7% vs 41.4%; zero new confident-wrongs).

    Stated cost: pure temperature fog is still followed (any divergence
    with fixed point p = q follows a retreating teacher - not removable
    inside the KL family), but the softening is transported along the
    teacher's own shape instead of a p-weighted scalar squeeze, bounded and
    terminal, on ~0.3% of tokens.
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))
    scores = torch.log_softmax(t_lp, dim=-1).exp() - torch.log_softmax(s_lp, dim=-1).exp()
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)


def compute_rkl_dt_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Baseline reverse-KL cell rewards against a one-sided entropy-detempered teacher.

    The reward formula is the baseline's own  r_c = -p_c * (log p_c - log q_c);
    the single change is the target. Per position, with p and q renormalized
    over the valid cells:

        if H(q) > H(p):  q~ = q^lam / Z(lam)  with lam >= 1 solving H(q~) = H(p)
        else:            q~ = q                       (teacher kept as-is)

        r_c = -p_c * (log p_c - log q~_c)

    Why (validated offline on rollout trajectories):

    * On student-sampled contexts the teacher's extra softness is mostly
      off-manifold epistemic noise, not "many valid continuations". Matching
      it (or reverse-KL against it) keeps ordering the student to spread out;
      the symmetric-TRI run showed the resulting entropy runaway. Tempering
      is rank-preserving (log q~ = lam * log q - log Z), so the teacher's
      preferences pass through at the student's own contrast level, and the
      "sharpen up" edits (teacher sharper than student) pass untouched.
    * At confidently-wrong positions the sharpened q~ makes the wrong top's
      log-ratio large, so the suppression cascade (re-computed each step)
      hands the position to the teacher's choice: 6/6 confidently-wrong and
      ~99%/88% of rank-2/rank-3+ flips fixed in iterated one-step probes.

    lam is found by bisection on [1, 64] (entropy is monotone in lam);
    50 iterations of (batch, seq)-vectorized tensor ops, negligible cost.

    Args:
        student_log_probs: (batch, seq, K) student log probs on the cells.
        teacher_log_probs: (batch, seq, K) teacher log probs on the same cells.
        valid_mask: (batch, seq, K) bool mask of cells that participate.

    Returns:
        (batch, seq, K) rewards, zero on invalid cells, dtype of the input.
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))

    # renormalize both distributions on the shared cell support
    s = torch.log_softmax(s_lp, dim=-1)
    t = torch.log_softmax(t_lp, dim=-1)

    h_p = _cell_entropy(s)
    h_q = _cell_entropy(t)
    need = h_q > h_p + 1e-4

    lo = torch.ones_like(h_p)
    hi = torch.full_like(h_p, _LAM_MAX)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        t_mid = torch.log_softmax(
            torch.where(valid_mask, t * mid.unsqueeze(-1), torch.full_like(t, neg_inf)), dim=-1
        )
        too_soft = _cell_entropy(t_mid) > h_p
        lo = torch.where(too_soft, mid, lo)
        hi = torch.where(too_soft, hi, mid)

    lam = torch.where(need, 0.5 * (lo + hi), torch.ones_like(h_p))
    t_dt = torch.log_softmax(
        torch.where(valid_mask, t * lam.unsqueeze(-1), torch.full_like(t, neg_inf)), dim=-1
    )

    scores = -s.exp() * (s - t_dt)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)


def compute_mu_dt_scores(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Baseline reverse-KL rewards against the mu-detempered teacher (clamp-free).

        L = KL(p || sg[q~]),   q~ = q^(1/sqrt(mu)) / Z,
        mu = KL(q||u) / KL(p||u) = (logK - H(q)) / (logK - H(p)),

    u the uniform distribution on the K valid cells, and the cell force (the
    logit gradient of L up to the per-position baseline) is the baseline's own

        r_c = -p_c * (log p_c - log q~_c).

    mu is the teacher/student knowledge-radius ratio: KL(.||u) measures how far
    a distribution stands from ignorance, tempering moves along the q-u
    geodesic (q^b u^(1-b) / Z = q^b / Z), and along that fiber KL(.||u) is
    approximately quadratic, so the radius-matching exponent has the closed
    form b = mu^(-1/2). No bisection (rkl_dt), no entropy gate (one-sided
    rkl_dt), no second-moment gauge (rkl_cdt). Equivalently: G-OPD with the
    reference model replaced by u and the reward-scaling factor measured per
    position, lambda_t = mu^(-1/2) - no third model, no extra forward pass.
    Self-extinguishing: H(p) = H(q) gives mu = 1 and q~ = q, i.e. baseline OPD
    is recovered exactly at radius match.

    Why sqrt (adjudicated on 270k real val tokens + synthetic anatomies):
    b = mu^(-1/2) is the geometric half-step between "do not move" (1) and
    naive level matching (1/mu). It never crosses the student's radius (0.0%
    of FS/fog positions end with a target sharper than the student), while
    1/mu crosses on 18-20% of them (the chase reverses direction, the
    rkl_sdt collapse channel) and kills thin semantic branches; the
    arithmetic mixture (1-a)q + au can only soften (negentropy convexity),
    so it is structurally mute on exactly the lesions, which need the
    teacher sharpened.

    Real-trajectory audit (38 val trajectories, 270k tokens, steps 0-200):
    fog level force -42% (two-sided: teacher-softer AND teacher-sharper),
    FS-directed level force -21%, healthy tokens untouched (class-median
    mu = 1.000 at every checkpoint), confident-wrong the only class whose
    force RISES (+2.9%; promotion boosted on 96% of CW positions), rejected
    -branch overkill trimmed (equal-budget overshoot 41.4% -> 36.9%),
    true-fork alt-force keep ratio 0.991 (median), total budget -11.6% with
    89.7% of the change landing on non-healthy tokens.

    Hyperparameter-free: the only guards are clamp_min(0) on the student
    negentropy (>= 0 by theorem, float roundoff can dip below) and _EPS in
    the denominator against literal division by zero - IEEE hygiene, not
    behavioral knobs. Unclamped exponents measured on the audit span
    [0.23, 2.23]; a [1/16, 16] clamp binds on 1 token in 270k and changes no
    metric, hence none is used. The theoretical singularity (teacher exactly
    uniform on the cells, KL(q||u) -> 0) is empirically unreachable: min
    KL(q||u) = 0.13 nats across the audit; if the cell construction ever
    changes (not top-K renormalized), re-measure that floor. Applying the
    same rule iteratively converges to the exact two-sided radius match
    (monotone improvement); kept out of this implementation to stay
    parameter-free.

    Args:
        student_log_probs: (batch, seq, K) student log probs on the cells.
        teacher_log_probs: (batch, seq, K) teacher log probs on the same cells.
        valid_mask: (batch, seq, K) bool mask of cells that participate.

    Returns:
        (batch, seq, K) rewards, zero on invalid cells, dtype of the input.
    """
    neg_inf = torch.finfo(torch.float32).min
    s_lp = torch.where(valid_mask, student_log_probs.float(), torch.full_like(student_log_probs.float(), neg_inf))
    t_lp = torch.where(valid_mask, teacher_log_probs.float(), torch.full_like(teacher_log_probs.float(), neg_inf))

    # renormalize both distributions on the shared cell support
    s = torch.log_softmax(s_lp, dim=-1)
    t = torch.log_softmax(t_lp, dim=-1)

    log_k = valid_mask.sum(dim=-1).clamp_min(1).to(s.dtype).log()
    n_p = (log_k - _cell_entropy(s)).clamp_min(0.0)  # KL(p||u) >= 0 by theorem
    n_q = (log_k - _cell_entropy(t)).clamp_min(_EPS)  # _EPS: division guard only
    beta = (n_p / n_q).sqrt().unsqueeze(-1)  # = mu^(-1/2)

    t_mudt = torch.log_softmax(
        torch.where(valid_mask, beta * t, torch.full_like(t, neg_inf)), dim=-1
    )

    scores = -s.exp() * (s - t_mudt)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    return torch.nan_to_num(scores, nan=0.0).to(student_log_probs.dtype)
