# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Online probes for the SC-ratio OPD advantage adv = a * (1 + w).

One readout per pre-registered claim:

  * align_mean            E[a] = -KL(S||T): the debt-repayment progress bar
                          (starts ~ -0.2, rises to 0; rent is impossible, so it
                          must never settle at a positive plateau).
  * bonus_mean            E[w*a]: the net bonus force. Contrast with G-OPD's
                          frozen-tilt plateau (+0.025) / first4k (+0.019): this
                          series has no positive-plateau failure mode.
  * weight_mean/clamp_frac  bonus budget and firewall bite. Expected shape:
                          clamp_frac ~2% at start, rises during the length
                          explosion (loop states), drifts toward ~50% at
                          convergence as the bonus retires (harmless: the
                          unclamped half's w is ~0 there too).
  * seg{k} num/den        depth profile [0,2k)/[2k,4k)/[4k,8k)/[8k,+): the
                          force amplification 1 + wabsa_num/absa_num should
                          reproduce the offline prediction (~1.23/1.24/1.19/1.13
                          vs G-OPD's flat 1.25) and anneal to 1; absa_num/tok_den
                          is the debt ladder collapsing over training.
  * capped rows           runaway starving and the ledger flip: weight and
                          (signed) bonus mass flowing to full-buffer rows, plus
                          the teacher-SC collapse (offline: ~14.5 capped-deep
                          vs ~22.5 normal) that drives it.
  * terminal              executed-stop probe: a and w on the last valid token
                          of terminated rows (the anti-early-stop channel).

Conventions (inherited from the earlier probe suite's hardening):
  * float32 throughout: bf16 summation over 16k-token rows loses percent-level
    accuracy in num/den comparisons.
  * Raw values, deliberately NOT multiplied by loss_scale_factor.
  * *_num / *_den are per-micro sums; the metric pipeline mean-reduces over
    micro-batches, which preserves sum ratios exactly. Divide the two logged
    series in W&B to recover the true ratio (per-micro ratios can degenerate
    when a micro-batch holds a single sequence).
  * Rows whose valid length fills the whole response buffer are treated as
    capped (runaway proxy); zero-length rows are guarded out of the terminal
    probe.
"""

from __future__ import annotations

import torch

_SEG_BOUNDS = (2048, 4096, 8192)
_CENTERED_SEG_BOUNDS = (1024, 2048, 4096, 8192, 12288)
_TERMINAL_WINDOW = 256
_ACTIVE_EPS = 0.05
_STRONG_A = 0.5
_SC_FLOOR = 1e-6

__all__ = ["compute_sc_probe_metrics", "compute_sc_centered_probe_metrics"]


@torch.no_grad()
def compute_sc_probe_metrics(
    align_adv: torch.Tensor,
    sc_weight: torch.Tensor,
    sc_student: torch.Tensor,
    sc_teacher: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Probes for the SC-ratio advantage. All inputs are (bsz, response_len).

    Args:
        align_adv: alignment debt a = logT - logS on the sampled token.
        sc_weight: the applied bonus weight w = clamp(1 - SC_S/SC_T, 0, 1).
        sc_student: student self-certainty SC_S per position.
        sc_teacher: teacher self-certainty SC_T per position.
        response_mask: valid-token mask.
    """
    if align_adv.dim() != 2 or sc_weight.dim() != 2:
        return {}

    a = align_adv.float()
    w = sc_weight.float()
    sc_s = sc_student.float()
    sc_t = sc_teacher.float()
    mask = response_mask.to(torch.float32)
    n_tok = mask.sum().clamp_min(1.0)
    bonus = w * a
    out: dict[str, float] = {}

    def full_mean(x: torch.Tensor) -> float:
        return ((x * mask).sum() / n_tok).item()

    out["sc_probe/align_mean"] = full_mean(a)
    out["sc_probe/align_abs_mean"] = full_mean(a.abs())  # remaining token-debt mass
    out["sc_probe/bonus_mean"] = full_mean(bonus)
    out["sc_probe/weight_mean"] = full_mean(w)
    out["sc_probe/clamp_frac"] = full_mean((w <= 0).float())
    out["sc_probe/sc_student_mean"] = full_mean(sc_s)
    out["sc_probe/sc_teacher_mean"] = full_mean(sc_t)

    # --- depth segments (num/den pairs) --------------------------------------
    positions = torch.arange(mask.shape[-1], device=mask.device)
    lo = 0
    for k, hi in enumerate((*_SEG_BOUNDS, mask.shape[-1] + 1)):
        seg = ((positions >= lo) & (positions < hi)).float().unsqueeze(0) * mask
        out[f"sc_probe/tok_den_seg{k}"] = seg.sum().item()
        out[f"sc_probe/absa_num_seg{k}"] = (a.abs() * seg).sum().item()
        out[f"sc_probe/wabsa_num_seg{k}"] = (w * a.abs() * seg).sum().item()
        out[f"sc_probe/weight_num_seg{k}"] = (w * seg).sum().item()
        out[f"sc_probe/clamp_num_seg{k}"] = ((w <= 0).float() * seg).sum().item()
        lo = hi

    # --- trajectory ledger: capped (runaway proxy) rows ----------------------
    lengths = mask.sum(dim=-1)
    capped_row = (lengths >= mask.shape[-1]).float()
    capped_col = capped_row.unsqueeze(-1)
    out["sc_probe/capped_seq_frac"] = capped_row.mean().item()
    out["sc_probe/tok_total_den"] = n_tok.item()
    out["sc_probe/tok_capped_den"] = (capped_col * mask).sum().item()
    out["sc_probe/weight_capped_num"] = (w * capped_col * mask).sum().item()
    out["sc_probe/bonus_capped_num"] = (bonus * capped_col * mask).sum().item()
    out["sc_probe/sct_capped_num"] = (sc_t * capped_col * mask).sum().item()

    # --- executed-stop probe: last valid token of terminated rows ------------
    terminated_row = 1.0 - capped_row
    last_idx = (lengths.long() - 1).clamp_min(0)
    row_idx = torch.arange(mask.shape[0], device=mask.device)
    terminal_mask = torch.zeros_like(mask)
    terminal_mask[row_idx, last_idx] = terminated_row
    # Guard zero-length rows: their clamped last_idx points at padding.
    terminal_mask = terminal_mask * mask
    out["sc_probe/terminal_cnt_den"] = terminal_mask.sum().item()
    out["sc_probe/terminal_a_num"] = (a * terminal_mask).sum().item()
    out["sc_probe/terminal_w_num"] = (w * terminal_mask).sum().item()

    return out


@torch.no_grad()
def compute_sc_centered_probe_metrics(
    align_adv: torch.Tensor,
    centered_tilt: torch.Tensor,
    sc_student: torch.Tensor,
    sc_teacher: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Probes for the SC-centered advantage adv = a + c, c = g - mean_traj(g).

    Two pre-registered propositions, both resolved per depth segment:

      P1 (selection) c lands on the tokens the frozen G-OPD extrapolation force
                     lands on, and with the same sign.
      P2 (payment)   c's per-token force is of the same order as the
                     extrapolation force's, segment by segment.

    Reference-free anchor, and its expiry date. This arm never instantiates a
    reference worker, so d = logT - logR is not observable as a tensor. It
    enters through a single identity: at step 0 the rollout policy IS the init
    model (REFERENCE_MODEL_PATH defaults to ACTOR_MODEL_PATH), hence
    a = logT - logS equals d exactly and 0.25 * |a| is the force budget G-OPD
    would have spent on the extrapolation term. That lambda = 1.25 belongs to
    the G-OPD comparison arm (opd-baseline-qwen3-4b's GOPD_LAMBDA default); this
    arm runs GOPD_LAMBDA=1.0 and the centered branch ignores lambda_vals
    altogether, so the actor/gopd_lambda series (= 1.0 * loss_scale_factor) has
    nothing to do with the 0.25 above. d is frozen for
    the whole run; a is not. From step 1 on, a = d - (logS_t - logR) is the live
    debt and it typically melts, so a ratio taken against the same-step a
    inflates on its own and would confirm P2 by artefact. Both propositions are
    therefore read against a step-0 denominator (see below). This is an honest
    limitation of the reference-free design, not a derivation: the dose law and
    the first4k analysis both put the extrapolation force's value in the early
    accumulation phase, which is the window the anchor covers.

    Segments (_CENTERED_SEG_BOUNDS, response-token index, prompt excluded):
      0 [0, 1k)    the "front 1k" of the offline +0.082 baseline
      1 [1k, 2k)   continuity with the existing readouts
      2 [2k, 4k)   inside the first4k window
      3 [4k, 8k)   first4k's outer edge (first4k stops paying here)
      4 [8k, 12k)  long but still-terminating trajectories
      5 [12k, +)   loop deep tail
    _SEG_BOUNDS is shared with compute_sc_probe_metrics (SC-ratio arm) and must
    stay untouched: changing it would retroactively redefine seg0..3 of the
    already-finished scratio run.

    Derived readings (divide the two logged series in W&B):

      * substitution rate, upper bound (P2, per segment)
            [cabs_num_seg{k} / tok_den_seg{k}] at step t
            / (0.25 * [absa_num_seg{k} / tok_den_seg{k}] at step 0)
      * substitution rate, effective (P2, per segment)
            same with csigna_num_seg{k} in the numerator: c projected onto the
            extrapolation force's direction. The cabs version credits
            anti-aligned force as if it had substituted, so it only bounds.
            Per-token normalisation is mandatory in both: numerator and
            denominator come from different steps, whose per-segment token
            counts differ, so the raw sums no longer cancel. The step-0
            denominator also has to exist: the init student rarely reaches the
            deep segments (offline, its longest correct trajectory is 6.8k), so
            a segment whose tok_den_seg{k} at step 0 is small carries a noisy or
            empty denominator. Gate on it (a few thousand tokens) and report
            "no data" for that segment rather than a large ratio.
      * fingerprint overlap (P1, per segment)
            agree_num_seg{k} / strong_den_seg{k}; offline ~67% front, ~54% deep.
            Not readable on its own: the chance level of sign agreement is
            p_a * p_c + (1 - p_a) * (1 - p_c), with p_a = strong_apos_num /
            strong_den and p_c = strong_cpos_num / strong_den, and both
            marginals drift over training and across segments (c is zero-sum,
            a is early-positive). Always report overlap minus chance. The table
            degenerates where c is exactly 0 (single-token rows center to zero):
            such tokens count towards neither agree_num nor strong_cpos_num, so
            treat the overlap as unreadable in a batch full of very short
            responses (watch tok_den_seg0 against the response-length series).
      * payment schedule (per segment)
            c_num_seg{k} / tok_den_seg{k}: front positive, deep negative is the
            claim. Offline front-1k minus 8k+ contrast +0.082 (first4k itself
            +0.027, gopd whole-trajectory +0.016).
      * deep-force attribution
            c_capped_num_seg{k} / tok_capped_den_seg{k} against the same
            segment's normal-row mean, which is the complement
            (c_num_seg{k} - c_capped_num_seg{k}) / (tok_den_seg{k} -
            tok_capped_den_seg{k}): the negative force beyond 8k should land on
            runaway rows (offline, the longest correct trajectory is 6.8k).
            Read this on seg4: seg5 is filled almost exclusively by capped rows,
            so its capped-vs-normal contrast has no denominator left.
      * force budget / kill line
            c_abs_mean. Offline (init-student proxy) ~0.078, live student
            estimated 0.02-0.04. A sustained < 0.02 by step ~15 means the arm
            is underpowered: report it as such, do not raise beta to compensate.
      * conflict rate
            conflict_num / active_den. Same axis as the overlap family (sign
            relation between c and a) but at the lower |.| > 0.05 threshold and
            global only; kept solely for comparability with the gopd terminal
            60% reading (offline SC-centered: ~30%, bounded real conflict).
      * profile attribution
            logsct_num_seg{k}, logscs_num_seg{k}. Their difference is exactly
            g's segment sum (g = log SC_T - log SC_S), so the uncentered
            profile needs no family of its own, and the split says whether a
            flattening depth profile came from the teacher getting clearer in
            the deep tail or from the student's tail suppression catching up --
            two situations whose responses are opposite. Logged as means of
            log SC rather than of SC because g is a difference of logs and only
            the log means decompose it exactly (E[g] != log(E[SC_T]/E[SC_S]));
            SC levels in nats remain available as sc_{student,teacher}_mean.
      * zero-sum self-check
            c_mean, ~0 by construction (centering + mask plumbing).
      * terminal window
            c_term_num / term_den over the last 256 valid tokens of terminated
            rows (offline ~-0.13 on correct endings; watch for early-truncation
            side effects). A safety watchdog, not a proposition.

    Deliberately not logged, recoverable by algebra:
      * positive / negative tilt mass per segment:
        (cabs_num_seg{k} +/- c_num_seg{k}) / 2.
      * normal (non-runaway) rows: total minus capped, e.g.
        c_num_seg{k} - c_capped_num_seg{k}, denominator likewise.
      * the full 2x2 sign table on the strong-a set: its four cells follow from
        strong_den, agree_num, strong_apos and strong_cpos.
      * uncentered g per segment: logsct_num_seg{k} - logscs_num_seg{k}.
      * global means of |a|, |c| and g: sum a segment family and divide by
        sum(tok_den_seg). c_abs_mean is kept anyway, as the kill line.
      * runaway row share: the trainer already logs response_length/clip_ratio
        over the whole batch. A per-micro row mean here would additionally have
        broken this file's num/den convention (it is a ratio taken inside the
        micro, so it only survives the mean-reduce while every micro-batch holds
        the same number of rows).
      * align_mean is kept although it is recoverable too: c is zero-sum per
        row, so actor/gopd_adv_mean = align_mean * loss_scale_factor exactly,
        and actor/gopd_lambda is that same factor while lambda_vals is 1.0.
        Recovering it means dividing two loss_scale_factor-contaminated series
        and relying on this arm's lambda; the probe suite keeps raw readings.

    Deliberately no longer measured:
      * negative-force share on capped rows. The pre-registered "66-71%
        negative beyond 8k" was a token-count share, carried by the retired
        cneg_capped_num global; the offline script that produced the figure is
        gone, so it cannot be re-derived or its convention re-checked. It stays
        on record as history only, and the deep-force claim rests on the
        mean-vs-mean attribution above, which is what the proposition needs.
    """
    if align_adv.dim() != 2 or centered_tilt.dim() != 2:
        return {}

    a = align_adv.float()
    c = centered_tilt.float()
    sc_s = sc_student.float()
    sc_t = sc_teacher.float()
    mask = response_mask.to(torch.float32)
    n_tok = mask.sum().clamp_min(1.0)
    log_sc_s = torch.log(sc_s.clamp_min(_SC_FLOOR))
    log_sc_t = torch.log(sc_t.clamp_min(_SC_FLOOR))
    sign_a = torch.sign(a)
    strong_a = (a.abs() > _STRONG_A).float()
    # torch.sign(0) = 0, so a zero tilt never counts as agreement (on the strong
    # set sign(a) is +/-1, so only sign(c) can be the zero).
    agree = (torch.sign(c) == sign_a).float() * strong_a
    out: dict[str, float] = {}

    def full_mean(x: torch.Tensor) -> float:
        return ((x * mask).sum() / n_tok).item()

    out["sc_centered/align_mean"] = full_mean(a)
    out["sc_centered/c_mean"] = full_mean(c)
    out["sc_centered/c_abs_mean"] = full_mean(c.abs())
    out["sc_centered/sc_student_mean"] = full_mean(sc_s)
    out["sc_centered/sc_teacher_mean"] = full_mean(sc_t)

    # --- conflict with the alignment debt -------------------------------------
    active = ((c.abs() > _ACTIVE_EPS) & (a.abs() > _ACTIVE_EPS)).float() * mask
    conflict = (torch.sign(c) != sign_a).float() * active
    out["sc_centered/active_den"] = active.sum().item()
    out["sc_centered/conflict_num"] = conflict.sum().item()

    # --- runaway targeting: capped (full-buffer) rows --------------------------
    lengths = mask.sum(dim=-1)
    capped_row = (lengths >= mask.shape[-1]).float()
    capped_col = capped_row.unsqueeze(-1)

    # --- depth segments (num/den pairs) ---------------------------------------
    positions = torch.arange(mask.shape[-1], device=mask.device)
    lo = 0
    for k, hi in enumerate((*_CENTERED_SEG_BOUNDS, mask.shape[-1] + 1)):
        seg = ((positions >= lo) & (positions < hi)).float().unsqueeze(0) * mask
        seg_capped = seg * capped_col
        out[f"sc_centered/tok_den_seg{k}"] = seg.sum().item()
        out[f"sc_centered/c_num_seg{k}"] = (c * seg).sum().item()
        out[f"sc_centered/cabs_num_seg{k}"] = (c.abs() * seg).sum().item()
        out[f"sc_centered/absa_num_seg{k}"] = (a.abs() * seg).sum().item()
        out[f"sc_centered/csigna_num_seg{k}"] = (c * sign_a * seg).sum().item()
        out[f"sc_centered/strong_den_seg{k}"] = (strong_a * seg).sum().item()
        out[f"sc_centered/agree_num_seg{k}"] = (agree * seg).sum().item()
        out[f"sc_centered/strong_apos_num_seg{k}"] = ((a > _STRONG_A).float() * seg).sum().item()
        out[f"sc_centered/strong_cpos_num_seg{k}"] = (strong_a * (c > 0).float() * seg).sum().item()
        out[f"sc_centered/tok_capped_den_seg{k}"] = seg_capped.sum().item()
        out[f"sc_centered/c_capped_num_seg{k}"] = (c * seg_capped).sum().item()
        out[f"sc_centered/logsct_num_seg{k}"] = (log_sc_t * seg).sum().item()
        out[f"sc_centered/logscs_num_seg{k}"] = (log_sc_s * seg).sum().item()
        lo = hi

    # --- terminal window: last 256 valid tokens of terminated rows -------------
    terminated_col = (1.0 - capped_row).unsqueeze(-1)
    start = (lengths - _TERMINAL_WINDOW).clamp_min(0.0).unsqueeze(-1)
    window = (positions.unsqueeze(0) >= start).float() * mask * terminated_col
    out["sc_centered/term_den"] = window.sum().item()
    out["sc_centered/c_term_num"] = (c * window).sum().item()

    return out
