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
"""CPU tests for the entropy-tempered extrapolation (ET-OPD) advantage."""

import math

import numpy as np
import pytest
import torch

from verl.trainer.ppo.metric_utils import compute_validation_completion_metrics
from verl.utils.etopd import (
    compute_etopd_advantages,
    compute_etopd_probe_metrics,
    entropy_tempered_alpha,
    entropy_tempered_residual,
)


def _logp(*probs):
    return torch.tensor(probs, dtype=torch.float32).log()


def test_alpha_is_at_least_one_and_equals_one_at_1_over_e():
    t = torch.linspace(1e-6, 1.0 - 1e-6, 20001)
    alpha = entropy_tempered_alpha(t.log())
    assert torch.all(alpha >= 1.0 - 1e-6)
    alpha_e = entropy_tempered_alpha(torch.tensor([1.0 / math.e]).log())
    assert torch.allclose(alpha_e, torch.ones(1), atol=1e-5)
    # alpha grows without bound toward both ends
    assert entropy_tempered_alpha(_logp(1e-4)).item() > 100.0
    assert entropy_tempered_alpha(_logp(0.999)).item() > 100.0


def test_alpha_finite_for_teacher_prob_one_and_tiny():
    alpha = entropy_tempered_alpha(torch.tensor([0.0, -40.0, -1e3]))
    assert torch.isfinite(alpha).all()
    assert torch.all(alpha >= 1.0)


def test_residual_matches_float64_power_form():
    torch.manual_seed(0)
    log_t = torch.rand(4, 7).clamp_min(1e-3).log()
    log_r = torch.rand(4, 7).clamp_min(1e-3).log()
    alpha = entropy_tempered_alpha(log_t)
    got = entropy_tempered_residual(log_t, log_r, alpha)
    t64 = log_t.double().exp()
    r64 = log_r.double().exp()
    a64 = alpha.double()
    want = (t64**a64 - r64**a64) / a64
    assert torch.allclose(got.double(), want, atol=1e-6)


def test_residual_sign_consistent_and_bounded():
    torch.manual_seed(1)
    log_t = torch.rand(64, 33).clamp(1e-4, 1 - 1e-4).log()
    log_r = torch.rand(64, 33).clamp(1e-4, 1 - 1e-4).log()
    alpha = entropy_tempered_alpha(log_t)
    res = entropy_tempered_residual(log_t, log_r, alpha)
    # Never the wrong sign. The residual may be exactly 0 where alpha is so large that both
    # T^alpha and R^alpha underflow (T -> 0 or T -> 1): that is the intended limit, not a flip.
    target_sign = torch.sign(log_t.exp() - log_r.exp())
    assert torch.all((res == 0) | (torch.sign(res) == target_sign))
    # The underflow-to-zero case only happens at very cold comparison temperatures.
    assert torch.all(alpha[(res == 0) & (target_sign != 0)] > 20.0)
    assert torch.all(res.abs() <= 1.0 / alpha + 1e-6)
    assert torch.all(res.abs() <= 1.0)


def test_residual_vanishes_on_eos_cliff_and_negligible_tokens():
    # optional-stop EOS: teacher ~1e-4, base 0.08 (log-space residual would be ~ -6.7)
    res_eos = entropy_tempered_residual(_logp(1e-4), _logp(0.08), entropy_tempered_alpha(_logp(1e-4)))
    assert abs(res_eos.item()) < 1e-6
    # nobody-cares token
    res_tail = entropy_tempered_residual(_logp(0.008), _logp(0.001), entropy_tempered_alpha(_logp(0.008)))
    assert abs(res_tail.item()) < 1e-4
    # teacher-certain corridor token: strongly damped relative to the L2 residual
    res_corr = entropy_tempered_residual(_logp(0.99), _logp(0.6), entropy_tempered_alpha(_logp(0.99)))
    assert 0.0 < res_corr.item() < 0.1
    # contested decision token stays close to the mass difference
    res_dec = entropy_tempered_residual(_logp(0.56), _logp(0.2), entropy_tempered_alpha(_logp(0.56)))
    assert 0.25 < res_dec.item() < 0.4


def test_fixed_alpha_one_recovers_l2_residual():
    torch.manual_seed(2)
    log_t = torch.rand(5, 9).clamp_min(1e-3).log()
    log_r = torch.rand(5, 9).clamp_min(1e-3).log()
    alpha = entropy_tempered_alpha(log_t, fixed_alpha=1.0)
    res = entropy_tempered_residual(log_t, log_r, alpha)
    assert torch.allclose(res, log_t.exp() - log_r.exp(), atol=1e-6)


def test_advantage_is_alignment_plus_residual_and_keeps_dtype():
    torch.manual_seed(3)
    log_t = torch.rand(3, 11).clamp_min(1e-3).log().to(torch.bfloat16)
    log_s = torch.rand(3, 11).clamp_min(1e-3).log().to(torch.bfloat16)
    log_r = torch.rand(3, 11).clamp_min(1e-3).log().to(torch.bfloat16)
    adv, align, residual, alpha = compute_etopd_advantages(log_t, log_s, log_r)
    assert adv.dtype == torch.bfloat16
    assert torch.allclose(align, log_t.float() - log_s.float())
    assert torch.allclose(adv.float(), (align + residual).to(torch.bfloat16).float())
    assert torch.all(alpha >= 1.0)


def test_probe_metrics_eos_capped_and_bins():
    # two rows, response buffer 6: row 0 terminates after 4 tokens, row 1 is capped (full buffer)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    log_t = torch.tensor(
        [[0.9, 0.5, 0.05, 1e-4, 0.5, 0.5], [0.95, 0.4, 0.3, 0.2, 0.7, 0.6]], dtype=torch.float32
    ).log()
    log_r = torch.tensor(
        [[0.6, 0.2, 0.05, 0.08, 0.5, 0.5], [0.9, 0.1, 0.3, 0.4, 0.5, 0.3]], dtype=torch.float32
    ).log()
    log_s = torch.tensor(
        [[0.7, 0.3, 0.1, 0.05, 0.5, 0.5], [0.9, 0.2, 0.3, 0.3, 0.6, 0.4]], dtype=torch.float32
    ).log()
    adv, align, residual, alpha = compute_etopd_advantages(log_t, log_s, log_r)
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, log_r, mask)

    # one terminated row -> one EOS position (index 3 of row 0), whose residual is ~0
    assert m["etopd/eos_den"] == 1.0
    assert abs(m["etopd/eos_res_num"]) < 1e-6
    assert abs(m["etopd/eos_align_num"] - (math.log(1e-4) - math.log(0.05))) < 1e-4
    # the same EOS under the counterfactual log-space residual is the cliff: log(1e-4/0.08) = -6.7
    assert abs(m["etopd/cf_log_eos_num"] - math.log(1e-4 / 0.08)) < 1e-4
    assert abs(m["etopd/cf_l2_eos_num"] - (1e-4 - 0.08)) < 1e-6
    assert abs(m["etopd/eos_logT_num"] - math.log(1e-4)) < 1e-4
    assert abs(m["etopd/eos_logR_num"] - math.log(0.08)) < 1e-4
    assert abs(m["etopd/eos_logS_num"] - math.log(0.05)) < 1e-4
    # capped row bookkeeping for the alignment term
    assert abs(m["etopd/capped_align_num"] - (align[1] * mask[1]).sum().item()) < 1e-5
    # counterfactual and actual |residual| totals partition over the 3 coarse bins
    for cf in ("cf_log", "cf_l2", "cf_noe", "cf_noinv"):
        assert m[f"etopd/{cf}_abs_den"] >= m[f"etopd/{cf}_abs_num_tlow"] + m[f"etopd/{cf}_abs_num_thigh"]
    # fine bins partition the coarse middle bin
    assert abs(m["etopd/tok_num_t1"] + m["etopd/tok_num_t2"] + m["etopd/tok_num_t3"] - m["etopd/tok_num_tmid"]) < 1e-6
    # alpha quantiles are ordered and >= 0 (alpha >= 1)
    assert 0.0 <= m["etopd/log_alpha_p10"] <= m["etopd/log_alpha_p50"] <= m["etopd/log_alpha_p90"]
    # beyond-teacher realisation: row 1 token 1 has T=0.4 > R=0.1 and S=0.2 < T -> negative contribution
    expected_beyond = (torch.sign(log_t.exp() - log_r.exp()) * (log_s - log_t) * mask).sum().item()
    assert abs(m["etopd/beyond_num"] - expected_beyond) < 1e-4
    assert m["etopd/opposed_den"] <= m["etopd/tok_den"]
    assert 0.0 <= m["etopd/opposed_num"] <= m["etopd/opposed_den"]
    # capped row contributes all 6 tokens
    assert m["etopd/capped_tok_den"] == 6.0
    assert m["etopd/tok_den"] == 10.0
    # token bins partition the valid tokens
    assert m["etopd/tok_num_tlow"] + m["etopd/tok_num_tmid"] + m["etopd/tok_num_thigh"] == 10.0
    assert m["etopd/tok_num_tlow"] == 2.0  # 0.05 and 1e-4 in row 0
    # |residual| shares sum to the total
    tot = m["etopd/resabs_num_tlow"] + m["etopd/resabs_num_tmid"] + m["etopd/resabs_num_thigh"]
    assert abs(tot - m["etopd/resabs_den"]) < 1e-6
    # depth segments partition the valid tokens (all positions < 1024 here)
    assert m["etopd/tok_den_seg0"] == 10.0
    assert sum(m[f"etopd/tok_den_seg{k}"] for k in range(5)) == 10.0
    # masked positions never contribute
    assert m["etopd/tok_den"] == mask.sum().item()


@pytest.mark.parametrize("fixed_alpha", [0.0, 1.0, 2.0])
def test_zero_length_row_is_ignored(fixed_alpha):
    mask = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.float32)
    log_t = torch.full((2, 3), math.log(0.5))
    log_r = torch.full((2, 3), math.log(0.4))
    log_s = torch.full((2, 3), math.log(0.45))
    adv, align, residual, alpha = compute_etopd_advantages(log_t, log_s, log_r, fixed_alpha=fixed_alpha)
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, log_r, mask)
    assert m["etopd/eos_den"] == 1.0  # only the second row terminates
    assert m["etopd/tok_den"] == 2.0
    assert torch.isfinite(adv).all()


def test_probe_metrics_all_finite_and_no_nan_on_extreme_inputs():
    torch.manual_seed(4)
    mask = torch.ones(3, 40)
    mask[0, 25:] = 0
    log_t = torch.rand(3, 40).clamp(1e-6, 1 - 1e-6).log()
    log_t[1, 5] = 0.0  # T == 1 exactly
    log_t[2, 7] = -60.0  # T ~ 0
    log_r = torch.rand(3, 40).clamp(1e-6, 1 - 1e-6).log()
    log_s = torch.rand(3, 40).clamp(1e-6, 1 - 1e-6).log()
    adv, align, residual, alpha = compute_etopd_advantages(log_t, log_s, log_r)
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, log_r, mask)
    assert len(m) > 60
    assert all(math.isfinite(v) for v in m.values()), [k for k, v in m.items() if not math.isfinite(v)]
    assert torch.isfinite(adv).all()


def test_rent_counts_only_agreement_corridors():
    mask = torch.ones(1, 4)
    log_t = torch.tensor([[0.9, 0.9, 0.2, 0.6]]).log()
    log_r = torch.tensor([[0.8, 0.3, 0.1, 0.55]]).log()
    log_s = torch.tensor([[0.85, 0.5, 0.15, 0.6]]).log()
    adv, align, residual, alpha = compute_etopd_advantages(log_t, log_s, log_r)
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, log_r, mask)
    # tokens 0 and 3 have both T > 0.5 and R > 0.5
    assert m["etopd/rent_den"] == 2.0
    assert abs(m["etopd/rent_num"] - (residual[0, 0] + residual[0, 3]).item()) < 1e-6


def test_validation_completion_decomposition():
    ds = ["A", "A", "A", "A", "B", "B"]
    infos = {
        "acc": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "format_score": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "clipped": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "response_length": [100.0, 200.0, 16384.0, 300.0, 50.0, 70.0],
    }
    m = compute_validation_completion_metrics(ds, infos)
    assert m["val-aux/A/completion/completion_rate"] == 0.5
    assert m["val-aux/A/completion/acc_given_completed"] == 0.5
    assert m["val-aux/A/completion/clipped_rate"] == 0.25
    assert m["val-aux/A/completion/finished_unboxed_rate"] == 0.25  # sample 3: no box, not clipped
    assert abs(m["val-aux/A/completion/acc_given_unclipped"] - 1 / 3) < 1e-9
    assert m["val-aux/A/completion/response_length_completed"] == 150.0
    assert m["val-aux/B/completion/completion_rate"] == 1.0
    assert m["val-aux/B/completion/acc_given_completed"] == 1.0
    assert "val-aux/B/completion/finished_unboxed_rate" in m
    # acc = completion_rate * acc_given_completed holds pooled as well
    pooled = m["val-aux/all/completion/completion_rate"] * m["val-aux/all/completion/acc_given_completed"]
    assert abs(pooled - np.mean(infos["acc"])) < 1e-9
    # missing keys -> no metrics rather than a crash
    assert compute_validation_completion_metrics(ds, {"acc": infos["acc"]}) == {}


# ---------------------------------------------------------------------------------------
# alpha_source = "ratio": alpha = c_T / c_R
# ---------------------------------------------------------------------------------------
def _ratio_res(t, r, coef=1.0):
    lt, lr = _logp(t), _logp(r)
    a = entropy_tempered_alpha(lt, ref_log_prob=lr, alpha_source="ratio")
    return coef * entropy_tempered_residual(lt, lr, a).item(), a.item()


def test_ratio_alpha_is_hot_on_resolutions_and_cold_on_creations():
    # RL committed to a token the base was unsure about -> c_T < c_R -> alpha < 1 (log-like)
    _, a = _ratio_res(0.98, 0.7)
    assert a < 1.0
    # RL abandoned a token the base used -> c_T < c_R -> hot
    _, a = _ratio_res(0.005, 0.05)
    assert a < 1.0
    # RL raised a token the base essentially never produced (hedge / rare boost) -> cold
    _, a = _ratio_res(0.5, 0.01)
    assert a > 5.0
    _, a = _ratio_res(0.008, 0.001)
    assert a > 5.0
    # contested stop: base certain to stop (R ~ 1) -> c_R -> 0 -> alpha huge -> residual ~ 0
    res, a = _ratio_res(0.38, 0.9998)
    assert a > 1e3 and abs(res) < 1e-3
    # identical probabilities -> alpha = 1 exactly and residual 0
    res, a = _ratio_res(0.3, 0.3)
    assert abs(a - 1.0) < 1e-6 and abs(res) < 1e-7


def test_ratio_residual_matches_gopd_log_in_the_hot_limit_and_vanishes_when_cold():
    # cliff: T = 1e-4, R = 0.08 -> alpha ~ 0.005 -> residual ~ log(T/R) = -6.68
    res, a = _ratio_res(1e-4, 0.08)
    assert a < 0.01
    assert abs(res - math.log(1e-4 / 0.08)) < 0.25
    # scaled by G-OPD's lambda - 1 it reproduces G-OPD's cliff force
    res_l, _ = _ratio_res(1e-4, 0.08, coef=0.25)
    assert abs(res_l - 0.25 * math.log(1e-4 / 0.08)) < 0.07
    # top boost at lambda-1: about G-OPD's +0.084
    res_top, _ = _ratio_res(0.98, 0.7, coef=0.25)
    assert 0.06 < res_top < 0.10
    # hedge-like boost the base never produced: essentially zero (G-OPD would give +0.98)
    res_h, _ = _ratio_res(0.5, 0.01, coef=0.25)
    assert abs(res_h) < 1e-3
    # rare boost: zero (G-OPD +0.52)
    res_r, _ = _ratio_res(0.008, 0.001, coef=0.25)
    assert abs(res_r) < 1e-4


def test_ratio_residual_sign_consistent_bounded_by_log_and_finite_at_extremes():
    torch.manual_seed(5)
    lt = torch.rand(256, 64).clamp(1e-6, 1 - 1e-6).log()
    lr = torch.rand(256, 64).clamp(1e-6, 1 - 1e-6).log()
    lt[0, 0], lr[0, 1], lt[1, 2], lr[1, 3] = 0.0, 0.0, -80.0, -80.0  # T == 1, R == 1, T ~ 0, R ~ 0
    a = entropy_tempered_alpha(lt, ref_log_prob=lr, alpha_source="ratio")
    res = entropy_tempered_residual(lt, lr, a)
    assert torch.isfinite(res).all() and torch.isfinite(a).all()
    target = torch.sign(lt.exp() - lr.exp())
    assert torch.all((res == 0) | (torch.sign(res) == target))
    # |residual| never exceeds the log residual (alpha -> 0 limit) by more than float error
    assert torch.all(res.abs() <= (lt - lr).abs() + 1e-4)


def test_expm1_residual_equals_exp_form_for_alpha_ge_1():
    torch.manual_seed(6)
    lt = torch.rand(8, 16).clamp_min(1e-3).log()
    lr = torch.rand(8, 16).clamp_min(1e-3).log()
    a = entropy_tempered_alpha(lt)  # teacher source, alpha >= 1
    got = entropy_tempered_residual(lt, lr, a)
    want = (torch.exp(a * lt) - torch.exp(a * lr)) / a
    assert torch.allclose(got, want, atol=1e-6)


def test_compute_advantages_with_ratio_source_and_lambda_coef():
    torch.manual_seed(7)
    lt = torch.rand(3, 9).clamp_min(1e-3).log()
    ls = torch.rand(3, 9).clamp_min(1e-3).log()
    lr = torch.rand(3, 9).clamp_min(1e-3).log()
    adv, align, res, alpha = compute_etopd_advantages(lt, ls, lr, alpha_source="ratio", residual_coef=0.25)
    a_ref = entropy_tempered_alpha(lt, ref_log_prob=lr, alpha_source="ratio")
    assert torch.allclose(alpha, a_ref)
    assert torch.allclose(res, 0.25 * entropy_tempered_residual(lt, lr, a_ref))
    assert torch.allclose(adv, align + res)
    with pytest.raises(ValueError):
        entropy_tempered_alpha(lt, alpha_source="nonsense")
    with pytest.raises(ValueError):
        entropy_tempered_alpha(lt, alpha_source="ratio")  # ref_log_prob missing


def test_probe_edit_type_buckets_partition_and_ratio_claims():
    mask = torch.ones(1, 6)
    #        top boost   hedge-like   slip       contested-stop  rare boost  agree
    T = torch.tensor([[0.98, 0.5, 0.005, 0.38, 0.008, 0.9]])
    R = torch.tensor([[0.70, 0.01, 0.05, 0.9998, 0.001, 0.9]])
    S = torch.tensor([[0.90, 0.30, 0.02, 0.60, 0.004, 0.9]])
    lt, lr, ls = T.log(), R.log(), S.log()
    adv, align, res, alpha = compute_etopd_advantages(lt, ls, lr, alpha_source="ratio", residual_coef=0.25)
    m = compute_etopd_probe_metrics(align, res, alpha, lt, lr, mask)
    n = m["etopd/tok_num_resolution"] + m["etopd/tok_num_creation"] + m["etopd/tok_num_uncommit"] + m["etopd/tok_num_tie"]
    assert n == m["etopd/tok_den"] == 6.0
    # top boost and slip are resolutions; hedge-like and rare boost are creations;
    # contested stop is uncommit; the agree token (T == R) is a tie
    assert m["etopd/tok_num_resolution"] == 2.0
    assert m["etopd/tok_num_creation"] == 2.0
    assert m["etopd/tok_num_uncommit"] == 1.0
    assert m["etopd/tok_num_tie"] == 1.0
    # the ratio residual puts (almost) nothing on creation/uncommit tokens
    assert m["etopd/resabs_num_creation"] / m["etopd/resabs_den"] < 0.01
    assert m["etopd/resabs_num_uncommit"] / m["etopd/resabs_den"] < 0.01
    # while G-OPD's log residual would spend a large share there
    assert m["etopd/cf_log_abs_num_creation"] / m["etopd/cf_log_abs_den"] > 0.4
    # hot fraction: resolutions with c_T < c_R are the alpha < 1 tokens
    assert m["etopd/alpha_lt1_frac"] == pytest.approx(2 / 6, abs=1e-6)
    # counterfactual columns exist and the applied residual is 0.25 x the unscaled ratio residual
    assert abs(m["etopd/residual_num"] - 0.25 * m["etopd/cf_ratio_num"]) < 1e-5
    assert "etopd/cf_aT_abs_den" in m
    assert all(math.isfinite(v) for v in m.values())
