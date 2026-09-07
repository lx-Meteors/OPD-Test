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

import pytest
import torch

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
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, mask)

    # one terminated row -> one EOS position (index 3 of row 0), whose residual is ~0
    assert m["etopd/eos_den"] == 1.0
    assert abs(m["etopd/eos_res_num"]) < 1e-6
    assert abs(m["etopd/eos_align_num"] - (math.log(1e-4) - math.log(0.05))) < 1e-4
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
    m = compute_etopd_probe_metrics(align, residual, alpha, log_t, mask)
    assert m["etopd/eos_den"] == 1.0  # only the second row terminates
    assert m["etopd/tok_den"] == 2.0
    assert torch.isfinite(adv).all()
