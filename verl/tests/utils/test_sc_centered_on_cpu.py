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
"""CPU tests for the SC-centered OPD advantage: c = g - mean_traj(g),
g = log(SC_T/SC_S), adv = a + c. Covers the centering helper
(verl/utils/self_certainty.py) and its probe suite (verl/utils/sc_probe.py)."""

import math

import torch

from verl.utils.sc_probe import compute_sc_centered_probe_metrics
from verl.utils.self_certainty import centered_log_sc_ratio


def _make_case(bsz=5, resp_len=97, seed=0):
    torch.manual_seed(seed)
    sc_s = torch.rand(bsz, resp_len) * 30.0 + 0.5
    sc_t = torch.rand(bsz, resp_len) * 30.0 + 0.5
    lengths = torch.randint(low=1, high=resp_len + 1, size=(bsz,))
    lengths[0] = resp_len  # one capped (runaway proxy) row
    lengths[1] = 1  # one single-token row
    mask = (torch.arange(resp_len).unsqueeze(0) < lengths.unsqueeze(-1)).float()
    return sc_s, sc_t, mask, lengths


def _bruteforce_centered(sc_s, sc_t, mask):
    """Per-row python-loop reference in float64."""
    out = torch.zeros_like(sc_s, dtype=torch.float64)
    for i in range(sc_s.shape[0]):
        valid = mask[i] > 0
        g = torch.log(sc_t[i][valid].double().clamp_min(1e-6) / sc_s[i][valid].double().clamp_min(1e-6))
        out[i][valid] = g - g.mean()
    return out.float()


def test_centered_matches_bruteforce():
    sc_s, sc_t, mask, _ = _make_case()
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    ref = _bruteforce_centered(sc_s, sc_t, mask)
    assert c.shape == sc_s.shape
    assert torch.allclose(c, ref, atol=1e-5), (c - ref).abs().max()


def test_per_row_masked_sum_is_zero():
    """The zero-net-force identity: within-trajectory total tilt is exactly 0."""
    sc_s, sc_t, mask, _ = _make_case(seed=1)
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    row_sums = (c * mask).sum(dim=-1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-4), row_sums


def test_masked_positions_are_zero():
    sc_s, sc_t, mask, _ = _make_case(seed=2)
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    assert (c[mask == 0] == 0).all()


def test_single_token_row_centers_to_zero():
    sc_s, sc_t, mask, lengths = _make_case(seed=3)
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    single = lengths == 1
    assert single.any()
    assert torch.allclose(c[single], torch.zeros_like(c[single]), atol=1e-6)


def test_all_masked_row_is_safe():
    sc_s = torch.rand(2, 8) + 0.5
    sc_t = torch.rand(2, 8) + 0.5
    mask = torch.zeros(2, 8)
    mask[0, :4] = 1.0  # row 1 fully masked
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    assert torch.isfinite(c).all()
    assert (c[1] == 0).all()


def test_sc_floor_guards_zero_and_negative_inputs():
    """SC values are >= 0 up to rounding; the eps floor must keep c finite."""
    sc_s = torch.tensor([[0.0, 1.0, 2.0]])
    sc_t = torch.tensor([[1.0, -1e-9, 2.0]])
    mask = torch.ones(1, 3)
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    assert torch.isfinite(c).all()


def test_bf16_inputs_are_finite_and_close():
    sc_s, sc_t, mask, _ = _make_case(seed=4)
    c32 = centered_log_sc_ratio(sc_s, sc_t, mask)
    cbf = centered_log_sc_ratio(sc_s.bfloat16(), sc_t.bfloat16(), mask)
    assert torch.isfinite(cbf).all()
    assert (c32 - cbf).abs().max() < 0.05


def test_scale_invariance_of_the_ratio():
    """g depends only on SC_T/SC_S: rescaling both by the same factor is a no-op."""
    sc_s, sc_t, mask, _ = _make_case(seed=5)
    c1 = centered_log_sc_ratio(sc_s, sc_t, mask)
    c2 = centered_log_sc_ratio(sc_s * 3.7, sc_t * 3.7, mask)
    assert torch.allclose(c1, c2, atol=1e-5)


def test_uniform_gap_centers_to_zero():
    """SC_T = k * SC_S everywhere: g is constant per row, so c must vanish.
    The speed component (the constant) is exactly what the centering removes."""
    sc_s, _, mask, _ = _make_case(seed=6)
    sc_t = sc_s * 1.5
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-5)


def test_advantage_formula():
    """adv = a + c, elementwise (the dp_actor branch is this one-liner)."""
    sc_s, sc_t, mask, _ = _make_case(seed=7)
    a = torch.randn_like(sc_s) * 0.3
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    adv = a + c
    assert torch.allclose(adv - a, c)


# --------------------------------------------------------------------------
# Probe suite
# --------------------------------------------------------------------------


def test_probe_matches_bruteforce():
    sc_s, sc_t, mask, lengths = _make_case(bsz=6, resp_len=64, seed=8)
    a = torch.randn_like(sc_s) * 0.4
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    out = compute_sc_centered_probe_metrics(
        align_adv=a, centered_tilt=c, sc_student=sc_s, sc_teacher=sc_t, response_mask=mask
    )

    m = mask.bool()
    n_tok = m.sum().item()
    g = torch.log(sc_t.clamp_min(1e-6) / sc_s.clamp_min(1e-6))

    assert math.isclose(out["sc_centered/c_abs_mean"], c[m].abs().mean().item(), rel_tol=1e-5)
    # zero-net-force identity surfaces as c_mean ~ 0
    assert abs(out["sc_centered/c_mean"]) < 1e-5
    assert math.isclose(out["sc_centered/g_mean"], g[m].mean().item(), rel_tol=1e-5)
    assert math.isclose(out["sc_centered/align_mean"], a[m].mean().item(), rel_tol=1e-5, abs_tol=1e-7)

    # conflict accounting
    active = (c.abs() > 0.05) & (a.abs() > 0.05) & m
    conflict = (torch.sign(c) != torch.sign(a)) & active
    assert out["sc_centered/active_den"] == active.sum().item()
    assert out["sc_centered/conflict_num"] == conflict.sum().item()

    # depth segments partition the valid tokens exactly
    seg_den = sum(out[f"sc_centered/tok_den_seg{k}"] for k in range(4))
    assert math.isclose(seg_den, n_tok, rel_tol=1e-6)
    seg_c = sum(out[f"sc_centered/c_num_seg{k}"] for k in range(4))
    assert abs(seg_c) < 1e-3  # zero-sum identity again, via the segment route

    # capped (runaway proxy) row accounting
    capped = lengths >= mask.shape[-1]
    assert math.isclose(out["sc_centered/capped_seq_frac"], capped.float().mean().item(), rel_tol=1e-6)
    cap_tokens = m[capped].sum().item()
    assert out["sc_centered/tok_capped_den"] == cap_tokens
    assert math.isclose(
        out["sc_centered/c_capped_num"], c[capped][m[capped]].sum().item(), rel_tol=1e-4, abs_tol=1e-4
    )
    assert out["sc_centered/cneg_capped_num"] == (c[capped][m[capped]] < 0).sum().item()

    # terminal window: last 256 valid tokens of terminated rows only
    term_expected = 0.0
    den_expected = 0
    for i in range(mask.shape[0]):
        if capped[i]:
            continue
        length = int(lengths[i].item())
        lo = max(0, length - 256)
        term_expected += c[i, lo:length].sum().item()
        den_expected += length - lo
    assert out["sc_centered/term_den"] == den_expected
    assert math.isclose(out["sc_centered/c_term_num"], term_expected, rel_tol=1e-4, abs_tol=1e-4)


def test_probe_rejects_topk_shapes():
    x = torch.randn(2, 8, 4)
    out = compute_sc_centered_probe_metrics(
        align_adv=x, centered_tilt=x, sc_student=x, sc_teacher=x, response_mask=torch.ones(2, 8)
    )
    assert out == {}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
