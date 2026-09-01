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

_SEG_RANGES = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 12288), (12288, 16384))
_N_SEG = len(_SEG_RANGES)

_FAMILIES = (
    "tok_den",
    "c_num",
    "cabs_num",
    "absa_num",
    "csigna_num",
    "strong_den",
    "agree_num",
    "strong_apos_num",
    "strong_cpos_num",
    "tok_capped_den",
    "c_capped_num",
    "logsct_num",
    "logscs_num",
)

# Globals dropped in favour of the segment families (algebraically recoverable),
# of no service to either proposition, or already logged by the trainer
# (capped_seq_frac vs response_length/clip_ratio).
_DELETED_KEYS = (
    "sc_centered/align_abs_mean",
    "sc_centered/g_mean",
    "sc_centered/g_abs_mean",
    "sc_centered/tok_capped_den",
    "sc_centered/c_capped_num",
    "sc_centered/cneg_capped_num",
    "sc_centered/capped_seq_frac",
)

# capped row / the 12288 boundary / one token past it / a seg4-only row /
# a single-token row (c == 0) / a zero-length row.
_DEEP_LENGTHS = (16384, 12288, 12289, 9000, 1, 0)


def _probe(a, c, sc_s, sc_t, mask):
    return compute_sc_centered_probe_metrics(
        align_adv=a, centered_tilt=c, sc_student=sc_s, sc_teacher=sc_t, response_mask=mask
    )


def _make_deep_case(seed=11):
    """A batch that actually populates all six centered segments.

    resp_len is the production response buffer (16384) so that seg5 = [12k, +)
    is reachable at all; without it the deep-segment counters are untested.
    """
    torch.manual_seed(seed)
    resp_len = _DEEP_LENGTHS[0]
    bsz = len(_DEEP_LENGTHS)
    sc_s = torch.rand(bsz, resp_len) * 30.0 + 0.5
    sc_t = torch.rand(bsz, resp_len) * 30.0 + 0.5
    lengths = torch.tensor(_DEEP_LENGTHS, dtype=torch.long)
    mask = (torch.arange(resp_len).unsqueeze(0) < lengths.unsqueeze(-1)).float()
    a = torch.randn(bsz, resp_len) * 0.4  # ~21% of tokens above the strong line
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    return a, c, sc_s, sc_t, mask, lengths


def _bruteforce_segments(a, c, sc_s, sc_t, mask):
    """float64 python-loop reference for every segment counter family."""
    resp_len = mask.shape[-1]
    bounds = tuple(hi for _, hi in _SEG_RANGES[:-1]) + (resp_len + 1,)
    acc = [{name: 0.0 for name in _FAMILIES} for _ in bounds]
    for i in range(mask.shape[0]):
        mask_row = mask[i].tolist()
        a_row = a[i].tolist()
        c_row = c[i].tolist()
        sct_row = sc_t[i].tolist()
        scs_row = sc_s[i].tolist()
        capped = sum(mask_row) >= resp_len
        for j in range(resp_len):
            if mask_row[j] <= 0:
                continue
            k = next(idx for idx, hi in enumerate(bounds) if j < hi)
            d = acc[k]
            av, cv = a_row[j], c_row[j]
            sign_a = 0.0 if av == 0.0 else math.copysign(1.0, av)
            sign_c = 0.0 if cv == 0.0 else math.copysign(1.0, cv)
            d["tok_den"] += 1.0
            d["c_num"] += cv
            d["cabs_num"] += abs(cv)
            d["absa_num"] += abs(av)
            d["csigna_num"] += cv * sign_a
            if capped:
                d["tok_capped_den"] += 1.0
                d["c_capped_num"] += cv
            if abs(av) > 0.5:
                d["strong_den"] += 1.0
                if sign_c == sign_a:
                    d["agree_num"] += 1.0
                if av > 0.0:
                    d["strong_apos_num"] += 1.0
                if cv > 0.0:
                    d["strong_cpos_num"] += 1.0
            d["logsct_num"] += math.log(max(sct_row[j], 1e-6))
            d["logscs_num"] += math.log(max(scs_row[j], 1e-6))
    return acc


def test_probe_matches_bruteforce():
    sc_s, sc_t, mask, lengths = _make_case(bsz=6, resp_len=64, seed=8)
    a = torch.randn_like(sc_s) * 0.4
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    out = _probe(a, c, sc_s, sc_t, mask)

    m = mask.bool()
    n_tok = m.sum().item()

    assert math.isclose(out["sc_centered/c_abs_mean"], c[m].abs().mean().item(), rel_tol=1e-5)
    # zero-net-force identity surfaces as c_mean ~ 0
    assert abs(out["sc_centered/c_mean"]) < 1e-5
    assert math.isclose(out["sc_centered/align_mean"], a[m].mean().item(), rel_tol=1e-5, abs_tol=1e-7)
    assert math.isclose(out["sc_centered/sc_teacher_mean"], sc_t[m].mean().item(), rel_tol=1e-5)

    # conflict accounting
    active = (c.abs() > 0.05) & (a.abs() > 0.05) & m
    conflict = (torch.sign(c) != torch.sign(a)) & active
    assert out["sc_centered/active_den"] == active.sum().item()
    assert out["sc_centered/conflict_num"] == conflict.sum().item()

    # depth segments partition the valid tokens exactly (six of them now)
    seg_den = sum(out[f"sc_centered/tok_den_seg{k}"] for k in range(_N_SEG))
    assert math.isclose(seg_den, n_tok, rel_tol=1e-6)
    seg_c = sum(out[f"sc_centered/c_num_seg{k}"] for k in range(_N_SEG))
    assert abs(seg_c) < 1e-3  # zero-sum identity again, via the segment route

    capped = lengths >= mask.shape[-1]
    cap_tokens = m[capped].sum().item()
    seg_cap_den = sum(out[f"sc_centered/tok_capped_den_seg{k}"] for k in range(_N_SEG))
    assert math.isclose(seg_cap_den, cap_tokens, rel_tol=1e-6)

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


def test_all_segment_counters_match_bruteforce():
    """Every family x every segment against a float64 python-loop reference."""
    a, c, sc_s, sc_t, mask, _ = _make_deep_case()
    out = _probe(a, c, sc_s, sc_t, mask)
    ref = _bruteforce_segments(a, c, sc_s, sc_t, mask)
    for k in range(_N_SEG):
        assert ref[k]["tok_den"] > 0, f"seg{k} has no data: the case is vacuous"
        for name in _FAMILIES:
            got = out[f"sc_centered/{name}_seg{k}"]
            exp = ref[k][name]
            assert math.isclose(got, exp, rel_tol=2e-4, abs_tol=1e-3), (k, name, got, exp)


def test_segment_partition_and_subset_identities():
    a, c, sc_s, sc_t, mask, lengths = _make_deep_case(seed=12)
    out = _probe(a, c, sc_s, sc_t, mask)

    assert math.isclose(
        sum(out[f"sc_centered/tok_den_seg{k}"] for k in range(_N_SEG)), mask.sum().item(), rel_tol=1e-6
    )
    # zero-sum survives the segment route (fp32 sums over ~50k tokens)
    assert abs(sum(out[f"sc_centered/c_num_seg{k}"] for k in range(_N_SEG))) < 5e-2

    capped = lengths >= mask.shape[-1]
    assert math.isclose(
        sum(out[f"sc_centered/tok_capped_den_seg{k}"] for k in range(_N_SEG)),
        mask[capped].sum().item(),
        rel_tol=1e-6,
    )
    for k in range(_N_SEG):
        tok = out[f"sc_centered/tok_den_seg{k}"]
        assert out[f"sc_centered/tok_capped_den_seg{k}"] <= tok
        assert out[f"sc_centered/strong_den_seg{k}"] <= tok
        assert out[f"sc_centered/agree_num_seg{k}"] <= out[f"sc_centered/strong_den_seg{k}"]
        assert out[f"sc_centered/strong_apos_num_seg{k}"] <= out[f"sc_centered/strong_den_seg{k}"]
        assert out[f"sc_centered/strong_cpos_num_seg{k}"] <= out[f"sc_centered/strong_den_seg{k}"]
        assert abs(out[f"sc_centered/c_capped_num_seg{k}"]) <= out[f"sc_centered/cabs_num_seg{k}"] + 1e-3
        assert abs(out[f"sc_centered/csigna_num_seg{k}"]) <= out[f"sc_centered/cabs_num_seg{k}"] + 1e-3


def test_boundary_lengths_land_in_expected_segments():
    a, c, sc_s, sc_t, mask, _ = _make_deep_case(seed=13)
    out = _probe(a, c, sc_s, sc_t, mask)
    # lengths (16384, 12288, 12289, 9000, 1, 0); seg widths are 1024/1024/2048/4096/4096
    assert out["sc_centered/tok_den_seg0"] == 1024.0 * 4 + 1.0
    assert out["sc_centered/tok_den_seg4"] == 4096.0 * 3 + 808.0
    # the 12288-long row stops exactly at the seg5 boundary and contributes nothing
    assert out["sc_centered/tok_den_seg5"] == 4096.0 + 1.0
    # only the full-buffer row is capped, so its tokens are the whole buffer
    assert out["sc_centered/tok_capped_den_seg5"] == 4096.0
    assert sum(out[f"sc_centered/tok_capped_den_seg{k}"] for k in range(_N_SEG)) == 16384.0


def test_strong_line_is_strictly_greater():
    """|a| > 0.5, so a token sitting exactly on the line is not a fingerprint."""
    a = torch.tensor([[0.5, -0.5, 0.5000001, -0.6, 0.4]])
    c = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]])
    sc = torch.full((1, 5), 2.0)
    out = _probe(a, c, sc, sc, torch.ones(1, 5))
    assert out["sc_centered/strong_den_seg0"] == 2.0  # 0.5000001 and -0.6 only
    assert out["sc_centered/strong_apos_num_seg0"] == 1.0
    assert out["sc_centered/strong_cpos_num_seg0"] == 1.0
    assert out["sc_centered/agree_num_seg0"] == 2.0


def test_zero_tilt_never_counts_as_agreement():
    """torch.sign(0) = 0: a single-token row's c = 0 must not read as agreement."""
    a = torch.tensor([[0.9, -0.9]])
    c = torch.zeros(1, 2)
    sc = torch.full((1, 2), 3.0)
    out = _probe(a, c, sc, sc, torch.ones(1, 2))
    assert out["sc_centered/strong_den_seg0"] == 2.0
    assert out["sc_centered/agree_num_seg0"] == 0.0
    assert out["sc_centered/strong_cpos_num_seg0"] == 0.0
    assert out["sc_centered/csigna_num_seg0"] == 0.0


def test_sign_table_is_fully_determined():
    """strong_den + agree + the two marginals pin all four cells of the table."""
    torch.manual_seed(17)
    bsz, resp_len = 4, 200
    sc_s = torch.rand(bsz, resp_len) * 30.0 + 0.5
    sc_t = torch.rand(bsz, resp_len) * 30.0 + 0.5
    lengths = torch.tensor([200, 150, 90, 40])  # no single-token row: c has no exact zeros
    mask = (torch.arange(resp_len).unsqueeze(0) < lengths.unsqueeze(-1)).float()
    a = torch.randn(bsz, resp_len) * 0.4
    c = centered_log_sc_ratio(sc_s, sc_t, mask)
    out = _probe(a, c, sc_s, sc_t, mask)

    strong = (a.abs() > 0.5) & mask.bool()
    for k, (lo, hi) in enumerate(_SEG_RANGES):
        seg = torch.zeros_like(mask, dtype=torch.bool)
        seg[:, lo : min(hi, resp_len)] = True
        n_strong = out[f"sc_centered/strong_den_seg{k}"]
        agree = out[f"sc_centered/agree_num_seg{k}"]
        m_a = out[f"sc_centered/strong_apos_num_seg{k}"]
        m_c = out[f"sc_centered/strong_cpos_num_seg{k}"]
        cell_pp = (m_a + m_c + agree - n_strong) / 2  # #(a > 0 and c > 0)
        expected = ((a > 0) & (c > 0) & strong & seg).sum().item()
        assert math.isclose(cell_pp, expected, abs_tol=1e-6), (k, cell_pp, expected)


def test_recoverable_readings_match_direct_computation():
    """The algebra that justifies not logging g_num_seg and the dropped globals."""
    a, c, sc_s, sc_t, mask, _ = _make_deep_case(seed=15)
    out = _probe(a, c, sc_s, sc_t, mask)
    m = mask.bool()
    n_tok = mask.sum().item()

    # uncentered g per segment = logsct - logscs
    g = torch.log(sc_t.clamp_min(1e-6) / sc_s.clamp_min(1e-6))
    g_from_logs = sum(
        out[f"sc_centered/logsct_num_seg{k}"] - out[f"sc_centered/logscs_num_seg{k}"] for k in range(_N_SEG)
    )
    assert math.isclose(g_from_logs, g[m].sum().item(), rel_tol=1e-4)

    # the dropped global means come back out of the segment families
    absa = sum(out[f"sc_centered/absa_num_seg{k}"] for k in range(_N_SEG)) / n_tok
    assert math.isclose(absa, a[m].abs().mean().item(), rel_tol=1e-4)
    g_mean = g_from_logs / n_tok
    assert math.isclose(g_mean, g[m].mean().item(), rel_tol=1e-4)
    cabs = sum(out[f"sc_centered/cabs_num_seg{k}"] for k in range(_N_SEG)) / n_tok
    assert math.isclose(cabs, out["sc_centered/c_abs_mean"], rel_tol=1e-4)

    # positive / negative tilt mass, and the normal-row complement
    for k in range(_N_SEG):
        cabs_k = out[f"sc_centered/cabs_num_seg{k}"]
        c_k = out[f"sc_centered/c_num_seg{k}"]
        pos, neg = (cabs_k + c_k) / 2, (cabs_k - c_k) / 2
        assert pos >= -1e-6 and neg >= -1e-6
        assert math.isclose(pos + neg, cabs_k, rel_tol=1e-6, abs_tol=1e-6)
        normal_den = out[f"sc_centered/tok_den_seg{k}"] - out[f"sc_centered/tok_capped_den_seg{k}"]
        assert normal_den >= 0.0


def test_deleted_keys_are_absent():
    a, c, sc_s, sc_t, mask, _ = _make_deep_case(seed=16)
    out = _probe(a, c, sc_s, sc_t, mask)
    for key in _DELETED_KEYS:
        assert key not in out, key
    for k in range(_N_SEG):
        # recoverable as logsct_num_seg{k} - logscs_num_seg{k}
        assert f"sc_centered/g_num_seg{k}" not in out
        # the linear SC families were replaced by their log versions
        assert f"sc_centered/sct_num_seg{k}" not in out
        assert f"sc_centered/scs_num_seg{k}" not in out
        # dropped: its only reading (negative mass share on capped rows) was not
        # the token-count share the retired offline baseline referred to
        assert f"sc_centered/cabs_capped_num_seg{k}" not in out
    assert "sc_centered/tok_den_seg6" not in out  # exactly six segments


def test_degenerate_rows_are_safe():
    """Zero-length rows (deep case) and an all-masked batch must stay finite."""
    a, c, sc_s, sc_t, mask, _ = _make_deep_case(seed=18)
    out = _probe(a, c, sc_s, sc_t, mask)
    assert all(math.isfinite(v) for v in out.values())

    empty_mask = torch.zeros(2, 16)
    out0 = _probe(
        torch.randn(2, 16), torch.randn(2, 16), torch.rand(2, 16) + 0.5, torch.rand(2, 16) + 0.5, empty_mask
    )
    assert all(math.isfinite(v) for v in out0.values())
    assert out0["sc_centered/tok_den_seg0"] == 0.0
    assert out0["sc_centered/term_den"] == 0.0
    assert out0["sc_centered/tok_capped_den_seg0"] == 0.0


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
