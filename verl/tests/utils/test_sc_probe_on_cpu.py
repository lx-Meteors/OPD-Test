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
"""Brute-force reference tests for the SC-ratio probe suite."""

import torch

from verl.utils.sc_probe import _SEG_BOUNDS, compute_sc_probe_metrics


def _bruteforce(a, w, sc_s, sc_t, mask):
    """Independent python-loop reference for every probe value."""
    bsz, resp = mask.shape
    a = a.double()
    w = w.double()
    sc_s = sc_s.double()
    sc_t = sc_t.double()
    m = mask.double()

    tot = {"tok": 0.0, "a": 0.0, "absa": 0.0, "bonus": 0.0, "w": 0.0, "clamp": 0.0,
           "scs": 0.0, "sct": 0.0}
    seg_bounds = (*_SEG_BOUNDS, resp + 1)
    seg = [{"tok": 0.0, "absa": 0.0, "wabsa": 0.0, "w": 0.0, "clamp": 0.0} for _ in seg_bounds]
    cap = {"tok": 0.0, "w": 0.0, "bonus": 0.0, "sct": 0.0}
    term = {"cnt": 0.0, "a": 0.0, "w": 0.0}
    capped_rows = 0

    for i in range(bsz):
        length = int(m[i].sum().item())
        row_capped = length >= resp
        capped_rows += int(row_capped)
        for t in range(resp):
            if m[i, t] == 0:
                continue
            tot["tok"] += 1
            tot["a"] += a[i, t].item()
            tot["absa"] += abs(a[i, t].item())
            tot["bonus"] += (w[i, t] * a[i, t]).item()
            tot["w"] += w[i, t].item()
            tot["clamp"] += float(w[i, t].item() <= 0)
            tot["scs"] += sc_s[i, t].item()
            tot["sct"] += sc_t[i, t].item()
            k = sum(t >= b for b in _SEG_BOUNDS)
            seg[k]["tok"] += 1
            seg[k]["absa"] += abs(a[i, t].item())
            seg[k]["wabsa"] += (w[i, t] * abs(a[i, t])).item()
            seg[k]["w"] += w[i, t].item()
            seg[k]["clamp"] += float(w[i, t].item() <= 0)
            if row_capped:
                cap["tok"] += 1
                cap["w"] += w[i, t].item()
                cap["bonus"] += (w[i, t] * a[i, t]).item()
                cap["sct"] += sc_t[i, t].item()
        if not row_capped and length > 0:
            term["cnt"] += 1
            term["a"] += a[i, length - 1].item()
            term["w"] += w[i, length - 1].item()

    n = max(tot["tok"], 1.0)
    ref = {
        "sc_probe/align_mean": tot["a"] / n,
        "sc_probe/align_abs_mean": tot["absa"] / n,
        "sc_probe/bonus_mean": tot["bonus"] / n,
        "sc_probe/weight_mean": tot["w"] / n,
        "sc_probe/clamp_frac": tot["clamp"] / n,
        "sc_probe/sc_student_mean": tot["scs"] / n,
        "sc_probe/sc_teacher_mean": tot["sct"] / n,
        "sc_probe/capped_seq_frac": capped_rows / bsz,
        "sc_probe/tok_total_den": tot["tok"],
        "sc_probe/tok_capped_den": cap["tok"],
        "sc_probe/weight_capped_num": cap["w"],
        "sc_probe/bonus_capped_num": cap["bonus"],
        "sc_probe/sct_capped_num": cap["sct"],
        "sc_probe/terminal_cnt_den": term["cnt"],
        "sc_probe/terminal_a_num": term["a"],
        "sc_probe/terminal_w_num": term["w"],
    }
    for k, s in enumerate(seg):
        ref[f"sc_probe/tok_den_seg{k}"] = s["tok"]
        ref[f"sc_probe/absa_num_seg{k}"] = s["absa"]
        ref[f"sc_probe/wabsa_num_seg{k}"] = s["wabsa"]
        ref[f"sc_probe/weight_num_seg{k}"] = s["w"]
        ref[f"sc_probe/clamp_num_seg{k}"] = s["clamp"]
    return ref


def _random_case(bsz=6, resp=9000, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(bsz, resp, generator=g) * 1.5
    w = torch.clamp(torch.randn(bsz, resp, generator=g) * 0.4 + 0.2, 0.0, 1.0)
    sc_s = torch.rand(bsz, resp, generator=g) * 25
    sc_t = torch.rand(bsz, resp, generator=g) * 25 + 1
    lengths = torch.randint(0, resp + 1, (bsz,), generator=g)
    lengths[0] = resp  # one capped row
    lengths[1] = 0     # one zero-length row
    mask = (torch.arange(resp).unsqueeze(0) < lengths.unsqueeze(-1)).float()
    return a.to(dtype), w.to(dtype), sc_s.to(dtype), sc_t.to(dtype), mask


def _compare(out, ref, atol=1e-4, rtol=1e-4):
    assert set(out) == set(ref), set(out) ^ set(ref)
    for k, v in ref.items():
        assert abs(out[k] - v) <= atol + rtol * abs(v), (k, out[k], v)


def test_matches_bruteforce_reference():
    a, w, sc_s, sc_t, mask = _random_case()
    _compare(compute_sc_probe_metrics(a, w, sc_s, sc_t, mask), _bruteforce(a, w, sc_s, sc_t, mask))


def test_bf16_inputs_are_cast_and_close():
    a, w, sc_s, sc_t, mask = _random_case(seed=1)
    out = compute_sc_probe_metrics(a.bfloat16(), w.bfloat16(), sc_s.bfloat16(), sc_t.bfloat16(), mask)
    ref = _bruteforce(a.bfloat16().float(), w.bfloat16().float(), sc_s.bfloat16().float(),
                      sc_t.bfloat16().float(), mask)
    _compare(out, ref, atol=1e-2, rtol=1e-3)


def test_all_capped_batch_has_no_terminal():
    a, w, sc_s, sc_t, _ = _random_case(bsz=3, resp=100, seed=2)
    mask = torch.ones(3, 100)
    out = compute_sc_probe_metrics(a[:, :100], w[:, :100], sc_s[:, :100], sc_t[:, :100], mask)
    assert out["sc_probe/capped_seq_frac"] == 1.0
    assert out["sc_probe/terminal_cnt_den"] == 0.0
    assert out["sc_probe/tok_capped_den"] == out["sc_probe/tok_total_den"]


def test_short_buffer_leaves_deep_segments_empty():
    a, w, sc_s, sc_t, _ = _random_case(bsz=2, resp=100, seed=3)
    mask = torch.ones(2, 100)
    out = compute_sc_probe_metrics(a[:, :100], w[:, :100], sc_s[:, :100], sc_t[:, :100], mask)
    for k in (1, 2, 3):
        assert out[f"sc_probe/tok_den_seg{k}"] == 0.0
        assert out[f"sc_probe/wabsa_num_seg{k}"] == 0.0


def test_bonus_identity_with_advantage():
    """bonus == adv - a for adv = a*(1+w), elementwise and in the probe mean."""
    a, w, sc_s, sc_t, mask = _random_case(seed=4)
    adv = a * (1.0 + w)
    out = compute_sc_probe_metrics(a, w, sc_s, sc_t, mask)
    n = mask.sum()
    expected_bonus_mean = (((adv - a) * mask).sum() / n).item()
    assert abs(out["sc_probe/bonus_mean"] - expected_bonus_mean) < 1e-5


if __name__ == "__main__":
    test_matches_bruteforce_reference()
    test_bf16_inputs_are_cast_and_close()
    test_all_capped_batch_has_no_terminal()
    test_short_buffer_leaves_deep_segments_empty()
    test_bonus_identity_with_advantage()
    print("all sc_probe tests passed")
