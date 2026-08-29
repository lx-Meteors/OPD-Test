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
"""CPU tests for the SC-ratio OPD building blocks (verl/utils/self_certainty.py)."""

import math

import torch

from verl.utils.self_certainty import sc_ratio_weight, self_certainty_from_logits


def _bruteforce_kl_uniform_to_p(logits: torch.Tensor) -> torch.Tensor:
    """KL(U || softmax(logits)) computed from the definition in float64."""
    log_p = torch.log_softmax(logits.double(), dim=-1)
    vocab = logits.shape[-1]
    return (-math.log(vocab) - log_p.mean(dim=-1)).float()


def test_self_certainty_matches_bruteforce_kl():
    torch.manual_seed(0)
    logits = torch.randn(7, 11, 512) * 5.0
    sc = self_certainty_from_logits(logits)
    ref = _bruteforce_kl_uniform_to_p(logits)
    assert sc.shape == logits.shape[:-1]
    assert torch.allclose(sc, ref, atol=1e-4), (sc - ref).abs().max()
    # Gibbs: KL(U || p) >= 0
    assert (sc >= -1e-5).all()


def test_uniform_distribution_gives_zero():
    logits = torch.full((3, 128), 2.5)  # constant logits -> uniform p
    sc = self_certainty_from_logits(logits)
    assert torch.allclose(sc, torch.zeros_like(sc), atol=1e-5)


def test_chunked_matches_unchunked():
    torch.manual_seed(1)
    logits = torch.randn(300, 257)
    assert torch.allclose(
        self_certainty_from_logits(logits, chunk_size=7),
        self_certainty_from_logits(logits, chunk_size=10**9),
        atol=1e-6,
    )


def test_bf16_input_is_finite_and_close():
    torch.manual_seed(2)
    logits = torch.randn(64, 1024) * 8.0
    sc_bf16 = self_certainty_from_logits(logits.bfloat16())
    sc_fp32 = self_certainty_from_logits(logits)
    assert torch.isfinite(sc_bf16).all()
    assert (sc_bf16 - sc_fp32).abs().max() < 0.15


def test_no_grad_statistic():
    logits = torch.randn(4, 33, requires_grad=True)
    sc = self_certainty_from_logits(logits)
    assert not sc.requires_grad


def test_weight_semantics():
    sc_t = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0])
    sc_s = torch.tensor([4.0, 0.0, 8.0, 2.0, -1.0])  # equal / zero / sharper / half / fp-noise
    w = sc_ratio_weight(sc_s, sc_t)
    expected = torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0])
    assert torch.allclose(w, expected, atol=1e-6)
    assert (w >= 0).all() and (w <= 1).all()


def test_weight_survives_degenerate_teacher():
    w = sc_ratio_weight(torch.tensor([0.5]), torch.tensor([0.0]))
    assert torch.isfinite(w).all() and (w >= 0).all() and (w <= 1).all()


def test_advantage_mode_formulas():
    """Replicates the dp_actor only_reverse_kl advantage synthesis on toy tensors."""
    torch.manual_seed(3)
    log_s = torch.log(torch.rand(2, 9).clamp(0.05, 0.95))
    log_t = torch.log(torch.rand(2, 9).clamp(0.05, 0.95))
    log_r = torch.log(torch.rand(2, 9).clamp(0.05, 0.95))
    a = log_t - log_s
    d = log_t - log_r
    lam = 1.25

    # classic G-OPD: -[(logS - logR) - lam*(logT - logR)] == a + (lam-1)*d
    classic = -((log_s - log_r) - lam * (log_t - log_r))
    assert torch.allclose(classic, a + (lam - 1.0) * d, atol=1e-6)

    # live clock: lam * a
    live = lam * (log_t - log_s)
    assert torch.allclose(live, lam * a, atol=1e-6)

    # SC-ratio: a * (1 + w), force between a and 2a (sign-parallel, capped x2)
    sc_s = torch.rand(2, 9) * 10.0
    sc_t = torch.rand(2, 9) * 10.0
    w = sc_ratio_weight(sc_s, sc_t)
    adv = (log_t - log_s) * (1.0 + w)
    assert torch.allclose(adv.sign() * adv.abs(), adv, atol=0)  # tautology guard for shape
    assert (adv.sign() * a.sign() >= 0).all()  # never fights the alignment debt
    assert (adv.abs() >= a.abs() - 1e-6).all() and (adv.abs() <= 2 * a.abs() + 1e-6).all()
    # where the student is at least as certain, the bonus vanishes exactly
    off = sc_s >= sc_t
    assert torch.allclose(adv[off], a[off], atol=1e-6)


if __name__ == "__main__":
    test_self_certainty_matches_bruteforce_kl()
    test_uniform_distribution_gives_zero()
    test_chunked_matches_unchunked()
    test_bf16_input_is_finite_and_close()
    test_no_grad_statistic()
    test_weight_semantics()
    test_weight_survives_degenerate_teacher()
    test_advantage_mode_formulas()
    print("all self-certainty tests passed")
