# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.utils.horizon_opd import HorizonInvariantOPDTracker, normalize_horizon_weights_for_loss


def _response_mask(lengths: list[int], max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length).unsqueeze(0)
    return positions < torch.tensor(lengths).unsqueeze(1)


def test_reference_batch_uses_uniform_float_weights():
    tracker = HorizonInvariantOPDTracker({"enable": True, "bin_size": 4, "reference_steps": 1}, max_response_length=8)
    mask = _response_mask([8, 4], max_length=8)

    weights, metrics = tracker.compute(mask)

    torch.testing.assert_close(weights, mask.float())
    assert weights.dtype == torch.float32
    assert metrics["horizon_opd/reference_ready"] == 1.0
    assert metrics["horizon_opd/correction_active"] == 0.0
    torch.testing.assert_close(tracker.reference_mass, torch.tensor([2 / 3, 1 / 3], dtype=torch.float64))


def test_importance_weights_restore_reference_bin_mass():
    tracker = HorizonInvariantOPDTracker(
        {
            "enable": True,
            "bin_size": 4,
            "reference_steps": 1,
            "alpha": 1.0,
            "min_weight": 0.01,
            "max_weight": 100.0,
        },
        max_response_length=8,
    )
    tracker.compute(_response_mask([8, 4], max_length=8))
    current_mask = _response_mask([8, 8], max_length=8)

    weights, metrics = tracker.compute(current_mask)
    normalized = normalize_horizon_weights_for_loss(weights, current_mask, "seq-mean-token-sum")
    weighted_bin_mass = torch.stack([normalized[:, :4].sum(), normalized[:, 4:].sum()])
    weighted_bin_mass /= weighted_bin_mass.sum()

    torch.testing.assert_close(weighted_bin_mass, torch.tensor([2 / 3, 1 / 3]))
    assert metrics["horizon_opd/correction_active"] == 1.0
    assert metrics["horizon_opd/corrected_mass_tv"] == pytest.approx(0.0, abs=1e-6)


def test_alpha_zero_disables_correction():
    tracker = HorizonInvariantOPDTracker(
        {"enable": True, "bin_size": 4, "reference_steps": 1, "alpha": 0.0}, max_response_length=8
    )
    tracker.compute(_response_mask([8, 4], max_length=8))
    current_mask = _response_mask([8, 8], max_length=8)

    weights, _ = tracker.compute(current_mask)

    torch.testing.assert_close(weights, current_mask.float())


def test_weight_clipping_is_applied():
    tracker = HorizonInvariantOPDTracker(
        {
            "enable": True,
            "bin_size": 4,
            "reference_steps": 1,
            "min_weight": 0.8,
            "max_weight": 1.2,
        },
        max_response_length=8,
    )
    tracker.compute(_response_mask([8, 4], max_length=8))

    weights, _ = tracker.compute(_response_mask([8, 8], max_length=8))
    valid_weights = weights[weights > 0]

    assert valid_weights.max().item() == pytest.approx(1.2)
    assert valid_weights.min().item() == pytest.approx(0.8)


def test_uniform_seq_sum_normalization_matches_token_mean():
    mask = _response_mask([4, 2], max_length=4)
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 0.0, 0.0]])
    weights = normalize_horizon_weights_for_loss(mask.float(), mask, "seq-mean-token-sum")

    weighted_seq_mean = (losses * weights).sum(dim=-1).mean()
    token_mean = (losses * mask).sum() / mask.sum()

    torch.testing.assert_close(weighted_seq_mean, token_mean)


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="alpha"):
        HorizonInvariantOPDTracker({"alpha": 1.1}, max_response_length=8)

    tracker = HorizonInvariantOPDTracker({"bin_size": 4}, max_response_length=8)
    with pytest.raises(ValueError, match="no valid response tokens"):
        tracker.compute(torch.zeros(2, 8, dtype=torch.bool))

    with pytest.raises(ValueError, match="must match response_mask"):
        normalize_horizon_weights_for_loss(torch.ones(2, 7), torch.ones(2, 8), "seq-mean-token-sum")
