# Copyright 2026 Bridge-OPD authors
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
"""CPU tests for Bridge-OPD prefix-state weighting."""

import numpy as np
import torch

from verl.utils.bridge_opd import apply_bridge_opd_to_scores, compute_bridge_opd_weights


def _batch():
    student = torch.zeros(4, 4)
    teacher = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
            [-2.0, -2.0, 0.0, 0.0],
        ]
    )
    mask = torch.ones_like(student)
    uid = np.array(["prompt-0"] * 4, dtype=object)
    return student, teacher, mask, uid


def test_beta_zero_is_exactly_standard_opd():
    student, teacher, mask, uid = _batch()
    scores = torch.randn(4, 4, 3)
    weighted, aux = apply_bridge_opd_to_scores(
        scores, student, teacher, mask, uid, {"beta": 0.0, "adaptive_beta": False}
    )

    torch.testing.assert_close(weighted, scores)
    torch.testing.assert_close(aux["bridge_opd_weights"], torch.ones_like(mask))


def test_current_token_does_not_weight_its_own_state():
    student, teacher, mask, uid = _batch()
    weights, _ = compute_bridge_opd_weights(
        student, teacher, mask, uid, {"beta": 0.2, "adaptive_beta": False}
    )

    # Every response begins at the same prompt state, regardless of y_0.
    torch.testing.assert_close(weights[:, 0], torch.ones(4))
    # At s_1 the first sampled token is part of the prefix, so it can change the state weight.
    assert weights[0, 1] > weights[1, 1] > weights[2, 1] > weights[3, 1]


def test_weights_have_mean_one_per_prompt_and_position():
    student, teacher, mask, uid = _batch()
    mask[-1, -1] = 0
    weights, _ = compute_bridge_opd_weights(
        student, teacher, mask, uid, {"beta": 0.3, "adaptive_beta": False}
    )

    for position in range(mask.shape[1]):
        active = mask[:, position].bool()
        torch.testing.assert_close(weights[active, position].mean(), torch.tensor(1.0))
        assert torch.all(weights[~active, position] == 0)


def test_adaptive_beta_respects_the_ess_floor():
    student, teacher, mask, uid = _batch()
    # Create an extreme occupancy mismatch that would collapse fixed-beta weights.
    teacher[:, 0] = torch.tensor([20.0, 0.0, -10.0, -20.0])
    weights, aux = compute_bridge_opd_weights(
        student,
        teacher,
        mask,
        uid,
        {"beta": 1.0, "adaptive_beta": True, "min_ess_ratio": 0.5, "beta_search_steps": 20},
    )

    assert aux["bridge_opd_ess"][:, 1].min() >= 2.0 - 1e-4
    assert aux["bridge_opd_beta"][:, 1].max() < 1.0
    torch.testing.assert_close(weights[:, 1].mean(), torch.tensor(1.0))


def test_prompt_groups_are_normalized_independently():
    student = torch.zeros(4, 2)
    teacher = torch.tensor([[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0], [-4.0, 0.0]])
    mask = torch.ones_like(student)
    uid = np.array(["a", "a", "b", "b"], dtype=object)

    weights, _ = compute_bridge_opd_weights(
        student, teacher, mask, uid, {"beta": 0.5, "adaptive_beta": False}
    )

    torch.testing.assert_close(weights[:2, 1].mean(), torch.tensor(1.0))
    torch.testing.assert_close(weights[2:, 1].mean(), torch.tensor(1.0))


def test_two_dimensional_scores_are_supported():
    student, teacher, mask, uid = _batch()
    scores = torch.ones(4, 4)
    weighted, aux = apply_bridge_opd_to_scores(
        scores, student, teacher, mask, uid, {"beta": 0.2, "adaptive_beta": False}
    )
    torch.testing.assert_close(weighted, aux["bridge_opd_weights"])
