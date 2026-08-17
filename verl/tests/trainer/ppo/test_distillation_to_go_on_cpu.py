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

from verl.utils.distillation_to_go import compute_distillation_to_go_advantages


def _response_mask(lengths: list[int], max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length).unsqueeze(0)
    return positions < torch.tensor(lengths).unsqueeze(1)


def _terminal_rewards(scores: list[float], max_length: int) -> torch.Tensor:
    rewards = torch.zeros(len(scores), max_length)
    rewards[:, -1] = torch.tensor(scores)
    return rewards


def test_future_disagreement_rewards_better_sibling_trajectories():
    mask = _response_mask([8, 8, 8, 8], max_length=8)
    local_costs = torch.tensor(
        [
            [0.1] * 8,
            [0.1] * 8,
            [0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ]
    )

    advantages, metrics = compute_distillation_to_go_advantages(
        response_mask=mask,
        group_ids=["prompt-1"] * 4,
        config={"block_size": 2, "block_gamma": 1.0, "outcome_weight": 0.0},
        token_level_rewards=local_costs,
    )

    assert advantages[0, :2].mean().item() > 0
    assert advantages[1, :2].mean().item() > 0
    assert advantages[2, :2].mean().item() < 0
    assert advantages[3, :2].mean().item() < 0
    assert metrics["dtg/local_disagreement_mean"] > 0


def test_terminal_outcome_prevents_incorrect_early_exit_loophole():
    mask = _response_mask([8, 8, 8, 8], max_length=8)
    local_costs = torch.zeros(4, 8)
    true_rewards = _terminal_rewards([1.0, 1.0, 0.0, 0.0], max_length=8)

    advantages, _ = compute_distillation_to_go_advantages(
        response_mask=mask,
        group_ids=["prompt-1"] * 4,
        config={"block_size": 2, "outcome_weight": 0.25},
        true_reward_score=true_rewards,
        token_level_rewards=local_costs,
    )

    assert advantages[:2].mean().item() > 0
    assert advantages[2:].mean().item() < 0


def test_padding_and_single_survivor_have_zero_advantage():
    mask = _response_mask([8, 6, 4, 2], max_length=8)
    local_costs = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32

    advantages, _ = compute_distillation_to_go_advantages(
        response_mask=mask,
        group_ids=["prompt-1"] * 4,
        config={"block_size": 2, "outcome_weight": 0.0},
        token_level_rewards=local_costs,
    )

    torch.testing.assert_close(advantages * (~mask), torch.zeros_like(advantages))
    # Only the longest trajectory reaches the final block, so it has no peer
    # against which to estimate a counterfactual future cost.
    torch.testing.assert_close(advantages[0, 6:8], torch.zeros(2))


def test_topk_total_variation_is_used_when_log_probs_are_available():
    mask = _response_mask([4, 4], max_length=4)
    student_probs = torch.tensor([0.7, 0.2]).view(1, 1, 2).expand(2, 4, 2)
    teacher_probs = torch.tensor([0.4, 0.4]).view(1, 1, 2).expand(2, 4, 2)

    _, metrics = compute_distillation_to_go_advantages(
        response_mask=mask,
        group_ids=["prompt-1", "prompt-1"],
        config={"block_size": 2, "outcome_weight": 0.0},
        student_top_k_log_probs=student_probs.log(),
        teacher_on_student_log_probs=teacher_probs.log(),
    )

    assert metrics["dtg/disagreement_source_topk_tv"] == 1.0
    assert metrics["dtg/local_disagreement_mean"] == pytest.approx(0.25)


def test_groups_without_siblings_produce_no_dtg_update():
    mask = _response_mask([4, 4], max_length=4)
    advantages, metrics = compute_distillation_to_go_advantages(
        response_mask=mask,
        group_ids=["prompt-1", "prompt-2"],
        config={"block_size": 2},
        token_level_rewards=torch.ones(2, 4),
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    assert metrics["dtg/valid_comparison_ratio"] == 0.0


def test_invalid_configuration_and_shapes_raise_clear_errors():
    mask = _response_mask([4, 4], max_length=4)
    with pytest.raises(ValueError, match="block_gamma"):
        compute_distillation_to_go_advantages(
            response_mask=mask,
            group_ids=["prompt-1", "prompt-1"],
            config={"block_gamma": 1.1},
            token_level_rewards=torch.ones(2, 4),
        )

    with pytest.raises(ValueError, match="one group id per response"):
        compute_distillation_to_go_advantages(
            response_mask=mask,
            group_ids=["prompt-1"],
            config={},
            token_level_rewards=torch.ones(2, 4),
        )
