# Copyright 2026
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    build_set_opd_teacher_policy_features,
    compute_set_opd_advantage,
    compute_set_opd_sequence_advantages,
)


def test_set_opd_rewards_unique_correct_trajectory_over_duplicate_correct_trajectories():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    correctness = torch.tensor([1.0, 1.0, 1.0, 0.0])
    group_ids = np.array(["same-prompt"] * 4, dtype=object)

    advantage, raw_score = compute_set_opd_sequence_advantages(
        features,
        correctness,
        group_ids,
        normalize_by_std=False,
    )

    assert raw_score[2] > raw_score[0]
    assert raw_score[0] == raw_score[1]
    assert advantage[2] > advantage[0]
    assert torch.allclose(advantage.sum(), torch.tensor(0.0), atol=1e-6)


def test_set_opd_has_no_set_signal_when_the_whole_group_is_incorrect():
    features = torch.eye(4)
    correctness = torch.zeros(4)
    group_ids = np.array(["same-prompt"] * 4, dtype=object)

    advantage, raw_score = compute_set_opd_sequence_advantages(features, correctness, group_ids)

    assert torch.count_nonzero(advantage) == 0
    assert torch.count_nonzero(raw_score) == 0


def test_set_opd_estimator_keeps_topk_opd_and_returns_separate_2d_set_advantage():
    batch_size, response_length, top_k = 4, 5, 3
    token_rewards = torch.full((batch_size, response_length, top_k), -0.01)
    response_mask = torch.ones(batch_size, response_length)
    teacher_ids = torch.arange(
        batch_size * response_length * top_k, dtype=torch.long
    ).reshape(batch_size, response_length, top_k)
    teacher_log_probs = torch.log_softmax(
        torch.tensor([2.0, 1.0, 0.0]).expand(batch_size, response_length, top_k), dim=-1
    )
    true_reward = torch.zeros(batch_size, response_length)
    true_reward[0, -1] = 1.0
    true_reward[1, -1] = 1.0
    group_ids = np.array(["same-prompt"] * 4, dtype=object)
    config = SimpleNamespace(
        set_opd_weight=0.05,
        set_opd_feature_dim=16,
        set_opd_position_bins=4,
        set_opd_max_positions=5,
        set_opd_logdet_scale=1.0,
        set_opd_quality_weight=1.0,
        set_opd_diversity_weight=1.0,
        set_opd_normalize_by_std=True,
        set_opd_correct_threshold=0.5,
    )

    advantages, returns, extras = compute_set_opd_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_ids,
        true_reward_score=true_reward,
        teacher_top_k_ids=teacher_ids,
        teacher_top_k_log_probs=teacher_log_probs,
        config=config,
    )

    assert advantages.shape == token_rewards.shape
    assert returns.shape == token_rewards.shape
    assert torch.equal(advantages, token_rewards)
    assert extras["set_opd_advantages"].shape == response_mask.shape
    assert extras["set_opd_sequence_advantage"].shape == (batch_size,)
    assert torch.isfinite(extras["set_opd_advantages"]).all()


def test_teacher_policy_feature_builder_is_deterministic_and_normalized():
    teacher_ids = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
        ]
    )
    teacher_log_probs = torch.log_softmax(torch.ones(2, 3, 2), dim=-1)
    response_mask = torch.ones(2, 3)

    features = build_set_opd_teacher_policy_features(
        teacher_ids,
        teacher_log_probs,
        response_mask,
        feature_dim=16,
        position_bins=4,
        max_positions=3,
    )

    assert torch.allclose(features[0], features[1])
    assert torch.allclose(features.norm(dim=-1), torch.ones(2), atol=1e-6)
