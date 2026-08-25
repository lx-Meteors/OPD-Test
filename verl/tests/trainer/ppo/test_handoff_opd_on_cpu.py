# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.handoff_opd import select_best_verified_handoff_candidate


def test_select_best_verified_handoff_candidate_restores_parent_batch():
    parent_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 0])
    handoff_mask = torch.tensor([True] * 8 + [False, False])
    reward_scores = torch.tensor([[0.6], [0.9], [0.8], [0.2], [0.1], [0.4], [0.2], [0.3], [1.0], [1.0]])

    opd_mask = torch.zeros(10, 4)
    opd_mask[0, :2] = 1
    opd_mask[4, :3] = 1
    opd_mask[8, :4] = 1

    ce_mask = torch.zeros(10, 4)
    ce_mask[:8, 2:] = 1

    data = DataProto.from_dict(
        tensors={
            "row_id": torch.arange(10),
            "handoff_parent_index": parent_indices,
            "handoff_opd_mask": opd_mask,
            "handoff_ce_mask": ce_mask,
            "handoff_mask": handoff_mask,
        },
        non_tensors={"label": np.array([f"row-{index}" for index in range(10)], dtype=object)},
    )

    selected, stats = select_best_verified_handoff_candidate(
        data,
        reward_scores=reward_scores,
        correct_threshold=0.5,
    )

    assert len(selected) == 3
    assert selected.batch["row_id"].tolist() == [1, 4, 8]
    assert selected.non_tensor_batch["label"].tolist() == ["row-1", "row-4", "row-8"]
    assert "handoff_parent_index" not in selected.batch
    assert selected.batch["handoff_opd_mask"][0].tolist() == [1, 1, 0, 0]
    assert selected.batch["handoff_ce_mask"][1].sum().item() == 0
    assert selected.batch["handoff_opd_mask"][2].tolist() == [1, 1, 1, 1]
    assert selected.batch["handoff_ce_mask"][2].sum().item() == 0
    assert stats == {
        "parent_count": 3,
        "handoff_parent_count": 2,
        "candidate_handoff_count": 8,
        "candidate_correct_count": 3,
        "selected_correct_count": 1,
    }


def test_select_best_verified_handoff_candidate_breaks_reward_ties_stably():
    data = DataProto.from_dict(
        tensors={
            "row_id": torch.arange(4),
            "handoff_parent_index": torch.zeros(4, dtype=torch.long),
            "handoff_opd_mask": torch.tensor([[1, 1], [0, 0], [0, 0], [0, 0]]),
            "handoff_ce_mask": torch.ones(4, 2),
            "handoff_mask": torch.ones(4, dtype=torch.bool),
        }
    )

    selected, _ = select_best_verified_handoff_candidate(
        data,
        reward_scores=torch.ones(4),
        correct_threshold=0.5,
    )

    assert selected.batch["row_id"].item() == 0
    assert selected.batch["handoff_opd_mask"].tolist() == [[1, 1]]
    assert selected.batch["handoff_ce_mask"].tolist() == [[1, 1]]
