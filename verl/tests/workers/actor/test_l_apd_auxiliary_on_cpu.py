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

import torch

from verl.trainer.ppo.l_apd import compute_l_apd_token_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_l_apd_reuses_precomputed_policy_log_probs():
    actor = object.__new__(DataParallelPPOActor)
    actor.l_apd_config = {
        "candidate_source": "student",
        "tail_candidate": False,
        "complement_candidate": True,
        "normalize_weights": True,
        "pair_divergence": "reverse_kl",
    }

    def fail_if_forwarded(*args, **kwargs):
        raise AssertionError("auxiliary L-APD must reuse the OPD policy forward")

    actor._forward_micro_batch = fail_if_forwarded

    responses = torch.tensor([[7, 8]])
    candidate_ids = torch.tensor([[[1, 2], [3, 4]]])
    response_mask = torch.ones(1, 2)
    student_anchor = torch.log(torch.tensor([[0.40, 0.30]], requires_grad=True))
    student_candidates = torch.log(
        torch.tensor([[[0.20, 0.10], [0.25, 0.15]]], requires_grad=True)
    )
    student_anchor.retain_grad()
    student_candidates.retain_grad()
    teacher_anchor = torch.log(torch.tensor([[0.50, 0.35]]))
    teacher_candidates = torch.log(torch.tensor([[[0.15, 0.05], [0.20, 0.10]]]))

    model_inputs = {
        "responses": responses,
        "student_top_k_ids": candidate_ids,
        "teacher_on_student_log_probs": teacher_candidates,
        "teacher_log_probs": teacher_anchor,
    }

    entropy, returned_anchor, loss, metrics = actor._compute_l_apd_loss(
        model_inputs=model_inputs,
        response_mask=response_mask,
        temperature=1.0,
        calculate_entropy=False,
        loss_agg_mode="token-mean",
        student_anchor_log_probs=student_anchor,
        student_candidate_log_probs=student_candidates,
        entropy=None,
    )

    expected_token_loss, _ = compute_l_apd_token_loss(
        student_anchor_log_probs=student_anchor,
        student_candidate_log_probs=student_candidates,
        teacher_anchor_log_probs=teacher_anchor,
        teacher_candidate_log_probs=teacher_candidates,
        candidate_mask=candidate_ids != responses.unsqueeze(-1),
        response_mask=response_mask,
        tail_candidate=False,
        complement_candidate=True,
        normalize_weights=True,
        pair_divergence="reverse_kl",
    )
    expected_loss = expected_token_loss.sum() / response_mask.sum()

    assert entropy is None
    assert returned_anchor is student_anchor
    torch.testing.assert_close(loss, expected_loss)
    assert "actor/l_apd_pair_kl" in metrics

    loss.backward()
    assert student_anchor.grad is not None
    assert student_candidates.grad is not None
