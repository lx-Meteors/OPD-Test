# Copyright 2026
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

from verl.trainer.ppo.attention_distill import LastAttentionQK, compute_full_attention_rkl_token_loss


def _states(query, key):
    return LastAttentionQK(query=query, key=key, scaling=query.shape[-1] ** -0.5, num_key_value_groups=2)


def test_identical_attention_has_zero_loss_and_student_gradient():
    torch.manual_seed(0)
    query = torch.randn(1, 4, 6, 3, requires_grad=True)
    key = torch.randn(1, 2, 6, 3, requires_grad=True)
    teacher_query = query.detach().clone()
    teacher_key = key.detach().clone()
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    loss = compute_full_attention_rkl_token_loss(
        _states(query, key),
        _states(teacher_query, teacher_key),
        attention_mask,
        response_mask,
        query_chunk_size=2,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    loss.sum().backward()
    assert query.grad is not None
    assert key.grad is not None


def test_prompt_queries_are_ignored_but_prompt_keys_remain_context():
    torch.manual_seed(1)
    student_query = torch.randn(1, 4, 5, 3, requires_grad=True)
    student_key = torch.randn(1, 2, 5, 3, requires_grad=True)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    response_mask = torch.tensor([[1, 1]], dtype=torch.long)

    teacher_query = student_query.detach().clone()
    teacher_query[:, :, :3] += 100
    unchanged_key = student_key.detach().clone()
    prompt_query_only_loss = compute_full_attention_rkl_token_loss(
        _states(student_query, student_key),
        _states(teacher_query, unchanged_key),
        attention_mask,
        response_mask,
        query_chunk_size=1,
    )
    torch.testing.assert_close(
        prompt_query_only_loss,
        torch.zeros_like(prompt_query_only_loss),
        atol=1e-6,
        rtol=0,
    )

    changed_prompt_key = unchanged_key.clone()
    changed_prompt_key[:, :, 0] += 4
    prompt_key_loss = compute_full_attention_rkl_token_loss(
        _states(student_query, student_key),
        _states(student_query.detach(), changed_prompt_key),
        attention_mask,
        response_mask,
        query_chunk_size=1,
    )
    assert torch.all(prompt_key_loss > 0)


def test_remove_padding_layout_matches_padded_layout_for_multiple_samples():
    torch.manual_seed(2)
    student_query = torch.randn(2, 4, 6, 3, requires_grad=True)
    student_key = torch.randn(2, 2, 6, 3, requires_grad=True)
    teacher_query = student_query.detach().clone()
    teacher_key = student_key.detach().clone()
    teacher_key[0, :, 1] += 1
    teacher_key[1, :, 2] -= 1
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    padded_loss = compute_full_attention_rkl_token_loss(
        _states(student_query, student_key),
        _states(teacher_query, teacher_key),
        attention_mask,
        response_mask,
        query_chunk_size=2,
    )

    packed_student_query = torch.cat(
        [student_query[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_student_key = torch.cat(
        [student_key[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_teacher_query = torch.cat(
        [teacher_query[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_teacher_key = torch.cat(
        [teacher_key[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_loss = compute_full_attention_rkl_token_loss(
        _states(packed_student_query, packed_student_key),
        _states(packed_teacher_query, packed_teacher_key),
        attention_mask,
        response_mask,
        query_chunk_size=2,
    )

    torch.testing.assert_close(packed_loss, padded_loss)
