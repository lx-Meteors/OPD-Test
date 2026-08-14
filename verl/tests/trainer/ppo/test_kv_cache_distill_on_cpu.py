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

from types import SimpleNamespace

import torch
from torch import nn

from verl.trainer.ppo.kv_cache_distill import (
    KVCachedStates,
    LayerKVStates,
    compute_kv_cosine_token_loss,
    forward_with_kv_cached_states,
)


def _states(key, value, layer_idx=0):
    return KVCachedStates(layers=(LayerKVStates(layer_idx=layer_idx, key=key, value=value),))


class _DummyQwen2Attention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )
        self.head_dim = 2
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)

    def forward(self, hidden_states):
        return self.k_proj(hidden_states) + self.v_proj(hidden_states)


class _DummyQwen2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_DummyQwen2Attention(idx) for idx in range(3)])

    def forward(self, hidden_states):
        return tuple(layer(hidden_states) for layer in self.layers)


def test_capture_selects_layers_and_reshapes_pre_rope_kv():
    model = _DummyQwen2Model()
    hidden_states = torch.randn(2, 5, 8, requires_grad=True)
    output, states = forward_with_kv_cached_states(
        model,
        layer_indices=[0, -1],
        hidden_states=hidden_states,
    )

    assert len(output) == 3
    assert [layer.layer_idx for layer in states.layers] == [0, 2]
    assert states.layers[0].key.shape == (2, 2, 5, 2)
    assert states.layers[0].value.shape == (2, 2, 5, 2)
    (states.layers[0].key.sum() + states.layers[1].value.sum()).backward()
    assert hidden_states.grad is not None


def test_identical_kv_has_zero_loss_and_student_gradient():
    torch.manual_seed(0)
    key = torch.randn(1, 2, 6, 4, requires_grad=True)
    value = torch.randn(1, 2, 6, 4, requires_grad=True)
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    loss = compute_kv_cosine_token_loss(
        _states(key, value),
        _states(key.detach().clone(), value.detach().clone()),
        attention_mask,
        response_mask,
        token_chunk_size=2,
    )

    torch.testing.assert_close(loss.key, torch.zeros_like(loss.key), atol=1e-6, rtol=0)
    torch.testing.assert_close(loss.value, torch.zeros_like(loss.value), atol=1e-6, rtol=0)
    (loss.key.sum() + loss.value.sum()).backward()
    assert key.grad is not None
    assert value.grad is not None


def test_all_scope_supervises_prompt_while_response_scope_omits_it():
    torch.manual_seed(1)
    student_key = torch.randn(1, 2, 5, 3, requires_grad=True)
    student_value = torch.randn(1, 2, 5, 3, requires_grad=True)
    teacher_key = student_key.detach().clone()
    teacher_value = student_value.detach().clone()
    teacher_key[:, :, 0] *= -1
    teacher_value[:, :, 1] *= -1
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    response_mask = torch.tensor([[1, 1]], dtype=torch.long)

    all_loss = compute_kv_cosine_token_loss(
        _states(student_key, student_value),
        _states(teacher_key, teacher_value),
        attention_mask,
        response_mask,
        token_scope="all",
    )
    assert all_loss.key[0, 0] > 0
    assert all_loss.value[0, 1] > 0

    response_loss = compute_kv_cosine_token_loss(
        _states(student_key, student_value),
        _states(teacher_key, teacher_value),
        attention_mask,
        response_mask,
        token_scope="response",
    )
    torch.testing.assert_close(response_loss.key, torch.zeros_like(response_loss.key), atol=1e-6, rtol=0)
    torch.testing.assert_close(response_loss.value, torch.zeros_like(response_loss.value), atol=1e-6, rtol=0)


def test_remove_padding_layout_matches_padded_layout_for_multiple_samples():
    torch.manual_seed(2)
    student_key = torch.randn(2, 2, 6, 3, requires_grad=True)
    student_value = torch.randn(2, 2, 6, 3, requires_grad=True)
    teacher_key = student_key.detach().clone()
    teacher_value = student_value.detach().clone()
    teacher_key[0, :, 1] *= -1
    teacher_value[1, :, 3] *= -1
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    padded_loss = compute_kv_cosine_token_loss(
        _states(student_key, student_value),
        _states(teacher_key, teacher_value),
        attention_mask,
        response_mask,
    )

    packed_student_key = torch.cat(
        [student_key[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_student_value = torch.cat(
        [student_value[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_teacher_key = torch.cat(
        [teacher_key[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_teacher_value = torch.cat(
        [teacher_value[idx][:, attention_mask[idx].bool(), :] for idx in range(2)], dim=1
    ).unsqueeze(0)
    packed_loss = compute_kv_cosine_token_loss(
        _states(packed_student_key, packed_student_value),
        _states(packed_teacher_key, packed_teacher_value),
        attention_mask,
        response_mask,
    )

    torch.testing.assert_close(packed_loss.key, padded_loss.key)
    torch.testing.assert_close(packed_loss.value, padded_loss.value)


def test_multiple_layers_are_averaged():
    key = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    value = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
    opposite_key = -key.detach()
    opposite_value = -value.detach()
    student = KVCachedStates(
        layers=(
            LayerKVStates(0, key, value),
            LayerKVStates(1, key, value),
        )
    )
    teacher = KVCachedStates(
        layers=(
            LayerKVStates(0, key.detach().clone(), value.detach().clone()),
            LayerKVStates(1, opposite_key, opposite_value),
        )
    )

    loss = compute_kv_cosine_token_loss(
        student,
        teacher,
        torch.ones(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.long),
    )
    torch.testing.assert_close(loss.key, torch.ones_like(loss.key))
    torch.testing.assert_close(loss.value, torch.ones_like(loss.value))


def test_response_scope_handles_sample_without_valid_response_tokens():
    key = torch.randn(1, 2, 3, 4, requires_grad=True)
    value = torch.randn(1, 2, 3, 4, requires_grad=True)
    loss = compute_kv_cosine_token_loss(
        _states(key, value),
        _states(key.detach().clone(), value.detach().clone()),
        torch.ones(1, 3, dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.long),
        token_scope="response",
    )
    assert loss.key.shape == (1, 2)
    assert loss.value.shape == (1, 2)
    torch.testing.assert_close(loss.key, torch.zeros_like(loss.key))
    torch.testing.assert_close(loss.value, torch.zeros_like(loss.value))
