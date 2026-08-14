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

"""Direct Qwen2 KV-cache distillation on aligned teacher/student layers.

Keys are captured before RoPE so the loss does not mix content alignment with
position rotation. Values are captured directly from ``v_proj``. The intended
use is a same-architecture teacher/student pair whose internal coordinate
systems are already aligned (for example, two checkpoints derived from the same
base model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LayerKVStates:
    """Pre-RoPE key/value states from one decoder layer."""

    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class KVCachedStates:
    """Ordered KV states from all configured decoder layers."""

    layers: tuple[LayerKVStates, ...]


@dataclass
class KVTokenLoss:
    """Per-token key and value cosine distances."""

    key: torch.Tensor
    value: torch.Tensor


def _find_qwen2_attention_layers(model: nn.Module) -> list[tuple[int, nn.Module]]:
    candidates: dict[int, nn.Module] = {}
    for _, module in model.named_modules():
        if not all(hasattr(module, name) for name in ("k_proj", "v_proj", "layer_idx")):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            continue
        model_type = getattr(getattr(module, "config", None), "model_type", None)
        if model_type != "qwen2":
            continue
        layer_idx = int(layer_idx)
        previous = candidates.get(layer_idx)
        if previous is not None and previous is not module:
            raise ValueError(f"Found multiple Qwen2 attention modules for layer {layer_idx}")
        candidates[layer_idx] = module

    if not candidates:
        raise ValueError("Could not find Qwen2 decoder attention layers with k_proj/v_proj")
    return sorted(candidates.items())


def _resolve_layer_positions(num_layers: int, layer_indices: Sequence[int]) -> list[int]:
    if not layer_indices:
        raise ValueError("kv_cache_distill.layer_indices must contain at least one layer")

    resolved = []
    for requested_idx in layer_indices:
        position = int(requested_idx)
        if position < 0:
            position += num_layers
        if position < 0 or position >= num_layers:
            raise ValueError(
                f"KV distillation layer index {requested_idx} is out of range for a {num_layers}-layer model"
            )
        if position in resolved:
            raise ValueError(f"KV distillation layer index {requested_idx} selects a duplicate layer")
        resolved.append(position)
    return resolved


class KVCachedStatesCapture:
    """Capture pre-RoPE K/V projections from selected Qwen2 layers."""

    def __init__(self, model: nn.Module, layer_indices: Sequence[int]):
        all_layers = _find_qwen2_attention_layers(model)
        positions = _resolve_layer_positions(len(all_layers), layer_indices)
        self.attention_layers = [all_layers[position] for position in positions]
        self._key_projections: dict[int, torch.Tensor] = {}
        self._value_projections: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for layer_idx, attention in self.attention_layers:
            self._handles.extend(
                [
                    attention.k_proj.register_forward_hook(self._capture_key(layer_idx)),
                    attention.v_proj.register_forward_hook(self._capture_value(layer_idx)),
                ]
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _capture_key(self, layer_idx: int):
        def hook(_module, _inputs, output):
            self._key_projections[layer_idx] = output

        return hook

    def _capture_value(self, layer_idx: int):
        def hook(_module, _inputs, output):
            self._value_projections[layer_idx] = output

        return hook

    @staticmethod
    def _reshape_projection(projection: torch.Tensor, attention: nn.Module, name: str) -> torch.Tensor:
        config = attention.config
        num_heads = int(config.num_attention_heads)
        num_key_value_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(attention, "head_dim", config.hidden_size // num_heads))
        expected_width = num_key_value_heads * head_dim
        if projection.shape[-1] != expected_width:
            raise ValueError(
                f"Unexpected Qwen2 {name} projection width {projection.shape[-1]}; expected {expected_width}"
            )
        return projection.view(*projection.shape[:-1], num_key_value_heads, head_dim).transpose(1, 2)

    def build(self, detach: bool = False) -> KVCachedStates:
        layers = []
        for layer_idx, attention in self.attention_layers:
            if layer_idx not in self._key_projections or layer_idx not in self._value_projections:
                raise RuntimeError(f"Qwen2 layer {layer_idx} K/V projections were not executed during capture")
            key = self._reshape_projection(self._key_projections[layer_idx], attention, "key")
            value = self._reshape_projection(self._value_projections[layer_idx], attention, "value")
            if detach:
                key = key.detach()
                value = value.detach()
            layers.append(LayerKVStates(layer_idx=layer_idx, key=key, value=value))
        return KVCachedStates(layers=tuple(layers))


def forward_with_kv_cached_states(
    model: nn.Module, *, layer_indices: Sequence[int], detach: bool = False, **model_inputs
) -> tuple[object, KVCachedStates]:
    """Run one model forward and return the selected pre-RoPE K/V states."""

    with KVCachedStatesCapture(model, layer_indices) as capture:
        output = model(**model_inputs)
    return output, capture.build(detach=detach)


def _split_valid_sequences(
    layer: LayerKVStates, attention_mask: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert padded or remove-padding KV states into valid per-sample sequences."""

    batch_size, padded_length = attention_mask.shape
    key, value = layer.key, layer.value
    valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()

    if key.shape[0] == batch_size and key.shape[-2] == padded_length:
        sequences = []
        for batch_idx in range(batch_size):
            valid = attention_mask[batch_idx].bool()
            sequences.append((key[batch_idx][:, valid, :], value[batch_idx][:, valid, :]))
        return sequences

    total_valid = sum(valid_lengths)
    if key.shape[0] == 1 and key.shape[-2] == total_valid:
        sequences = []
        offset = 0
        for length in valid_lengths:
            sequences.append(
                (
                    key[0, :, offset : offset + length, :],
                    value[0, :, offset : offset + length, :],
                )
            )
            offset += length
        return sequences

    raise ValueError(
        "Cannot align captured KV states with attention_mask: "
        f"K shape={tuple(key.shape)}, mask shape={tuple(attention_mask.shape)}, total_valid={total_valid}"
    )


def _cosine_token_loss(student: torch.Tensor, teacher: torch.Tensor, token_chunk_size: int) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            "Direct KV distillation requires matching teacher/student KV shapes; "
            f"got student={tuple(student.shape)} and teacher={tuple(teacher.shape)}"
        )

    chunks = []
    for start in range(0, student.shape[-2], token_chunk_size):
        student_chunk = student[:, start : start + token_chunk_size, :].float()
        teacher_chunk = teacher[:, start : start + token_chunk_size, :].float()
        # Average over KV heads, retaining one loss value per sequence position.
        chunks.append((1.0 - F.cosine_similarity(student_chunk, teacher_chunk, dim=-1)).mean(dim=0))
    if chunks:
        return torch.cat(chunks)
    return student.sum().reshape(1)[:0].float()


def compute_kv_cosine_token_loss(
    student: KVCachedStates,
    teacher: KVCachedStates,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    token_scope: str = "all",
    token_chunk_size: int = 1024,
) -> KVTokenLoss:
    """Return layer/head-averaged K/V cosine distance for each supervised token.

    ``token_scope='all'`` supervises every valid prompt and response cache entry
    and returns tensors shaped like ``attention_mask``. ``token_scope='response'``
    supervises only valid response cache entries and returns tensors shaped like
    ``response_mask``.
    """

    if token_scope not in {"all", "response"}:
        raise ValueError("token_scope must be either 'all' or 'response'")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if attention_mask.shape[0] != response_mask.shape[0]:
        raise ValueError("attention_mask and response_mask batch sizes must match")
    if len(student.layers) != len(teacher.layers):
        raise ValueError(
            "Student and teacher must expose the same number of KV distillation layers; "
            f"got {len(student.layers)} and {len(teacher.layers)}"
        )
    if not student.layers:
        raise ValueError("At least one KV layer is required")

    layer_key_losses = []
    layer_value_losses = []
    for layer_position, (student_layer, teacher_layer) in enumerate(
        zip(student.layers, teacher.layers, strict=True)
    ):
        student_sequences = _split_valid_sequences(student_layer, attention_mask)
        teacher_sequences = _split_valid_sequences(teacher_layer, attention_mask)
        key_rows = []
        value_rows = []

        for batch_idx, ((student_key, student_value), (teacher_key, teacher_value)) in enumerate(
            zip(student_sequences, teacher_sequences, strict=True)
        ):
            if student_key.shape[-2] != teacher_key.shape[-2]:
                raise ValueError(
                    "Student/teacher valid sequence lengths differ at selected layer position "
                    f"{layer_position}, sample {batch_idx}: {student_key.shape[-2]} != {teacher_key.shape[-2]}"
                )

            valid_key_loss = _cosine_token_loss(student_key, teacher_key, token_chunk_size)
            valid_value_loss = _cosine_token_loss(student_value, teacher_value, token_chunk_size)

            if token_scope == "all":
                output_columns = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten()
                output_length = attention_mask.shape[-1]
            else:
                output_columns = torch.nonzero(response_mask[batch_idx].bool(), as_tuple=False).flatten()
                output_length = response_mask.shape[-1]
                num_response_tokens = output_columns.numel()
                if num_response_tokens > valid_key_loss.numel():
                    raise ValueError(
                        f"Response has {num_response_tokens} valid tokens but sequence has {valid_key_loss.numel()}"
                    )
                if num_response_tokens > 0:
                    valid_key_loss = valid_key_loss[-num_response_tokens:]
                    valid_value_loss = valid_value_loss[-num_response_tokens:]
                else:
                    valid_key_loss = valid_key_loss[:0]
                    valid_value_loss = valid_value_loss[:0]

            key_row = valid_key_loss.new_zeros(output_length)
            value_row = valid_value_loss.new_zeros(output_length)
            key_rows.append(key_row.scatter(0, output_columns, valid_key_loss))
            value_rows.append(value_row.scatter(0, output_columns, valid_value_loss))

        layer_key_losses.append(torch.stack(key_rows))
        layer_value_losses.append(torch.stack(value_rows))

    return KVTokenLoss(
        key=torch.stack(layer_key_losses).mean(dim=0),
        value=torch.stack(layer_value_losses).mean(dim=0),
    )
