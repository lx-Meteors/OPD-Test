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

"""Exact last-layer attention distillation without materializing model attentions.

The module captures the final Qwen2 attention layer's query/key projections and
reconstructs its post-RoPE attention probabilities. Attention heads are averaged
within each model before the student-to-teacher reverse KL is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class LastAttentionQK:
    """Post-RoPE query/key states required to reconstruct last-layer attention."""

    query: torch.Tensor
    key: torch.Tensor
    scaling: float
    num_key_value_groups: int


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def _find_qwen2_attention_and_rotary(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    attention_candidates = []
    rotary_candidates = []

    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "layer_idx"):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                attention_candidates.append((int(layer_idx), name, module))
        if name.endswith("rotary_emb"):
            rotary_candidates.append((name, module))

    if not attention_candidates:
        raise ValueError("Could not find a decoder attention layer with q_proj/k_proj")
    if not rotary_candidates:
        raise ValueError("Could not find the model rotary embedding module")

    _, _, attention = max(attention_candidates, key=lambda item: item[0])
    model_type = getattr(getattr(attention, "config", None), "model_type", None)
    if model_type != "qwen2":
        raise NotImplementedError(
            "Full attention distillation currently reconstructs Qwen2 attention only; "
            f"found model_type={model_type!r}"
        )

    # Qwen2 has one shared rotary embedding module. Prefer the shortest matching
    # name in case a wrapper exposes the same module through multiple paths.
    _, rotary = min(rotary_candidates, key=lambda item: (item[0].count("."), len(item[0])))
    return attention, rotary


class LastAttentionQKCapture:
    """Capture and reconstruct the final Qwen2 layer's post-RoPE Q/K tensors."""

    def __init__(self, model: nn.Module):
        self.attention, self.rotary = _find_qwen2_attention_and_rotary(model)
        self._query_projection = None
        self._key_projection = None
        self._cos = None
        self._sin = None
        self._handles = []

    def __enter__(self):
        self._handles = [
            self.attention.q_proj.register_forward_hook(self._capture_query),
            self.attention.k_proj.register_forward_hook(self._capture_key),
            self.rotary.register_forward_hook(self._capture_rotary),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _capture_query(self, _module, _inputs, output):
        self._query_projection = output

    def _capture_key(self, _module, _inputs, output):
        self._key_projection = output

    def _capture_rotary(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("Expected Qwen2 rotary_emb to return (cos, sin)")
        self._cos, self._sin = output

    def build(self, detach: bool = False) -> LastAttentionQK:
        if self._query_projection is None or self._key_projection is None:
            raise RuntimeError("The final attention projections were not executed during the captured forward pass")
        if self._cos is None or self._sin is None:
            raise RuntimeError("The rotary embedding module was not executed during the captured forward pass")

        config = self.attention.config
        num_heads = int(config.num_attention_heads)
        num_key_value_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(self.attention, "head_dim", config.hidden_size // num_heads))

        query = self._query_projection
        key = self._key_projection
        expected_query_width = num_heads * head_dim
        expected_key_width = num_key_value_heads * head_dim
        if query.shape[-1] != expected_query_width or key.shape[-1] != expected_key_width:
            raise ValueError(
                "Unexpected Qwen2 projection widths: "
                f"query={query.shape[-1]} (expected {expected_query_width}), "
                f"key={key.shape[-1]} (expected {expected_key_width})"
            )

        query = query.view(*query.shape[:-1], num_heads, head_dim).transpose(1, 2)
        key = key.view(*key.shape[:-1], num_key_value_heads, head_dim).transpose(1, 2)
        cos = self._cos.unsqueeze(1)
        sin = self._sin.unsqueeze(1)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        if detach:
            query = query.detach()
            key = key.detach()

        scaling = float(getattr(self.attention, "scaling", head_dim**-0.5))
        return LastAttentionQK(
            query=query,
            key=key,
            scaling=scaling,
            num_key_value_groups=num_heads // num_key_value_heads,
        )


def forward_with_last_attention_qk(
    model: nn.Module, *, detach: bool = False, **model_inputs
) -> tuple[object, LastAttentionQK]:
    """Run a model forward and return its output plus reconstructed final Q/K."""

    with LastAttentionQKCapture(model) as capture:
        output = model(**model_inputs)
    return output, capture.build(detach=detach)


def _repeat_key_value(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, groups, sequence_length, head_dim)
    return hidden_states.reshape(num_key_value_heads * groups, sequence_length, head_dim)


def _mean_head_log_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Return log of the head-averaged causal attention distribution."""

    key = _repeat_key_value(key, num_key_value_groups)
    if query.shape[0] != key.shape[0]:
        raise ValueError(f"Query/key head mismatch after GQA expansion: {query.shape[0]} != {key.shape[0]}")

    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    key_positions = torch.arange(key.shape[-2], device=query.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    head_log_probs = torch.log_softmax(scores, dim=-1, dtype=torch.float32)
    return torch.logsumexp(head_log_probs, dim=0) - torch.log(
        torch.tensor(query.shape[0], dtype=torch.float32, device=query.device)
    )


def _split_valid_sequences(
    states: LastAttentionQK, attention_mask: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert padded or remove-padding Q/K states into per-sample valid sequences."""

    batch_size, padded_length = attention_mask.shape
    query, key = states.query, states.key
    valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()

    if query.shape[0] == batch_size and query.shape[-2] == padded_length:
        sequences = []
        for batch_idx in range(batch_size):
            valid = attention_mask[batch_idx].bool()
            sequences.append((query[batch_idx][:, valid, :], key[batch_idx][:, valid, :]))
        return sequences

    total_valid = sum(valid_lengths)
    if query.shape[0] == 1 and query.shape[-2] == total_valid:
        sequences = []
        offset = 0
        for length in valid_lengths:
            sequences.append(
                (
                    query[0, :, offset : offset + length, :],
                    key[0, :, offset : offset + length, :],
                )
            )
            offset += length
        return sequences

    raise ValueError(
        "Cannot align captured attention states with attention_mask: "
        f"Q shape={tuple(query.shape)}, mask shape={tuple(attention_mask.shape)}, total_valid={total_valid}"
    )


def _attention_reverse_kl_block(
    student_query: torch.Tensor,
    student_key: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_key: torch.Tensor,
    query_positions: torch.Tensor,
    student_scaling: float,
    teacher_scaling: float,
    student_key_value_groups: int,
    teacher_key_value_groups: int,
) -> torch.Tensor:
    student_log_attention = _mean_head_log_attention(
        student_query,
        student_key,
        query_positions,
        student_scaling,
        student_key_value_groups,
    )
    teacher_log_attention = _mean_head_log_attention(
        teacher_query,
        teacher_key,
        query_positions,
        teacher_scaling,
        teacher_key_value_groups,
    )
    student_attention = student_log_attention.exp()
    token_terms = student_attention * (student_log_attention - teacher_log_attention)
    token_terms = torch.where(torch.isfinite(student_log_attention), token_terms, torch.zeros_like(token_terms))
    return token_terms.sum(dim=-1)


def compute_full_attention_rkl_token_loss(
    student: LastAttentionQK,
    teacher: LastAttentionQK,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    query_chunk_size: int,
) -> torch.Tensor:
    """Compute exact response-query attention RKL as a response-shaped tensor.

    Every valid response token is used as a query. Prompt queries are omitted, while
    valid prompt tokens remain in the causal key set. Chunking changes only peak
    memory use and does not subsample queries or keys.
    """

    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if attention_mask.shape[0] != response_mask.shape[0]:
        raise ValueError("attention_mask and response_mask batch sizes must match")

    student_sequences = _split_valid_sequences(student, attention_mask)
    teacher_sequences = _split_valid_sequences(teacher, attention_mask)
    response_length = response_mask.shape[-1]
    loss_rows = []

    for batch_idx, ((student_query, student_key), (teacher_query, teacher_key)) in enumerate(
        zip(student_sequences, teacher_sequences, strict=True)
    ):
        if student_query.shape[-2] != teacher_query.shape[-2]:
            raise ValueError(
                f"Student/teacher valid sequence lengths differ for sample {batch_idx}: "
                f"{student_query.shape[-2]} != {teacher_query.shape[-2]}"
            )

        response_columns = torch.nonzero(response_mask[batch_idx].bool(), as_tuple=False).flatten()
        num_response_tokens = response_columns.numel()
        sequence_length = student_query.shape[-2]
        if num_response_tokens > sequence_length:
            raise ValueError(
                f"Response has {num_response_tokens} valid tokens but sequence has length {sequence_length}"
            )

        query_positions = torch.arange(
            sequence_length - num_response_tokens,
            sequence_length,
            device=student_query.device,
            dtype=torch.long,
        )
        block_losses = []
        for start in range(0, num_response_tokens, query_chunk_size):
            block_positions = query_positions[start : start + query_chunk_size]
            student_query_block = student_query[:, block_positions, :]
            teacher_query_block = teacher_query[:, block_positions, :]

            def loss_block(sq, sk, tq, tk, positions):
                return _attention_reverse_kl_block(
                    sq,
                    sk,
                    tq,
                    tk,
                    positions,
                    student.scaling,
                    teacher.scaling,
                    student.num_key_value_groups,
                    teacher.num_key_value_groups,
                )

            block_losses.append(
                checkpoint(
                    loss_block,
                    student_query_block,
                    student_key,
                    teacher_query_block,
                    teacher_key,
                    block_positions,
                    use_reentrant=False,
                )
            )

        if block_losses:
            valid_response_loss = torch.cat(block_losses)
        else:
            valid_response_loss = student_query.sum().reshape(1)[:0].to(dtype=torch.float32)

        loss_row = torch.zeros(response_length, dtype=valid_response_loss.dtype, device=student_query.device)
        loss_rows.append(loss_row.scatter(0, response_columns, valid_response_loss))

    return torch.stack(loss_rows)
