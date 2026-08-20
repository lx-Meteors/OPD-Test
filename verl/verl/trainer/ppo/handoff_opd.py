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

"""Utilities for prefix-OPD / teacher-continuation SFT handoff training."""

import torch

from verl import DataProto


def expand_handoff_candidates(
    student_output: DataProto,
    cutoff: int,
    eos_token_id: int | None,
    num_teacher_samples: int,
    dispatch_size: int,
) -> DataProto:
    """Repeat only handoff-eligible Student prefixes for Teacher sampling.

    The returned ``handoff_parent_index`` maps each expanded row back to the
    ordinary Student rollout batch.  Only the first Teacher candidate retains
    prefix OPD supervision, preventing ``num_teacher_samples`` identical copies
    of the same prefix from multiplying its OPD weight.
    """

    if num_teacher_samples <= 1:
        student_output.batch["handoff_parent_index"] = torch.arange(
            len(student_output), device=student_output.batch.device, dtype=torch.long
        )
        return student_output
    if dispatch_size <= 0:
        raise ValueError("dispatch_size must be positive.")

    responses = student_output.batch["responses"]
    if responses.shape[-1] != cutoff:
        raise ValueError(f"Expected student response width {cutoff}, got {responses.shape[-1]}.")

    response_attention = student_output.batch["attention_mask"][:, -cutoff:].bool()
    valid_lengths = response_attention.sum(dim=-1)
    handoff_mask = valid_lengths == cutoff
    if eos_token_id is not None:
        last_indices = (valid_lengths - 1).clamp_min(0).long()
        last_tokens = responses.gather(1, last_indices.unsqueeze(-1)).squeeze(-1)
        handoff_mask &= last_tokens != eos_token_id

    repeat_counts = torch.where(
        handoff_mask,
        torch.full_like(valid_lengths, num_teacher_samples),
        torch.ones_like(valid_lengths),
    )
    parent_indices = torch.repeat_interleave(
        torch.arange(responses.shape[0], device=responses.device), repeat_counts
    )
    prefix_loss_rows = torch.ones(parent_indices.shape[0], dtype=torch.bool, device=responses.device)
    candidate_active = torch.ones(parent_indices.shape[0], dtype=torch.bool, device=responses.device)
    row_start = 0
    for repeat_count in repeat_counts.detach().cpu().tolist():
        if repeat_count > 1:
            prefix_loss_rows[row_start + 1 : row_start + repeat_count] = False
        row_start += repeat_count

    # Reward-worker dispatch requires equal chunks.  Pad with inactive copies
    # rather than duplicating real loss-bearing candidates.
    padding = (-parent_indices.shape[0]) % dispatch_size
    if padding:
        padding_parents = parent_indices.new_zeros(padding)
        parent_indices = torch.cat((parent_indices, padding_parents), dim=0)
        prefix_loss_rows = torch.cat(
            (prefix_loss_rows, torch.zeros(padding, dtype=torch.bool, device=responses.device)), dim=0
        )
        candidate_active = torch.cat(
            (candidate_active, torch.zeros(padding, dtype=torch.bool, device=responses.device)), dim=0
        )

    expanded = student_output.select_idxs(parent_indices)

    expanded.batch["handoff_parent_index"] = parent_indices
    expanded.batch["handoff_prefix_loss_row"] = prefix_loss_rows
    expanded.batch["handoff_candidate_active"] = candidate_active
    return expanded


def attach_teacher_continuations(student_output: DataProto, teacher_output: DataProto) -> DataProto:
    """Append fixed-width teacher continuations and construct disjoint loss masks.

    ``student_output`` contains a rollout padded to the handoff boundary.  The
    teacher worker returns a continuation padded to the remaining response
    budget.  Short student responses have an all-zero continuation mask and
    therefore remain ordinary full-response OPD examples.
    """

    responses = student_output.batch["responses"]
    suffixes = teacher_output.batch["teacher_suffixes"].to(responses.device)
    suffix_mask = teacher_output.batch["teacher_suffix_mask"].to(responses.device)
    handoff_mask = teacher_output.batch["handoff_mask"].to(responses.device)
    response_length = responses.shape[-1]
    student_response_mask = student_output.batch["attention_mask"][:, -response_length:].to(suffix_mask.dtype)

    student_output.batch["responses"] = torch.cat((responses, suffixes), dim=-1)
    student_output.batch["input_ids"] = torch.cat((student_output.batch["input_ids"], suffixes), dim=-1)
    student_output.batch["attention_mask"] = torch.cat(
        (student_output.batch["attention_mask"], suffix_mask.to(student_output.batch["attention_mask"].dtype)), dim=-1
    )

    old_position_ids = student_output.batch["position_ids"]
    suffix_length = suffixes.shape[-1]
    delta = torch.arange(1, suffix_length + 1, device=old_position_ids.device)
    delta = delta.view(1, -1).expand(old_position_ids.shape[0], -1)
    if old_position_ids.dim() == 3:
        delta = delta.view(old_position_ids.shape[0], 1, -1).expand(
            old_position_ids.shape[0], old_position_ids.shape[1], -1
        )
    suffix_position_ids = old_position_ids[..., -1:] + delta
    student_output.batch["position_ids"] = torch.cat((old_position_ids, suffix_position_ids), dim=-1)

    prefix_zeros = torch.zeros_like(student_response_mask)
    suffix_zeros = torch.zeros_like(suffix_mask)
    prefix_loss_row = None
    if "handoff_prefix_loss_row" in student_output.batch:
        prefix_loss_row = student_output.batch.pop("handoff_prefix_loss_row")
    if "handoff_candidate_active" in student_output.batch:
        student_output.batch.pop("handoff_candidate_active")
    if prefix_loss_row is not None:
        student_response_mask = student_response_mask * prefix_loss_row.unsqueeze(-1).to(student_response_mask.dtype)
    student_output.batch["handoff_opd_mask"] = torch.cat((student_response_mask, suffix_zeros), dim=-1)
    student_output.batch["handoff_ce_mask"] = torch.cat((prefix_zeros, suffix_mask), dim=-1)
    student_output.batch["handoff_mask"] = handoff_mask

    # Rollout log-probabilities only exist for the student prefix.  Extend the
    # tensor for shape compatibility; these values are never used by the CE
    # suffix and its OPD advantage is masked to zero.
    if "rollout_log_probs" in student_output.batch:
        suffix_log_probs = torch.zeros_like(suffix_mask, dtype=student_output.batch["rollout_log_probs"].dtype)
        student_output.batch["rollout_log_probs"] = torch.cat(
            (student_output.batch["rollout_log_probs"], suffix_log_probs), dim=-1
        )

    return student_output


def gate_teacher_suffix_sft_by_reward(
    data: DataProto,
    reward_scores: torch.Tensor,
    correct_threshold: float = 0.5,
) -> torch.Tensor:
    """Keep suffix SFT tokens only for verifier-correct teacher continuations.

    The verifier scores the complete response (Student prefix plus Teacher
    continuation).  A continuation is accepted only when its trajectory-level
    score reaches ``correct_threshold``.  Non-handoff examples never acquire a
    suffix loss, even if their Student response is correct.

    Returns a per-example boolean mask indicating accepted handoff examples.
    """

    if "handoff_ce_mask" not in data.batch or "handoff_mask" not in data.batch:
        raise KeyError("Verified suffix SFT requires handoff_ce_mask and handoff_mask.")

    if reward_scores.ndim == 1:
        trajectory_scores = reward_scores
    else:
        trajectory_scores = reward_scores.reshape(reward_scores.shape[0], -1).sum(dim=-1)

    ce_mask = data.batch["handoff_ce_mask"]
    handoff_mask = data.batch["handoff_mask"].to(device=ce_mask.device, dtype=torch.bool)
    correct_mask = trajectory_scores.to(ce_mask.device) >= correct_threshold
    accepted_mask = handoff_mask & correct_mask

    data.batch["handoff_ce_mask"] = ce_mask * accepted_mask.unsqueeze(-1).to(ce_mask.dtype)
    return accepted_mask


def select_best_verified_handoff_candidate(
    data: DataProto,
    reward_scores: torch.Tensor,
    correct_threshold: float = 0.5,
) -> tuple[DataProto, dict[str, int]]:
    """Collapse Best-of-N Teacher suffixes to one Actor-training row per Student rollout.

    Teacher generation and verifier scoring still use every sampled suffix.  Before
    computing advantages and updating the Actor, this function keeps the
    highest-scoring correct suffix for each Student prefix.  If no suffix is
    correct, it keeps the canonical row that owns the prefix OPD mask and leaves
    its suffix CE mask empty.

    Moving the canonical OPD mask to a selected non-canonical candidate preserves
    exactly one copy of prefix OPD supervision per Student rollout.  Collapsing
    the rows restores the Actor batch size (and therefore its optimizer-step
    count) to the ordinary OPD budget.
    """

    required_keys = {
        "handoff_parent_index",
        "handoff_opd_mask",
        "handoff_ce_mask",
        "handoff_mask",
    }
    missing_keys = required_keys.difference(data.batch.keys())
    if missing_keys:
        raise KeyError(f"Best-of-N handoff selection is missing batch keys: {sorted(missing_keys)}")

    accepted_mask = gate_teacher_suffix_sft_by_reward(
        data,
        reward_scores=reward_scores,
        correct_threshold=correct_threshold,
    )

    if reward_scores.ndim == 1:
        trajectory_scores = reward_scores
    else:
        trajectory_scores = reward_scores.reshape(reward_scores.shape[0], -1).sum(dim=-1)

    parent_indices = data.batch["handoff_parent_index"].long()
    opd_mask = data.batch["handoff_opd_mask"]
    handoff_mask = data.batch["handoff_mask"].bool()
    trajectory_scores = trajectory_scores.to(parent_indices.device)
    accepted_mask = accepted_mask.to(parent_indices.device)

    selected_rows: list[torch.Tensor] = []
    handoff_parent_count = 0
    selected_correct_count = 0

    for parent_index in torch.unique(parent_indices, sorted=True):
        group_rows = torch.nonzero(parent_indices == parent_index, as_tuple=False).squeeze(-1)

        # Exactly one real row owns prefix OPD.  Padded dispatch rows and the
        # extra Teacher candidates have an all-zero prefix mask.
        opd_token_counts = opd_mask[group_rows].reshape(group_rows.numel(), -1).sum(dim=-1)
        canonical_row = group_rows[torch.argmax(opd_token_counts)]

        group_handoff_mask = handoff_mask[group_rows]
        if group_handoff_mask.any():
            handoff_parent_count += 1

        correct_rows = group_rows[accepted_mask[group_rows]]
        if correct_rows.numel() > 0:
            candidate_scores = torch.nan_to_num(
                trajectory_scores[correct_rows], nan=-torch.inf, neginf=-torch.inf
            )
            selected_row = correct_rows[torch.argmax(candidate_scores)]
            selected_correct_count += 1
        else:
            selected_row = canonical_row

        if selected_row.item() != canonical_row.item():
            opd_mask[selected_row] = opd_mask[canonical_row].clone()
        selected_rows.append(selected_row)

    selected_indices = torch.stack(selected_rows)
    selected_data = data.select_idxs(selected_indices)
    selected_data.batch.pop("handoff_parent_index")

    candidate_handoff_count = int(handoff_mask.sum().item())
    candidate_correct_count = int(accepted_mask.sum().item())
    stats = {
        "parent_count": len(selected_rows),
        "handoff_parent_count": handoff_parent_count,
        "candidate_handoff_count": candidate_handoff_count,
        "candidate_correct_count": candidate_correct_count,
        "selected_correct_count": selected_correct_count,
    }
    return selected_data, stats
