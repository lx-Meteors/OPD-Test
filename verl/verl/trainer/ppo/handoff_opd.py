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
