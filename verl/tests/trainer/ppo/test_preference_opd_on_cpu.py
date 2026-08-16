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

from verl.utils.preference_opd import apply_preference_opd_to_scores, fit_candidate_temperature


def _log_probs(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).log().view(1, 1, -1)


def test_confidence_beta_one_exactly_recovers_standard_opd() -> None:
    student = _log_probs([0.5, 0.3, 0.2])
    teacher = _log_probs([0.8, 0.15, 0.05])
    valid = torch.ones_like(student, dtype=torch.bool)
    weights = torch.softmax(student, dim=-1)
    expected = (teacher - student) * weights

    scores, aux = apply_preference_opd_to_scores(
        student,
        teacher,
        valid,
        weights,
        {"enable": True, "confidence_beta": 1.0},
    )

    assert torch.equal(scores, expected)
    assert aux["preference_opd_temperature"].shape == (1, 1)


def test_pure_temperature_difference_has_no_preference_signal() -> None:
    student = _log_probs([0.5, 0.3, 0.2])
    teacher = torch.log_softmax(3.0 * student, dim=-1)
    valid = torch.ones_like(student, dtype=torch.bool)
    weights = torch.softmax(student, dim=-1)

    scores, aux = apply_preference_opd_to_scores(
        student,
        teacher,
        valid,
        weights,
        {"enable": True, "confidence_beta": 0.0, "newton_steps": 8},
    )

    torch.testing.assert_close(aux["preference_opd_temperature"], torch.tensor([[3.0]]), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(scores, torch.zeros_like(scores), atol=1e-5, rtol=0.0)
    assert aux["preference_opd_explained_ratio"].item() > 0.999


def test_candidate_mass_change_is_classified_as_confidence() -> None:
    student = _log_probs([0.3, 0.2, 0.1])
    student_cond = student - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_cond = torch.log_softmax(2.0 * student_cond, dim=-1)
    teacher = teacher_cond + torch.tensor(0.7).log()
    valid = torch.ones_like(student, dtype=torch.bool)
    weights = torch.softmax(student, dim=-1)

    scores, aux = apply_preference_opd_to_scores(
        student,
        teacher,
        valid,
        weights,
        {"enable": True, "confidence_beta": 0.0, "newton_steps": 8},
    )

    torch.testing.assert_close(scores, torch.zeros_like(scores), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(aux["preference_opd_teacher_candidate_mass"], torch.tensor([[0.7]]))
    assert aux["preference_opd_confidence_abs"].item() > 0.0


def test_changed_preference_survives_temperature_removal() -> None:
    student = _log_probs([0.5, 0.3, 0.2])
    teacher = _log_probs([0.2, 0.7, 0.1])
    valid = torch.ones_like(student, dtype=torch.bool)
    weights = torch.softmax(student, dim=-1)

    scores, aux = apply_preference_opd_to_scores(
        student,
        teacher,
        valid,
        weights,
        {"enable": True, "confidence_beta": 0.0},
    )

    assert scores.abs().sum().item() > 0.1
    assert aux["preference_opd_top1_agreement"].item() == 0.0
    assert aux["preference_opd_preference_abs"].item() > 0.1


def test_masked_candidates_do_not_affect_temperature_fit() -> None:
    student = _log_probs([0.6, 0.3, 0.1])
    teacher = _log_probs([0.8, 0.15, 0.05])
    valid = torch.tensor([[[True, True, False]]])

    scale_a, calibrated_a, _, _ = fit_candidate_temperature(student, teacher, valid, newton_steps=8)

    modified_student = student.clone()
    modified_teacher = teacher.clone()
    modified_student[..., 2] = 100.0
    modified_teacher[..., 2] = -100.0
    scale_b, calibrated_b, _, _ = fit_candidate_temperature(modified_student, modified_teacher, valid, newton_steps=8)

    torch.testing.assert_close(scale_a, scale_b)
    torch.testing.assert_close(calibrated_a[valid], calibrated_b[valid])


def test_invalid_confidence_beta_is_rejected() -> None:
    student = _log_probs([0.5, 0.3, 0.2])
    valid = torch.ones_like(student, dtype=torch.bool)

    with pytest.raises(ValueError, match="confidence_beta"):
        apply_preference_opd_to_scores(
            student,
            student,
            valid,
            torch.ones_like(student),
            {"enable": True, "confidence_beta": 1.1},
        )
