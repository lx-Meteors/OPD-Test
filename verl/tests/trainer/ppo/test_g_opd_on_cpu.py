# Copyright 2026 G-OPD reproduction authors
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
"""CPU tests for Generalized On-Policy Distillation."""

import pytest
import torch

from verl.utils.g_opd import compute_g_opd_scores, compute_g_opd_target_log_probs, g_opd_metrics


def _log_probs(values):
    return torch.tensor(values, dtype=torch.float32).log()


def _batch():
    student = _log_probs([[[0.55, 0.30, 0.10], [0.40, 0.35, 0.15]]])
    teacher = _log_probs([[[0.60, 0.25, 0.10], [0.50, 0.30, 0.10]]])
    reference = _log_probs([[[0.45, 0.35, 0.15], [0.35, 0.40, 0.15]]])
    mask = torch.ones(1, 2)
    return student, teacher, reference, mask


def test_lambda_one_is_exactly_standard_topk_opd():
    student, teacher, reference, mask = _batch()
    target, _ = compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": 1.0})
    torch.testing.assert_close(target, teacher, rtol=0.0, atol=1e-7)

    scores, _ = compute_g_opd_scores(student, teacher, reference, mask, {"lambda": 1.0})
    expected = -(student - teacher) * torch.softmax(student, dim=-1)
    torch.testing.assert_close(scores, expected, rtol=0.0, atol=1e-7)


def test_lambda_zero_recovers_reference_target():
    student, teacher, reference, mask = _batch()
    target, _ = compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": 0.0})
    torch.testing.assert_close(target, reference, rtol=0.0, atol=1e-7)


def test_target_matches_exponential_family_solution():
    student, teacher, reference, mask = _batch()
    lambda_value = 1.25
    target, _ = compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": lambda_value})

    expected_log_score = (1.0 - lambda_value) * reference + lambda_value * teacher
    torch.testing.assert_close(target, expected_log_score, rtol=0.0, atol=3e-7)
    torch.testing.assert_close(torch.softmax(target, dim=-1), torch.softmax(expected_log_score, dim=-1))


def test_extrapolation_scales_teacher_reference_direction():
    student, teacher, reference, mask = _batch()
    lambda_value = 1.25
    target, _ = compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": lambda_value})
    torch.testing.assert_close(target - reference, lambda_value * (teacher - reference))


def test_masked_positions_have_zero_scores_and_diagnostics():
    student, teacher, reference, mask = _batch()
    mask[:, 1] = 0
    scores, aux = compute_g_opd_scores(student, teacher, reference, mask, {"lambda": 1.25})

    assert torch.all(scores[:, 1] == 0)
    for key, value in aux.items():
        if key != "g_opd_target_log_probs":
            assert torch.all(value[:, 1] == 0)


def test_nonfinite_padding_does_not_contaminate_scores():
    student, teacher, reference, mask = _batch()
    mask[:, 1] = 0
    student[:, 1] = torch.nan
    teacher[:, 1] = -torch.inf
    reference[:, 1] = torch.inf

    scores, aux = compute_g_opd_scores(student, teacher, reference, mask, {"lambda": 1.25})

    assert torch.isfinite(scores).all()
    assert torch.all(scores[:, 1] == 0)
    assert all(torch.isfinite(value).all() for value in aux.values())


def test_metrics_use_only_active_positions():
    student, teacher, reference, mask = _batch()
    mask[:, 1] = 0
    _, aux = compute_g_opd_scores(student, teacher, reference, mask, {"lambda": 1.25})
    metrics = g_opd_metrics(aux, mask)

    assert metrics["g_opd/lambda"] == pytest.approx(1.25)
    assert set(metrics) == {
        "g_opd/lambda",
        "g_opd/implicit_reward_mean",
        "g_opd/implicit_reward_std",
        "g_opd/target_shift_tv",
        "g_opd/student_target_kl",
        "g_opd/teacher_target_kl",
    }


@pytest.mark.parametrize("lambda_value", [-0.1, float("nan"), float("inf")])
def test_invalid_lambda_is_rejected(lambda_value):
    student, teacher, reference, mask = _batch()
    with pytest.raises(ValueError):
        compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": lambda_value})


def test_nonfinite_active_log_probability_is_rejected():
    student, teacher, reference, mask = _batch()
    teacher[:, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="active response position"):
        compute_g_opd_target_log_probs(student, teacher, reference, mask, {"lambda": 1.25})
