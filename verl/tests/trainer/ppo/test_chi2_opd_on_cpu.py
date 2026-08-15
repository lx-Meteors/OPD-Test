# Copyright 2026 Chi2-OPD authors
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
"""CPU tests for Pearson-chi-squared tangent OPD."""

import pytest
import torch

from verl.utils.chi2_opd import compute_chi2_opd_scores, compute_chi2_target_log_probs


def _log_probs(values):
    return torch.tensor(values, dtype=torch.float32).log()


def _batch():
    student = _log_probs([[[0.55, 0.30, 0.10], [0.40, 0.35, 0.15]]])
    teacher = _log_probs([[[0.60, 0.25, 0.10], [0.50, 0.30, 0.10]]])
    reference = _log_probs([[[0.45, 0.35, 0.15], [0.35, 0.40, 0.15]]])
    mask = torch.ones(1, 2)
    return student, teacher, reference, mask


def test_zero_kappa_is_exactly_standard_topk_opd():
    student, teacher, reference, mask = _batch()
    target, _ = compute_chi2_target_log_probs(student, teacher, reference, mask, {"kappa": 0.0})
    torch.testing.assert_close(target, teacher, rtol=0.0, atol=1e-7)

    scores, _ = compute_chi2_opd_scores(student, teacher, reference, mask, {"kappa": 0.0})
    expected = -(student - teacher) * torch.softmax(student, dim=-1)
    torch.testing.assert_close(scores, expected)


def test_target_preserves_teacher_candidate_mass():
    student, teacher, reference, mask = _batch()
    target, _ = compute_chi2_target_log_probs(student, teacher, reference, mask, {"kappa": 0.2})
    torch.testing.assert_close(target.exp().sum(dim=-1), teacher.exp().sum(dim=-1), rtol=1e-6, atol=1e-7)


def test_positive_reward_is_linearly_amplified():
    student, teacher, reference, mask = _batch()
    target, _ = compute_chi2_target_log_probs(student, teacher, reference, mask, {"kappa": 0.2})
    density_ratio = (target - teacher).exp()

    reward = teacher - reference
    teacher_cond = torch.softmax(teacher, dim=-1)
    centered = reward - (teacher_cond * reward).sum(dim=-1, keepdim=True)
    expected = 1.0 + 0.2 * centered
    torch.testing.assert_close(density_ratio, expected, rtol=1e-5, atol=1e-6)


def test_adaptive_kappa_respects_density_bounds():
    student = _log_probs([[[0.4, 0.3, 0.2]]])
    teacher = _log_probs([[[0.8, 0.1, 0.05]]])
    reference = _log_probs([[[0.01, 0.49, 0.45]]])
    mask = torch.ones(1, 1)
    target, aux = compute_chi2_target_log_probs(
        student,
        teacher,
        reference,
        mask,
        {"kappa": 10.0, "adaptive_kappa": True, "min_density_ratio": 0.2, "max_density_ratio": 2.0},
    )
    density_ratio = (target - teacher).exp()

    assert density_ratio.min() >= 0.2 - 1e-6
    assert density_ratio.max() <= 2.0 + 1e-6
    assert aux["chi2_opd_kappa_shrunk"].item() == 1.0
    assert aux["chi2_opd_kappa"].item() < 10.0


def test_masked_positions_have_zero_scores_and_diagnostics():
    student, teacher, reference, mask = _batch()
    mask[:, 1] = 0
    scores, aux = compute_chi2_opd_scores(student, teacher, reference, mask, {"kappa": 0.2})
    assert torch.all(scores[:, 1] == 0)
    for key, value in aux.items():
        if key != "chi2_opd_target_log_probs":
            assert torch.all(value[:, 1] == 0)


def test_nonfinite_padding_does_not_contaminate_scores():
    student, teacher, reference, mask = _batch()
    mask[:, 1] = 0
    student[:, 1] = torch.nan
    teacher[:, 1] = -torch.inf
    reference[:, 1] = torch.inf

    scores, aux = compute_chi2_opd_scores(student, teacher, reference, mask, {"kappa": 0.25})

    assert torch.isfinite(scores).all()
    assert torch.all(scores[:, 1] == 0)
    assert all(torch.isfinite(value).all() for value in aux.values())


@pytest.mark.parametrize(
    "config",
    [
        {"kappa": -0.1},
        {"kappa": float("nan")},
        {"kappa": float("inf")},
        {"min_density_ratio": 0.0},
        {"min_density_ratio": 1.1},
        {"min_density_ratio": float("nan")},
        {"max_density_ratio": 0.9},
        {"max_density_ratio": float("inf")},
    ],
)
def test_invalid_config_is_rejected(config):
    student, teacher, reference, mask = _batch()
    with pytest.raises(ValueError):
        compute_chi2_target_log_probs(student, teacher, reference, mask, config)
