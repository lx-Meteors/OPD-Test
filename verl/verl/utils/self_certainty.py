# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Self-certainty (INTUITOR-style) utilities for the SC-ratio OPD advantage.

Self-certainty of a next-token distribution p = softmax(z) is the KL divergence
from the uniform distribution U to p:

    SC = KL(U || p) = -log|V| - (1/|V|) * sum_v log p(v)
       = logsumexp(z) - mean(z) - log|V|

It is >= 0 (Gibbs), equals 0 iff p is uniform, and is dominated by how deeply
the model suppresses the non-head candidates (tail suppression depth), which is
what distinguishes it from entropy (head-dominated).

The SC-ratio advantage replaces the frozen G-OPD extrapolation term
(lambda - 1) * (log T - log R) with a live, reference-free state-level weight:

    adv_t = (log T - log S) * (1 + w_t),
    w_t   = clamp(1 - SC_S(s_t) / SC_T(s_t), min=0, max=1)

so the bonus force stays parallel to the alignment debt (no quadrant-3 buyout by
construction), is capped at doubling it, pays out only where the teacher's
distribution shape is still sharper than the student's, and retires token by
token as SC_S -> SC_T. No reference model, no position window, no free
constants: the only inputs are the two models' own distributions.
"""

import math

import torch

__all__ = ["SC_FLOOR", "self_certainty_from_logits", "sc_ratio_weight", "centered_log_sc_ratio"]

# Numerical floor for SC before any division or log. SC >= 0 by Gibbs, but
# padding positions are exactly 0 (pad_input zero-fills the rmpad path) and fp32
# cancellation can return ~-1e-9 on a near-uniform distribution. Public because
# sc_probe's log-SC readouts must use the same floor, or the g they report stops
# being the g that produced the tilt.
SC_FLOOR = 1e-6


def self_certainty_from_logits(logits: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Compute SC = logsumexp(z) - mean(z) - log|V| over the last dim, in fp32.

    Args:
        logits: (..., vocab_size). Any float dtype; reduced in float32. The
            tensor is detached: SC is always a no-grad statistic.
        chunk_size: rows per chunk over the flattened leading dims, bounding
            the fp32 temporary to chunk_size x vocab.

    Returns:
        (...,) float32 tensor of per-position self-certainty.
    """
    original_shape = logits.shape
    vocab_size = original_shape[-1]
    log_v = math.log(vocab_size)

    flat = logits.detach().reshape(-1, vocab_size)
    out = torch.empty(flat.size(0), dtype=torch.float32, device=flat.device)
    for start in range(0, flat.size(0), chunk_size):
        chunk = flat[start : start + chunk_size].to(torch.float32)
        out[start : start + chunk.size(0)] = torch.logsumexp(chunk, dim=-1) - chunk.mean(dim=-1) - log_v
    return out.view(original_shape[:-1])


def centered_log_sc_ratio(
    student_self_certainty: torch.Tensor,
    teacher_self_certainty: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = SC_FLOOR,
) -> torch.Tensor:
    """Within-trajectory centered log SC-ratio for the SC-centered advantage.

        g_t = log(SC_T(s_t) / SC_S(s_t)),   c_t = g_t - mean_traj(g)

    where the mean runs over each row's valid response tokens. The centered
    tilt c is added to the sampled-token alignment debt: adv = a + c.

    Why centered: raw g is empirically almost everywhere positive (the student's
    tail suppression lags the teacher's), so adding it uncentered is a
    length-coupled rent — a uniform positive advantage offset that acts as pure
    self-sharpening speed and props up runaway loops. Subtracting the
    per-trajectory mean removes exactly that speed component and leaves a
    zero-sum redistribution of commitment: positive where the teacher's
    distribution shape is relatively clearer than the student's (front of the
    trajectory, execution corridors), negative where it is relatively more
    confused (deep wandering, loops). Per-trajectory total force is identically
    zero, so no length rent and no clock speedup by construction. It is also
    immune to the global sign flips of g observed live (the student's SC
    overshoots the teacher's during the early entropy-collapse sprint).

    Single-rollout self-contained: no cross-sample statistics, no reference
    model, no window, no free constants (the natural nats scale of the log
    ratio is on the same order as G-OPD's tilt budget, so beta = 1).

    Args:
        student_self_certainty: (bsz, response_len) SC_S per position.
        teacher_self_certainty: (bsz, response_len) SC_T per position.
        response_mask: (bsz, response_len) valid-token mask.
        eps: numerical floor for both SC values (SC >= 0 up to rounding).

    Returns:
        (bsz, response_len) float32 tensor c, zeroed outside the mask.
        Rows with a single valid token center to exactly zero.
    """
    sc_s = student_self_certainty.to(torch.float32).clamp_min(eps)
    sc_t = teacher_self_certainty.to(torch.float32).clamp_min(eps)
    mask = response_mask.to(torch.float32)
    g = torch.log(sc_t / sc_s) * mask
    row_mean = g.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (g - row_mean) * mask


def sc_ratio_weight(
    student_self_certainty: torch.Tensor,
    teacher_self_certainty: torch.Tensor,
    eps: float = SC_FLOOR,
) -> torch.Tensor:
    """State-level bonus weight w = clamp(1 - SC_S / SC_T, 0, 1).

    w = 0 wherever the student's distribution is already at least as sharp as
    the teacher's (including the loop/runaway signature SC_S > SC_T, where any
    non-zero bonus would subsidize the student's own over-commitment), and
    approaches 1 only where the teacher is certain and the student is not.
    Clamps guard fp edge cases: SC values are >= 0 up to rounding, and eps
    keeps the ratio finite if SC_T underflows to zero.
    """
    sc_s = student_self_certainty.to(torch.float32).clamp_min(0.0)
    sc_t = teacher_self_certainty.to(torch.float32).clamp_min(eps)
    return torch.clamp(1.0 - sc_s / sc_t, min=0.0, max=1.0)
