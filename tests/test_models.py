from __future__ import annotations

import numpy as np

from timeflowcodec.models import (
    fit_linear_time_model,
    reconstruct_constant_sequence,
    reconstruct_linear_sequence,
)


def test_fit_linear_exact():
    t = np.arange(12, dtype=float)
    s = -1.5 + 0.25 * t
    a, b = fit_linear_time_model(s)
    assert abs(a + 1.5) < 1e-6
    assert abs(b - 0.25) < 1e-6
    s_hat = reconstruct_linear_sequence(a, b, len(t))
    assert np.allclose(s_hat, s)


def test_fit_linear_noisy():
    rng = np.random.default_rng(1)
    t = np.arange(40, dtype=float)
    s = 3.0 + 0.4 * t + rng.normal(0, 0.05, size=t.shape)
    a, b = fit_linear_time_model(s)
    assert abs(a - 3.0) < 0.15
    assert abs(b - 0.4) < 0.02


def test_reconstruct_constant():
    a = 11.0
    seq = reconstruct_constant_sequence(a, 6)
    assert np.allclose(seq, np.full(6, a))
