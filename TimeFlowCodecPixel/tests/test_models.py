from __future__ import annotations

import numpy as np

from timeflowcodec.models import (
    fit_linear_time_model,
    reconstruct_constant_sequence,
    reconstruct_linear_sequence,
)


def test_fit_linear_exact():
    t = np.arange(10, dtype=float)
    s = 2.0 + 0.5 * t
    a, b = fit_linear_time_model(s)
    assert abs(a - 2.0) < 1e-6
    assert abs(b - 0.5) < 1e-6
    s_hat = reconstruct_linear_sequence(a, b, len(t))
    assert np.allclose(s_hat, s)


def test_fit_linear_noisy():
    rng = np.random.default_rng(0)
    t = np.arange(50, dtype=float)
    s = 5.0 - 0.2 * t + rng.normal(0, 0.1, size=t.shape)
    a, b = fit_linear_time_model(s)
    assert abs(a - 5.0) < 0.2
    assert abs(b + 0.2) < 0.02


def test_reconstruct_constant():
    a = 7.0
    seq = reconstruct_constant_sequence(a, 5)
    assert np.allclose(seq, np.full(5, a))
