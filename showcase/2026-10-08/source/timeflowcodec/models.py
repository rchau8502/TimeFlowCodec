"""Temporal models for per-pixel per-channel RGB TimeFlowCodec."""
from __future__ import annotations

import numpy as np


def fit_linear_time_model(signal: np.ndarray) -> tuple[float, float]:
    """
    Fit s[t] ≈ a + b * t via closed-form least squares for t=0..T-1.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    T = signal.shape[0]
    if T == 0:
        return 0.0, 0.0
    s = signal.astype(np.float64, copy=False)
    n = float(T)
    sum_t = n * (n - 1.0) / 2.0
    sum_t2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    sum_s = float(np.sum(s))
    t = np.arange(T, dtype=np.float64)
    sum_ts = float(np.sum(t * s))
    denom = n * sum_t2 - sum_t * sum_t
    if denom == 0.0:
        b = 0.0
    else:
        b = (n * sum_ts - sum_t * sum_s) / denom
    a = (sum_s - b * sum_t) / n
    return float(a), float(b)


def reconstruct_linear_sequence(a: float, b: float, T: int) -> np.ndarray:
    """Return s_hat[t] = a + b * t for t=0..T-1."""
    t = np.arange(T, dtype=np.float32)
    return (a + b * t).astype(np.float32)


def reconstruct_constant_sequence(a: float, T: int) -> np.ndarray:
    """Return s_hat[t] = a for t=0..T-1."""
    return np.full((T,), a, dtype=np.float32)
