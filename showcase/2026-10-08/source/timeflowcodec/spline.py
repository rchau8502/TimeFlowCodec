"""Higher-order temporal spline modeling for TimeFlowCodec v2.0.

Provides closed-form OLS solvers for quadratic (2nd-order) temporal regression
and piecewise spline evaluation with O(1) Gram matrix precomputation.
"""

from __future__ import annotations

import numpy as np


def precompute_quadratic_gram_inv(T: int) -> np.ndarray:
    """Precompute the 3x3 inverse Gram matrix for temporal indices t in [0, T-1].

    Normal equations:
        [S0 S1 S2] [a]   [sum s(t)   ]
        [S1 S2 S3] [b] = [sum t s(t) ]
        [S2 S3 S4] [c]   [sum t^2 s(t)]

    where S_k = sum_{t=0}^{T-1} t^k.
    """
    if T < 3:
        raise ValueError("Quadratic temporal regression requires at least T >= 3 frames.")

    t = np.arange(T, dtype=np.float64)
    s0 = float(T)
    s1 = float(np.sum(t))
    s2 = float(np.sum(t**2))
    s3 = float(np.sum(t**3))
    s4 = float(np.sum(t**4))

    gram = np.array([
        [s0, s1, s2],
        [s1, s2, s3],
        [s2, s3, s4],
    ], dtype=np.float64)

    return np.linalg.inv(gram)


def fit_quadratic_temporal_batch(
    sum_s: np.ndarray,
    sum_ts: np.ndarray,
    sum_t2s: np.ndarray,
    sum_s2: np.ndarray,
    T: int,
    gram_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a batch of pixels with quadratic models: s(t) = a + b*t + c*t^2.

    Parameters
    ----------
    sum_s : np.ndarray
        Array of shape (N,) containing sum_{t=0}^{T-1} s_i(t).
    sum_ts : np.ndarray
        Array of shape (N,) containing sum_{t=0}^{T-1} t * s_i(t).
    sum_t2s : np.ndarray
        Array of shape (N,) containing sum_{t=0}^{T-1} t^2 * s_i(t).
    sum_s2 : np.ndarray
        Array of shape (N,) containing sum_{t=0}^{T-1} s_i(t)^2.
    T : int
        Temporal segment duration in frames.
    gram_inv : np.ndarray, optional
        Precomputed 3x3 inverse Gram matrix.

    Returns
    -------
    a : np.ndarray of shape (N,)
        Intercept.
    b : np.ndarray of shape (N,)
        Linear slope.
    c : np.ndarray of shape (N,)
        Quadratic curvature.
    sse : np.ndarray of shape (N,)
        Sum of squared residual errors.
    """
    if gram_inv is None:
        gram_inv = precompute_quadratic_gram_inv(T)

    g = gram_inv
    # Direct vectorized matrix-vector multiplication for N coordinates:
    # [a, b, c]^T = G @ [sum_s, sum_ts, sum_t2s]^T
    a = g[0, 0] * sum_s + g[0, 1] * sum_ts + g[0, 2] * sum_t2s
    b = g[1, 0] * sum_s + g[1, 1] * sum_ts + g[1, 2] * sum_t2s
    c = g[2, 0] * sum_s + g[2, 1] * sum_ts + g[2, 2] * sum_t2s

    # Closed-form residual SSE = sum(s^2) - a*sum(s) - b*sum(ts) - c*sum(t^2s)
    sse = sum_s2 - a * sum_s - b * sum_ts - c * sum_t2s
    sse = np.maximum(0.0, sse)

    return a, b, c, sse


def evaluate_quadratic(a: np.ndarray, b: np.ndarray, c: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Evaluate quadratic temporal model at time t."""
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim == 0:
        return a + b * t_arr + c * (t_arr**2)
    return a[:, None] + b[:, None] * t_arr[None, :] + c[:, None] * (t_arr[None, :] ** 2)
