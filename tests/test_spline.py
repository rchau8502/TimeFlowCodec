"""Unit tests for TimeFlowCodec v2.0 spline and quadratic modeling."""

import numpy as np

from timeflowcodec.spline import (
    evaluate_quadratic,
    fit_quadratic_temporal_batch,
    precompute_quadratic_gram_inv,
)


def test_quadratic_solver_exact_polynomial():
    """Verify quadratic solver exactly recovers true quadratic functions."""
    T = 16
    N = 100
    np.random.seed(42)

    # Generate synthetic true parameters
    true_a = np.random.uniform(10.0, 200.0, size=N)
    true_b = np.random.uniform(-5.0, 5.0, size=N)
    true_c = np.random.uniform(-0.5, 0.5, size=N)

    # Compute trajectory for all pixels over T frames
    t_vals = np.arange(T, dtype=np.float64)
    # trajectories: shape (N, T)
    trajectories = true_a[:, None] + true_b[:, None] * t_vals[None, :] + true_c[:, None] * (t_vals[None, :] ** 2)

    # Compute temporal accumulator moments
    sum_s = np.sum(trajectories, axis=1)
    sum_ts = np.sum(trajectories * t_vals[None, :], axis=1)
    sum_t2s = np.sum(trajectories * (t_vals[None, :] ** 2), axis=1)
    sum_s2 = np.sum(trajectories**2, axis=1)

    gram_inv = precompute_quadratic_gram_inv(T)
    est_a, est_b, est_c, sse = fit_quadratic_temporal_batch(sum_s, sum_ts, sum_t2s, sum_s2, T, gram_inv)

    np.testing.assert_allclose(est_a, true_a, atol=1e-8)
    np.testing.assert_allclose(est_b, true_b, atol=1e-8)
    np.testing.assert_allclose(est_c, true_c, atol=1e-8)
    np.testing.assert_allclose(sse, 0.0, atol=1e-7)


def test_quadratic_solver_against_numpy_polyfit():
    """Verify quadratic solver matches numpy.polyfit on noisy data."""
    T = 24
    N = 10
    np.random.seed(123)

    t_vals = np.arange(T, dtype=np.float64)
    noisy_trajectories = np.random.uniform(0.0, 255.0, size=(N, T))

    sum_s = np.sum(noisy_trajectories, axis=1)
    sum_ts = np.sum(noisy_trajectories * t_vals[None, :], axis=1)
    sum_t2s = np.sum(noisy_trajectories * (t_vals[None, :] ** 2), axis=1)
    sum_s2 = np.sum(noisy_trajectories**2, axis=1)

    est_a, est_b, est_c, sse = fit_quadratic_temporal_batch(sum_s, sum_ts, sum_t2s, sum_s2, T)

    for i in range(N):
        # np.polyfit fits y = c*x^2 + b*x + a
        poly_c, poly_b, poly_a = np.polyfit(t_vals, noisy_trajectories[i], deg=2)
        assert np.isclose(est_a[i], poly_a, atol=1e-6)
        assert np.isclose(est_b[i], poly_b, atol=1e-6)
        assert np.isclose(est_c[i], poly_c, atol=1e-6)

        # Check SSE match
        pred = poly_a + poly_b * t_vals + poly_c * (t_vals**2)
        expected_sse = np.sum((noisy_trajectories[i] - pred) ** 2)
        assert np.isclose(sse[i], expected_sse, atol=1e-5)


def test_quadratic_evaluation():
    """Verify evaluate_quadratic handles scalars and 2D arrays correctly."""
    a = np.array([10.0, 20.0])
    b = np.array([2.0, -1.0])
    c = np.array([0.5, 0.1])

    # Scalar evaluation
    val_0 = evaluate_quadratic(a, b, c, 0.0)
    np.testing.assert_allclose(val_0, [10.0, 20.0])

    val_2 = evaluate_quadratic(a, b, c, 2.0)
    np.testing.assert_allclose(val_2, [10.0 + 4.0 + 2.0, 20.0 - 2.0 + 0.4])

    # Array evaluation
    arr = evaluate_quadratic(a, b, c, [0.0, 1.0, 2.0])
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr[0], [10.0, 12.5, 16.0])
