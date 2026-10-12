"""Bjontegaard Delta (BD-Rate and BD-PSNR) metric implementation.

Reference:
    G. Bjontegaard, "Calculation of average PSNR differences between RD-curves",
    ITU-T SG16 Q.6 Document VCEG-M33, Austin, TX, USA, April 2001.

Standard JVET / VCEG method to compute the average bitrate saving (BD-Rate in %)
and average PSNR improvement (BD-PSNR in dB) between two Rate-Distortion curves.
"""

from __future__ import annotations

import numpy as np


def _fit_curve(rates: list[float] | np.ndarray, dists: list[float] | np.ndarray, piecewise: bool = False):
    """Fit a 3rd-order polynomial to log-rate vs distortion, or piecewise cubic."""
    r = np.array(rates, dtype=np.float64)
    d = np.array(dists, dtype=np.float64)
    d = np.nan_to_num(d, nan=0.0, posinf=100.0, neginf=0.0)
    if len(r) != len(d) or len(r) < 4:
        raise ValueError("BD metrics require at least 4 rate-distortion points.")
    # Sort strictly by distortion/rate
    order = np.argsort(d)
    d = d[order]
    r = r[order]

    log_r = np.log10(np.maximum(r, 1e-9))
    if not piecewise:
        # Standard VCEG 3rd-order polynomial: log(R) = a + b*D + c*D^2 + d*D^3
        poly = np.polyfit(d, log_r, deg=3)
        return poly, float(d[0]), float(d[-1])
    else:
        # Piecewise interpolation
        from scipy.interpolate import PchipInterpolator
        interp = PchipInterpolator(d, log_r)
        return interp, float(d[0]), float(d[-1])


def bd_rate(
    r1: list[float] | np.ndarray,
    d1: list[float] | np.ndarray,
    r2: list[float] | np.ndarray,
    d2: list[float] | np.ndarray,
    piecewise: bool = False,
) -> float:
    """Calculate Bjontegaard Delta Rate (BD-Rate in %).

    A negative value indicates that curve 2 (test codec) achieves the same distortion
    at a lower bitrate than curve 1 (anchor / reference codec).
    e.g., -25.4% means 25.4% average bitrate savings for equal PSNR.
    """
    poly1, min_d1, max_d1 = _fit_curve(r1, d1, piecewise=piecewise)
    poly2, min_d2, max_d2 = _fit_curve(r2, d2, piecewise=piecewise)

    # Integration interval is the overlap region of distortion
    min_int = max(min_d1, min_d2)
    max_int = min(max_d1, max_d2)

    if min_int >= max_int:
        return 0.0

    if not piecewise:
        # Analytical integration of polynomials: poly[0]*x^3 + poly[1]*x^2 + poly[2]*x + poly[3]
        int1 = np.polyint(poly1)
        int2 = np.polyint(poly2)

        v1 = np.polyval(int1, max_int) - np.polyval(int1, min_int)
        v2 = np.polyval(int2, max_int) - np.polyval(int2, min_int)
    else:
        # Numerical integration for piecewise
        from scipy.integrate import quad
        v1, _ = quad(poly1, min_int, max_int)
        v2, _ = quad(poly2, min_int, max_int)

    avg_diff = (v2 - v1) / (max_int - min_int)
    # Convert log10 difference back to percentage rate difference
    bd_rate_pct = (10.0 ** avg_diff - 1.0) * 100.0
    return float(bd_rate_pct)


def bd_psnr(
    r1: list[float] | np.ndarray,
    d1: list[float] | np.ndarray,
    r2: list[float] | np.ndarray,
    d2: list[float] | np.ndarray,
    piecewise: bool = False,
) -> float:
    """Calculate Bjontegaard Delta PSNR (BD-PSNR in dB).

    A positive value indicates that curve 2 (test codec) achieves higher average
    distortion (quality) than curve 1 (anchor codec) across the same bitrate range.
    """
    r1_arr = np.array(r1, dtype=np.float64)
    d1_arr = np.array(d1, dtype=np.float64)
    d1_arr = np.nan_to_num(d1_arr, nan=0.0, posinf=100.0, neginf=0.0)
    r2_arr = np.array(r2, dtype=np.float64)
    d2_arr = np.array(d2, dtype=np.float64)
    d2_arr = np.nan_to_num(d2_arr, nan=0.0, posinf=100.0, neginf=0.0)

    if len(r1_arr) < 4 or len(r2_arr) < 4:
        raise ValueError("BD metrics require at least 4 rate-distortion points.")

    log_r1 = np.log10(np.maximum(r1_arr, 1e-9))
    log_r2 = np.log10(np.maximum(r2_arr, 1e-9))

    order1 = np.argsort(log_r1)
    order2 = np.argsort(log_r2)

    log_r1 = log_r1[order1]
    d1_sorted = d1_arr[order1]
    log_r2 = log_r2[order2]
    d2_sorted = d2_arr[order2]

    min_int = max(float(log_r1[0]), float(log_r2[0]))
    max_int = min(float(log_r1[-1]), float(log_r2[-1]))

    if min_int >= max_int:
        return 0.0

    if not piecewise:
        poly1 = np.polyfit(log_r1, d1_sorted, deg=3)
        poly2 = np.polyfit(log_r2, d2_sorted, deg=3)

        int1 = np.polyint(poly1)
        int2 = np.polyint(poly2)

        v1 = np.polyval(int1, max_int) - np.polyval(int1, min_int)
        v2 = np.polyval(int2, max_int) - np.polyval(int2, min_int)
    else:
        from scipy.integrate import quad
        from scipy.interpolate import PchipInterpolator
        interp1 = PchipInterpolator(log_r1, d1_sorted)
        interp2 = PchipInterpolator(log_r2, d2_sorted)
        v1, _ = quad(interp1, min_int, max_int)
        v2, _ = quad(interp2, min_int, max_int)

    avg_psnr_diff = (v2 - v1) / (max_int - min_int)
    return float(avg_psnr_diff)
