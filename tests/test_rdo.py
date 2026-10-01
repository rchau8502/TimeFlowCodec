"""Tests for Rate-Distortion Optimization (RDO) in TimeFlowCodec."""

import numpy as np

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import (
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
    _encode_plane_from_stats,
    encode_video_to_tfc,
)


def test_rdo_prefers_const_for_subtle_gradient():
    """Verify that RDO favors CONST over LINEAR when the distortion delta is tiny compared to bit cost."""
    T = 16
    N = 100
    # Create very subtle linear drift: drift of 0.05 across 16 frames (basically imperceptible)
    t = np.arange(T, dtype=np.float64)
    # Signal: s(t) = 100 + 0.05 * t
    signal = 100.0 + 0.05 * t[None, :]  # shape (1, T)
    signals = np.repeat(signal, N, axis=0)  # shape (N, T)

    sum_s = np.sum(signals, axis=1)
    sum_ts = np.sum(signals * t[None, :], axis=1)
    sum_s2 = np.sum(signals**2, axis=1)
    counts = np.ones(N, dtype=np.int64)

    tau = 2.0
    slope_threshold = 0.01  # slope is 0.05 > 0.01, so heuristic would pick LINEAR

    # Without RDO: heuristic picks LINEAR because slope (0.05) > slope_threshold (0.01)
    modes_heur, _, _, _, _, _, _, _ = _encode_plane_from_stats(
        sum_s, sum_ts, sum_s2, counts, T, tau=tau, slope_threshold=slope_threshold, rdo=False
    )
    assert np.all(modes_heur == MODE_TFC_LINEAR), "Heuristic should have chosen LINEAR based on slope"

    # With RDO: with reasonable lambda, CONST saves 16 bits per pixel with negligible MSE increase
    modes_rdo, _, _, _, _, _, _, _ = _encode_plane_from_stats(
        sum_s, sum_ts, sum_s2, counts, T, tau=tau, slope_threshold=slope_threshold, rdo=True, lambda_rdo=10.0
    )
    assert np.all(modes_rdo == MODE_TFC_CONST), "RDO should have chosen CONST to save 16 bits per pixel"


def test_rdo_preserves_linear_when_distortion_large():
    """Verify that RDO still selects LINEAR when the slope is substantial."""
    T = 16
    N = 100
    t = np.arange(T, dtype=np.float64)
    # Steep slope: 5.0 per frame -> CONST would have massive MSE (~ 5^2 * T^2 / 12 ~ 500)
    signal = 50.0 + 5.0 * t[None, :]
    signals = np.repeat(signal, N, axis=0)

    sum_s = np.sum(signals, axis=1)
    sum_ts = np.sum(signals * t[None, :], axis=1)
    sum_s2 = np.sum(signals**2, axis=1)
    counts = np.ones(N, dtype=np.int64)

    tau = 2.0
    slope_threshold = 0.01

    modes_rdo, _, _, _, _, _, _, _ = _encode_plane_from_stats(
        sum_s, sum_ts, sum_s2, counts, T, tau=tau, slope_threshold=slope_threshold, rdo=True, lambda_rdo=1.0
    )
    assert np.all(modes_rdo == MODE_TFC_LINEAR), "RDO should select LINEAR when CONST distortion is huge"


def test_rdo_end_to_end_encode_decode(tmp_path):
    """Verify end-to-end video encode and decode with RDO enabled."""
    H, W, T = 32, 32, 16
    np.random.seed(42)
    # Synthesize test video: background with subtle drift, one moving block
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        video[t, :, :, :] = (120 + t // 4) % 256
        video[t, 10:20, 10:20, :] = 200

    out_heur = tmp_path / "heur.tfc"
    out_rdo = tmp_path / "rdo.tfc"

    encode_video_to_tfc(video, str(out_heur), tau=2.0, rdo=False)
    encode_video_to_tfc(video, str(out_rdo), tau=2.0, rdo=True, lambda_rdo=10.0)

    # Both should decode back accurately
    dec_heur = decode_tfc_to_video(str(out_heur), None)
    dec_rdo = decode_tfc_to_video(str(out_rdo), None)

    assert dec_heur.shape == video.shape
    assert dec_rdo.shape == video.shape

    # PSNR for both should be high (> 40 dB)
    mse_heur = np.mean((video.astype(np.float32) - dec_heur.astype(np.float32)) ** 2)
    mse_rdo = np.mean((video.astype(np.float32) - dec_rdo.astype(np.float32)) ** 2)

    psnr_heur = 10 * np.log10(255.0**2 / max(mse_heur, 1e-10))
    psnr_rdo = 10 * np.log10(255.0**2 / max(mse_rdo, 1e-10))

    assert psnr_heur > 40.0
    assert psnr_rdo > 40.0

    # RDO file size should be <= heuristic file size because subtle drift is absorbed into CONST
    size_heur = out_heur.stat().st_size
    size_rdo = out_rdo.stat().st_size
    assert size_rdo <= size_heur, f"RDO size ({size_rdo}) should be <= heuristic size ({size_heur})"
