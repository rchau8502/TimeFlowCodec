from __future__ import annotations

from timeflowcodec.rate_control import (
    RateController,
    crf_to_parameters,
    parse_bitrate_string,
)


def test_crf_to_parameters_monotonic():
    crfs = [0, 10, 18, 23, 28, 35, 50, 63]
    taus = [crf_to_parameters(c)["tau"] for c in crfs]

    # As CRF increases, tolerance tau must monotonically increase
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1]

    # CRF 0 should be fine tolerance
    assert taus[0] < 1e-4
    # CRF 63 should be loose tolerance
    assert taus[-1] > 0.01


def test_crf_clamping():
    p_neg = crf_to_parameters(-10)
    p_0 = crf_to_parameters(0)
    assert p_neg["tau"] == p_0["tau"]

    p_high = crf_to_parameters(100)
    p_63 = crf_to_parameters(63)
    assert p_high["tau"] == p_63["tau"]


def test_crf_tunes():
    p_screen = crf_to_parameters(23, tune="screen")
    p_anime = crf_to_parameters(23, tune="anime")
    p_nat = crf_to_parameters(23, tune="natural")

    # Natural video has optical noise so base tau is higher than clean screen
    assert p_screen["tau"] < p_anime["tau"] < p_nat["tau"]


def test_parse_bitrate_string():
    assert parse_bitrate_string("1500k") == 1500.0
    assert parse_bitrate_string("1500kbps") == 1500.0
    assert parse_bitrate_string("2M") == 2000.0
    assert parse_bitrate_string("2mbps") == 2000.0
    assert parse_bitrate_string("800") == 800.0
    assert parse_bitrate_string(800000) == 800.0


def test_rate_controller_buffer_feedback():
    target_kbps = 500.0
    rc = RateController(target_bitrate_kbps=target_kbps, fps=30.0, mode="cbr")

    initial_params = rc.get_chunk_parameters(chunk_frames=30)
    tau_init = initial_params["tau"]

    # Simulate an oversized chunk that overshoots target bits by 3x
    # 1 second of video at 500 kbps is 500,000 bits = 62,500 bytes
    heavy_bytes = 62500 * 3
    rc.update(chunk_frames=30, actual_bytes=heavy_bytes)

    # Next chunk must demand higher tau (lower quality) to drain buffer
    next_params = rc.get_chunk_parameters(chunk_frames=30)
    assert next_params["tau"] > tau_init

    # Now simulate 5 consecutive light chunks to allow buffer to drain
    for _ in range(5):
        rc.update(chunk_frames=30, actual_bytes=5000)

    recovered_params = rc.get_chunk_parameters(chunk_frames=30)
    assert recovered_params["tau"] < next_params["tau"]

    stats = rc.get_stats()
    assert stats.frames_encoded == 180
    assert stats.chunks_encoded == 6
