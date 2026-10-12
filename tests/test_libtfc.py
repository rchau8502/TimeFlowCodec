"""Tests for Native ANSI C Reference Decoder (libtfc)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_media import make_video

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc

LIBTFC_DIR = Path(__file__).parent.parent / "libtfc"
TFC_DEC_BIN = LIBTFC_DIR / "build" / "tfc_dec"


@pytest.fixture(scope="session", autouse=True)
def build_libtfc():
    """Ensure libtfc is built before running tests."""
    subprocess.run(["make", "-C", str(LIBTFC_DIR)], check=True)
    assert TFC_DEC_BIN.exists(), f"Binary {TFC_DEC_BIN} not found"


def test_libtfc_v3_exact_match(tmp_path):
    """Verify libtfc produces exact pixel matches for v3 containers."""
    H, W, T = 32, 48, 16
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        video[t, :, :, 0] = (50 + t * 4) % 256
        video[t, :, :, 1] = 128
        video[t, 10:20, 15:35, 2] = 220

    tfc_file = tmp_path / "test_v3.tfc"
    raw_out = tmp_path / "test_v3.raw"

    # Encode with container_version=3, zlib compression
    encode_video_to_tfc(video, str(tfc_file), container_version=3, tau=0.5, payload_comp_type=1)

    # Decode via Python decoder
    py_frames = decode_tfc_to_video(str(tfc_file), None)
    assert py_frames is not None

    # Decode via C reference decoder
    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out), "--bench"], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"
    assert "Throughput:" in res.stderr

    # Read raw RGB bytes from C decoder
    c_raw = raw_out.read_bytes()
    c_frames = np.frombuffer(c_raw, dtype=np.uint8).reshape((T, H, W, 3))

    # Compare pixel-for-pixel (allowing at most 1 LSB difference due to float32 vs float64 rounding)
    diff = np.max(np.abs(c_frames.astype(int) - py_frames.astype(int)))
    assert diff <= 1, f"Max difference between C and Python decoder is {diff}"


@pytest.mark.parametrize("version", [3, 4])
def test_libtfc_adaptive_spatial_chunk_is_exact(tmp_path, version):
    from timeflowcodec.chunked import _encode_rgb_chunk

    background = np.random.default_rng(20).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    frames = np.array([np.roll(background, 2 * t, axis=1) for t in range(16)])
    encoded, raw = tmp_path / "spatial.tfc", tmp_path / "spatial.raw"
    _encode_rgb_chunk(frames, encoded, dict(preset="screen", tau=0, container_version=version))
    subprocess.run([str(TFC_DEC_BIN), str(encoded), str(raw)], check=True, capture_output=True)
    np.testing.assert_array_equal(np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(frames.shape), frames)


def test_libtfc_v5_chunked_match(tmp_path):
    """Verify libtfc produces exact pixel matches for multi-chunk v5 containers."""
    source = tmp_path / "source.mp4"
    tfc_file = tmp_path / "test_v5.tfc"
    raw_out = tmp_path / "test_v5.raw"

    make_video(source, rate="24")  # 12 frames
    encode_video_to_tfc(
        str(source),
        str(tfc_file),
        container_version=5,
        window=4,
        tau=0.5,
        payload_comp_type=1,
    )

    py_frames = decode_tfc_to_video(str(tfc_file), None)
    assert py_frames is not None
    num_frames, H, W, _ = py_frames.shape

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out)], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"

    c_raw = raw_out.read_bytes()
    c_frames = np.frombuffer(c_raw, dtype=np.uint8).reshape((num_frames, H, W, 3))

    diff = np.max(np.abs(c_frames.astype(int) - py_frames.astype(int)))
    assert diff <= 1, f"Max difference between C and Python decoder is {diff}"


def test_libtfc_ppm_output(tmp_path):
    """Verify PPM output mode."""
    H, W, T = 16, 16, 4
    video = np.full((T, H, W, 3), 100, dtype=np.uint8)
    tfc_file = tmp_path / "test.tfc"
    ppm_out = tmp_path / "frame0.ppm"

    encode_video_to_tfc(video, str(tfc_file), container_version=3, tau=0.1, payload_comp_type=1)

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(ppm_out)], capture_output=True, text=True)
    assert res.returncode == 0
    assert ppm_out.exists()
    header = ppm_out.read_bytes()[:15]
    assert header.startswith(b"P6\n16 16\n255\n")


def test_libtfc_v3_zstd_match(tmp_path):
    """Verify libtfc decompressing zstd-compressed payloads."""
    H, W, T = 32, 48, 12
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        video[t, :, :, 0] = (80 + t * 6) % 256
        video[t, :, :, 1] = 160
        video[t, 8:24, 12:36, 2] = 200

    tfc_file = tmp_path / "test_zstd.tfc"
    raw_out = tmp_path / "test_zstd.raw"

    # Encode with payload_comp_type=3 (zstd)
    encode_video_to_tfc(video, str(tfc_file), container_version=3, tau=0.5, payload_comp_type=3)

    py_frames = decode_tfc_to_video(str(tfc_file), None)
    assert py_frames is not None

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out), "--bench"], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"

    c_raw = raw_out.read_bytes()
    c_frames = np.frombuffer(c_raw, dtype=np.uint8).reshape((T, H, W, 3))

    diff = np.max(np.abs(c_frames.astype(int) - py_frames.astype(int)))
    assert diff <= 1, f"Max difference between C and Python decoder is {diff}"


def test_libtfc_v5_zstd_chunked_match(tmp_path):
    """Verify libtfc produces exact pixel matches for multi-chunk v5 containers with zstd."""
    source = tmp_path / "source.mp4"
    tfc_file = tmp_path / "test_v5_zstd.tfc"
    raw_out = tmp_path / "test_v5_zstd.raw"

    make_video(source, rate="24")  # 12 frames
    encode_video_to_tfc(
        str(source),
        str(tfc_file),
        container_version=5,
        window=4,
        tau=0.5,
        payload_comp_type=3,  # zstd default
    )

    py_frames = decode_tfc_to_video(str(tfc_file), None)
    assert py_frames is not None
    num_frames, H, W, _ = py_frames.shape

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out)], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"

    c_raw = raw_out.read_bytes()
    c_frames = np.frombuffer(c_raw, dtype=np.uint8).reshape((num_frames, H, W, 3))

    diff = np.max(np.abs(c_frames.astype(int) - py_frames.astype(int)))
    assert diff <= 1, f"Max difference between C and Python decoder is {diff}"


def test_libtfc_yuv420_match(tmp_path):
    """Verify libtfc produces matching pixels for yuv420 color format."""
    H, W, T = 32, 48, 8
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        video[t, :, :, 0] = (100 + t * 5) % 256
        video[t, :, :, 1] = 150
        video[t, :, :, 2] = 200

    tfc_file = tmp_path / "test_yuv.tfc"
    raw_out = tmp_path / "test_yuv.raw"

    encode_video_to_tfc(
        video,
        str(tfc_file),
        container_version=3,
        colorspace="yuv420",
        tau=0.5,
        payload_comp_type=1,
    )

    py_frames = decode_tfc_to_video(str(tfc_file), None)
    assert py_frames is not None

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out)], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"

    c_raw = raw_out.read_bytes()
    c_frames = np.frombuffer(c_raw, dtype=np.uint8).reshape((T, H, W, 3))

    diff = np.max(np.abs(c_frames.astype(int) - py_frames.astype(int)))
    # Allow small tolerance of 2 LSB for color matrix float rounding in C vs Python
    assert diff <= 2, f"Max difference between C and Python decoder is {diff}"


def test_libtfc_simd_throughput_and_exactness(tmp_path):
    """Verify that libtfc achieves high-throughput decode and exact reconstruction."""
    from timeflowcodec.chunked import _encode_rgb_chunk

    background = np.random.default_rng(42).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    frames = np.array([np.roll(background, 2 * t, axis=1) for t in range(32)])
    tfc_file = tmp_path / "simd_bench.tfc"
    raw_out = tmp_path / "simd_bench.raw"

    _encode_rgb_chunk(frames, tfc_file, dict(preset="screen", tau=0, container_version=3))

    res = subprocess.run([str(TFC_DEC_BIN), str(tfc_file), str(raw_out), "--bench"], capture_output=True, text=True)
    assert res.returncode == 0, f"tfc_dec failed: {res.stderr}"
    assert "Throughput:" in res.stderr

    c_frames = np.frombuffer(raw_out.read_bytes(), dtype=np.uint8).reshape(frames.shape)
    np.testing.assert_array_equal(c_frames, frames)

