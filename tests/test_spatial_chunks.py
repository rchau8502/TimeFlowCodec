"""Adaptive native packing preserves pixels and keeps the smaller representation."""

import numpy as np
import pytest

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.format import read_header


@pytest.mark.parametrize("version", [3, 4])
def test_spatial_parser_keeps_const_values_as_arrays(tmp_path, version):
    import struct

    import timeflowcodec.chunked as chunked
    from timeflowcodec.format import parse_plane_payload_v3, parse_plane_payload_v4

    background = np.random.default_rng(20).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    frames = np.array([np.roll(background, 2 * t, axis=1) for t in range(16)])
    destination = tmp_path / "spatial.tfc"
    chunked._encode_rgb_chunk(frames, destination, dict(preset="screen", tau=0, container_version=version))
    with destination.open("rb") as stream:
        header = read_header(stream)
        size = struct.unpack("<I", stream.read(4))[0]
        parser = parse_plane_payload_v3 if version == 3 else parse_plane_payload_v4
        segments = parser(stream.read(size), header["payload_comp_type"], compact_const=True)
    for index, segment in enumerate(segments):
        assert segment["tfc_params"] == {}
        np.testing.assert_array_equal(segment["const_values"], frames[index, :, :, 0].reshape(-1))


@pytest.mark.parametrize("version", [3, 4])
def test_moving_texture_selects_smaller_spatial_packing(tmp_path, version):
    import timeflowcodec.chunked as chunked

    background = np.random.default_rng(20).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    frames = np.array([np.roll(background, 2 * t, axis=1) for t in range(16)])
    destination = tmp_path / "moving.tfc"
    chunked._encode_rgb_chunk(frames, destination, dict(preset="screen", tau=0, container_version=version))
    assert destination.stat().st_size < 15_100
    np.testing.assert_array_equal(decode_tfc_to_video(str(destination), None), frames)
    with destination.open("rb") as stream:
        assert read_header(stream)["segment_count"] == len(frames)


def test_adaptive_packing_keeps_efficient_temporal_model(tmp_path):
    import timeflowcodec.chunked as chunked

    frames = np.array([np.full((24, 32, 3), 20 + 2 * t, dtype=np.uint8) for t in range(16)])
    original, adaptive = tmp_path / "temporal.tfc", tmp_path / "adaptive.tfc"
    options = dict(preset="screen", tau=0, container_version=3)
    encode_video_to_tfc(frames, str(original), **options)
    chunked._encode_rgb_chunk(frames, adaptive, options)
    assert adaptive.read_bytes() == original.read_bytes()


@pytest.mark.parametrize("options", [dict(preset="anime"), dict(preset="screen", q_step=8), dict(tiling=2)])
def test_spatial_candidate_preserves_other_encoding_policies(tmp_path, options):
    import timeflowcodec.chunked as chunked

    frames = np.random.default_rng(8).integers(0, 256, (4, 8, 8, 3), dtype=np.uint8)
    original, adaptive = tmp_path / "original.tfc", tmp_path / "adaptive.tfc"
    encode_video_to_tfc(frames, str(original), **options)
    chunked._encode_rgb_chunk(frames, adaptive, options)
    assert adaptive.read_bytes() == original.read_bytes()
