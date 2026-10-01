"""Tests for Multi-Threaded Parallel Chunk Engine in TimeFlowCodec."""

import imageio.v2 as imageio
import numpy as np
import pytest
from test_media import make_video

from timeflowcodec.chunked import iter_frames, records
from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.format import read_header


def test_invalid_threads_rejected(tmp_path):
    target = tmp_path / "video.tfc"
    with pytest.raises(ValueError, match="threads must be a non-negative integer"):
        encode_video_to_tfc("missing.mp4", str(target), container_version=5, threads=-1)


def test_multithreaded_chunked_roundtrip(tmp_path):
    source = tmp_path / "source.mp4"
    encoded_seq = tmp_path / "seq.tfc"
    encoded_mt = tmp_path / "mt.tfc"

    make_video(source, "30000/1001")

    # Encode with 1 thread
    encode_video_to_tfc(str(source), str(encoded_seq), container_version=5, window=3, threads=1)
    # Encode with 2 threads
    encode_video_to_tfc(str(source), str(encoded_mt), container_version=5, window=3, threads=2)

    with encoded_mt.open("rb") as stream:
        header = read_header(stream)
    assert (header["fps_num"], header["fps_den"]) == (30000, 1001)

    chunks_seq = list(records(encoded_seq))
    chunks_mt = list(records(encoded_mt))
    assert len(chunks_mt) == 5
    assert len(chunks_seq) == 5

    # Reconstruct frames from both
    frames_seq = np.stack(list(iter_frames(encoded_seq, threads=1)))
    frames_mt = np.stack(list(iter_frames(encoded_mt, threads=2)))

    reader = imageio.get_reader(source)
    try:
        expected = np.stack(list(reader.iter_data()))
    finally:
        reader.close()

    assert frames_mt.shape == expected.shape
    assert frames_seq.shape == expected.shape
    # Reconstructed multi-threaded output matches single-threaded output bit-for-bit
    assert np.array_equal(frames_mt, frames_seq)


def test_multithreaded_decode(tmp_path):
    source = tmp_path / "source.mp4"
    encoded = tmp_path / "video.tfc"
    restored = tmp_path / "restored.mp4"

    make_video(source)
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=3, threads=2)

    # Decode with threads=2 to output file
    decode_tfc_to_video(str(encoded), str(restored), stream_output=True, threads=2)

    # Decode with threads=2 to numpy array
    frames_arr = decode_tfc_to_video(str(encoded), None, threads=2)
    assert frames_arr is not None
    assert frames_arr.shape[0] == 12


def test_threads_zero_autodetect(tmp_path):
    source = tmp_path / "source.mp4"
    encoded = tmp_path / "auto.tfc"

    make_video(source)
    # threads=0 activates auto-detection of core count
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=3, threads=0)

    frames = decode_tfc_to_video(str(encoded), None, threads=0)
    assert frames is not None
    assert frames.shape[0] == 12
