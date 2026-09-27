import struct

import imageio.v2 as imageio
import numpy as np
import pytest
from test_media import make_video

from timeflowcodec.chunked import RECORD, iter_frames, records
from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.format import read_header
from timeflowcodec.jobs import JobCancelled
from timeflowcodec.media import probe_media


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_window_rejected_before_source_access(tmp_path, window):
    target = tmp_path / "video.tfc"
    target.write_bytes(b"original")
    with pytest.raises(ValueError, match="window must be a positive integer"):
        encode_video_to_tfc("missing-source.mp4", str(target), container_version=5, window=window)
    assert target.read_bytes() == b"original"


def test_chunked_roundtrip_and_exact_timing(tmp_path):
    source, encoded, restored = [tmp_path / n for n in ("source.mp4", "encoded.tfc", "restored.mp4")]
    make_video(source, "30000/1001")
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=3, tau=0, payload_comp_type=3)
    with encoded.open("rb") as stream:
        header = read_header(stream)
    assert (header["fps_num"], header["fps_den"]) == (30000, 1001)
    chunks = list(records(encoded))
    assert len(chunks) == 5
    frames = np.stack(list(iter_frames(encoded)))
    reader = imageio.get_reader(source)
    try:
        expected = np.stack(list(reader.iter_data()))
    finally:
        reader.close()
    assert frames.shape == expected.shape
    assert np.max(np.abs(frames.astype(int) - expected.astype(int))) <= 1
    decode_tfc_to_video(str(encoded), str(restored), stream_output=True)
    info = probe_media(restored)
    assert (info["fps_num"], info["fps_den"], info["num_frames"]) == (30000, 1001, 15)


def test_chunked_audio_roundtrip(tmp_path):
    source, encoded, restored = [tmp_path / n for n in ("source.mp4", "encoded.tfc", "restored.mp4")]
    make_video(source, audio=True)
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=4)
    decode_tfc_to_video(str(encoded), str(restored), stream_output=True)
    assert probe_media(restored)["has_audio"]


@pytest.mark.parametrize("mutation", ["truncated", "checksum", "trailing", "size"])
def test_chunked_corruption_preserves_output(tmp_path, mutation):
    source, encoded, restored = [tmp_path / n for n in ("source.mp4", "encoded.tfc", "restored.mp4")]
    make_video(source)
    encode_video_to_tfc(str(source), str(encoded), container_version=5)
    data = bytearray(encoded.read_bytes())
    if mutation == "truncated":
        data = data[:-1]
    elif mutation == "checksum":
        data[-RECORD.size - 1] ^= 1
    elif mutation == "trailing":
        data += b"junk"
    else:
        offset = 8 + struct.unpack("<I", data[4:8])[0]
        data[offset + 4 : offset + 12] = struct.pack("<Q", 2**40)
    encoded.write_bytes(data)
    restored.write_bytes(b"original")
    with pytest.raises(ValueError):
        decode_tfc_to_video(str(encoded), str(restored), stream_output=True)
    assert restored.read_bytes() == b"original"


def test_cancel_before_encode_preserves_output(tmp_path):
    target = tmp_path / "video.tfc"
    target.write_bytes(b"original")
    with pytest.raises(JobCancelled):
        encode_video_to_tfc("unused", str(target), container_version=5, cancel_check=lambda: True)
    assert target.read_bytes() == b"original"


def test_chunked_disk_cache_is_exact_and_uses_source_rate(tmp_path):
    from timeflowcodec.player_decode import DiskFrames, decode_cache

    source, encoded, cache = [tmp_path / n for n in ("source.mp4", "encoded.tfc", "frames.npy")]
    make_video(source, "30000/1001", audio=True)
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=3)
    decode_cache(encoded, cache)
    disk = DiskFrames(cache)
    try:
        expected = decode_tfc_to_video(str(encoded), None)
        assert disk.shape == expected.shape
        for index in (0, 7, 14):
            np.testing.assert_array_equal(disk[index], expected[index])
    finally:
        disk._mmap.close()
    import json

    metadata = json.loads(cache.with_suffix(".json").read_text())
    assert (metadata["fps_num"], metadata["fps_den"]) == (30000, 1001)
    assert cache.with_suffix(".mka").stat().st_size > 0
