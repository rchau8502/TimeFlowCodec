import io
import struct

import numpy as np
import pytest

from timeflowcodec import format as fmt


@pytest.mark.parametrize("comp", [0, 1, 2, 3])
def test_decompression_ceiling(comp):
    compressed = fmt._compress_payload(b"x" * 1024, comp)
    with pytest.raises(ValueError, match="limit"):
        fmt._decompress_payload(compressed, comp, max_decoded_bytes=100)


def test_rle_ceiling_and_zero_run():
    for data in [struct.pack("<HB", 101, 0), struct.pack("<HB", 0, 0)]:
        with pytest.raises(ValueError):
            fmt._rle_decode(data, max_decoded_bytes=100)


@pytest.mark.parametrize(
    "updates",
    [
        dict(width=0),
        dict(num_frames=0),
        dict(width=2**31),
        dict(color_format=99),
        dict(bits_per_mode=8),
        dict(segment_count=0),
    ],
)
def test_invalid_headers(updates):
    header = dict(version=4, width=2, height=2, num_frames=2)
    header.update(updates)
    stream = io.BytesIO()
    fmt.write_header(stream, header)
    stream.seek(0)
    with pytest.raises(ValueError):
        fmt.read_header(stream)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize(
    "field,value", [("length", 0), ("mode", 3), ("scale", float("nan")), ("trailing", 1), ("logical", 2**31)]
)
def test_invalid_scan_payloads(version, field, value):
    length, logical, mode, scale = 2, 1, 1, 1.0
    if field == "length":
        length = value
    if field == "logical":
        logical = value
    if field == "mode":
        mode = value
    if field == "scale":
        scale = value
    header = struct.pack("<IIIffIII", length, logical, 3, scale, 1.0, int(mode == 1), 0, 0)
    if version == 4:
        header += struct.pack("<I", 0)
    payload = struct.pack("<I", 1) + header + struct.pack("<HB", 1, mode) + (b"\x20" if mode == 1 else b"")
    if field == "trailing":
        payload += b"x"
    with pytest.raises(ValueError):
        getattr(fmt, f"parse_plane_payload_v{version}")(payload, 0)


def test_v1_invalid_value_count_is_valueerror():
    payload = struct.pack("<IB3sIBBH", 1, 0, b"\0" * 3, 0, 1, 0, 0)
    with pytest.raises(ValueError):
        fmt.parse_plane_payload_v1(payload, 2, 0)


def test_v2_rejects_index_outside_mode_map():
    segment = dict(length=2, modes=np.array([1], dtype=np.uint8), tfc_params={9: dict(mode=1, a=20)}, fb_params={})
    with pytest.raises(ValueError):
        fmt.parse_plane_payload_v2(fmt.build_plane_payload_v2([segment], 0), 0, 1)


def test_invalid_motion_flag_and_code():
    for flag, code in [(2, 0), (1, 255)]:
        payload = (
            struct.pack("<IIIIffIIII", 1, 2, 1, 3, 1.0, 1.0, 1, 0, 0, flag)
            + struct.pack("<HB", 1, 1)
            + bytes([code, 20])
        )
        with pytest.raises(ValueError):
            fmt.parse_plane_payload_v4(payload, 0)


@pytest.mark.parametrize("comp", [1, 2, 3])
@pytest.mark.parametrize("mutation", ["truncated", "trailing"])
def test_compressed_payload_must_be_complete(comp, mutation):
    payload = fmt._compress_payload(b"hello" * 50, comp)
    payload = payload[:-1] if mutation == "truncated" else payload + b"junk"
    with pytest.raises(ValueError):
        fmt._decompress_payload(payload, comp)


def test_atomic_destination_preserves_existing_file(tmp_path):
    from timeflowcodec.atomic import atomic_destination

    target = tmp_path / "output.tfc"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with atomic_destination(target) as temporary:
            from pathlib import Path

            Path(temporary).write_bytes(b"incomplete")
            raise RuntimeError("simulated write failure")
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_encode_payload_failure_preserves_existing_file(tmp_path, monkeypatch):
    from timeflowcodec import encoder

    target = tmp_path / "output.tfc"
    target.write_bytes(b"original")
    monkeypatch.setattr(encoder.imageio, "get_reader", lambda _: iter(np.zeros((2, 2, 2, 3), dtype=np.uint8)))

    def fail(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(encoder, "build_plane_payload_v3", fail)
    with pytest.raises(RuntimeError, match="serialization failed"):
        encoder.encode_video_to_tfc("synthetic", str(target))
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_export_writer_failure_closes_writer_and_preserves_file(tmp_path, monkeypatch):
    from timeflowcodec import decoder, encoder

    source, target = tmp_path / "input.tfc", tmp_path / "output.mp4"
    target.write_bytes(b"original")
    monkeypatch.setattr(encoder.imageio, "get_reader", lambda _: iter(np.zeros((2, 2, 2, 3), dtype=np.uint8)))
    encoder.encode_video_to_tfc("synthetic", str(source))

    class Writer:
        closed = False

        def append_data(self, frame):
            raise RuntimeError("export failed")

        def close(self):
            self.closed = True

    writer = Writer()
    monkeypatch.setattr(decoder.imageio, "get_writer", lambda *a, **kw: writer)
    with pytest.raises(RuntimeError, match="export failed"):
        decoder.decode_tfc_to_video(str(source), str(target), stream_output=True)
    assert writer.closed
    assert target.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.tfc", "output.mp4"]
