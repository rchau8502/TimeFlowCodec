"""Binary format helpers for RGB TimeFlowCodec (.tfc)."""

from __future__ import annotations

import io
import struct
import zlib
import lzma
from typing import Dict, Tuple

import numpy as np

from .constants import BITS_PER_MODE, COLOR_FORMAT_RGB, MODE_TFC_CONST, MODE_TFC_LINEAR

MAGIC = b"TFC1"
VERSION_V1 = 1
VERSION_V2 = 2
HEADER_SIZE_V1 = 32
HEADER_SIZE_V2 = 40


def pack_modes(modes: np.ndarray, bits_per_mode: int = BITS_PER_MODE) -> bytes:
    if bits_per_mode != 2:
        raise ValueError("Only 2-bit modes supported")
    N = int(modes.size)
    out = bytearray((N + 3) // 4)
    for i in range(0, N, 4):
        m0 = int(modes[i]) & 0x3
        m1 = int(modes[i + 1]) & 0x3 if i + 1 < N else 0
        m2 = int(modes[i + 2]) & 0x3 if i + 2 < N else 0
        m3 = int(modes[i + 3]) & 0x3 if i + 3 < N else 0
        out[i // 4] = m0 | (m1 << 2) | (m2 << 4) | (m3 << 6)
    return bytes(out)


def unpack_modes(buf: bytes, N: int, bits_per_mode: int = BITS_PER_MODE) -> np.ndarray:
    if bits_per_mode != 2:
        raise ValueError("Only 2-bit modes supported")
    expected_len = (N + 3) // 4
    if len(buf) < expected_len:
        raise ValueError("Buffer too small for mode table")
    modes = np.zeros((N,), dtype=np.uint8)
    for i in range(N):
        byte_idx = i // 4
        shift = (i % 4) * 2
        modes[i] = (buf[byte_idx] >> shift) & 0x3
    return modes


def _rle_encode(data: bytes) -> bytes:
    out = bytearray()
    if not data:
        return b""
    run_val = data[0]
    run_len = 1
    for b in data[1:]:
        if b == run_val and run_len < 0xFFFF:
            run_len += 1
        else:
            out += struct.pack("<HB", run_len, run_val)
            run_val = b
            run_len = 1
    out += struct.pack("<HB", run_len, run_val)
    return bytes(out)


def _rle_decode(data: bytes) -> bytes:
    out = bytearray()
    bio = io.BytesIO(data)
    while True:
        hdr = bio.read(3)
        if not hdr:
            break
        if len(hdr) != 3:
            raise ValueError("Corrupt RLE stream")
        run_len, run_val = struct.unpack("<HB", hdr)
        out += bytes([run_val]) * run_len
    return bytes(out)


def write_header(f, header: dict) -> None:
    """Write header for v1 or v2."""
    version = int(header.get("version", VERSION_V2))
    if version == VERSION_V1:
        git_hash_bytes = header.get("encoder_git_hash", b"")
        if isinstance(git_hash_bytes, str):
            git_hash_bytes = git_hash_bytes.encode("utf-8")
        git_hash_bytes = git_hash_bytes[:9].ljust(9, b"\x00")
        packed = struct.pack(
            "<4sHHIII BBB9s",
            MAGIC,
            VERSION_V1,
            HEADER_SIZE_V1,
            int(header["width"]),
            int(header["height"]),
            int(header["num_frames"]),
            int(header.get("color_format", COLOR_FORMAT_RGB)),
            int(header.get("bits_per_mode", BITS_PER_MODE)),
            int(header.get("payload_comp_type", 0)),
            git_hash_bytes,
        )
        f.write(packed)
    else:
        git_hash_bytes = header.get("encoder_git_hash", b"")
        if isinstance(git_hash_bytes, str):
            git_hash_bytes = git_hash_bytes.encode("utf-8")
        git_hash_bytes = git_hash_bytes[:8].ljust(8, b"\x00")
        packed = struct.pack(
            "<4sHHIII BBB HH 5s 8s",
            MAGIC,
            VERSION_V2,
            HEADER_SIZE_V2,
            int(header["width"]),
            int(header["height"]),
            int(header["num_frames"]),
            int(header.get("color_format", COLOR_FORMAT_RGB)),
            int(header.get("bits_per_mode", BITS_PER_MODE)),
            int(header.get("payload_comp_type", 0)),
            int(header.get("tiling", 0)),
            int(header.get("segment_count", 1)),
            b"\x00" * 5,
            git_hash_bytes,
        )
        f.write(packed)


def read_header(f) -> dict:
    prefix = f.read(4)
    if prefix != MAGIC:
        raise ValueError("Invalid magic")
    version_bytes = f.read(2)
    if len(version_bytes) != 2:
        raise ValueError("Failed to read version")
    version = struct.unpack("<H", version_bytes)[0]
    if version == VERSION_V1:
        rest = f.read(HEADER_SIZE_V1 - 6)
        if len(rest) != HEADER_SIZE_V1 - 6:
            raise ValueError("Incomplete header")
        (
            header_size,
            width,
            height,
            num_frames,
            color_format,
            bits_per_mode,
            payload_comp_type,
            git_hash,
        ) = struct.unpack("<HIII BBB9s", rest)
        if header_size != HEADER_SIZE_V1:
            raise ValueError("Unsupported header size")
        return {
            "magic": MAGIC,
            "version": VERSION_V1,
            "header_size": header_size,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "color_format": color_format,
            "bits_per_mode": bits_per_mode,
            "payload_comp_type": payload_comp_type,
            "encoder_git_hash": git_hash.rstrip(b"\x00").decode(
                "utf-8", errors="ignore"
            ),
            "tiling": 0,
            "segment_count": 1,
        }
    elif version == VERSION_V2:
        rest = f.read(HEADER_SIZE_V2 - 6)
        if len(rest) != HEADER_SIZE_V2 - 6:
            raise ValueError("Incomplete header")
        (
            header_size,
            width,
            height,
            num_frames,
            color_format,
            bits_per_mode,
            payload_comp_type,
            tiling,
            segment_count,
            _reserved,
            git_hash,
        ) = struct.unpack("<HIII BBB HH 5s 8s", rest)
        if header_size != HEADER_SIZE_V2:
            raise ValueError("Unsupported header size")
        return {
            "magic": MAGIC,
            "version": VERSION_V2,
            "header_size": header_size,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "color_format": color_format,
            "bits_per_mode": bits_per_mode,
            "payload_comp_type": payload_comp_type,
            "tiling": tiling,
            "segment_count": segment_count,
            "encoder_git_hash": git_hash.rstrip(b"\x00").decode(
                "utf-8", errors="ignore"
            ),
        }
    else:
        raise ValueError(f"Unsupported version {version}")


def _compress_payload(buf: bytes, comp_type: int) -> bytes:
    if comp_type == 0:
        return buf
    if comp_type == 1:
        return zlib.compress(buf)
    if comp_type == 2:
        return lzma.compress(buf)
    raise ValueError(f"Unknown compression type {comp_type}")


def _decompress_payload(buf: bytes, comp_type: int) -> bytes:
    if comp_type == 0:
        return buf
    if comp_type == 1:
        return zlib.decompress(buf)
    if comp_type == 2:
        return lzma.decompress(buf)
    raise ValueError(f"Unknown compression type {comp_type}")


def build_plane_payload_v1(
    tfc_params: Dict[int, Dict[str, float]],
    fb_params: Dict[int, np.ndarray],
    T: int,
    payload_comp_type: int,
) -> bytes:
    """Legacy float32 payload (v1)."""
    buf = io.BytesIO()
    tfc_items = sorted(tfc_params.items(), key=lambda kv: kv[0])
    buf.write(struct.pack("<IB3s", len(tfc_items), 0, b"\x00" * 3))
    for pixel_index, param in tfc_items:
        mode = param["mode"]
        if mode == MODE_TFC_CONST:
            values: Tuple[float, ...] = (float(param["a"]),)
        elif mode == MODE_TFC_LINEAR:
            values = (float(param["a"]), float(param.get("b", 0.0)))
        else:
            raise ValueError(f"Invalid mode in params: {mode}")
        value_count = len(values)
        buf.write(struct.pack("<IBBH", pixel_index, mode, value_count, 0))
        buf.write(struct.pack(f"<{value_count}f", *values))

    fb_items = sorted(fb_params.items(), key=lambda kv: kv[0])
    buf.write(struct.pack("<IB3s", len(fb_items), 0, b"\x00" * 3))
    for pixel_index, seq in fb_items:
        seq_u8 = np.asarray(seq, dtype=np.uint8).reshape(T)
        buf.write(struct.pack("<I", pixel_index))
        buf.write(seq_u8.tobytes())

    return _compress_payload(buf.getvalue(), payload_comp_type)


def parse_plane_payload_v1(comp_payload: bytes, T: int, payload_comp_type: int):
    payload = _decompress_payload(comp_payload, payload_comp_type)
    bio = io.BytesIO(payload)
    tfc_params: Dict[int, Dict[str, float]] = {}
    fb_params: Dict[int, np.ndarray] = {}

    raw = bio.read(8)
    if len(raw) != 8:
        raise ValueError("Incomplete payload (param header)")
    num_tfc, enc_type, _ = struct.unpack("<IB3s", raw)
    if enc_type != 0:
        raise ValueError(f"Unsupported param encoding {enc_type}")
    for _ in range(num_tfc):
        entry_hdr = bio.read(8)
        if len(entry_hdr) != 8:
            raise ValueError("Incomplete TFC entry")
        pixel_index, mode, value_count, _reserved = struct.unpack("<IBBH", entry_hdr)
        values_raw = bio.read(4 * value_count)
        if len(values_raw) != 4 * value_count:
            raise ValueError("Incomplete TFC values")
        values = struct.unpack(f"<{value_count}f", values_raw)
        param = {"mode": mode, "a": values[0]}
        if value_count == 2:
            param["b"] = values[1]
        tfc_params[pixel_index] = param

    raw = bio.read(8)
    if len(raw) != 8:
        raise ValueError("Incomplete payload (raw header)")
    num_raw, raw_enc, _ = struct.unpack("<IB3s", raw)
    if raw_enc != 0:
        raise ValueError(f"Unsupported raw encoding {raw_enc}")
    for _ in range(num_raw):
        idx_raw = bio.read(4)
        if len(idx_raw) != 4:
            raise ValueError("Incomplete raw index")
        pixel_index = struct.unpack("<I", idx_raw)[0]
        seq_raw = bio.read(T)
        if len(seq_raw) != T:
            raise ValueError("Incomplete raw sequence")
        fb_params[pixel_index] = np.frombuffer(seq_raw, dtype=np.uint8).copy()

    return tfc_params, fb_params


def build_plane_payload_v2(
    segments: list[dict],
    payload_comp_type: int,
) -> bytes:
    """Version 2 payload with quantized params and RLE mode map, per segment."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(segments)))
    for seg in segments:
        modes: np.ndarray = seg["modes"]
        tfc_params: Dict[int, Dict[str, float]] = seg["tfc_params"]
        fb_params: Dict[int, np.ndarray] = seg["fb_params"]
        T = int(seg["length"])

        mode_bytes = pack_modes(modes, bits_per_mode=BITS_PER_MODE)
        mode_rle = _rle_encode(mode_bytes)
        buf.write(struct.pack("<II", T, len(mode_rle)))
        buf.write(mode_rle)

        linear_params = [p for p in tfc_params.values() if p["mode"] == MODE_TFC_LINEAR]
        max_abs_b = max((abs(p.get("b", 0.0)) for p in linear_params), default=0.0)
        b_scale = 1.0 if max_abs_b == 0 else max_abs_b / 32767.0

        const_items = [
            (idx, p) for idx, p in tfc_params.items() if p["mode"] == MODE_TFC_CONST
        ]
        buf.write(struct.pack("<I", len(const_items)))
        for pixel_index, param in sorted(const_items, key=lambda kv: kv[0]):
            a_q = np.uint8(np.clip(round(param["a"]), 0, 255))
            buf.write(struct.pack("<IB", pixel_index, a_q))

        linear_items = [
            (idx, p) for idx, p in tfc_params.items() if p["mode"] == MODE_TFC_LINEAR
        ]
        buf.write(struct.pack("<If", len(linear_items), float(b_scale)))
        for pixel_index, param in sorted(linear_items, key=lambda kv: kv[0]):
            a_q = np.uint8(np.clip(round(param["a"]), 0, 255))
            b_q = int(np.clip(round(param.get("b", 0.0) / b_scale), -32768, 32767))
            buf.write(struct.pack("<IBh", pixel_index, a_q, b_q))

        fb_items = sorted(fb_params.items(), key=lambda kv: kv[0])
        buf.write(struct.pack("<I", len(fb_items)))
        for pixel_index, seq in fb_items:
            seq_u8 = np.asarray(seq, dtype=np.uint8).reshape(T)
            buf.write(struct.pack("<I", pixel_index))
            buf.write(seq_u8.tobytes())

    return _compress_payload(buf.getvalue(), payload_comp_type)


def parse_plane_payload_v2(comp_payload: bytes, payload_comp_type: int, N: int):
    payload = _decompress_payload(comp_payload, payload_comp_type)
    bio = io.BytesIO(payload)
    seg_count_raw = bio.read(4)
    if len(seg_count_raw) != 4:
        raise ValueError("Incomplete v2 payload (segment count)")
    seg_count = struct.unpack("<I", seg_count_raw)[0]
    segments = []
    for _ in range(seg_count):
        seg_hdr = bio.read(8)
        if len(seg_hdr) != 8:
            raise ValueError("Incomplete v2 segment header")
        T, mode_len = struct.unpack("<II", seg_hdr)
        mode_rle = bio.read(mode_len)
        mode_bytes = _rle_decode(mode_rle)
        modes = unpack_modes(mode_bytes, N, bits_per_mode=BITS_PER_MODE)

        tfc_params: Dict[int, Dict[str, float]] = {}
        fb_params: Dict[int, np.ndarray] = {}

        const_count_raw = bio.read(4)
        if len(const_count_raw) != 4:
            raise ValueError("Incomplete const header")
        const_count = struct.unpack("<I", const_count_raw)[0]
        for _ in range(const_count):
            rec = bio.read(5)
            if len(rec) != 5:
                raise ValueError("Incomplete const entry")
            pixel_index, a_q = struct.unpack("<IB", rec)
            tfc_params[pixel_index] = {"mode": MODE_TFC_CONST, "a": float(a_q)}

        linear_hdr = bio.read(8)
        if len(linear_hdr) != 8:
            raise ValueError("Incomplete linear header")
        linear_count, b_scale = struct.unpack("<If", linear_hdr)
        for _ in range(linear_count):
            rec = bio.read(7)
            if len(rec) != 7:
                raise ValueError("Incomplete linear entry")
            pixel_index, a_q, b_q = struct.unpack("<IBh", rec)
            tfc_params[pixel_index] = {
                "mode": MODE_TFC_LINEAR,
                "a": float(a_q),
                "b": float(b_q) * float(b_scale),
            }

        raw_count_raw = bio.read(4)
        if len(raw_count_raw) != 4:
            raise ValueError("Incomplete raw header")
        raw_count = struct.unpack("<I", raw_count_raw)[0]
        for _ in range(raw_count):
            idx_raw = bio.read(4)
            if len(idx_raw) != 4:
                raise ValueError("Incomplete raw index")
            pixel_index = struct.unpack("<I", idx_raw)[0]
            seq_raw = bio.read(T)
            if len(seq_raw) != T:
                raise ValueError("Incomplete raw sequence")
            fb_params[pixel_index] = np.frombuffer(seq_raw, dtype=np.uint8).copy()

        segments.append(
            {
                "length": T,
                "modes": modes,
                "tfc_params": tfc_params,
                "fb_params": fb_params,
            }
        )

    return segments
