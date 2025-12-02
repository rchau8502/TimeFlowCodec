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
VERSION = 1
HEADER_SIZE = 32


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


def write_header(f, header: dict) -> None:
    packed = struct.pack(
        "<4sHHIII BBB9s",
        MAGIC,
        int(header.get("version", VERSION)),
        HEADER_SIZE,
        int(header["width"]),
        int(header["height"]),
        int(header["num_frames"]),
        int(header.get("color_format", COLOR_FORMAT_RGB)),
        int(header.get("bits_per_mode", BITS_PER_MODE)),
        int(header.get("payload_comp_type", 0)),
        b"\x00" * 9,
    )
    f.write(packed)


def read_header(f) -> dict:
    data = f.read(HEADER_SIZE)
    if len(data) != HEADER_SIZE:
        raise ValueError("Failed to read header")
    magic, version, header_size, width, height, num_frames, color_format, bits_per_mode, payload_comp_type, _ = struct.unpack(
        "<4sHHIII BBB9s", data
    )
    if magic != MAGIC:
        raise ValueError("Invalid magic")
    if header_size != HEADER_SIZE:
        raise ValueError("Unsupported header size")
    return {
        "magic": magic,
        "version": version,
        "header_size": header_size,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "color_format": color_format,
        "bits_per_mode": bits_per_mode,
        "payload_comp_type": payload_comp_type,
    }


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


def build_plane_payload(
    tfc_params: Dict[int, Dict[str, float]],
    fb_params: Dict[int, np.ndarray],
    T: int,
    payload_comp_type: int,
) -> bytes:
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


def parse_plane_payload(comp_payload: bytes, T: int, payload_comp_type: int):
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
