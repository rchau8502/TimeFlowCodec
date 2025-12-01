"""Binary format helpers for the TimeFlowCodec (.tfc) container."""
from __future__ import annotations

import io
import struct
import zlib
import lzma
from typing import Dict, Tuple

import numpy as np

from .constants import MODE_TFC_CONST, MODE_TFC_LINEAR

MAGIC = b"TFC1"
VERSION = 1
HEADER_SIZE = 32  # fixed-size header


def pack_modes(modes: np.ndarray, bits_per_mode: int = 2) -> bytes:
    """
    Pack per-pixel modes (values 0-3) into a compact byte array.
    For bits_per_mode=2, four modes are stored per byte, lowest index in LSBs.
    """
    if bits_per_mode != 2:
        raise ValueError("Only 2 bits_per_mode supported in this implementation")
    N = int(modes.size)
    out = bytearray((N + 3) // 4)
    for i in range(0, N, 4):
        m0 = int(modes[i]) & 0x3
        m1 = int(modes[i + 1]) & 0x3 if i + 1 < N else 0
        m2 = int(modes[i + 2]) & 0x3 if i + 2 < N else 0
        m3 = int(modes[i + 3]) & 0x3 if i + 3 < N else 0
        out[i // 4] = m0 | (m1 << 2) | (m2 << 4) | (m3 << 6)
    return bytes(out)


def unpack_modes(buf: bytes, N: int, bits_per_mode: int = 2) -> np.ndarray:
    """
    Unpack packed modes into an array of length N.
    """
    if bits_per_mode != 2:
        raise ValueError("Only 2 bits_per_mode supported in this implementation")
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
    """Write fixed-size header to file-like object."""
    payload_comp_type = int(header.get("payload_comp_type", 0))
    packed = struct.pack(
        "<4sHHIII BB10s",
        MAGIC,
        int(header.get("version", VERSION)),
        HEADER_SIZE,
        int(header["width"]),
        int(header["height"]),
        int(header["num_frames"]),
        int(header.get("modes_bits_per_pix", 2)),
        payload_comp_type,
        b"\x00" * 10,
    )
    f.write(packed)


def read_header(f) -> dict:
    """Read header from file-like object and return a dict."""
    data = f.read(HEADER_SIZE)
    if len(data) != HEADER_SIZE:
        raise ValueError("Failed to read header")
    magic, version, header_size, width, height, num_frames, modes_bits_per_pix, payload_comp_type, _ = struct.unpack(
        "<4sHHIII BB10s", data
    )
    if magic != MAGIC:
        raise ValueError("Invalid magic: %s" % magic)
    if header_size != HEADER_SIZE:
        raise ValueError("Unsupported header size: %d" % header_size)
    return {
        "magic": magic,
        "version": version,
        "header_size": header_size,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "modes_bits_per_pix": modes_bits_per_pix,
        "payload_comp_type": payload_comp_type,
    }


def _compress_payload(buf: bytes, comp_type: int) -> bytes:
    if comp_type == 0:
        return buf
    if comp_type == 1:
        return zlib.compress(buf)
    if comp_type == 2:
        return lzma.compress(buf)
    raise ValueError("Unknown payload compression type: %d" % comp_type)


def _decompress_payload(buf: bytes, comp_type: int) -> bytes:
    if comp_type == 0:
        return buf
    if comp_type == 1:
        return zlib.decompress(buf)
    if comp_type == 2:
        return lzma.decompress(buf)
    raise ValueError("Unknown payload compression type: %d" % comp_type)


def build_payload(
    modes: np.ndarray,
    tfc_params: Dict[int, Dict[str, float]],
    fb_params: Dict[int, np.ndarray],
    header: dict,
    payload_comp_type: int,
) -> bytes:
    """Build and compress the payload buffer according to the .tfc format."""
    buf = io.BytesIO()

    # Param section
    tfc_items = sorted(tfc_params.items(), key=lambda kv: kv[0])
    buf.write(struct.pack("<IB3s", len(tfc_items), 0, b"\x00" * 3))
    for pixel_index, param in tfc_items:
        mode = param["mode"]
        values: Tuple[float, ...]
        if mode == MODE_TFC_CONST:
            values = (float(param["a"]),)
        elif mode == MODE_TFC_LINEAR:
            values = (float(param["a"]), float(param.get("b", 0.0)))
        else:
            raise ValueError("Invalid TFC mode in params: %s" % mode)
        value_count = len(values)
        buf.write(struct.pack("<IBBH", pixel_index, mode, value_count, 0))
        buf.write(struct.pack("<%df" % value_count, *values))

    # Fallback section
    fb_items = sorted(fb_params.items(), key=lambda kv: kv[0])
    buf.write(struct.pack("<IB3s", len(fb_items), 0, b"\x00" * 3))
    T = int(header["num_frames"])
    for pixel_index, seq in fb_items:
        seq_u8 = np.asarray(seq, dtype=np.uint8).reshape(T)
        buf.write(struct.pack("<I", pixel_index))
        buf.write(seq_u8.tobytes())

    comp = _compress_payload(buf.getvalue(), payload_comp_type)
    return comp


def parse_payload(comp_payload: bytes, header: dict, modes: np.ndarray):
    """
    Decompress comp_payload based on payload_comp_type in header,
    parse into:
      - tfc_params: dict[int, dict]  # pixel_index -> {"mode":..., "a":..., "b":...}
      - fb_params: dict[int, np.ndarray]  # pixel_index -> uint8 array of shape (T,)
    """
    payload = _decompress_payload(comp_payload, int(header.get("payload_comp_type", 0)))
    bio = io.BytesIO(payload)

    tfc_params: Dict[int, Dict[str, float]] = {}
    fb_params: Dict[int, np.ndarray] = {}

    # Param section
    raw = bio.read(8)
    if len(raw) != 8:
        raise ValueError("Incomplete payload (param header)")
    num_tfc_pixels, param_encoding_type, _ = struct.unpack("<IB3s", raw)
    if param_encoding_type != 0:
        raise ValueError("Unsupported param encoding type: %d" % param_encoding_type)
    for _ in range(num_tfc_pixels):
        entry_hdr = bio.read(8)
        if len(entry_hdr) != 8:
            raise ValueError("Incomplete TFC param entry")
        pixel_index, mode, value_count, _reserved = struct.unpack("<IBBH", entry_hdr)
        values_raw = bio.read(4 * value_count)
        if len(values_raw) != 4 * value_count:
            raise ValueError("Incomplete TFC param values")
        values = struct.unpack("<%df" % value_count, values_raw)
        param = {"mode": mode, "a": values[0]}
        if value_count == 2:
            param["b"] = values[1]
        tfc_params[pixel_index] = param

    # Fallback section
    raw = bio.read(8)
    if len(raw) != 8:
        raise ValueError("Incomplete payload (fallback header)")
    num_fb_pixels, fb_encoding_type, _ = struct.unpack("<IB3s", raw)
    if fb_encoding_type != 0:
        raise ValueError("Unsupported fallback encoding type: %d" % fb_encoding_type)
    T = int(header["num_frames"])
    for _ in range(num_fb_pixels):
        idx_raw = bio.read(4)
        if len(idx_raw) != 4:
            raise ValueError("Incomplete fallback pixel index")
        pixel_index = struct.unpack("<I", idx_raw)[0]
        seq_raw = bio.read(T)
        if len(seq_raw) != T:
            raise ValueError("Incomplete fallback pixel sequence")
        fb_params[pixel_index] = np.frombuffer(seq_raw, dtype=np.uint8).copy()

    return tfc_params, fb_params
