"""RGB per-pixel decoder for TimeFlowCodec."""
from __future__ import annotations

import struct
import numpy as np

from .constants import (
    BITS_PER_MODE,
    COLOR_FORMAT_RGB,
    MODE_FB_RAW,
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
    PLANE_B,
    PLANE_G,
    PLANE_R,
)
from .format import parse_plane_payload, read_header, unpack_modes
from .models import reconstruct_constant_sequence, reconstruct_linear_sequence
from .utils import save_video_from_rgb


def _reconstruct_plane(modes, tfc_params, fb_params, T: int, N: int) -> np.ndarray:
    flat = np.zeros((T, N), dtype=np.uint8)
    for p in range(N):
        mode = int(modes[p])
        if mode == MODE_TFC_CONST:
            param = tfc_params.get(p)
            if param is None:
                raise ValueError(f"Missing params for pixel {p} (CONST)")
            s_hat = reconstruct_constant_sequence(float(param["a"]), T)
        elif mode == MODE_TFC_LINEAR:
            param = tfc_params.get(p)
            if param is None:
                raise ValueError(f"Missing params for pixel {p} (LINEAR)")
            s_hat = reconstruct_linear_sequence(float(param["a"]), float(param.get("b", 0.0)), T)
        elif mode == MODE_FB_RAW:
            seq = fb_params.get(p)
            if seq is None:
                raise ValueError(f"Missing raw data for pixel {p}")
            s_hat = seq.astype(np.float32)
        else:
            raise ValueError(f"Unknown mode {mode} for pixel {p}")
        flat[:, p] = np.clip(np.rint(s_hat), 0, 255).astype(np.uint8)
    return flat


def decode_tfc_to_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
) -> None:
    """Decode .tfc RGB file back into an RGB video."""

    with open(input_path, "rb") as f:
        header = read_header(f)
        if header.get("color_format") != COLOR_FORMAT_RGB:
            raise ValueError("Unsupported color format in TFC file")
        if header.get("bits_per_mode") != BITS_PER_MODE:
            raise ValueError("Unexpected bits_per_mode in TFC file")
        W = int(header["width"])
        H = int(header["height"])
        T = int(header["num_frames"])
        N = W * H
        payload_comp_type = int(header.get("payload_comp_type", 0))

        remaining = f.read()

    offset = 0
    mode_table_size = (N + 3) // 4
    plane_modes = {}
    plane_params = {}

    for plane in (PLANE_R, PLANE_G, PLANE_B):
        if offset + mode_table_size > len(remaining):
            raise ValueError("Incomplete mode table data")
        mode_bytes = remaining[offset:offset + mode_table_size]
        offset += mode_table_size
        modes = unpack_modes(mode_bytes, N, bits_per_mode=BITS_PER_MODE)

        if offset + 4 > len(remaining):
            raise ValueError("Missing payload size")
        payload_size = struct.unpack("<I", remaining[offset:offset + 4])[0]
        offset += 4
        if offset + payload_size > len(remaining):
            raise ValueError("Incomplete payload data")
        comp_payload = remaining[offset:offset + payload_size]
        offset += payload_size

        tfc_params, fb_params = parse_plane_payload(
            comp_payload, T, payload_comp_type
        )
        plane_modes[plane] = modes
        plane_params[plane] = (tfc_params, fb_params)

    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    flat = frames.reshape(T, W * H, 3)

    for plane in (PLANE_R, PLANE_G, PLANE_B):
        modes = plane_modes[plane]
        tfc_params, fb_params = plane_params[plane]
        channel_flat = _reconstruct_plane(modes, tfc_params, fb_params, T, N)
        flat[:, :, plane] = channel_flat

    save_video_from_rgb(frames, output_path, fps=fps)
    print(f"Decoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}")
