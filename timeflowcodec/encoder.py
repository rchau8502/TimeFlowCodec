"""RGB per-pixel encoder for TimeFlowCodec."""
from __future__ import annotations

import struct
import numpy as np

from .constants import (
    BITS_PER_MODE,
    COLOR_FORMAT_RGB,
    DEFAULT_SLOPE_THRESHOLD,
    DEFAULT_TAU,
    MODE_FB_RAW,
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
    PLANE_B,
    PLANE_G,
    PLANE_R,
)
from .format import build_plane_payload, pack_modes, write_header
from .utils import load_video_rgb


def _encode_plane(channel_data: np.ndarray, channel_u8: np.ndarray, tau: float, slope_threshold: float):
    """
    Vectorized per-plane encode: fit CONST/LINEAR per pixel/channel and gather fallbacks.
    """
    T, N = channel_data.shape
    modes = np.full((N,), MODE_FB_RAW, dtype=np.uint8)
    tfc_params: dict[int, dict] = {}
    fb_params: dict[int, np.ndarray] = {}

    n = float(T)
    sum_t = n * (n - 1.0) / 2.0
    sum_t2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    t = np.arange(T, dtype=np.float64)

    sum_s = np.sum(channel_data, axis=0, dtype=np.float64)
    sum_ts = np.sum(channel_data * t[:, None], axis=0, dtype=np.float64)
    denom = n * sum_t2 - sum_t * sum_t
    if denom == 0.0:
        b = np.zeros_like(sum_s, dtype=np.float64)
    else:
        b = (n * sum_ts - sum_t * sum_s) / denom
    a = (sum_s - b * sum_t) / n

    s_lin = a[None, :] + b[None, :] * t[:, None]
    diff = channel_data - s_lin
    D_tfc = np.mean(diff * diff, axis=0)
    D_sig = np.mean(channel_data * channel_data, axis=0) + 1e-8
    r = D_tfc / D_sig

    modeled = r <= tau
    const_mask = modeled & (np.abs(b) < slope_threshold)
    zero_mask = ~np.any(channel_data != 0, axis=0)
    const_mask |= zero_mask
    linear_mask = modeled & (~const_mask)
    fallback_mask = ~(const_mask | linear_mask)

    modes[const_mask] = MODE_TFC_CONST
    modes[linear_mask] = MODE_TFC_LINEAR

    for idx in np.nonzero(const_mask)[0]:
        tfc_params[idx] = {"mode": MODE_TFC_CONST, "a": float(a[idx])}
    for idx in np.nonzero(linear_mask)[0]:
        tfc_params[idx] = {"mode": MODE_TFC_LINEAR, "a": float(a[idx]), "b": float(b[idx])}
    for idx in np.nonzero(fallback_mask)[0]:
        fb_params[idx] = channel_u8[:, idx].copy()

    return modes, tfc_params, fb_params, int(const_mask.sum()), int(linear_mask.sum()), int(fallback_mask.sum())


def encode_video_to_tfc(
    input_path: str,
    output_path: str,
    tau: float = DEFAULT_TAU,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    payload_comp_type: int = 1,
    max_frames: int | None = None,
) -> None:
    """
    Encode an RGB video into .tfc using per-pixel temporal modeling per channel.
    """

    frames = load_video_rgb(input_path, max_frames=max_frames)
    T, H, W, _ = frames.shape
    N = H * W
    flat_u8 = frames.reshape(T, N, 3)
    flat = flat_u8.astype(np.float32)

    plane_results = {}
    for plane, name in zip((PLANE_R, PLANE_G, PLANE_B), "RGB"):
        modes, tfc_params, fb_params, c_const, c_lin, c_raw = _encode_plane(
            flat[:, :, plane], flat_u8[:, :, plane], tau, slope_threshold
        )
        plane_results[plane] = {
            "modes": modes,
            "tfc_params": tfc_params,
            "fb_params": fb_params,
            "counts": (c_const, c_lin, c_raw),
        }
        print(
            f"Plane {name}: Const={c_const}, Linear={c_lin}, Raw={c_raw}"
        )

    header = {
        "version": 1,
        "width": W,
        "height": H,
        "num_frames": T,
        "color_format": COLOR_FORMAT_RGB,
        "bits_per_mode": BITS_PER_MODE,
        "payload_comp_type": payload_comp_type,
    }

    with open(output_path, "wb") as f:
        write_header(f, header)
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            modes = plane_results[plane]["modes"]
            f.write(pack_modes(modes, bits_per_mode=BITS_PER_MODE))
            payload = build_plane_payload(
                plane_results[plane]["tfc_params"],
                plane_results[plane]["fb_params"],
                T,
                payload_comp_type,
            )
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    print(f"Encoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}")
