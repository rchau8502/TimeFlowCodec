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
from .models import fit_linear_time_model, reconstruct_linear_sequence
from .utils import load_video_rgb, mse


def _encode_plane(channel_data: np.ndarray, tau: float, slope_threshold: float, zeros: np.ndarray):
    T, N = channel_data.shape
    modes = np.zeros((N,), dtype=np.uint8)
    tfc_params: dict[int, dict] = {}
    fb_params: dict[int, np.ndarray] = {}
    count_const = count_linear = count_raw = 0

    for p in range(N):
        s = channel_data[:, p]
        if not np.any(s):
            modes[p] = MODE_TFC_CONST
            tfc_params[p] = {"mode": MODE_TFC_CONST, "a": 0.0}
            count_const += 1
            continue

        a, b = fit_linear_time_model(s)
        s_lin = reconstruct_linear_sequence(a, b, T)
        D_tfc = mse(s, s_lin)
        D_sig = mse(s, zeros) + 1e-8
        r = D_tfc / D_sig

        if r <= tau:
            if abs(b) < slope_threshold:
                modes[p] = MODE_TFC_CONST
                tfc_params[p] = {"mode": MODE_TFC_CONST, "a": float(a)}
                count_const += 1
            else:
                modes[p] = MODE_TFC_LINEAR
                tfc_params[p] = {"mode": MODE_TFC_LINEAR, "a": float(a), "b": float(b)}
                count_linear += 1
        else:
            modes[p] = MODE_FB_RAW
            fb_params[p] = s.astype(np.uint8).copy()
            count_raw += 1

    return modes, tfc_params, fb_params, count_const, count_linear, count_raw


def encode_video_to_tfc(
    input_path: str,
    output_path: str,
    tau: float = DEFAULT_TAU,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    payload_comp_type: int = 2,
    max_frames: int | None = None,
) -> None:
    """
    Encode an RGB video into .tfc using per-pixel temporal modeling per channel.
    """

    frames = load_video_rgb(input_path, max_frames=max_frames)
    T, H, W, _ = frames.shape
    N = H * W
    flat = frames.reshape(T, N, 3).astype(np.float32)

    zeros = np.zeros((T,), dtype=np.float32)

    plane_results = {}
    for plane, name in zip((PLANE_R, PLANE_G, PLANE_B), "RGB"):
        modes, tfc_params, fb_params, c_const, c_lin, c_raw = _encode_plane(
            flat[:, :, plane], tau, slope_threshold, zeros
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
