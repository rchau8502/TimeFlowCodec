"""Per-pixel encoder for TimeFlowCodec."""
from __future__ import annotations

import numpy as np

from .constants import (
    DEFAULT_SLOPE_THRESHOLD,
    DEFAULT_TAU,
    MODE_FB_RAW,
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
)
from .format import build_payload, pack_modes, write_header
from .models import fit_linear_time_model, reconstruct_linear_sequence
from .utils import load_video_to_array, mse


def encode_video_to_tfc(
    input_path: str,
    output_path: str,
    tau: float = DEFAULT_TAU,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    payload_comp_type: int = 2,
    max_frames: int | None = None,
) -> None:
    """
    Per-pixel TimeFlow codec:
    - load video
    - flatten spatial dims
    - fit per-pixel temporal models
    - decide per-pixel modes
    - build and write .tfc file
    """

    frames = load_video_to_array(input_path, max_frames=max_frames)
    T, H, W = frames.shape
    N = H * W
    flat_frames = frames.reshape(T, N)
    data = flat_frames.astype(np.float32)

    modes = np.zeros((N,), dtype=np.uint8)
    tfc_params: dict[int, dict] = {}
    fb_params: dict[int, np.ndarray] = {}

    zeros = np.zeros((T,), dtype=np.float32)
    count_linear = count_const = count_fb = 0

    for p in range(N):
        s = data[:, p]
        if not np.any(s):
            # Skip computation for static black pixels
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
            fb_params[p] = flat_frames[:, p].astype(np.uint8).copy()
            count_fb += 1

    header = {
        "version": 1,
        "width": W,
        "height": H,
        "num_frames": T,
        "modes_bits_per_pix": 2,
        "payload_comp_type": payload_comp_type,
    }

    with open(output_path, "wb") as f:
        write_header(f, header)
        mode_table = pack_modes(modes, bits_per_mode=2)
        f.write(mode_table)
        payload = build_payload(modes, tfc_params, fb_params, header, payload_comp_type)
        f.write(payload)

    print(
        f"Encoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}, "
        f"Const={count_const}, Linear={count_linear}, Raw={count_fb}"
    )
