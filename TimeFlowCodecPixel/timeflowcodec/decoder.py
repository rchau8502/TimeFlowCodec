"""Per-pixel decoder for TimeFlowCodec."""
from __future__ import annotations

import numpy as np

from .constants import MODE_FB_RAW, MODE_TFC_CONST, MODE_TFC_LINEAR
from .format import read_header, unpack_modes, parse_payload
from .models import reconstruct_constant_sequence, reconstruct_linear_sequence
from .utils import save_array_to_video


def decode_tfc_to_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
) -> None:
    """
    Read a .tfc file produced by encode_video_to_tfc and reconstruct
    the grayscale video to output_path.
    """

    with open(input_path, "rb") as f:
        header = read_header(f)
        W = int(header["width"])
        H = int(header["height"])
        T = int(header["num_frames"])
        N = W * H

        mode_table_size = (N + 3) // 4
        mode_table = f.read(mode_table_size)
        if len(mode_table) != mode_table_size:
            raise ValueError("Incomplete mode table")
        modes = unpack_modes(mode_table, N, bits_per_mode=2)

        comp_payload = f.read()
        tfc_params, fb_params = parse_payload(comp_payload, header, modes)

    frames = np.zeros((T, H, W), dtype=np.uint8)
    flat = frames.reshape(T, N)

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
                raise ValueError(f"Missing fallback data for pixel {p}")
            s_hat = seq.astype(np.float32)
        else:
            raise ValueError(f"Unknown mode {mode} for pixel {p}")

        s_u8 = np.clip(np.rint(s_hat), 0, 255).astype(np.uint8)
        flat[:, p] = s_u8

    save_array_to_video(frames, output_path, fps=fps)
    print(f"Decoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}")
