"""RGB per-pixel encoder for TimeFlowCodec."""
from __future__ import annotations

import struct
import numpy as np
import imageio.v2 as imageio

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
from .utils import _ensure_rgb  # type: ignore


def _encode_plane_from_stats(sum_s, sum_ts, sum_s2, T: int, tau: float, slope_threshold: float):
    """
    Vectorized per-plane encode using precomputed sums to avoid storing full video in memory.
    """
    n = float(T)
    N = sum_s.shape[0]
    modes = np.full((N,), MODE_FB_RAW, dtype=np.uint8)
    tfc_params: dict[int, dict] = {}

    sum_t = n * (n - 1.0) / 2.0
    sum_t2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    denom = n * sum_t2 - sum_t * sum_t
    b = np.where(denom == 0.0, 0.0, (n * sum_ts - sum_t * sum_s) / denom)
    a = (sum_s - b * sum_t) / n

    rss = sum_s2 - 2 * a * sum_s - 2 * b * sum_ts + (a * a) * n + 2 * a * b * sum_t + (b * b) * sum_t2
    D_tfc = rss / n
    D_sig = sum_s2 / n + 1e-8
    r = D_tfc / D_sig

    modeled = r <= tau
    const_mask = modeled & (np.abs(b) < slope_threshold)
    zero_mask = sum_s2 == 0
    const_mask |= zero_mask
    linear_mask = modeled & (~const_mask)
    fallback_mask = ~(const_mask | linear_mask)

    modes[const_mask] = MODE_TFC_CONST
    modes[linear_mask] = MODE_TFC_LINEAR

    for idx in np.nonzero(const_mask)[0]:
        tfc_params[idx] = {"mode": MODE_TFC_CONST, "a": float(a[idx])}
    for idx in np.nonzero(linear_mask)[0]:
        tfc_params[idx] = {"mode": MODE_TFC_LINEAR, "a": float(a[idx]), "b": float(b[idx])}

    return modes, tfc_params, fallback_mask, int(const_mask.sum()), int(linear_mask.sum()), int(fallback_mask.sum())


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
    Streaming implementation to reduce memory usage.
    """

    reader = imageio.get_reader(input_path)
    frames_iter = iter(reader)
    try:
        first = next(frames_iter)
    except StopIteration as exc:  # noqa: B904
        raise ValueError("No frames found in input video") from exc
    first_rgb = _ensure_rgb(first)
    H, W, _ = first_rgb.shape
    N = H * W

    # Stats per plane
    sum_s = [np.zeros((N,), dtype=np.float64) for _ in range(3)]
    sum_ts = [np.zeros((N,), dtype=np.float64) for _ in range(3)]
    sum_s2 = [np.zeros((N,), dtype=np.float64) for _ in range(3)]

    t = 0
    frames_consumed = 0

    def accumulate(frame_arr: np.ndarray, t_idx: int) -> None:
        flat = frame_arr.reshape(-1, 3).astype(np.float64)
        for c in range(3):
            ch = flat[:, c]
            sum_s[c] += ch
            sum_ts[c] += ch * t_idx
            sum_s2[c] += ch * ch

    accumulate(first_rgb, t)
    frames_consumed += 1

    for frame in frames_iter:
        if max_frames is not None and frames_consumed >= max_frames:
            break
        t += 1
        accumulate(_ensure_rgb(frame), t)
        frames_consumed += 1

    T = frames_consumed

    plane_results = {}
    fallback_indices = {}
    for plane, name in zip((PLANE_R, PLANE_G, PLANE_B), "RGB"):
        modes, tfc_params, fb_mask, c_const, c_lin, c_raw = _encode_plane_from_stats(
            sum_s[plane], sum_ts[plane], sum_s2[plane], T, tau, slope_threshold
        )
        plane_results[plane] = {
            "modes": modes,
            "tfc_params": tfc_params,
            "fb_params": {},  # filled later if needed
            "counts": (c_const, c_lin, c_raw),
        }
        fallback_indices[plane] = np.nonzero(fb_mask)[0]
        print(f"Plane {name}: Const={c_const}, Linear={c_lin}, Raw={c_raw}")

    # Collect fallback samples only for required pixels (second pass)
    needs_fb = any(len(fallback_indices[p]) > 0 for p in (PLANE_R, PLANE_G, PLANE_B))
    if needs_fb:
        fb_buffers = {}
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            idxs = fallback_indices[plane]
            fb_buffers[plane] = np.empty((len(idxs), T), dtype=np.uint8)

        reader2 = imageio.get_reader(input_path)
        for frame_idx, frame in enumerate(reader2):
            if frame_idx >= T:
                break
            flat = _ensure_rgb(frame).reshape(-1, 3).astype(np.uint8)
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                idxs = fallback_indices[plane]
                if len(idxs) == 0:
                    continue
                fb_buffers[plane][:, frame_idx] = flat[idxs, plane]

        for plane in (PLANE_R, PLANE_G, PLANE_B):
            idxs = fallback_indices[plane]
            fb_params = {}
            for buf_idx, pix_idx in enumerate(idxs):
                fb_params[int(pix_idx)] = fb_buffers[plane][buf_idx].copy()
            plane_results[plane]["fb_params"] = fb_params

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
