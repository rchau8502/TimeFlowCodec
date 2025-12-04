"""RGB per-pixel encoder for TimeFlowCodec (streaming, bounded memory, segments)."""

from __future__ import annotations

import struct
import tempfile
import warnings
from pathlib import Path

import imageio.v2 as imageio
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
from .format import (
    build_plane_payload_v1,
    build_plane_payload_v2,
    pack_modes,
    write_header,
)
from .utils import _ensure_rgb  # type: ignore
from .version import get_build_meta


def _tile_indices(H: int, W: int, tile: int) -> tuple[np.ndarray, int, int]:
    tiles_y = (H + tile - 1) // tile
    tiles_x = (W + tile - 1) // tile
    tile_idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    tiles = np.zeros_like(tile_idx)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile
            x0 = tx * tile
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            tiles[y0:y1, x0:x1] = ty * tiles_x + tx
    return tiles.reshape(-1), tiles_y, tiles_x


def _encode_plane_from_stats(
    sum_s, sum_ts, sum_s2, T: int, tau: float, slope_threshold: float
):
    """
    Vectorized per-plane encode using precomputed sums to avoid storing frames.
    Returns per-pixel modes/params and fallback mask (bool array).
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

    rss = (
        sum_s2
        - 2 * a * sum_s
        - 2 * b * sum_ts
        + (a * a) * n
        + 2 * a * b * sum_t
        + (b * b) * sum_t2
    )
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
        tfc_params[int(idx)] = {"mode": MODE_TFC_CONST, "a": float(a[idx])}
    for idx in np.nonzero(linear_mask)[0]:
        tfc_params[int(idx)] = {
            "mode": MODE_TFC_LINEAR,
            "a": float(a[idx]),
            "b": float(b[idx]),
        }

    return (
        modes,
        tfc_params,
        fallback_mask,
        int(const_mask.sum()),
        int(linear_mask.sum()),
        int(fallback_mask.sum()),
    )


def encode_video_to_tfc(
    input_path: str,
    output_path: str,
    tau: float = DEFAULT_TAU,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    payload_comp_type: int = 1,
    container_version: int = 2,
    max_frames: int | None = None,
    window: int | None = None,
    tiling: int | None = None,
    max_ram_mb: int | None = None,
    dtype: str = "uint8",
    scene_cut: str = "off",
    scene_threshold: float = 0.35,
) -> None:
    """
    Streaming RGB encoder with bounded memory and optional scene-cut segmentation.
    """

    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "float16": np.float16}
    if dtype not in dtype_map:
        raise ValueError("Unsupported dtype; choose from uint8,uint16,float16")
    np_dtype = dtype_map[dtype]
    tile_size = tiling if tiling and tiling > 1 else 1

    reader = imageio.get_reader(input_path)
    frames_iter = iter(reader)
    try:
        first = next(frames_iter)
    except StopIteration as exc:  # noqa: B904
        raise ValueError("No frames found in input video") from exc
    first_rgb = _ensure_rgb(first).astype(np_dtype, copy=False)
    H, W, _ = first_rgb.shape
    tile_idx_map, tiles_y, tiles_x = _tile_indices(H, W, tile_size)
    num_tiles = tiles_y * tiles_x

    def new_segment_accumulators():
        return (
            [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
            [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
            [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
        )

    sum_s, sum_ts, sum_s2 = new_segment_accumulators()
    segments_stats: list[dict] = []
    seg_lengths: list[int] = []
    frames_consumed = 0
    t_idx = 0
    prev_luma = None

    def accumulate(frame_arr: np.ndarray, t_val: int) -> None:
        flat = frame_arr.reshape(-1, 3)
        for c in range(3):
            ch = flat[:, c]
            ch_f = ch.astype(np.float64, copy=False)
            np.add.at(sum_s[c], tile_idx_map, ch_f)
            np.add.at(sum_ts[c], tile_idx_map, ch_f * t_val)
            np.add.at(sum_s2[c], tile_idx_map, ch_f * ch_f)

    accumulate(first_rgb, t_idx)
    frames_consumed += 1
    if scene_cut == "auto":
        luma = (
            0.299 * first_rgb[:, :, 0].astype(np.float32)
            + 0.587 * first_rgb[:, :, 1].astype(np.float32)
            + 0.114 * first_rgb[:, :, 2].astype(np.float32)
        )
        prev_luma = luma

    for frame in frames_iter:
        if max_frames is not None and frames_consumed >= max_frames:
            break
        frame_rgb = _ensure_rgb(frame).astype(np_dtype, copy=False)
        cut = False
        if scene_cut == "auto" and prev_luma is not None:
            luma = (
                0.299 * frame_rgb[:, :, 0].astype(np.float32)
                + 0.587 * frame_rgb[:, :, 1].astype(np.float32)
                + 0.114 * frame_rgb[:, :, 2].astype(np.float32)
            )
            diff = np.mean(np.abs(luma - prev_luma))
            if diff > scene_threshold * 255.0:
                cut = True
            prev_luma = luma

        if cut:
            segments_stats.append({"sum_s": sum_s, "sum_ts": sum_ts, "sum_s2": sum_s2})
            seg_lengths.append(frames_consumed - sum(seg_lengths))
            sum_s, sum_ts, sum_s2 = new_segment_accumulators()
            t_idx = 0
        else:
            t_idx += 1

        accumulate(frame_rgb, t_idx)
        frames_consumed += 1

    segments_stats.append({"sum_s": sum_s, "sum_ts": sum_ts, "sum_s2": sum_s2})
    seg_lengths.append(frames_consumed - sum(seg_lengths))

    T = frames_consumed

    plane_segments: dict[int, list[dict]] = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
    for seg_idx, seg_stat in enumerate(segments_stats):
        seg_len = seg_lengths[seg_idx]
        for plane, name in zip((PLANE_R, PLANE_G, PLANE_B), "RGB"):
            modes_tile, params_tile, fb_mask_tile, c_const, c_lin, _ = (
                _encode_plane_from_stats(
                    seg_stat["sum_s"][plane],
                    seg_stat["sum_ts"][plane],
                    seg_stat["sum_s2"][plane],
                    seg_len,
                    tau,
                    slope_threshold,
                )
            )
            modes = modes_tile[tile_idx_map].astype(np.uint8, copy=False)
            tfc_params = {}
            for pix, tile_id in enumerate(tile_idx_map):
                if modes_tile[tile_id] == MODE_TFC_CONST:
                    tfc_params[pix] = {
                        "mode": MODE_TFC_CONST,
                        "a": params_tile[int(tile_id)]["a"],
                    }
                elif modes_tile[tile_id] == MODE_TFC_LINEAR:
                    p = params_tile[int(tile_id)]
                    tfc_params[pix] = {
                        "mode": MODE_TFC_LINEAR,
                        "a": p["a"],
                        "b": p["b"],
                    }
            fb_mask_pix = fb_mask_tile[tile_idx_map]
            plane_segments[plane].append(
                {
                    "length": seg_len,
                    "modes": modes,
                    "tfc_params": tfc_params,
                    "fb_params": {},
                    "fb_mask": fb_mask_pix,
                    "counts": (
                        c_const * (tile_size * tile_size),
                        c_lin * (tile_size * tile_size),
                        int(fb_mask_pix.sum()),
                    ),
                }
            )

    needs_fb = any(
        any(seg["fb_mask"].any() for seg in plane_segments[p])
        for p in (PLANE_R, PLANE_G, PLANE_B)
    )
    if needs_fb:
        fb_indices = {
            p: [np.nonzero(seg["fb_mask"])[0] for seg in plane_segments[p]]
            for p in (PLANE_R, PLANE_G, PLANE_B)
        }
        fb_counts_total = sum(
            sum(len(idxs) for idxs in fb_indices[p])
            for p in (PLANE_R, PLANE_G, PLANE_B)
        )
        est_mem = fb_counts_total * T
        if max_ram_mb is not None and est_mem / (1024 * 1024) > max_ram_mb:
            warnings.warn(
                f"Estimated fallback buffer {est_mem/1e6:.2f} MB exceeds max_ram_mb={max_ram_mb}. Using memmap (disk)."
            )
        use_memmap = max_ram_mb is not None and est_mem / (1024 * 1024) > max_ram_mb

        fb_buffers = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
        tmpfiles = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            for seg_idx, idxs in enumerate(fb_indices[plane]):
                seg_len = seg_lengths[seg_idx]
                if len(idxs) == 0:
                    fb_buffers[plane].append(None)
                    tmpfiles[plane].append(None)
                    continue
                if use_memmap:
                    tmpfile = tempfile.NamedTemporaryFile(delete=False)
                    tmpfiles[plane].append(tmpfile.name)
                    fb_buffers[plane].append(
                        np.memmap(
                            tmpfile.name,
                            dtype=np.uint8,
                            mode="w+",
                            shape=(len(idxs), seg_len),
                        )
                    )
                else:
                    tmpfiles[plane].append(None)
                    fb_buffers[plane].append(
                        np.empty((len(idxs), seg_len), dtype=np.uint8)
                    )
        reader2 = imageio.get_reader(input_path)
        seg_idx = 0
        frame_in_seg = 0
        seg_frame_starts = np.cumsum([0] + seg_lengths)
        for frame_idx, frame in enumerate(reader2):
            if max_frames is not None and frame_idx >= T:
                break
            while (
                seg_idx < len(seg_lengths) - 1
                and frame_idx >= seg_frame_starts[seg_idx + 1]
            ):
                seg_idx += 1
                frame_in_seg = 0
            flat = _ensure_rgb(frame).reshape(-1, 3).astype(np.uint8, copy=False)
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                idxs = fb_indices[plane][seg_idx]
                buf = fb_buffers[plane][seg_idx]
                if buf is None or len(idxs) == 0:
                    continue
                buf[:, frame_in_seg] = flat[idxs, plane]
            frame_in_seg += 1
            if frame_in_seg >= seg_lengths[seg_idx]:
                frame_in_seg = 0

        for plane in (PLANE_R, PLANE_G, PLANE_B):
            for seg_idx, idxs in enumerate(fb_indices[plane]):
                fb_params = {}
                buf = fb_buffers[plane][seg_idx]
                if buf is None:
                    plane_segments[plane][seg_idx]["fb_params"] = fb_params
                    continue
                for buf_idx, pix_idx in enumerate(idxs):
                    fb_params[int(pix_idx)] = np.array(buf[buf_idx]).astype(
                        np.uint8, copy=False
                    )
                plane_segments[plane][seg_idx]["fb_params"] = fb_params
                if tmpfiles[plane][seg_idx]:
                    Path(tmpfiles[plane][seg_idx]).unlink(missing_ok=True)

    meta = get_build_meta()
    header = {
        "version": container_version,
        "width": W,
        "height": H,
        "num_frames": T,
        "color_format": COLOR_FORMAT_RGB,
        "bits_per_mode": BITS_PER_MODE,
        "payload_comp_type": payload_comp_type,
        "encoder_git_hash": meta.get("git_hash", ""),
        "tiling": tile_size if tile_size > 1 else 0,
        "segment_count": len(seg_lengths),
    }

    with open(output_path, "wb") as f:
        write_header(f, header)
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            if container_version == 1:
                payload = build_plane_payload_v1(
                    plane_segments[plane][0]["tfc_params"],
                    plane_segments[plane][0]["fb_params"],
                    T,
                    payload_comp_type,
                )
                f.write(
                    pack_modes(
                        plane_segments[plane][0]["modes"], bits_per_mode=BITS_PER_MODE
                    )
                )
            else:
                segments = [
                    {
                        "length": seg_lengths[idx],
                        "modes": plane_segments[plane][idx]["modes"],
                        "tfc_params": plane_segments[plane][idx]["tfc_params"],
                        "fb_params": plane_segments[plane][idx]["fb_params"],
                    }
                    for idx in range(len(seg_lengths))
                ]
                payload = build_plane_payload_v2(segments, payload_comp_type)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    print(
        f"Encoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}, Tiles={tiles_y}x{tiles_x}, Segments={len(seg_lengths)}"
    )
