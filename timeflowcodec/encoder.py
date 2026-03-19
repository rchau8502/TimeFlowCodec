"""TimeFlowCodec encoder (streaming, bounded memory, segmented)."""

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
    COLOR_FORMAT_YUV420,
    COLOR_FORMAT_YUV444,
    COMP_ZLIB,
    COMP_ZSTD,
    DEFAULT_SLOPE_THRESHOLD,
    DEFAULT_TAU,
    MODE_FB_RAW,
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
    MODE_TFC_MATRIX,
    PLANE_B,
    PLANE_G,
    PLANE_R,
)
from .format import (
    VERSION_V3,
    build_plane_payload_v1,
    build_plane_payload_v2,
    build_plane_payload_v3,
    pack_modes,
    write_header,
)
from .utils import (
    _ensure_rgb,
    plane_shape,
    rgb_to_planes,
    tile_grid_shape,
    tile_pixel_counts,
    tile_reduce_sum,
)
from .version import get_build_meta


def _apply_macbook_profile_defaults(
    *,
    payload_comp_type: int,
    tiling: int | None,
    max_ram_mb: int | None,
    dtype: str,
    scene_cut: str,
) -> tuple[int, int | None, int | None, str, str]:
    """Apply conservative defaults for Apple Silicon laptops."""
    if payload_comp_type == 2:
        # LZMA can be very slow and memory-heavy on laptop-class CPUs.
        payload_comp_type = COMP_ZLIB
    if tiling is None:
        tiling = 16
    if max_ram_mb is None:
        max_ram_mb = 1536
    if dtype not in {"uint8", "uint16"}:
        dtype = "uint8"
    if scene_cut == "off":
        scene_cut = "auto"
    return payload_comp_type, tiling, max_ram_mb, dtype, scene_cut


def _apply_preset_defaults(
    *,
    preset: str | None,
    container_version: int,
    color_format: int,
    tau: float,
    slope_threshold: float,
    payload_comp_type: int,
    tiling: int | None,
    max_ram_mb: int | None,
    scene_cut: str,
    scene_threshold: float,
    matrix_mode: bool,
    matrix_tau: float,
    matrix_rate_ratio: float,
) -> tuple[int, int, float, float, int, int | None, int | None, str, float, bool, float, float]:
    if preset is None or preset == "custom":
        return (
            container_version,
            color_format,
            tau,
            slope_threshold,
            payload_comp_type,
            tiling,
            max_ram_mb,
            scene_cut,
            scene_threshold,
            matrix_mode,
            matrix_tau,
            matrix_rate_ratio,
        )

    if preset == "anime":
        return (
            VERSION_V3 if container_version == 2 else container_version,
            COLOR_FORMAT_YUV420 if color_format == COLOR_FORMAT_RGB else color_format,
            0.06 if tau == DEFAULT_TAU else tau,
            0.0015 if slope_threshold == DEFAULT_SLOPE_THRESHOLD else slope_threshold,
            COMP_ZSTD if payload_comp_type == COMP_ZLIB else payload_comp_type,
            16 if tiling is None else tiling,
            2500 if max_ram_mb is None else max_ram_mb,
            "auto" if scene_cut == "off" else scene_cut,
            0.27 if scene_threshold == 0.35 else scene_threshold,
            matrix_mode,
            0.22 if matrix_tau == 0.12 else matrix_tau,
            1.05 if matrix_rate_ratio == 0.95 else matrix_rate_ratio,
        )

    if preset == "lownoise":
        return (
            VERSION_V3 if container_version == 2 else container_version,
            COLOR_FORMAT_YUV444 if color_format == COLOR_FORMAT_RGB else color_format,
            0.08 if tau == DEFAULT_TAU else tau,
            0.0015 if slope_threshold == DEFAULT_SLOPE_THRESHOLD else slope_threshold,
            COMP_ZSTD if payload_comp_type == COMP_ZLIB else payload_comp_type,
            16 if tiling is None else tiling,
            2500 if max_ram_mb is None else max_ram_mb,
            "auto" if scene_cut == "off" else scene_cut,
            0.32 if scene_threshold == 0.35 else scene_threshold,
            matrix_mode,
            0.14 if matrix_tau == 0.12 else matrix_tau,
            1.00 if matrix_rate_ratio == 0.95 else matrix_rate_ratio,
        )

    raise ValueError("preset must be one of: custom, anime, lownoise")


def _resolve_color_format(colorspace: str) -> int:
    mapping = {
        "rgb": COLOR_FORMAT_RGB,
        "yuv444": COLOR_FORMAT_YUV444,
        "yuv420": COLOR_FORMAT_YUV420,
    }
    try:
        return mapping[colorspace]
    except KeyError as exc:  # noqa: B904
        raise ValueError("colorspace must be one of: rgb, yuv444, yuv420") from exc


def _color_format_name(color_format: int) -> str:
    mapping = {
        COLOR_FORMAT_RGB: "rgb",
        COLOR_FORMAT_YUV444: "yuv444",
        COLOR_FORMAT_YUV420: "yuv420",
    }
    return mapping.get(color_format, f"unknown({color_format})")


def _fit_rank1_matrix(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit X ~= mean + outer(temporal, spatial) with rank-1 SVD.
    Returns mean, temporal vector, spatial vector, reconstructed matrix.
    """
    x = matrix.astype(np.float32, copy=False)
    mean = float(np.mean(x))
    centered = x - mean
    if np.allclose(centered, 0.0):
        temporal = np.zeros((x.shape[0],), dtype=np.float32)
        spatial = np.zeros((x.shape[1],), dtype=np.float32)
        recon = np.full_like(x, mean, dtype=np.float32)
        return mean, temporal, spatial, recon
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scale = float(np.sqrt(s[0]))
    temporal = (u[:, 0] * scale).astype(np.float32, copy=False)
    spatial = (vt[0, :] * scale).astype(np.float32, copy=False)
    recon = mean + np.outer(temporal, spatial)
    return mean, temporal, spatial, recon.astype(np.float32, copy=False)


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
    sum_s,
    sum_ts,
    sum_s2,
    tile_pixel_counts: np.ndarray,
    T: int,
    tau: float,
    slope_threshold: float,
):
    """
    Vectorized per-plane encode using precomputed sums to avoid storing frames.
    Returns per-pixel modes/params and fallback mask (bool array).
    """
    N = sum_s.shape[0]
    p_counts = tile_pixel_counts.astype(np.float64, copy=False)
    n = p_counts * float(T)
    modes = np.full((N,), MODE_FB_RAW, dtype=np.uint8)
    tfc_params: dict[int, dict] = {}

    sum_t_base = float(T) * (float(T) - 1.0) / 2.0
    sum_t2_base = (float(T) - 1.0) * float(T) * (2.0 * float(T) - 1.0) / 6.0
    sum_t = p_counts * sum_t_base
    sum_t2 = p_counts * sum_t2_base
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
    container_version: int = VERSION_V3,
    max_frames: int | None = None,
    window: int | None = None,
    tiling: int | None = None,
    max_ram_mb: int | None = None,
    dtype: str = "uint8",
    colorspace: str = "rgb",
    scene_cut: str = "off",
    scene_threshold: float = 0.35,
    macbook_profile: bool = False,
    matrix_mode: bool = False,
    matrix_tau: float = 0.12,
    matrix_rate_ratio: float = 0.95,
    preset: str | None = None,
) -> None:
    """
    Streaming encoder with bounded memory and optional scene-cut segmentation.
    """

    color_format = _resolve_color_format(colorspace)

    (
        container_version,
        color_format,
        tau,
        slope_threshold,
        payload_comp_type,
        tiling,
        max_ram_mb,
        scene_cut,
        scene_threshold,
        matrix_mode,
        matrix_tau,
        matrix_rate_ratio,
    ) = _apply_preset_defaults(
        preset=preset,
        container_version=container_version,
        color_format=color_format,
        tau=tau,
        slope_threshold=slope_threshold,
        payload_comp_type=payload_comp_type,
        tiling=tiling,
        max_ram_mb=max_ram_mb,
        scene_cut=scene_cut,
        scene_threshold=scene_threshold,
        matrix_mode=matrix_mode,
        matrix_tau=matrix_tau,
        matrix_rate_ratio=matrix_rate_ratio,
    )

    if window is not None:
        warnings.warn(
            "window is currently accepted for API compatibility but not used by the encoder."
        )

    if scene_cut not in {"off", "auto"}:
        raise ValueError("scene_cut must be 'off' or 'auto'")

    if macbook_profile:
        payload_comp_type, tiling, max_ram_mb, dtype, scene_cut = (
            _apply_macbook_profile_defaults(
                payload_comp_type=payload_comp_type,
                tiling=tiling,
                max_ram_mb=max_ram_mb,
                dtype=dtype,
                scene_cut=scene_cut,
            )
        )

    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "float16": np.float16}
    if dtype not in dtype_map:
        raise ValueError("Unsupported dtype; choose from uint8,uint16,float16")
    np_dtype = dtype_map[dtype]
    tile_size = tiling if tiling and tiling > 1 else 1
    if container_version >= VERSION_V3:
        matrix_mode = False
    if matrix_mode and tile_size <= 1:
        warnings.warn(
            "matrix_mode requested but tiling <= 1; matrix low-rank mode disabled."
        )
        matrix_mode = False
    if matrix_mode and container_version == 1:
        warnings.warn(
            "matrix_mode is only available in container version 2; disabling matrix mode."
        )
        matrix_mode = False
    if scene_cut == "auto" and container_version == 1:
        warnings.warn(
            "scene_cut auto requires container version 2 segmentation; disabling scene cuts."
        )
        scene_cut = "off"

    reader = imageio.get_reader(input_path)
    frames_iter = iter(reader)
    try:
        first = next(frames_iter)
    except StopIteration as exc:  # noqa: B904
        raise ValueError("No frames found in input video") from exc
    first_rgb = _ensure_rgb(first).astype(np_dtype, copy=False)
    H, W, _ = first_rgb.shape
    if container_version >= VERSION_V3:
        plane_infos = {
            plane: {
                "shape": plane_shape(H, W, color_format, plane),
                "tile_grid": tile_grid_shape(
                    *plane_shape(H, W, color_format, plane), tile_size
                ),
                "counts": tile_pixel_counts(
                    *plane_shape(H, W, color_format, plane), tile_size
                ),
            }
            for plane in (PLANE_R, PLANE_G, PLANE_B)
        }

        def new_segment_accumulators_v3():
            return {
                plane: {
                    "sum_s": np.zeros_like(plane_infos[plane]["counts"], dtype=np.float64),
                    "sum_ts": np.zeros_like(plane_infos[plane]["counts"], dtype=np.float64),
                    "sum_s2": np.zeros_like(plane_infos[plane]["counts"], dtype=np.float64),
                }
                for plane in (PLANE_R, PLANE_G, PLANE_B)
            }

        current_stats = new_segment_accumulators_v3()
        segments_stats: list[dict] = []
        seg_lengths: list[int] = []
        frames_consumed = 0
        t_idx = 0
        prev_luma = None

        def accumulate_v3(frame_rgb: np.ndarray, time_index: int) -> None:
            planes = rgb_to_planes(frame_rgb, color_format)
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                plane_arr = planes[plane]
                reduced = tile_reduce_sum(plane_arr, tile_size)
                current_stats[plane]["sum_s"] += reduced
                current_stats[plane]["sum_ts"] += reduced * float(time_index)
                current_stats[plane]["sum_s2"] += tile_reduce_sum(
                    plane_arr.astype(np.float64, copy=False) ** 2.0, tile_size
                )

        accumulate_v3(first_rgb, t_idx)
        frames_consumed += 1
        if scene_cut == "auto":
            prev_luma = (
                0.299 * first_rgb[:, :, 0].astype(np.float32)
                + 0.587 * first_rgb[:, :, 1].astype(np.float32)
                + 0.114 * first_rgb[:, :, 2].astype(np.float32)
            )

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
                segments_stats.append(current_stats)
                seg_lengths.append(frames_consumed - sum(seg_lengths))
                current_stats = new_segment_accumulators_v3()
                t_idx = 0
            else:
                t_idx += 1

            accumulate_v3(frame_rgb, t_idx)
            frames_consumed += 1

        segments_stats.append(current_stats)
        seg_lengths.append(frames_consumed - sum(seg_lengths))
        T = frames_consumed

        plane_segments: dict[int, list[dict]] = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
        for seg_idx, seg_stat in enumerate(segments_stats):
            seg_len = seg_lengths[seg_idx]
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                logical_counts = plane_infos[plane]["counts"]
                modes, tfc_params, fb_mask, _c_const, _c_lin, _c_raw = _encode_plane_from_stats(
                    seg_stat[plane]["sum_s"],
                    seg_stat[plane]["sum_ts"],
                    seg_stat[plane]["sum_s2"],
                    logical_counts,
                    seg_len,
                    tau,
                    slope_threshold,
                )
                plane_segments[plane].append(
                    {
                        "length": seg_len,
                        "modes": modes,
                        "tfc_params": tfc_params,
                        "raw_mask": fb_mask,
                        "raw_values": np.empty((0, seg_len), dtype=np.uint8),
                    }
                )

        needs_raw = any(
            any(np.any(seg["raw_mask"]) for seg in plane_segments[plane])
            for plane in (PLANE_R, PLANE_G, PLANE_B)
        )
        if needs_raw:
            raw_indices = {
                plane: [np.nonzero(seg["raw_mask"])[0] for seg in plane_segments[plane]]
                for plane in (PLANE_R, PLANE_G, PLANE_B)
            }
            raw_count_total = sum(
                sum(len(idxs) for idxs in raw_indices[plane])
                for plane in (PLANE_R, PLANE_G, PLANE_B)
            )
            est_mem = raw_count_total * T
            if max_ram_mb is not None and est_mem / (1024 * 1024) > max_ram_mb:
                warnings.warn(
                    "Estimated RAW tile buffer "
                    f"{est_mem/1e6:.2f} MB exceeds max_ram_mb={max_ram_mb}. "
                    "Using memmap (disk)."
                )
            use_memmap = max_ram_mb is not None and est_mem / (1024 * 1024) > max_ram_mb

            raw_buffers = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
            tmpfiles = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                for seg_idx, idxs in enumerate(raw_indices[plane]):
                    seg_len = seg_lengths[seg_idx]
                    if len(idxs) == 0:
                        raw_buffers[plane].append(None)
                        tmpfiles[plane].append(None)
                        continue
                    if use_memmap:
                        tmpfile = tempfile.NamedTemporaryFile(delete=False)
                        tmpfiles[plane].append(tmpfile.name)
                        raw_buffers[plane].append(
                            np.memmap(
                                tmpfile.name,
                                dtype=np.uint8,
                                mode="w+",
                                shape=(len(idxs), seg_len),
                            )
                        )
                    else:
                        tmpfiles[plane].append(None)
                        raw_buffers[plane].append(
                            np.empty((len(idxs), seg_len), dtype=np.uint8)
                        )

            reader2 = imageio.get_reader(input_path)
            seg_idx = 0
            frame_in_seg = 0
            seg_frame_starts = np.cumsum([0] + seg_lengths)
            for frame_idx, frame in enumerate(reader2):
                if max_frames is not None and frame_idx >= T:
                    break
                while seg_idx < len(seg_lengths) - 1 and frame_idx >= seg_frame_starts[seg_idx + 1]:
                    seg_idx += 1
                    frame_in_seg = 0
                planes = rgb_to_planes(_ensure_rgb(frame), color_format)
                for plane in (PLANE_R, PLANE_G, PLANE_B):
                    idxs = raw_indices[plane][seg_idx]
                    buf = raw_buffers[plane][seg_idx]
                    if buf is None or len(idxs) == 0:
                        continue
                    reduced = tile_reduce_sum(planes[plane], tile_size) / plane_infos[plane]["counts"]
                    buf[:, frame_in_seg] = np.clip(np.rint(reduced[idxs]), 0, 255).astype(np.uint8)
                frame_in_seg += 1
                if frame_in_seg >= seg_lengths[seg_idx]:
                    frame_in_seg = 0
            close_reader2 = getattr(reader2, "close", None)
            if callable(close_reader2):
                close_reader2()

            for plane in (PLANE_R, PLANE_G, PLANE_B):
                for seg_idx, idxs in enumerate(raw_indices[plane]):
                    buf = raw_buffers[plane][seg_idx]
                    if buf is not None:
                        plane_segments[plane][seg_idx]["raw_values"] = np.asarray(buf, dtype=np.uint8).copy()
                    if tmpfiles[plane][seg_idx]:
                        Path(tmpfiles[plane][seg_idx]).unlink(missing_ok=True)

    else:
        if color_format != COLOR_FORMAT_RGB:
            raise ValueError("Container versions 1/2 only support RGB; use version 3 for YUV codecs.")
        tile_idx_map, tiles_y, tiles_x = _tile_indices(H, W, tile_size)
        num_tiles = tiles_y * tiles_x
        tile_pixel_counts_arr = np.bincount(tile_idx_map, minlength=num_tiles).astype(np.int64)

        def new_segment_accumulators():
            return (
                [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
                [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
                [np.zeros((num_tiles,), dtype=np.float64) for _ in range(3)],
            )

        sum_s, sum_ts, sum_s2 = new_segment_accumulators()
        segments_stats = []
        seg_lengths = []
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

        plane_segments = {PLANE_R: [], PLANE_G: [], PLANE_B: []}
        for seg_idx, seg_stat in enumerate(segments_stats):
            seg_len = seg_lengths[seg_idx]
            for plane, _name in zip((PLANE_R, PLANE_G, PLANE_B), "RGB"):
                modes_tile, params_tile, fb_mask_tile, c_const, c_lin, _ = (
                    _encode_plane_from_stats(
                        seg_stat["sum_s"][plane],
                        seg_stat["sum_ts"][plane],
                        seg_stat["sum_s2"][plane],
                        tile_pixel_counts_arr,
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
                        "matrix_params": [],
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
                    "Estimated fallback buffer "
                    f"{est_mem/1e6:.2f} MB exceeds max_ram_mb={max_ram_mb}. "
                    "Using memmap (disk)."
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
            close_reader2 = getattr(reader2, "close", None)
            if callable(close_reader2):
                close_reader2()

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

    # Optional low-rank matrix mode for RAW-heavy tiles (v2 only).
    if matrix_mode and container_version == 2:
        tile_to_pixels: dict[int, np.ndarray] = {
            tile_id: np.nonzero(tile_idx_map == tile_id)[0]
            for tile_id in range(num_tiles)
        }
        matrix_tiles_used = 0
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            for seg_idx in range(len(seg_lengths)):
                seg = plane_segments[plane][seg_idx]
                modes = seg["modes"]
                fb_params = seg["fb_params"]
                if not fb_params:
                    continue
                raw_tile_ids = np.unique(tile_idx_map[modes == MODE_FB_RAW])
                for tile_id in raw_tile_ids.tolist():
                    pix = tile_to_pixels[int(tile_id)]
                    if pix.size <= 1:
                        continue
                    # Only transform full-RAW tiles to avoid overlap complexity.
                    if np.any(modes[pix] != MODE_FB_RAW):
                        continue
                    if not all(int(p) in fb_params for p in pix):
                        continue

                    x = np.stack([fb_params[int(p)] for p in pix], axis=1).astype(
                        np.float32
                    )
                    mean, temporal, spatial, recon = _fit_rank1_matrix(x)
                    d_tfc = float(np.mean((x - recon) ** 2))
                    d_sig = float(np.mean(x**2) + 1e-8)
                    r = d_tfc / d_sig
                    raw_bytes = int(x.size + 4 * pix.size)
                    matrix_bytes = int(16 + 2 * x.shape[0] + 6 * pix.size)
                    if r > matrix_tau or matrix_bytes >= int(
                        raw_bytes * matrix_rate_ratio
                    ):
                        continue

                    for p in pix:
                        fb_params.pop(int(p), None)
                    modes[pix] = MODE_TFC_MATRIX
                    seg["matrix_params"].append(
                        {
                            "mode": MODE_TFC_MATRIX,
                            "pixel_indices": pix.astype(np.int64),
                            "mean": mean,
                            "temporal": temporal,
                            "spatial": spatial.astype(np.float32, copy=False),
                        }
                    )
                    matrix_tiles_used += 1
        if matrix_tiles_used:
            print(f"Low-rank MATRIX tiles enabled: {matrix_tiles_used}")

    meta = get_build_meta()
    header = {
        "version": container_version,
        "width": W,
        "height": H,
        "num_frames": T,
        "color_format": color_format,
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
            elif container_version == 2:
                segments = [
                    {
                        "length": seg_lengths[idx],
                        "modes": plane_segments[plane][idx]["modes"],
                        "tfc_params": plane_segments[plane][idx]["tfc_params"],
                        "matrix_params": plane_segments[plane][idx]["matrix_params"],
                        "fb_params": plane_segments[plane][idx]["fb_params"],
                    }
                    for idx in range(len(seg_lengths))
                ]
                payload = build_plane_payload_v2(segments, payload_comp_type)
            else:
                segments = [
                    {
                        "length": seg_lengths[idx],
                        "modes": plane_segments[plane][idx]["modes"],
                        "tfc_params": plane_segments[plane][idx]["tfc_params"],
                        "raw_values": plane_segments[plane][idx]["raw_values"],
                    }
                    for idx in range(len(seg_lengths))
                ]
                payload = build_plane_payload_v3(segments, payload_comp_type)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    print(
        "Encoded "
        f"{input_path} -> {output_path}. Frames={T}, Size={H}x{W}, "
        f"Tiles={tile_grid_shape(H, W, tile_size)[0]}x{tile_grid_shape(H, W, tile_size)[1]}, "
        f"Segments={len(seg_lengths)}, Colorspace={_color_format_name(color_format)}"
    )
