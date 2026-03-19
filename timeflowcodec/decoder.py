"""TimeFlowCodec decoder (supports v1/v2/v3 containers)."""

from __future__ import annotations

import struct

import imageio.v2 as imageio
import numpy as np

from .constants import (
    BITS_PER_MODE,
    COLOR_FORMAT_RGB,
    COLOR_FORMAT_YUV420,
    COLOR_FORMAT_YUV444,
    MODE_FB_RAW,
    MODE_TFC_CONST,
    MODE_TFC_LINEAR,
    MODE_TFC_MATRIX,
    PLANE_B,
    PLANE_G,
    PLANE_R,
)
from .format import (
    VERSION_V1,
    VERSION_V2,
    VERSION_V3,
    parse_plane_payload_v1,
    parse_plane_payload_v2,
    parse_plane_payload_v3,
    read_header,
    unpack_modes,
)
from .utils import expand_tile_values, plane_shape, planes_to_rgb, save_video_from_rgb, tile_grid_shape


def _prepare_plane_arrays(
    T: int, N: int, modes: np.ndarray, tfc_params: dict, fb_params: dict
):
    a_params = np.zeros((N,), dtype=np.float32)
    b_params = np.zeros((N,), dtype=np.float32)
    if tfc_params:
        for idx, param in tfc_params.items():
            a_params[int(idx)] = float(param["a"])
            b_params[int(idx)] = float(param.get("b", 0.0))
    fb_indices = None
    fb_arr = None
    if fb_params:
        fb_indices = np.array(sorted(fb_params.keys()), dtype=np.int64)
        fb_arr = np.zeros((len(fb_indices), T), dtype=np.uint8)
        for row, pix in enumerate(fb_indices):
            fb_arr[row] = fb_params[int(pix)]
    return a_params, b_params, fb_indices, fb_arr


def decode_tfc_to_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
    stream_output: bool = False,
) -> None:
    """Decode .tfc RGB file back into an RGB video."""

    with open(input_path, "rb") as f:
        header = read_header(f)
        color_format = int(header.get("color_format", COLOR_FORMAT_RGB))
        if color_format not in {COLOR_FORMAT_RGB, COLOR_FORMAT_YUV444, COLOR_FORMAT_YUV420}:
            raise ValueError("Unsupported color format in TFC file")
        if header.get("bits_per_mode") != BITS_PER_MODE:
            raise ValueError("Unexpected bits_per_mode in TFC file")
        W = int(header["width"])
        H = int(header["height"])
        T = int(header["num_frames"])
        N = W * H
        payload_comp_type = int(header.get("payload_comp_type", 0))
        version = int(header.get("version", VERSION_V1))

        remaining = f.read()

    payloads = []
    offset = 0
    plane_data = {}
    if version == VERSION_V1:
        mode_table_size = (N + 3) // 4
        mode_tables = []
        for _ in range(3):
            if offset + mode_table_size > len(remaining):
                raise ValueError("Incomplete mode table data")
            mode_bytes = remaining[offset : offset + mode_table_size]
            offset += mode_table_size
            mode_tables.append(mode_bytes)
        for _ in range(3):
            if offset + 4 > len(remaining):
                raise ValueError("Missing payload size")
            payload_size = struct.unpack("<I", remaining[offset : offset + 4])[0]
            offset += 4
            if offset + payload_size > len(remaining):
                raise ValueError("Incomplete payload data")
            payloads.append(remaining[offset : offset + payload_size])
            offset += payload_size
        for plane, mode_bytes, payload in zip(
            (PLANE_R, PLANE_G, PLANE_B), mode_tables, payloads
        ):
            modes = unpack_modes(mode_bytes, N, bits_per_mode=BITS_PER_MODE)
            tfc_params, fb_params = parse_plane_payload_v1(
                payload, T, payload_comp_type
            )
            plane_data[plane] = (modes, tfc_params, fb_params)
    elif version == VERSION_V2:
        for _ in range(3):
            if offset + 4 > len(remaining):
                raise ValueError("Missing payload size")
            payload_size = struct.unpack("<I", remaining[offset : offset + 4])[0]
            offset += 4
            if offset + payload_size > len(remaining):
                raise ValueError("Incomplete payload data")
            payloads.append(remaining[offset : offset + payload_size])
            offset += payload_size
        for plane, payload in zip((PLANE_R, PLANE_G, PLANE_B), payloads):
            segments = parse_plane_payload_v2(payload, payload_comp_type, N)
            plane_data[plane] = segments
    elif version == VERSION_V3:
        for _ in range(3):
            if offset + 4 > len(remaining):
                raise ValueError("Missing payload size")
            payload_size = struct.unpack("<I", remaining[offset : offset + 4])[0]
            offset += 4
            if offset + payload_size > len(remaining):
                raise ValueError("Incomplete payload data")
            payloads.append(remaining[offset : offset + payload_size])
            offset += payload_size
        for plane, payload in zip((PLANE_R, PLANE_G, PLANE_B), payloads):
            plane_data[plane] = parse_plane_payload_v3(payload, payload_comp_type)
    else:
        raise ValueError(f"Unsupported version {version}")

    if version == VERSION_V1:
        # Precompute parameter arrays
        a_params = np.zeros((3, N), dtype=np.float32)
        b_params = np.zeros((3, N), dtype=np.float32)
        mode_arrays = np.zeros((3, N), dtype=np.uint8)
        fb_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for plane in (PLANE_R, PLANE_G, PLANE_B):
            modes, tfc_params, fb_params = plane_data[plane]
            mode_arrays[plane] = modes
            a_p, b_p, fb_idx, fb_arr = _prepare_plane_arrays(
                T, N, modes, tfc_params, fb_params
            )
            a_params[plane] = a_p
            b_params[plane] = b_p
            if fb_idx is not None and fb_arr is not None:
                fb_lookup[plane] = (fb_idx, fb_arr)

        t_range = np.arange(T, dtype=np.float32)
        if stream_output:
            writer = imageio.get_writer(output_path)
            flat_frame = np.empty((N, 3), dtype=np.float32)
            for t in range(T):
                for plane in (PLANE_R, PLANE_G, PLANE_B):
                    modes = mode_arrays[plane]
                    const_mask = modes == MODE_TFC_CONST
                    lin_mask = modes == MODE_TFC_LINEAR
                    fb_mask = modes == MODE_FB_RAW
                    vals = np.zeros((N,), dtype=np.float32)
                    if np.any(const_mask):
                        vals[const_mask] = a_params[plane, const_mask]
                    if np.any(lin_mask):
                        vals[lin_mask] = (
                            a_params[plane, lin_mask]
                            + b_params[plane, lin_mask] * t_range[t]
                        )
                    if np.any(fb_mask) and plane in fb_lookup:
                        fb_idx, fb_arr = fb_lookup[plane]
                        fb_map = {pix: row for row, pix in enumerate(fb_idx)}
                        fb_pix = np.nonzero(fb_mask)[0]
                        for pix in fb_pix:
                            vals[pix] = fb_arr[fb_map[int(pix)], t]
                    flat_frame[:, plane] = vals
                frame_u8 = np.clip(np.rint(flat_frame.reshape(H, W, 3)), 0, 255).astype(
                    np.uint8
                )
                writer.append_data(frame_u8)
            writer.close()
        else:
            frames = np.zeros((T, H, W, 3), dtype=np.uint8)
            flat = frames.reshape(T, N, 3)
            for plane in (PLANE_R, PLANE_G, PLANE_B):
                modes = mode_arrays[plane]
                const_mask = modes == MODE_TFC_CONST
                lin_mask = modes == MODE_TFC_LINEAR
                fb_mask = modes == MODE_FB_RAW
                vals = np.zeros((T, N), dtype=np.float32)
                if np.any(const_mask):
                    vals[:, const_mask] = a_params[plane, const_mask]
                if np.any(lin_mask):
                    vals[:, lin_mask] = (
                        a_params[plane, lin_mask][None, :]
                        + b_params[plane, lin_mask][None, :] * t_range[:, None]
                    )
                if np.any(fb_mask) and plane in fb_lookup:
                    fb_idx, fb_arr = fb_lookup[plane]
                    fb_map = {pix: row for row, pix in enumerate(fb_idx)}
                    fb_pix = np.nonzero(fb_mask)[0]
                    for pix in fb_pix:
                        vals[:, pix] = fb_arr[fb_map[int(pix)]]
                flat[:, :, plane] = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
            save_video_from_rgb(frames, output_path, fps=fps)
    elif version == VERSION_V2:
        # v2 segmented payload
        if stream_output:
            writer = imageio.get_writer(output_path, fps=fps)
            for seg_idx in range(len(plane_data[PLANE_R])):
                seg_len = plane_data[PLANE_R][seg_idx]["length"]
                frames = np.zeros((seg_len, H, W, 3), dtype=np.uint8)
                flat = frames.reshape(seg_len, N, 3)
                for plane in (PLANE_R, PLANE_G, PLANE_B):
                    seg = plane_data[plane][seg_idx]
                    modes = seg["modes"]
                    tfc_params = seg["tfc_params"]
                    fb_params = seg["fb_params"]
                    matrix_params = seg.get("matrix_params", [])
                    a_p, b_p, fb_idx, fb_arr = _prepare_plane_arrays(
                        seg_len, N, modes, tfc_params, fb_params
                    )
                    const_mask = modes == MODE_TFC_CONST
                    lin_mask = modes == MODE_TFC_LINEAR
                    matrix_mask = modes == MODE_TFC_MATRIX
                    fb_mask = modes == MODE_FB_RAW
                    t_range = np.arange(seg_len, dtype=np.float32)
                    vals = np.zeros((seg_len, N), dtype=np.float32)
                    if np.any(const_mask):
                        vals[:, const_mask] = a_p[const_mask]
                    if np.any(lin_mask):
                        vals[:, lin_mask] = (
                            a_p[lin_mask][None, :]
                            + b_p[lin_mask][None, :] * t_range[:, None]
                        )
                    if np.any(fb_mask) and fb_idx is not None and fb_arr is not None:
                        fb_map = {pix: row for row, pix in enumerate(fb_idx)}
                        fb_pix = np.nonzero(fb_mask)[0]
                        for pix in fb_pix:
                            vals[:, pix] = fb_arr[fb_map[int(pix)]]
                    if np.any(matrix_mask) and matrix_params:
                        for m in matrix_params:
                            pix = np.asarray(m["pixel_indices"], dtype=np.int64)
                            temporal = np.asarray(
                                m["temporal"], dtype=np.float32
                            ).reshape(seg_len)
                            spatial = np.asarray(
                                m["spatial"], dtype=np.float32
                            ).reshape(pix.size)
                            mean = float(m.get("mean", 0.0))
                            vals[:, pix] = mean + np.outer(temporal, spatial)
                    flat[:, :, plane] = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
                for frame in frames:
                    writer.append_data(frame)
            writer.close()
        else:
            all_frames: list[np.ndarray] = []
            for seg_idx in range(len(plane_data[PLANE_R])):
                seg_len = plane_data[PLANE_R][seg_idx]["length"]
                frames = np.zeros((seg_len, H, W, 3), dtype=np.uint8)
                flat = frames.reshape(seg_len, N, 3)
                for plane in (PLANE_R, PLANE_G, PLANE_B):
                    seg = plane_data[plane][seg_idx]
                    modes = seg["modes"]
                    tfc_params = seg["tfc_params"]
                    fb_params = seg["fb_params"]
                    matrix_params = seg.get("matrix_params", [])
                    a_p, b_p, fb_idx, fb_arr = _prepare_plane_arrays(
                        seg_len, N, modes, tfc_params, fb_params
                    )
                    const_mask = modes == MODE_TFC_CONST
                    lin_mask = modes == MODE_TFC_LINEAR
                    matrix_mask = modes == MODE_TFC_MATRIX
                    fb_mask = modes == MODE_FB_RAW
                    t_range = np.arange(seg_len, dtype=np.float32)
                    vals = np.zeros((seg_len, N), dtype=np.float32)
                    if np.any(const_mask):
                        vals[:, const_mask] = a_p[const_mask]
                    if np.any(lin_mask):
                        vals[:, lin_mask] = (
                            a_p[lin_mask][None, :]
                            + b_p[lin_mask][None, :] * t_range[:, None]
                        )
                    if np.any(fb_mask) and fb_idx is not None and fb_arr is not None:
                        fb_map = {pix: row for row, pix in enumerate(fb_idx)}
                        fb_pix = np.nonzero(fb_mask)[0]
                        for pix in fb_pix:
                            vals[:, pix] = fb_arr[fb_map[int(pix)]]
                    if np.any(matrix_mask) and matrix_params:
                        for m in matrix_params:
                            pix = np.asarray(m["pixel_indices"], dtype=np.int64)
                            temporal = np.asarray(
                                m["temporal"], dtype=np.float32
                            ).reshape(seg_len)
                            spatial = np.asarray(
                                m["spatial"], dtype=np.float32
                            ).reshape(pix.size)
                            mean = float(m.get("mean", 0.0))
                            vals[:, pix] = mean + np.outer(temporal, spatial)
                    flat[:, :, plane] = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
                all_frames.append(frames)
            frames_out = np.concatenate(all_frames, axis=0)
            save_video_from_rgb(frames_out, output_path, fps=fps)
    else:
        tile_size = int(header.get("tiling", 0) or 1)
        plane_dims = {
            plane: plane_shape(H, W, color_format, plane)
            for plane in (PLANE_R, PLANE_G, PLANE_B)
        }

        def reconstruct_segment(seg_idx: int) -> np.ndarray:
            seg_len = plane_data[PLANE_R][seg_idx]["length"]
            plane_frames: list[np.ndarray] = []
            for t in range(seg_len):
                reconstructed_planes = []
                for plane in (PLANE_R, PLANE_G, PLANE_B):
                    seg = plane_data[plane][seg_idx]
                    ph, pw = plane_dims[plane]
                    modes = np.asarray(seg["modes"], dtype=np.uint8).reshape(-1)
                    logical_count = modes.size
                    if logical_count != tile_grid_shape(ph, pw, tile_size)[0] * tile_grid_shape(ph, pw, tile_size)[1]:
                        raise ValueError("v3 logical mode count does not match plane tile grid")
                    values = np.zeros((logical_count,), dtype=np.float32)
                    raw_scan = np.nonzero(modes == MODE_FB_RAW)[0]
                    raw_values = np.asarray(seg.get("raw_values", []), dtype=np.uint8)
                    raw_lookup = {
                        int(idx): row for row, idx in enumerate(raw_scan.tolist())
                    } if raw_scan.size else {}
                    for logical_idx, param in seg["tfc_params"].items():
                        if param["mode"] == MODE_TFC_CONST:
                            values[int(logical_idx)] = float(param["a"])
                        elif param["mode"] == MODE_TFC_LINEAR:
                            values[int(logical_idx)] = float(param["a"]) + float(param.get("b", 0.0)) * float(t)
                    for logical_idx in raw_scan.tolist():
                        values[int(logical_idx)] = float(raw_values[raw_lookup[int(logical_idx)], t])
                    plane_img = expand_tile_values(values, ph, pw, tile_size)
                    reconstructed_planes.append(
                        np.clip(np.rint(plane_img), 0, 255).astype(np.uint8)
                    )
                plane_frames.append(
                    planes_to_rgb(
                        (
                            reconstructed_planes[0],
                            reconstructed_planes[1],
                            reconstructed_planes[2],
                        ),
                        color_format,
                        (H, W),
                    )
                )
            return np.stack(plane_frames, axis=0)

        if stream_output:
            writer = imageio.get_writer(output_path, fps=fps)
            for seg_idx in range(len(plane_data[PLANE_R])):
                seg_frames = reconstruct_segment(seg_idx)
                for frame in seg_frames:
                    writer.append_data(frame)
            writer.close()
        else:
            frames_out = np.concatenate(
                [reconstruct_segment(seg_idx) for seg_idx in range(len(plane_data[PLANE_R]))],
                axis=0,
            )
            save_video_from_rgb(frames_out, output_path, fps=fps)

    print(f"Decoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}")
