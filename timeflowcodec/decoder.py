"""RGB per-pixel decoder for TimeFlowCodec (streaming)."""
from __future__ import annotations

import struct
import warnings
from pathlib import Path

import imageio.v2 as imageio
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


def decode_tfc_to_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
    stream_output: bool = True,
) -> None:
    """Decode .tfc RGB file back into an RGB video. Streams frames to writer when stream_output=True."""

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

    # Precompute per-pixel a/b arrays for const/linear to avoid dict lookups in loop\n+    a_params = np.zeros((3, N), dtype=np.float32)\n+    b_params = np.zeros((3, N), dtype=np.float32)\n+    mode_arrays = np.zeros((3, N), dtype=np.uint8)\n+    fallback_arrays = {}\n+\n+    for plane in (PLANE_R, PLANE_G, PLANE_B):\n+        modes = plane_modes[plane]\n+        tfc_params, fb_params = plane_params[plane]\n+        mode_arrays[plane] = modes\n+        for idx, param in tfc_params.items():\n+            a_params[plane, int(idx)] = float(param[\"a\"])\n+            b_params[plane, int(idx)] = float(param.get(\"b\", 0.0))\n+        if fb_params:\n+            # stack fallback for fast lookup\n+            fb_arr = np.zeros((len(fb_params), T), dtype=np.uint8)\n+            fb_indices = np.array(sorted(fb_params.keys()), dtype=np.int64)\n+            for row, pix in enumerate(fb_indices):\n+                fb_arr[row] = fb_params[int(pix)]\n+            fallback_arrays[plane] = (fb_indices, fb_arr)\n+\n+    if stream_output:\n+        writer = imageio.get_writer(output_path, fps=fps)\n+        t_range = np.arange(T, dtype=np.float32)\n+        flat_frame = np.empty((N, 3), dtype=np.float32)\n+        for t in range(T):\n+            for plane in (PLANE_R, PLANE_G, PLANE_B):\n+                modes = mode_arrays[plane]\n+                flat_vals = np.empty((N,), dtype=np.float32)\n+                const_mask = modes == MODE_TFC_CONST\n+                lin_mask = modes == MODE_TFC_LINEAR\n+                fb_mask = modes == MODE_FB_RAW\n+                if np.any(const_mask):\n+                    flat_vals[const_mask] = a_params[plane, const_mask]\n+                if np.any(lin_mask):\n+                    flat_vals[lin_mask] = a_params[plane, lin_mask] + b_params[plane, lin_mask] * t_range[t]\n+                if np.any(fb_mask):\n+                    fb_idx, fb_arr = fallback_arrays.get(plane, (np.array([], dtype=np.int64), np.empty((0, T), dtype=np.uint8)))\n+                    if fb_arr.size:\n+                        # map fb_idx positions\n+                        fb_map = {pix: row for row, pix in enumerate(fb_idx)}\n+                        for pix in np.nonzero(fb_mask)[0]:\n+                            flat_vals[pix] = fb_arr[fb_map[int(pix)], t]\n+                flat_frame[:, plane] = flat_vals\n+            frame_u8 = np.clip(np.rint(flat_frame.reshape(H, W, 3)), 0, 255).astype(np.uint8)\n+            writer.append_data(frame_u8)\n+        writer.close()\n+    else:\n+        frames = np.zeros((T, H, W, 3), dtype=np.uint8)\n+        flat = frames.reshape(T, W * H, 3)\n+        t_range = np.arange(T, dtype=np.float32)\n+        for plane in (PLANE_R, PLANE_G, PLANE_B):\n+            modes = mode_arrays[plane]\n+            const_mask = modes == MODE_TFC_CONST\n+            lin_mask = modes == MODE_TFC_LINEAR\n+            fb_mask = modes == MODE_FB_RAW\n+            flat_vals = np.zeros((T, N), dtype=np.float32)\n+            if np.any(const_mask):\n+                flat_vals[:, const_mask] = a_params[plane, const_mask]\n+            if np.any(lin_mask):\n+                flat_vals[:, lin_mask] = a_params[plane, lin_mask][None, :] + b_params[plane, lin_mask][None, :] * t_range[:, None]\n+            if np.any(fb_mask):\n+                fb_idx, fb_arr = fallback_arrays.get(plane, (np.array([], dtype=np.int64), np.empty((0, T), dtype=np.uint8)))\n+                if fb_arr.size:\n+                    fb_map = {pix: row for row, pix in enumerate(fb_idx)}\n+                    for pix in np.nonzero(fb_mask)[0]:\n+                        flat_vals[:, pix] = fb_arr[fb_map[int(pix)]]\n+            flat[:, :, plane] = np.clip(np.rint(flat_vals), 0, 255).astype(np.uint8)\n+        # for tests / legacy\n+        import timeflowcodec.utils as utils\n+        utils.save_video_from_rgb(frames, output_path, fps=fps)\n+\n+    print(f\"Decoded {input_path} -> {output_path}. Frames={T}, Size={H}x{W}\")\n*** End Patch
