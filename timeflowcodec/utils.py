"""Utility helpers for TimeFlowCodec color conversion and I/O."""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np

from .constants import COLOR_FORMAT_RGB, COLOR_FORMAT_YUV420, COLOR_FORMAT_YUV444


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        # Grayscale -> replicate to RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.shape[2] == 3:
            return arr.astype(np.uint8)
    raise ValueError(f"Unsupported frame shape for RGB conversion: {arr.shape}")


def rgb_to_yuv444(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB to uint8 YUV444 using BT.601-style coefficients."""
    rgb = _ensure_rgb(frame).astype(np.float32, copy=False)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return (
        np.clip(np.rint(y), 0, 255).astype(np.uint8),
        np.clip(np.rint(u), 0, 255).astype(np.uint8),
        np.clip(np.rint(v), 0, 255).astype(np.uint8),
    )


def yuv444_to_rgb(
    y: np.ndarray, u: np.ndarray, v: np.ndarray, out_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """Convert uint8 YUV444-like planes to uint8 RGB."""
    if out_hw is not None and (u.shape != out_hw or v.shape != out_hw or y.shape != out_hw):
        raise ValueError("out_hw does not match YUV444 plane shapes")
    y_f = y.astype(np.float32, copy=False)
    u_f = u.astype(np.float32, copy=False) - 128.0
    v_f = v.astype(np.float32, copy=False) - 128.0
    r = y_f + 1.402 * v_f
    g = y_f - 0.344136 * u_f - 0.714136 * v_f
    b = y_f + 1.772 * u_f
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def downsample_420(plane: np.ndarray) -> np.ndarray:
    """Average-pool a plane into 4:2:0 chroma resolution."""
    h, w = plane.shape
    h2 = (h + 1) // 2
    w2 = (w + 1) // 2
    padded = np.pad(
        plane.astype(np.uint16, copy=False),
        ((0, h2 * 2 - h), (0, w2 * 2 - w)),
        mode="edge",
    )
    return (
        padded.reshape(h2, 2, w2, 2).mean(axis=(1, 3)).round().astype(np.uint8)
    )


def upsample_420(plane: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample a 4:2:0 plane back to full resolution."""
    h, w = out_hw
    up = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)
    return up[:h, :w]


def rgb_to_planes(frame: np.ndarray, color_format: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an RGB frame to three codec planes for the selected color format."""
    rgb = _ensure_rgb(frame)
    if color_format == COLOR_FORMAT_RGB:
        return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y, u, v = rgb_to_yuv444(rgb)
    if color_format == COLOR_FORMAT_YUV444:
        return y, u, v
    if color_format == COLOR_FORMAT_YUV420:
        return y, downsample_420(u), downsample_420(v)
    raise ValueError(f"Unsupported color format: {color_format}")


def planes_to_rgb(
    planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    color_format: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Convert codec planes back to uint8 RGB."""
    p0, p1, p2 = planes
    if color_format == COLOR_FORMAT_RGB:
        return np.stack([p0, p1, p2], axis=-1).astype(np.uint8, copy=False)
    if color_format == COLOR_FORMAT_YUV444:
        return yuv444_to_rgb(p0, p1, p2, out_hw=out_hw)
    if color_format == COLOR_FORMAT_YUV420:
        u = upsample_420(p1, out_hw)
        v = upsample_420(p2, out_hw)
        return yuv444_to_rgb(p0, u, v, out_hw=out_hw)
    raise ValueError(f"Unsupported color format: {color_format}")


def plane_shape(height: int, width: int, color_format: int, plane: int) -> tuple[int, int]:
    """Return logical plane shape for the selected colorspace."""
    if color_format in {COLOR_FORMAT_RGB, COLOR_FORMAT_YUV444} or plane == 0:
        return height, width
    if color_format == COLOR_FORMAT_YUV420:
        return (height + 1) // 2, (width + 1) // 2
    raise ValueError(f"Unsupported color format: {color_format}")


def tile_grid_shape(height: int, width: int, tile: int) -> tuple[int, int]:
    """Return tile-grid dimensions for a plane."""
    return (height + tile - 1) // tile, (width + tile - 1) // tile


def tile_pixel_counts(height: int, width: int, tile: int) -> np.ndarray:
    """Count samples per tile for a plane shape."""
    tiles_y, tiles_x = tile_grid_shape(height, width, tile)
    counts = np.zeros((tiles_y, tiles_x), dtype=np.int64)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile
            x0 = tx * tile
            counts[ty, tx] = min(tile, height - y0) * min(tile, width - x0)
    return counts.reshape(-1)


def tile_reduce_sum(plane: np.ndarray, tile: int) -> np.ndarray:
    """Sum a 2D plane into tile cells without scatter operations."""
    height, width = plane.shape
    tiles_y, tiles_x = tile_grid_shape(height, width, tile)
    padded = np.pad(
        plane.astype(np.float64, copy=False),
        ((0, tiles_y * tile - height), (0, tiles_x * tile - width)),
        mode="constant",
    )
    return padded.reshape(tiles_y, tile, tiles_x, tile).sum(axis=(1, 3)).reshape(-1)


def expand_tile_values(
    values: np.ndarray, height: int, width: int, tile: int
) -> np.ndarray:
    """Expand a tile-grid vector back to a full-resolution 2D plane."""
    tiles_y, tiles_x = tile_grid_shape(height, width, tile)
    grid = np.asarray(values).reshape(tiles_y, tiles_x)
    up = np.repeat(np.repeat(grid, tile, axis=0), tile, axis=1)
    return up[:height, :width]


def load_video_rgb(path: str, max_frames: int | None = None) -> np.ndarray:
    """
    Load a video file as uint8 RGB frames with shape (T, H, W, 3).
    """
    try:
        reader = imageio.get_reader(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not open video. Ensure ffmpeg plugin is available. "
            "Dependency imageio[ffmpeg] should auto-install a static ffmpeg binary."
        ) from exc
    frames = []
    for idx, frame in enumerate(reader):
        if max_frames is not None and idx >= max_frames:
            break
        frames.append(_ensure_rgb(frame))
    reader.close()
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    arr = np.stack(frames, axis=0)
    return arr.astype(np.uint8)


def save_video_from_rgb(frames: np.ndarray, path: str, fps: int = 30) -> None:
    """
    Save uint8 RGB frames with shape (T, H, W, 3) to a video file.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error between two arrays."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))
